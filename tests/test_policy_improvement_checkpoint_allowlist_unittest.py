#!/usr/bin/env fbpython
"""Adversarial tests for data-only checkpoint deserialization."""

from __future__ import annotations

import io
import os
import pickle
import pickletools
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from policy_improvement_checkpoint_allowlist import (
    ADDITIONAL_SAFE_GLOBALS,
    safe_checkpoint_globals,
    FIRST_PARTY_SAFE_GLOBALS,
    NUMPY_SAFE_GLOBALS,
    HISTORICAL_CHECKPOINT_GLOBALS,
    load_data_only_checkpoint,
    UnsafeCheckpointPayloadError,
)
from rl.replay import ReplayLatent, Transition


MARKER_PATH: list[str] = []


def _write_marker(marker: str) -> str:
    """Side effect a hostile reducer would achieve. Never legitimately called."""

    Path(marker).write_bytes(b"the reducer executed")
    return marker


class _MarkerReducer:
    """A payload whose ``__reduce__`` writes a marker file when unpickled."""

    def __init__(self, marker: Path) -> None:
        self._marker = str(marker)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (_write_marker, (self._marker,))


def _historical_shaped_payload() -> dict[str, object]:
    """Mirror the object graph a real Stage 0 checkpoint contains.

    Covers every global the opcode scan found in retained evidence: ordered
    state dicts, several storage dtypes, a NumPy RNG state, and the two replay
    dataclasses.
    """

    latent = ReplayLatent(z_H=torch.ones(2, 2), z_L=torch.zeros(2, 2))
    transition = Transition(
        x={"grid": torch.arange(4, dtype=torch.int32)},
        y=torch.arange(4, dtype=torch.long),
        action=torch.tensor([1]),
        reward=torch.tensor([0.5]),
        x_next={"grid": torch.arange(4, dtype=torch.int32)},
        y_next=torch.arange(4, dtype=torch.long),
        done=torch.tensor([True]),
        episode_id=0,
        timestep=0,
        latent=latent,
        next_latent=None,
        behavior_log_prob=torch.tensor([-0.1]),
        terminal_reason=None,
    )
    model_state = torch.nn.Linear(2, 2).state_dict()
    return {
        "checkpoint_schema_version": 5,
        "model_state_dict": model_state,
        "rollout_buffer": {"transitions": [transition]},
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "bool_flags": torch.tensor([True, False]),
        },
        "progress": {"env_steps": 32},
    }


class CheckpointAllowlistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.marker = self.root / "reducer-marker"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _save(self, value: object) -> Path:
        path = self.root / "checkpoint.pt"
        torch.save(value, path)
        return path

    @staticmethod
    def _pickle_globals(path: Path) -> set[str]:
        """Enumerate global references without executing the pickle."""

        names: set[str] = set()
        with zipfile.ZipFile(path) as archive:
            for entry in archive.namelist():
                if not entry.endswith("data.pkl"):
                    continue
                for opcode, argument, _ in pickletools.genops(archive.read(entry)):
                    if opcode.name == "GLOBAL":
                        names.add(str(argument).replace(" ", "."))
        return names

    # ---- the allowlist is minimal ---------------------------------------

    def test_the_added_allowlist_is_exactly_the_documented_minimum(self) -> None:
        self.assertEqual(
            {item.__name__ for item in FIRST_PARTY_SAFE_GLOBALS},
            {"ReplayLatent", "Transition"},
        )
        # The NumPy additions are justified by a property, not by an
        # enumeration that could drift: every one is either NumPy's array
        # reconstruction hook or an inert data descriptor type.
        reconstruct = [
            item for item in NUMPY_SAFE_GLOBALS if item.__name__ == "_reconstruct"
        ]
        self.assertEqual(len(reconstruct), 1)
        for item in NUMPY_SAFE_GLOBALS:
            with self.subTest(entry=getattr(item, "__name__", item)):
                if item in reconstruct:
                    continue
                self.assertTrue(isinstance(item, type), item)
                self.assertTrue(
                    item is np.ndarray or issubclass(item, np.dtype),
                    f"{item!r} is neither ndarray nor a dtype descriptor",
                )
                self.assertNotEqual(item.__name__, "ObjectDType")
        self.assertEqual(
            set(ADDITIONAL_SAFE_GLOBALS),
            set(FIRST_PARTY_SAFE_GLOBALS) | set(NUMPY_SAFE_GLOBALS),
        )
        for item in FIRST_PARTY_SAFE_GLOBALS:
            with self.subTest(cls=item.__name__):
                # A permitted class must not carry its own execution primitive.
                for dunder in ("__reduce__", "__reduce_ex__", "__setstate__"):
                    self.assertIs(
                        getattr(item, dunder, None),
                        getattr(object, dunder, None),
                        f"{item.__name__}.{dunder} is overridden",
                    )
                self.assertIs(item.__init__.__module__, item.__module__)

    def test_added_globals_are_load_bearing_and_the_scan_is_complete(self) -> None:
        """The additions are required, and the recorded scan covers the fixture.

        Named for what it actually asserts: without the additions the load
        fails, and the fixture names no global outside the recorded scan.  It
        does not assert anything about what Torch permits by default.
        """

        first_party = {"rl.replay.ReplayLatent", "rl.replay.Transition"}
        self.assertTrue(first_party.issubset(set(HISTORICAL_CHECKPOINT_GLOBALS)))
        payload = _historical_shaped_payload()
        path = self._save(payload)
        observed = self._pickle_globals(path)
        # The synthetic payload must actually exercise the historical shape.
        self.assertTrue(
            first_party.issubset(observed),
            f"fixture does not contain the replay dataclasses: {sorted(observed)}",
        )
        unexpected = observed - set(HISTORICAL_CHECKPOINT_GLOBALS)
        self.assertEqual(
            unexpected,
            set(),
            f"fixture names globals absent from the documented scan: {unexpected}",
        )
        # Without the additions the load must fail.  This shows the additions
        # are collectively load-bearing; it does not isolate which one, because
        # the fixture also carries NumPy RNG state that this Torch refuses by
        # default.  The isolating assertion is below.
        with path.open("rb") as handle:
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(handle, map_location="cpu", weights_only=True)

        # Isolate the first-party entries: grant only the NumPy set and the
        # load must still fail, which is attributable to rl.replay alone.
        with torch.serialization.safe_globals(list(NUMPY_SAFE_GLOBALS)):
            with path.open("rb") as handle:
                with self.assertRaises(pickle.UnpicklingError):
                    torch.load(handle, map_location="cpu", weights_only=True)

        # And symmetrically: grant only the first-party set and it still fails,
        # which is attributable to the NumPy entries alone.
        with torch.serialization.safe_globals(list(FIRST_PARTY_SAFE_GLOBALS)):
            with path.open("rb") as handle:
                with self.assertRaises(pickle.UnpicklingError):
                    torch.load(handle, map_location="cpu", weights_only=True)

    # ---- historical evidence still loads ---------------------------------

    def test_historical_shaped_checkpoint_round_trips(self) -> None:
        payload = _historical_shaped_payload()
        path = self._save(payload)
        with path.open("rb") as handle:
            restored = load_data_only_checkpoint(handle)
        self.assertEqual(restored["checkpoint_schema_version"], 5)
        transition = restored["rollout_buffer"]["transitions"][0]
        self.assertIsInstance(transition, Transition)
        self.assertIsInstance(transition.latent, ReplayLatent)
        self.assertTrue(torch.equal(transition.latent.z_H, torch.ones(2, 2)))
        self.assertEqual(
            sorted(restored["model_state_dict"]),
            sorted(payload["model_state_dict"]),
        )
        self.assertEqual(restored["rng_state"]["numpy"][0], "MT19937")

    # ---- hostile payloads cannot execute ---------------------------------

    def test_marker_reducer_is_never_executed(self) -> None:
        path = self._save({"payload": _MarkerReducer(self.marker)})
        self.assertIn("REDUCE", {op.name for op, _, _ in self._opcodes(path)})
        with path.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertFalse(self.marker.exists())

        # Positive control: the same bytes DO execute under the old policy, so
        # the assertion above is not passing for an unrelated reason.
        with path.open("rb") as handle:
            torch.load(handle, map_location="cpu", weights_only=False)
        self.assertTrue(self.marker.exists())

    def test_hostile_reducer_hidden_inside_a_historical_shape(self) -> None:
        """The realistic attack: valid-looking checkpoint, one hostile field."""

        payload = _historical_shaped_payload()
        payload["rollout_buffer"]["transitions"].append(_MarkerReducer(self.marker))
        path = self._save(payload)
        with path.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertFalse(self.marker.exists())

    def test_classic_os_system_reducer_is_rejected(self) -> None:
        marker = self.root / "os-marker"

        class _SystemReducer:
            def __reduce__(self) -> tuple[object, tuple[object, ...]]:
                import os

                return (os.system, (f"touch {marker}",))

        path = self._save({"payload": _SystemReducer()})
        with path.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertFalse(marker.exists())

    def test_allowlisted_class_in_reduce_position_calls_only_itself(self) -> None:
        """Permitting Transition must not permit calling anything else.

        Put an allowlisted class in the REDUCE position with attacker-chosen
        arguments.  It must construct that class and nothing more: no attribute
        of it may become a second callable the pickle can aim.
        """

        class _TransitionFactory:
            def __reduce__(self) -> tuple[object, tuple[object, ...]]:
                return (ReplayLatent, (1, 2))

        path = self._save({"payload": _TransitionFactory()})
        with path.open("rb") as handle:
            restored = load_data_only_checkpoint(handle)
        latent = restored["payload"]
        self.assertIsInstance(latent, ReplayLatent)
        self.assertEqual((latent.z_H, latent.z_L), (1, 2))
        self.assertFalse(self.marker.exists())

        # The allowlisted class does not launder a hostile callable placed in
        # one of its fields: that callable is still an unpermitted global.
        hostile = self._save(
            {"payload": ReplayLatent(z_H=_MarkerReducer(self.marker), z_L=0)}
        )
        with hostile.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertFalse(self.marker.exists())

    def test_loader_refuses_a_raw_pickle_rather_than_falling_back(self) -> None:
        payload = pickle.dumps({"payload": _MarkerReducer(self.marker)}, protocol=2)
        with self.assertRaises(UnsafeCheckpointPayloadError):
            load_data_only_checkpoint(io.BytesIO(payload))
        self.assertFalse(self.marker.exists())

    @staticmethod
    def _user_allowlist_state() -> list[str]:
        """Snapshot Torch's process-wide user allowlist."""

        getter = getattr(torch.serialization, "get_safe_globals", None)
        if getter is None:  # pragma: no cover - pinned Torch provides it
            raise unittest.SkipTest("Torch exposes no user-allowlist getter")
        return sorted(
            getattr(item, "__name__", repr(item)) for item in getter()
        )

    def test_safe_globals_state_is_identical_before_and_after_success(self) -> None:
        """Direct state check, not merely an observable side effect."""

        before = self._user_allowlist_state()
        path = self._save(_historical_shaped_payload())
        with path.open("rb") as handle:
            self.assertIsInstance(load_data_only_checkpoint(handle), dict)
        self.assertEqual(self._user_allowlist_state(), before)

        # And the grant really was in force during the load, so the equality
        # above is a restoration rather than the grant never happening.
        inside: list[list[str]] = []
        with safe_checkpoint_globals():
            inside.append(self._user_allowlist_state())
        self.assertEqual(self._user_allowlist_state(), before)
        self.assertNotEqual(inside[0], before)
        self.assertTrue(
            {"Transition", "ReplayLatent"}.issubset(set(inside[0])),
            inside[0],
        )

    def test_safe_globals_state_is_identical_after_an_exception(self) -> None:
        before = self._user_allowlist_state()
        hostile = self._save({"payload": _MarkerReducer(self.marker)})
        with hostile.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertEqual(self._user_allowlist_state(), before)

        # An exception raised by the caller inside the scope must unwind too.
        class _Boom(Exception):
            pass

        with self.assertRaises(_Boom):
            with safe_checkpoint_globals():
                raise _Boom()
        self.assertEqual(self._user_allowlist_state(), before)
        self.assertFalse(self.marker.exists())

    def test_effective_allowlist_excludes_the_ambient_torch_baseline(self) -> None:
        """`import torch` installs a large ambient set; it must not apply."""

        import torch.serialization as serialization

        baseline = serialization.get_safe_globals()
        self.assertGreater(
            len(baseline),
            10,
            "expected Torch to install an ambient baseline; test is stale",
        )
        with safe_checkpoint_globals():
            during = serialization.get_safe_globals()
        self.assertEqual(
            {id(item) for item in during},
            {id(item) for item in ADDITIONAL_SAFE_GLOBALS},
            "effective allowlist is not exactly ADDITIONAL_SAFE_GLOBALS",
        )
        self.assertEqual(len(during), len(ADDITIONAL_SAFE_GLOBALS))
        restored = serialization.get_safe_globals()
        self.assertEqual(
            sorted(map(id, restored)), sorted(map(id, baseline)), "baseline not restored"
        )

    def test_ambient_side_effect_callable_cannot_execute(self) -> None:
        """The decisive behavioural test for allowlist isolation."""

        import torch.serialization as serialization

        marker = self.root / "ambient-marker"

        class _AmbientReducer:
            def __reduce__(self) -> tuple[object, tuple[object, ...]]:
                return (_write_marker, (str(marker),))

        path = self._save({"payload": _AmbientReducer()})
        baseline = serialization.get_safe_globals()
        # Grant the hostile callable ambiently, exactly as importing a library
        # can.  The loader must still refuse it.
        serialization.add_safe_globals([_write_marker])
        try:
            self.assertIn(
                id(_write_marker),
                {id(item) for item in serialization.get_safe_globals()},
            )
            with path.open("rb") as handle:
                with self.assertRaises(UnsafeCheckpointPayloadError):
                    load_data_only_checkpoint(handle)
            self.assertFalse(marker.exists())

            # Positive control: with the ambient grant actually in force, the
            # very same bytes DO execute, so the refusal above is isolation and
            # not an unrelated rejection.
            with path.open("rb") as handle:
                torch.load(handle, map_location="cpu", weights_only=True)
            self.assertTrue(marker.exists())
        finally:
            serialization.clear_safe_globals()
            if baseline:
                serialization.add_safe_globals(list(baseline))
        self.assertEqual(
            sorted(map(id, serialization.get_safe_globals())),
            sorted(map(id, baseline)),
        )

    def test_overlapping_outer_grant_is_restored_exactly(self) -> None:
        """An outer grant that overlaps ours must survive unchanged."""

        import torch.serialization as serialization

        baseline = serialization.get_safe_globals()
        serialization.add_safe_globals([Transition])
        try:
            outer = serialization.get_safe_globals()
            self.assertIn(id(Transition), {id(item) for item in outer})
            with safe_checkpoint_globals():
                pass
            after = serialization.get_safe_globals()
            self.assertEqual(sorted(map(id, after)), sorted(map(id, outer)))
            self.assertIn(id(Transition), {id(item) for item in after})
        finally:
            serialization.clear_safe_globals()
            if baseline:
                serialization.add_safe_globals(list(baseline))

    def test_safe_globals_scope_nests_without_corrupting_state(self) -> None:
        """The audit loads several checkpoints; nesting must be clean."""

        before = self._user_allowlist_state()
        with safe_checkpoint_globals():
            outer = self._user_allowlist_state()
            with safe_checkpoint_globals():
                self.assertEqual(self._user_allowlist_state(), outer)
            self.assertEqual(self._user_allowlist_state(), outer)
        self.assertEqual(self._user_allowlist_state(), before)

    def test_allowlist_scope_does_not_leak_after_a_load(self) -> None:
        path = self._save(_historical_shaped_payload())
        with path.open("rb") as handle:
            load_data_only_checkpoint(handle)
        # Outside the context manager the replay dataclasses must be forbidden
        # again, so the process-wide unpickler is not permanently widened.
        with path.open("rb") as handle:
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(handle, map_location="cpu", weights_only=True)

    def test_allowlist_scope_is_restored_after_a_failed_load(self) -> None:
        hostile = self._save({"payload": _MarkerReducer(self.marker)})
        with hostile.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        good = self.root / "good.pt"
        torch.save(_historical_shaped_payload(), good)
        with good.open("rb") as handle:
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(handle, map_location="cpu", weights_only=True)
        with good.open("rb") as handle:
            self.assertIsInstance(load_data_only_checkpoint(handle), dict)

    def test_object_dtype_array_cannot_smuggle_a_nested_pickle(self) -> None:
        """The dangerous escape: NumPy unpickles object-array data itself.

        ``numpy.ndarray.__setstate__`` deserializes the data buffer of an
        object-dtype array with an ordinary unpickler.  If a hostile checkpoint
        can get an object-dtype ndarray built, it escapes the restricted
        unpickler entirely, regardless of ``ObjectDType`` being excluded from
        the allowlist, because ``numpy.dtype("O")`` reaches the same dtype from
        a plain string.
        """

        marker = self.root / "object-array-marker"

        class _Nested:
            def __reduce__(self) -> tuple[object, tuple[object, ...]]:
                return (_write_marker, (str(marker),))

        hostile = np.empty(1, dtype=object)
        hostile[0] = _Nested()
        path = self._save({"payload": hostile})
        with path.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertFalse(marker.exists())

        # Positive control: these bytes really do execute unrestricted.
        with path.open("rb") as handle:
            torch.load(handle, map_location="cpu", weights_only=False)
        self.assertTrue(marker.exists())

    def test_object_dtype_is_not_reachable_through_the_dtype_constructor(self) -> None:
        """Excluding ObjectDType is only meaningful if "O" is unreachable too."""

        marker = self.root / "dtype-string-marker"

        class _Nested:
            def __reduce__(self) -> tuple[object, tuple[object, ...]]:
                return (_write_marker, (str(marker),))

        hostile = np.array([_Nested()], dtype=np.dtype("O"))
        path = self._save({"payload": hostile})
        with path.open("rb") as handle:
            with self.assertRaises(UnsafeCheckpointPayloadError):
                load_data_only_checkpoint(handle)
        self.assertFalse(marker.exists())


    # ---- opt-in verification against real retained evidence -------------

    @unittest.skipUnless(
        os.environ.get("UPI_TRM_HISTORICAL_CHECKPOINT_ROOT"),
        "set UPI_TRM_HISTORICAL_CHECKPOINT_ROOT to verify real Stage 0 bytes",
    )
    def test_real_historical_checkpoints_load_data_only(self) -> None:
        """Compatibility proof against the actual retained Stage 0 evidence.

        Opt-in because it depends on machine-local evidence directories.  The
        load executes nothing by construction, so running it cannot be an
        experiment; it only answers whether historical bytes remain readable.
        """

        roots = [
            Path(item)
            for item in os.environ["UPI_TRM_HISTORICAL_CHECKPOINT_ROOT"].split(os.pathsep)
            if item
        ]
        expected = os.environ.get("UPI_TRM_HISTORICAL_CHECKPOINT_COUNT")
        self.assertIsNotNone(
            expected,
            "UPI_TRM_HISTORICAL_CHECKPOINT_COUNT is required so a mistyped or "
            "partial root cannot pass vacuously",
        )
        checkpoints = sorted(
            checkpoint for root in roots for checkpoint in root.rglob("*.pt")
        )
        self.assertTrue(checkpoints, f"no checkpoints under {roots}")
        self.assertEqual(
            len(checkpoints),
            int(str(expected)),
            f"expected {expected} checkpoints, found {len(checkpoints)}",
        )
        loaded = 0
        for checkpoint in checkpoints:
            with self.subTest(checkpoint=str(checkpoint)):
                with checkpoint.open("rb") as handle:
                    restored = load_data_only_checkpoint(handle)
                self.assertIsNotNone(restored)
                loaded += 1
        self.assertEqual(loaded, len(checkpoints))
        sys.stderr.write(
            f"data-only loaded {loaded} historical checkpoints from "
            f"{len(roots)} root(s)\n"
        )

    @staticmethod
    def _opcodes(path: Path) -> list[tuple[object, object, object]]:
        with zipfile.ZipFile(path) as archive:
            entry = next(n for n in archive.namelist() if n.endswith("data.pkl"))
            return list(pickletools.genops(archive.read(entry)))


if __name__ == "__main__":
    unittest.main()
