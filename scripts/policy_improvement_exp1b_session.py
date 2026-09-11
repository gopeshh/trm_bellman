#!/usr/bin/env fbpython
"""Experiment 1B train-only production session: ordering and access contract.

This module owns the properties the reduced-study training session has to
guarantee, and owns them in executable form:

1. **The run is one of the eight registered rows, and nothing else.** A session
   is built from an :class:`Exp1bTrainingRegistration`, an opaque value that
   only :func:`register_exp1b_training` can mint, and only from a protocol and
   registry that both validate and that cite each other by canonical digest. The
   raw registry row never leaves this module: a caller cannot hand-assemble a
   mapping with the right keys, and cannot reach in and read one out either.
2. **The effective configuration is the registered one.** The registration
   carries the reduced-study effective-config identity recomputed from the
   protocol's own inputs, and the trainer factory receives it. A run whose
   hyperparameters were merged from some other template cannot present a
   matching digest.
3. **Exactly one authenticated train load, before model construction.** The
   session performs the load itself: the registered dataset root, split
   ``train``, exactly 1,024 records, exactly once. The loader must return
   metadata -- resolved split, record count, ordered-record SHA-256, dataset and
   split manifest digests -- and every field must equal what the protocol
   registers. A loader that returns the right *count* from the wrong directory,
   or the right directory in the wrong order, is refused before the model
   exists. Refusals are recorded, so a test can assert validation was never
   *attempted*, not merely that it failed.
4. **Restore before derivation.** The admitted base policy is applied to the
   model *before* ``build_trainer`` runs. ``build_trainer`` copies weights into
   ``policy_model_old``, ``policy_model_candidate``, and ``target_model`` and
   builds optimizers over them, so restoring afterwards would leave the frozen
   base and its snapshots disagreeing. The ordering is enforced by a recorded
   call log, not by comment or convention.

Why the Torch objects are injected
----------------------------------
``policy_improvement_full_backend`` imports Torch at module scope. Putting the
ordering and access contract here, with the model factory, the restore callable,
and the trainer factory supplied by the caller, means the contract itself is
plain Python and is executed by the test suite on every run -- including in
environments without Torch. ``scripts/policy_improvement_exp1b_runtime.py`` is
the production caller and passes the real callables; nothing about the contract
changes between the two, because the contract *is* this module.

A missing base artifact is a hard error
---------------------------------------
Unlike the v2 full-run path, there is no ``base_policy is None ->
initialization_kind='random'`` fallback here. Experiment 1B requires the
admitted Experiment 0 artifact; its absence refuses the session.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.policy_improvement_exp1b_schema import (
    Exp1bSchemaError,
    METHOD_ID,
    REGISTERED_SEEDS,
    TERMINAL_ENVIRONMENT_INTERACTIONS,
    TRAINING_SPLIT,
    exp1b_document_sha256,
    validate_exp1b_protocol,
    validate_exp1b_registry,
)

__all__ = [
    "Exp1bSessionError",
    "Exp1bTrainOnlySession",
    "Exp1bTrainingRegistration",
    "LoadedTrainSplit",
    "TrainOnlyDatasetGuard",
    "build_exp1b_training_session",
    "register_exp1b_training",
]

_TRAIN_SPLIT = TRAINING_SPLIT

#: The ordered call sequence a conforming session must produce.
#:
#: ``seed_applied`` is first and is not decorative. Dataset sampling, model
#: initialization, environment construction, and trainer construction all draw
#: from the process RNG, so a seed applied after any of them leaves that step
#: determined by ambient state and the eight registered seeds label runs they do
#: not control.
_REQUIRED_ORDER = (
    "seed_applied",
    "train_split_loaded",
    "model_constructed",
    "base_policy_restored",
    "trainer_built",
)

#: Module-private construction token. An ``Exp1bTrainingRegistration`` that was
#: not minted by :func:`register_exp1b_training` cannot carry it, so a forged
#: instance is rejected by identity rather than by field inspection.
_REGISTRATION = object()

_ROW_FIELDS = frozenset(
    {
        "run_id",
        "seed",
        "seed_position",
        "method_id",
        "n",
        "K",
        "alpha",
        "terminal_environment_interactions",
        "training_split",
        "evaluation_population",
        "evaluation_split",
        "scientific_selection",
        "paper_evidence_eligible",
    }
)


class Exp1bSessionError(RuntimeError):
    """Raised when the reduced-study session contract would be violated."""


@dataclass(frozen=True)
class LoadedTrainSplit:
    """What a conforming Experiment 1B train loader must return.

    The dataset object itself is opaque to this module -- it is a Torch dataset
    in production and a synthetic stand-in under test. The identity fields are
    not opaque: they are exactly what the protocol registers, and the guard
    compares all of them.
    """

    dataset: Any
    dataset_root: Any
    split: str
    count: int
    ordered_record_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str


@dataclass(frozen=True)
class Exp1bTrainingRegistration:
    """One registered Experiment 1B run, validated end to end.

    Opaque by construction. The registry row is held privately; callers read the
    scalar accessors below. That is the point: ``build_exp1b_training_session``
    cannot be handed a dict that merely looks like a row, and nothing downstream
    can widen the run's identity by editing one.
    """

    _issued_by: Any
    _row: Mapping[str, Any]
    project_root: Path
    dataset_root: Path
    protocol_sha256: str
    registry_sha256: str
    effective_config_sha256: str
    dataset_name: str
    training_population_id: str
    ordered_record_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    train_record_count: int

    def __post_init__(self) -> None:
        if self._issued_by is not _REGISTRATION:
            raise Exp1bSessionError(
                "Experiment 1B training registration must be minted by "
                "register_exp1b_training."
            )

    @property
    def run_id(self) -> str:
        return str(self._row["run_id"])

    @property
    def seed(self) -> int:
        return int(self._row["seed"])

    @property
    def seed_position(self) -> int:
        return int(self._row["seed_position"])

    @property
    def method_id(self) -> str:
        return str(self._row["method_id"])

    @property
    def terminal_environment_interactions(self) -> int:
        return int(self._row["terminal_environment_interactions"])

    @property
    def training_split(self) -> str:
        return str(self._row["training_split"])


def register_exp1b_training(
    *,
    protocol: Mapping[str, object],
    registry: Mapping[str, object],
    project_root: Path,
    seed: int,
) -> Exp1bTrainingRegistration:
    """Validate the study documents and mint the registration for one seed.

    Both documents are validated in full, and the registry has to cite the
    protocol by the digest the protocol actually hashes to -- a registry lifted
    from an older protocol revision is refused here, not at first use.
    """

    checked_protocol = validate_exp1b_protocol(protocol)
    protocol_sha256 = exp1b_document_sha256(protocol)
    checked_registry = validate_exp1b_registry(
        registry, protocol_sha256=protocol_sha256
    )
    registry_sha256 = exp1b_document_sha256(registry)

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Exp1bSessionError("Experiment 1B seed must be an integer.")
    rows = checked_registry["rows"]
    assert isinstance(rows, list)
    matches = [row for row in rows if row["seed"] == seed]
    if len(matches) != 1:
        raise Exp1bSessionError(
            f"Seed {seed!r} is not exactly one registered Experiment 1B row."
        )
    row = _validated_row(matches[0])

    training = checked_protocol["training_population"]
    assert isinstance(training, Mapping)
    effective = checked_protocol["effective_config"]
    assert isinstance(effective, Mapping)

    root = Path(project_root)
    if not root.is_absolute():
        raise Exp1bSessionError("Experiment 1B project root must be absolute.")
    dataset_root = (root / str(training["dataset_root"])).resolve()
    try:
        dataset_root.relative_to(root.resolve())
    except ValueError as exc:
        raise Exp1bSessionError(
            "Experiment 1B dataset root escaped the project root."
        ) from exc

    return Exp1bTrainingRegistration(
        _issued_by=_REGISTRATION,
        _row=dict(row),
        project_root=root,
        dataset_root=dataset_root,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        effective_config_sha256=str(effective["effective_config_sha256"]),
        dataset_name=str(training["dataset_name"]),
        training_population_id=str(training["population_id"]),
        ordered_record_sha256=str(training["ordered_record_sha256"]),
        dataset_manifest_sha256=str(training["dataset_manifest_sha256"]),
        split_manifest_sha256=str(training["split_manifest_sha256"]),
        train_record_count=int(training["count"]),
    )


@dataclass
class TrainOnlyDatasetGuard:
    """A dataset loader seam that can only ever resolve the registered train split.

    Enforces the root, the split, the record count, **and** the identity of what
    came back. A load of one record is as wrong as a load of the validation
    split, because the registered study trains on all 1,024; and 1,024 records
    from an unregistered directory is as wrong as either, because the ordered
    record digest will not match. Every call is recorded, so a test can assert
    that validation was never attempted rather than merely refused.
    """

    registration: Exp1bTrainingRegistration
    loader: Callable[[Any, str, int], LoadedTrainSplit]
    resolved: list[tuple[Any, str, int]] = field(default_factory=list)
    refused: list[tuple[Any, str, int]] = field(default_factory=list)
    authenticated: list[LoadedTrainSplit] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.registration, Exp1bTrainingRegistration):
            raise Exp1bSessionError(
                "Experiment 1B loader guard requires a minted registration."
            )

    @property
    def dataset_root(self) -> Path:
        return self.registration.dataset_root

    @property
    def expected_count(self) -> int:
        return self.registration.train_record_count

    def load(self, root: Any, split: str, count: int) -> LoadedTrainSplit:
        if root != self.dataset_root or split != _TRAIN_SPLIT:
            self.refused.append((root, split, count))
            raise Exp1bSessionError(
                "Experiment 1B dataset loader received a non-train path."
            )
        if isinstance(count, bool) or not isinstance(count, int):
            self.refused.append((root, split, count))
            raise Exp1bSessionError("Experiment 1B train record count must be an integer.")
        if count != self.expected_count:
            self.refused.append((root, split, count))
            raise Exp1bSessionError(
                f"Experiment 1B trains on exactly {self.expected_count} records; "
                f"the loader was asked for {count}."
            )
        loaded = self.loader(root, split, count)
        try:
            self._authenticate(loaded, root=root, count=count)
        except Exp1bSessionError:
            self.refused.append((root, split, count))
            raise
        self.resolved.append((root, split, count))
        self.authenticated.append(loaded)
        return loaded

    def _authenticate(self, loaded: object, *, root: Any, count: int) -> None:
        """Prove the returned split is the registered one, not merely the right size."""

        if not isinstance(loaded, LoadedTrainSplit):
            raise Exp1bSessionError(
                "Experiment 1B train loader returned no authenticated split metadata."
            )
        registration = self.registration
        if loaded.dataset is None:
            raise Exp1bSessionError("Experiment 1B train loader returned no dataset.")
        if loaded.split != _TRAIN_SPLIT or loaded.dataset_root != root:
            raise Exp1bSessionError(
                "Experiment 1B train loader resolved a different split or root."
            )
        if (
            isinstance(loaded.count, bool)
            or not isinstance(loaded.count, int)
            or loaded.count != count
            or loaded.count != registration.train_record_count
        ):
            raise Exp1bSessionError(
                "Experiment 1B train loader resolved the wrong record count."
            )
        for name, expected in (
            ("ordered_record_sha256", registration.ordered_record_sha256),
            ("dataset_manifest_sha256", registration.dataset_manifest_sha256),
            ("split_manifest_sha256", registration.split_manifest_sha256),
        ):
            observed = getattr(loaded, name)
            if not isinstance(observed, str) or observed != expected:
                raise Exp1bSessionError(
                    f"Experiment 1B train loader {name} is not the registered value."
                )

    @property
    def resolved_splits(self) -> tuple[str, ...]:
        return tuple(split for _, split, _ in self.resolved)

    @property
    def touched_evaluation_data(self) -> bool:
        return any(split != _TRAIN_SPLIT for split in self.resolved_splits)


@dataclass(frozen=True)
class Exp1bTrainOnlySession:
    """One constructed reduced-study session and its verifiable provenance."""

    run_id: str
    seed: int
    seed_position: int
    method_id: str
    terminal_environment_interactions: int
    train_record_count: int
    dataset: Any
    model: Any
    trainer: Any
    initialization_kind: str
    initialization_artifact_sha256: str
    restored_model_state_sha256: str
    call_order: tuple[str, ...]
    dataset_guard: TrainOnlyDatasetGuard
    applied_seed: int
    effective_config_sha256: str
    train_ordered_record_sha256: str
    dataset_name: str
    training_population_id: str


def _validated_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise Exp1bSessionError("Experiment 1B session requires a registry row.")
    if set(row) != set(_ROW_FIELDS):
        raise Exp1bSessionError("Experiment 1B registry row inventory differs.")
    seed = row["seed"]
    position = row["seed_position"]
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position < len(REGISTERED_SEEDS)
        or REGISTERED_SEEDS[position] != seed
    ):
        raise Exp1bSessionError(
            "Experiment 1B registry row is not a registered seed in its order."
        )
    if row["method_id"] != METHOD_ID:
        raise Exp1bSessionError("Experiment 1B registers one method only.")
    if row["terminal_environment_interactions"] != TERMINAL_ENVIRONMENT_INTERACTIONS:
        raise Exp1bSessionError(
            "Experiment 1B trains to exactly "
            f"{TERMINAL_ENVIRONMENT_INTERACTIONS} environment interactions."
        )
    if row["training_split"] != _TRAIN_SPLIT:
        raise Exp1bSessionError("Experiment 1B trains on the train split only.")
    return row


def build_exp1b_training_session(
    *,
    registration: Exp1bTrainingRegistration,
    dataset_guard: TrainOnlyDatasetGuard,
    base_policy: Any,
    apply_seed: Callable[[int], None],
    model_factory: Callable[[Any], Any],
    restore_base_policy: Callable[[Any, Any], str],
    trainer_factory: Callable[[Any, Any, str], Any],
    model_state_digest: Callable[[Any], str],
) -> Exp1bTrainOnlySession:
    """Build one train-only session for one registered run.

    Order is seed -> authenticate-load -> construct -> restore -> build_trainer,
    enforced by the call log. ``apply_seed(seed)`` must set the global Python,
    NumPy, and Torch RNGs; ``restore_base_policy(model, base_policy)`` must
    return the restored model-state SHA-256; ``trainer_factory(model, dataset,
    effective_config_sha256)`` must build the trainer from the already-restored
    model under the registered effective configuration.
    """

    if not isinstance(registration, Exp1bTrainingRegistration):
        raise Exp1bSessionError(
            "Experiment 1B session requires a minted training registration."
        )
    if base_policy is None:
        # Deliberately no random-initialization fallback.
        raise Exp1bSessionError(
            "Experiment 1B requires the admitted Experiment 0 base-policy "
            "artifact; it has no random-initialization fallback."
        )
    if not isinstance(dataset_guard, TrainOnlyDatasetGuard):
        raise Exp1bSessionError("Experiment 1B requires the train-only loader guard.")
    if dataset_guard.registration is not registration:
        raise Exp1bSessionError(
            "Experiment 1B loader guard is bound to a different registration."
        )
    if dataset_guard.resolved:
        raise Exp1bSessionError(
            "Experiment 1B session requires a fresh loader guard; this one has "
            "already resolved a split."
        )

    if not callable(apply_seed):
        raise Exp1bSessionError(
            "Experiment 1B session requires the registered seeding callable."
        )

    call_order: list[str] = []

    # The registered seed determines the run, so it is applied before anything
    # that draws: before the dataset is sampled, before the model is
    # initialized, before the environment exists, before the trainer is built.
    apply_seed(registration.seed)
    call_order.append("seed_applied")

    # Exactly one authenticated load of the whole registered train split, and it
    # completes -- identity included -- before the model exists at all.
    loaded = dataset_guard.load(
        registration.dataset_root, _TRAIN_SPLIT, registration.train_record_count
    )
    call_order.append("train_split_loaded")
    if len(dataset_guard.resolved) != 1 or len(dataset_guard.authenticated) != 1:
        raise Exp1bSessionError(
            "Experiment 1B session performed more than one dataset load."
        )
    dataset = loaded.dataset

    model = model_factory(dataset)
    call_order.append("model_constructed")

    restored = restore_base_policy(model, base_policy)
    if not isinstance(restored, str) or not restored:
        raise Exp1bSessionError("Base-policy restore returned no model-state digest.")
    expected = getattr(base_policy, "model_state_sha256", None)
    if expected is not None and restored != expected:
        raise Exp1bSessionError(
            "Restored base policy differs from its registered identity."
        )
    call_order.append("base_policy_restored")

    # Verify the restore actually landed on the model the trainer will derive
    # from, before the trainer exists to derive from it.
    observed = model_state_digest(model)
    if observed != restored:
        raise Exp1bSessionError(
            "Model state at trainer construction differs from the restored base."
        )

    trainer = trainer_factory(model, dataset, registration.effective_config_sha256)
    call_order.append("trainer_built")

    if tuple(call_order) != _REQUIRED_ORDER:
        raise Exp1bSessionError(
            f"Experiment 1B session call order was {tuple(call_order)!r}; "
            f"it must be {_REQUIRED_ORDER!r}."
        )
    if dataset_guard.touched_evaluation_data:
        raise Exp1bSessionError("Experiment 1B session resolved evaluation data.")
    if len(dataset_guard.resolved) != 1:
        raise Exp1bSessionError(
            "Experiment 1B session performed more than one dataset load."
        )

    artifact_sha256 = getattr(base_policy, "checkpoint_sha256", None)
    if not isinstance(artifact_sha256, str) or not artifact_sha256:
        raise Exp1bSessionError("Admitted base policy carries no checkpoint digest.")
    initialization_kind = getattr(
        base_policy, "initialization_kind", "train_only_pretrained_base_policy"
    )
    return Exp1bTrainOnlySession(
        run_id=registration.run_id,
        seed=registration.seed,
        seed_position=registration.seed_position,
        method_id=registration.method_id,
        terminal_environment_interactions=(
            registration.terminal_environment_interactions
        ),
        train_record_count=registration.train_record_count,
        dataset=dataset,
        model=model,
        trainer=trainer,
        initialization_kind=str(initialization_kind),
        initialization_artifact_sha256=artifact_sha256,
        restored_model_state_sha256=restored,
        call_order=tuple(call_order),
        dataset_guard=dataset_guard,
        applied_seed=registration.seed,
        effective_config_sha256=registration.effective_config_sha256,
        train_ordered_record_sha256=loaded.ordered_record_sha256,
        dataset_name=registration.dataset_name,
        training_population_id=registration.training_population_id,
    )


def assert_no_evaluation_access(
    guard: TrainOnlyDatasetGuard,
    *,
    label: str = "Experiment 1B training",
) -> None:
    """Raise unless the guard resolved the registered train split only."""

    if not isinstance(guard, TrainOnlyDatasetGuard):
        raise Exp1bSessionError(f"{label} requires the train-only loader guard.")
    if guard.touched_evaluation_data:
        raise Exp1bSessionError(f"{label} resolved evaluation data.")
    if len(guard.resolved) != 1:
        raise Exp1bSessionError(f"{label} did not perform exactly one train load.")


def registered_run_ids(registry: Mapping[str, object]) -> tuple[str, ...]:
    """Ordered run IDs from a validated Experiment 1B registry."""

    checked = validate_exp1b_registry(registry)
    rows = checked["rows"]
    if not isinstance(rows, Sequence):
        raise Exp1bSchemaError("Experiment 1B registry has no rows.")
    return tuple(str(row["run_id"]) for row in rows)
