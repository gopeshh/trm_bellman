#!/usr/bin/env fbpython
"""Adversarial tests for the write-sealed checkpoint descriptor boundary."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from policy_improvement_sealed_evidence import (
    authenticate_sealed_checkpoint_field,
    authenticate_sealed_descriptor,
    seal_authenticated_checkpoint,
    seal_generation_checkpoint,
    SEALED_CHECKPOINT_SCHEMA_NAME,
    SealedCheckpointError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ALL_SEALS = (
    fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_WRITE
)


class SealedCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.generation = Path(self.temporary.name) / "generation"
        (self.generation / "checkpoints").mkdir(parents=True)
        self.checkpoint = self.generation / "checkpoints" / "checkpoint.pt"
        self.payload = b"authenticated checkpoint bytes"
        self.checkpoint.write_bytes(self.payload)
        self.sha256 = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _seal(self) -> int:
        return seal_authenticated_checkpoint(
            self.checkpoint,
            expected_sha256=self.sha256,
            expected_size_bytes=len(self.payload),
        )

    def test_sealed_descriptor_holds_exactly_the_authenticated_bytes(self) -> None:
        descriptor = self._seal()
        try:
            self.assertEqual(os.pread(descriptor, 4096, 0), self.payload)
            observed_sha256, observed_size = authenticate_sealed_descriptor(
                descriptor,
                expected_sha256=self.sha256,
                expected_size_bytes=len(self.payload),
            )
            self.assertEqual(observed_sha256, self.sha256)
            self.assertEqual(observed_size, len(self.payload))
            self.assertEqual(
                fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _ALL_SEALS,
                _ALL_SEALS,
            )
        finally:
            os.close(descriptor)

    def test_mutation_after_sealing_cannot_reach_a_loader(self) -> None:
        """Adversarial test 5: hash-then-load has no exploitable window."""

        descriptor = self._seal()
        try:
            hostile = b"hostile pickle bytes of the same length!".ljust(
                len(self.payload), b"."
            )[: len(self.payload)]
            self.assertNotEqual(hostile, self.payload)
            # Replace the named file, both in place and by relinking a new inode.
            self.checkpoint.write_bytes(hostile)
            replacement = self.generation / "checkpoints" / "replacement.pt"
            replacement.write_bytes(hostile)
            replacement.replace(self.checkpoint)
            # The sealed descriptor still reads exactly the authenticated bytes.
            self.assertEqual(os.pread(descriptor, 4096, 0), self.payload)
            self.assertEqual(
                authenticate_sealed_descriptor(descriptor)[0],
                self.sha256,
            )
            with self.assertRaises(OSError):
                os.pwrite(descriptor, b"x", 0)
            with self.assertRaises(OSError):
                os.ftruncate(descriptor, 0)
        finally:
            os.close(descriptor)

    def test_sealing_rejects_a_digest_or_size_that_was_not_authenticated(self) -> None:
        with self.assertRaises(SealedCheckpointError):
            seal_authenticated_checkpoint(
                self.checkpoint,
                expected_sha256="0" * 64,
                expected_size_bytes=len(self.payload),
            )
        with self.assertRaises(SealedCheckpointError):
            seal_authenticated_checkpoint(
                self.checkpoint,
                expected_sha256=self.sha256,
                expected_size_bytes=len(self.payload) + 1,
            )
        with self.assertRaises(SealedCheckpointError):
            seal_authenticated_checkpoint(
                self.checkpoint,
                expected_sha256="not a digest",
                expected_size_bytes=len(self.payload),
            )

    def test_sealing_rejects_symlinks_and_hard_linked_aliases(self) -> None:
        alias = self.generation / "checkpoints" / "alias.pt"
        alias.symlink_to(self.checkpoint)
        with self.assertRaises(SealedCheckpointError):
            seal_authenticated_checkpoint(
                alias,
                expected_sha256=self.sha256,
                expected_size_bytes=len(self.payload),
            )
        hard_link = self.generation / "checkpoints" / "hardlink.pt"
        os.link(self.checkpoint, hard_link)
        with self.assertRaises(SealedCheckpointError):
            seal_authenticated_checkpoint(
                self.checkpoint,
                expected_sha256=self.sha256,
                expected_size_bytes=len(self.payload),
            )

    def test_generation_relative_path_must_be_canonical(self) -> None:
        for hostile in (
            "../checkpoints/checkpoint.pt",
            "/checkpoints/checkpoint.pt",
            "checkpoints/./checkpoint.pt",
            "checkpoints\\checkpoint.pt",
            "",
        ):
            with self.assertRaises(SealedCheckpointError):
                seal_generation_checkpoint(
                    self.generation,
                    hostile,
                    expected_sha256=self.sha256,
                    expected_size_bytes=len(self.payload),
                )

    def test_unsealed_descriptor_is_rejected(self) -> None:
        descriptor = os.memfd_create("unsealed", os.MFD_ALLOW_SEALING)
        try:
            os.write(descriptor, self.payload)
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_descriptor(
                    descriptor,
                    expected_sha256=self.sha256,
                    expected_size_bytes=len(self.payload),
                )
        finally:
            os.close(descriptor)

    def test_partially_sealed_descriptor_is_rejected(self) -> None:
        descriptor = os.memfd_create("partial", os.MFD_ALLOW_SEALING)
        try:
            os.write(descriptor, self.payload)
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, fcntl.F_SEAL_GROW)
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_descriptor(descriptor)
        finally:
            os.close(descriptor)

    def test_regular_file_descriptor_is_rejected(self) -> None:
        descriptor = os.open(self.checkpoint, os.O_RDONLY)
        try:
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_descriptor(descriptor)
        finally:
            os.close(descriptor)

    def test_request_field_round_trips_and_fails_closed(self) -> None:
        sealed = seal_generation_checkpoint(
            self.generation,
            "checkpoints/checkpoint.pt",
            expected_sha256=self.sha256,
            expected_size_bytes=len(self.payload),
        )
        try:
            field = sealed.as_request_field()
            self.assertEqual(field["schema_name"], SEALED_CHECKPOINT_SCHEMA_NAME)
            self.assertNotIn("path", field)
            observed = authenticate_sealed_checkpoint_field(
                field, expected_sha256=self.sha256
            )
            self.assertEqual(observed.descriptor, sealed.descriptor)
            self.assertEqual(observed.sha256, self.sha256)

            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    field, expected_sha256="1" * 64
                )
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    {**field, "sha256": "1" * 64}, expected_sha256="1" * 64
                )
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    {**field, "size_bytes": len(self.payload) + 1},
                    expected_sha256=self.sha256,
                )
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    {**field, "schema_name": "other"}, expected_sha256=self.sha256
                )
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    {**field, "descriptor": True}, expected_sha256=self.sha256
                )
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    {**field, "extra": 1}, expected_sha256=self.sha256
                )
            with self.assertRaises(SealedCheckpointError):
                authenticate_sealed_checkpoint_field(
                    str(self.checkpoint), expected_sha256=self.sha256
                )
        finally:
            sealed.close()

    def test_full_runtime_shares_one_sealing_implementation(self) -> None:
        """The theory and evidence trust boundaries must not drift apart."""

        from scripts.policy_improvement_full_runtime import (
            _seal_authenticated_checkpoint,
            FullRuntimeError,
        )

        descriptor = _seal_authenticated_checkpoint(
            self.checkpoint,
            expected_sha256=self.sha256,
            expected_size_bytes=len(self.payload),
        )
        try:
            self.assertEqual(
                authenticate_sealed_descriptor(
                    descriptor,
                    expected_sha256=self.sha256,
                    expected_size_bytes=len(self.payload),
                ),
                (self.sha256, len(self.payload)),
            )
        finally:
            os.close(descriptor)
        with self.assertRaises(FullRuntimeError):
            _seal_authenticated_checkpoint(
                self.checkpoint,
                expected_sha256="0" * 64,
                expected_size_bytes=len(self.payload),
            )

    def test_sealing_leaks_no_descriptor_on_the_failure_path(self) -> None:
        before = sorted(os.listdir("/proc/self/fd"))
        for _ in range(16):
            with self.assertRaises(SealedCheckpointError):
                seal_authenticated_checkpoint(
                    self.checkpoint,
                    expected_sha256="0" * 64,
                    expected_size_bytes=len(self.payload),
                )
        after = sorted(os.listdir("/proc/self/fd"))
        self.assertEqual(len(after), len(before))

    def test_validator_factory_does_not_import_torch(self) -> None:
        """Import ordering: Torch must not load before evidence is sealed."""

        program = (
            "import sys\n"
            "from policy_improvement_sealed_evidence import "
            "load_sealed_checkpoint_validator\n"
            "import scripts.policy_improvement_audit\n"
            "import scripts.policy_improvement_evidence\n"
            "validator = load_sealed_checkpoint_validator()\n"
            "assert callable(validator)\n"
            "leaked = sorted(\n"
            "    name\n"
            "    for name in sys.modules\n"
            "    if name == 'torch' or name.startswith('torch.')\n"
            ")\n"
            "assert not leaked, leaked\n"
            "import scripts.policy_improvement_evidence as ev\n"
            "print(ev.__file__)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=str(REPOSITORY_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        resolved = Path(completed.stdout.strip()).resolve()
        self.assertEqual(
            resolved,
            (REPOSITORY_ROOT / "scripts" / "policy_improvement_evidence.py").resolve(),
            f"subprocess imported {resolved}, not the module under test",
        )

    def test_consumer_entrypoint_binds_the_validator_lazily(self) -> None:
        """Structurally reject any module-scope import of the Torch validator.

        Substring matching is defeatable: ``from X import y``, a one-line
        ``importlib.import_module("X")``, and ``__import__("X")`` all reintroduce
        the eager import while dodging naive text checks.  Parse instead.
        """

        entrypoint = REPOSITORY_ROOT / "policy_improvement_consumer_entrypoint.py"
        source = entrypoint.read_text(encoding="utf-8")
        self.assertIn("load_sealed_checkpoint_validator", source)
        tree = ast.parse(source, filename=str(entrypoint))
        forbidden = "policy_improvement_checkpoint_validator"
        offenders: list[str] = []
        for statement in tree.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Import):
                    offenders.extend(
                        f"import {alias.name} (line {node.lineno})"
                        for alias in node.names
                        if alias.name.split(".")[0] == forbidden
                    )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.split(".")[0] == forbidden:
                        offenders.append(f"from {module} import ... (line {node.lineno})")
                elif isinstance(node, ast.Call):
                    target = node.func
                    name = (
                        target.attr
                        if isinstance(target, ast.Attribute)
                        else target.id if isinstance(target, ast.Name) else ""
                    )
                    if name not in {"import_module", "__import__", "find_spec"}:
                        continue
                    offenders.extend(
                        f"{name}({argument.value!r}) (line {node.lineno})"
                        for argument in node.args
                        if isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                        and argument.value.split(".")[0] == forbidden
                    )
        self.assertEqual(offenders, [], f"eager validator import: {offenders}")

        # Guard the guard: the same parse must catch each reintroduction.
        for reintroduction in (
            "import policy_improvement_checkpoint_validator",
            "from policy_improvement_checkpoint_validator import validate_checkpoint",
            '_m = importlib.import_module("policy_improvement_checkpoint_validator")',
            '_m = __import__("policy_improvement_checkpoint_validator")',
        ):
            with self.subTest(reintroduction=reintroduction):
                self.assertTrue(
                    self._module_scope_imports_validator(
                        f"{source}\n{reintroduction}\n"
                    ),
                    reintroduction,
                )

    @staticmethod
    def _module_scope_imports_validator(source: str) -> bool:
        forbidden = "policy_improvement_checkpoint_validator"
        tree = ast.parse(source)
        for statement in tree.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Import) and any(
                    alias.name.split(".")[0] == forbidden for alias in node.names
                ):
                    return True
                if (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0] == forbidden
                ):
                    return True
                if isinstance(node, ast.Call):
                    target = node.func
                    name = (
                        target.attr
                        if isinstance(target, ast.Attribute)
                        else target.id if isinstance(target, ast.Name) else ""
                    )
                    if name in {"import_module", "__import__", "find_spec"} and any(
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                        and argument.value.split(".")[0] == forbidden
                        for argument in node.args
                    ):
                        return True
        return False


class EvidenceDeserializationClosureTest(unittest.TestCase):
    """No evidence call path may reach an unrestricted deserializer.

    This codifies the reachability audit so a future ``torch.load`` added
    anywhere in the evidence import closure fails the build instead of being
    caught only by review.
    """

    ENTRY_POINTS = (
        "scripts.policy_improvement_audit",
        "scripts.policy_improvement_analysis",
        "scripts.policy_improvement_test_open_cli",
        "scripts.policy_improvement_full_runtime",
        "scripts.policy_improvement_theory_backend",
        "policy_improvement_checkpoint_validator",
        "policy_improvement_full_backend",
        "policy_improvement_consumer_entrypoint",
    )
    # The single sanctioned deserializer.
    SANCTIONED = "policy_improvement_checkpoint_allowlist.py"
    # Import-reachable but not call-reachable, with the reason recorded.
    # Guarded below by asserting no evidence module imports the function.
    EXEMPT = {
        "rl/persistent_diagnostic_checkpoint.py": (
            "load_persistent_checkpoint",
            "diagnostics-only entry point; evidence modules import only "
            "state_dict_sha256 from this module",
        ),
    }

    @staticmethod
    def _module_path(name: str) -> Path | None:
        for candidate in (
            REPOSITORY_ROOT / (name.replace(".", "/") + ".py"),
            REPOSITORY_ROOT / name.replace(".", "/") / "__init__.py",
        ):
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def _imports(cls, path: Path) -> set[str]:
        names: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Call):
                target = node.func
                label = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else target.id if isinstance(target, ast.Name) else ""
                )
                if label in {"import_module", "__import__"}:
                    names.update(
                        a.value
                        for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    )
        return names

    @classmethod
    def _closure(cls) -> list[Path]:
        seen: set[str] = set()
        queue = list(cls.ENTRY_POINTS)
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            path = cls._module_path(name)
            if path is None:
                continue
            seen.add(name)
            queue.extend(cls._imports(path))
        return sorted({cls._module_path(n) for n in seen} - {None})

    @staticmethod
    def _torch_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
        """Names bound to the torch module, and names bound to torch.load."""

        modules = {"torch"}
        direct: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        modules.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[
                0
            ] == "torch":
                for alias in node.names:
                    if alias.name == "load":
                        direct.add(alias.asname or alias.name)
                    else:
                        modules.add(alias.asname or alias.name)
        return modules, direct

    @staticmethod
    def _is_torch_load(node: ast.Call, aliases: tuple[set[str], set[str]]) -> bool:
        modules, direct = aliases
        target = node.func
        if isinstance(target, ast.Name):
            return target.id in direct
        if isinstance(target, ast.Attribute) and target.attr == "load":
            base = target.value
            if isinstance(base, ast.Name):
                return base.id in modules
            # torch.serialization.load and deeper attribute chains.
            while isinstance(base, ast.Attribute):
                base = base.value
            return isinstance(base, ast.Name) and base.id in modules
        return False

    def test_alias_detection_catches_every_torch_load_spelling(self) -> None:
        """Guard the guard: the matcher must not be dodged by an alias."""

        spellings = (
            "import torch\ntorch.load(f)\n",
            "import torch as t\nt.load(f)\n",
            "from torch import load\nload(f)\n",
            "from torch import load as ld\nld(f)\n",
            "import torch.serialization\ntorch.serialization.load(f)\n",
            "from torch import serialization as s\ns.load(f)\n",
        )
        for source in spellings:
            with self.subTest(source=source.replace("\n", "; ")):
                tree = ast.parse(source)
                aliases = self._torch_aliases(tree)
                found = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and self._is_torch_load(node, aliases)
                ]
                self.assertEqual(len(found), 1, source)
        # And it must not fire on an unrelated load().
        tree = ast.parse("import json\njson.load(f)\n")
        aliases = self._torch_aliases(tree)
        self.assertFalse(
            [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and self._is_torch_load(node, aliases)
            ]
        )

    def test_closure_has_exactly_one_sanctioned_torch_load(self) -> None:
        closure = self._closure()
        self.assertGreater(len(closure), 40, "closure looks truncated")
        offenders: list[str] = []
        sanctioned_sites = 0
        for path in closure:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            aliases = self._torch_aliases(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not self._is_torch_load(node, aliases):
                    continue
                if relative == self.SANCTIONED:
                    sanctioned_sites += 1
                    keywords = {k.arg for k in node.keywords}
                    self.assertIn("weights_only", keywords)
                    value = next(
                        k.value for k in node.keywords if k.arg == "weights_only"
                    )
                    self.assertIs(getattr(value, "value", None), True)
                    continue
                if relative in self.EXEMPT:
                    continue
                offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "unsanctioned torch.load in the evidence closure: "
            f"{offenders}. Route it through load_data_only_checkpoint.",
        )
        self.assertEqual(sanctioned_sites, 1)

    def test_exempt_modules_expose_no_loader_to_the_evidence_path(self) -> None:
        """The exemptions are call-level, so pin what evidence modules import."""

        for relative, (function, _reason) in self.EXEMPT.items():
            module = relative[: -len(".py")].replace("/", ".")
            importers: list[str] = []
            for path in self._closure():
                text = path.read_text(encoding="utf-8")
                if f"from {module} import" not in text:
                    continue
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module == module
                        and any(a.name == function for a in node.names)
                    ):
                        importers.append(path.relative_to(REPOSITORY_ROOT).as_posix())
            self.assertEqual(
                importers,
                [],
                f"{module}.{function} is now imported by the evidence closure "
                f"({importers}); the call-level exemption no longer holds.",
            )

    def test_no_bare_pickle_load_in_the_evidence_closure(self) -> None:
        offenders: list[str] = []
        for path in self._closure():
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in {"load", "loads"}
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"pickle", "dill", "joblib"}
                ):
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], f"raw unpickling in evidence closure: {offenders}")

    def test_numpy_loads_in_the_closure_never_allow_pickle(self) -> None:
        offenders: list[str] = []
        for path in self._closure():
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                if not (
                    isinstance(target, ast.Attribute)
                    and target.attr == "load"
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"np", "numpy"}
                ):
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "allow_pickle" and getattr(
                        keyword.value, "value", None
                    ) is not False:
                        offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], f"np.load allowing pickle: {offenders}")

if __name__ == "__main__":
    unittest.main()
