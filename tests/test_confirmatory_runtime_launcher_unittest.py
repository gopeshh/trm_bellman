"""Tests for the pre-import confirmatory runtime launcher."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from confirmatory_runtime_launcher import (
    PAR_FILENAME_ENV,
    PRIVATE_UNPACK_BASE_ENV,
    PRIVATE_UNPACK_FD_ENV,
    SOURCE_MANIFEST_RELATIVE_PATH,
    VERIFIED_RUNTIME_PATH_ENV,
    VERIFIED_RUNTIME_SHA256_ENV,
    VERIFIED_RUNTIME_FD_ENV,
    ConfirmatoryRuntimeError,
    VerifiedRuntime,
    _assert_runtime_unchanged,
    launch_verified_runtime,
    validate_runtime_archive,
)


class TestConfirmatoryRuntimeLauncher(unittest.TestCase):
    def _sources(self) -> dict[str, bytes]:
        sources = {
            "confirmatory_runtime_launcher.py": b"# launcher\n",
            "puzzle_dataset.py": b"# puzzle\n",
            "upi_trm_train.py": b"# trainer\n",
        }
        for directory in ("dataset", "evaluators", "models", "rl", "utils"):
            sources[f"{directory}/module.py"] = f"# {directory}\n".encode()
        return sources

    def _manifest(self, sources: dict[str, bytes]) -> dict[str, object]:
        entries = {
            path: hashlib.sha256(content).hexdigest()
            for path, content in sources.items()
        }
        entries["configs/iclr_confirmatory/cell.yaml"] = hashlib.sha256(
            b"gamma: 0.9\n"
        ).hexdigest()
        return {
            "source_manifest_schema_version": 1,
            "sources": entries,
        }

    def _archive(
        self,
        path: Path,
        *,
        omit: str | None = None,
        extra: str | None = None,
        duplicate: str | None = None,
        bytecode: str | None = None,
        tamper: str | None = None,
    ) -> str:
        sources = self._sources()
        manifest = self._manifest(sources)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    SOURCE_MANIFEST_RELATIVE_PATH,
                    json.dumps(manifest, sort_keys=True).encode(),
                )
                config_content = b"gamma: 0.9\n"
                if tamper == "configs/iclr_confirmatory/cell.yaml":
                    config_content += b"# modified after manifest\n"
                if omit != "configs/iclr_confirmatory/cell.yaml":
                    archive.writestr(
                        "configs/iclr_confirmatory/cell.yaml",
                        config_content,
                    )
                for relative_path, content in sources.items():
                    if relative_path == omit:
                        continue
                    if relative_path == tamper:
                        content += b"# modified after manifest\n"
                    archive.writestr(relative_path, content)
                if extra is not None:
                    archive.writestr(extra, b"# extra\n")
                if duplicate is not None:
                    archive.writestr(duplicate, sources[duplicate])
                if bytecode is not None:
                    archive.writestr(bytecode, b"untrusted bytecode")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_archive_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(path)
            runtime = validate_runtime_archive(path, expected)
            try:
                self.assertEqual(runtime.path, path)
                self.assertEqual(runtime.sha256, expected)
                self.assertGreaterEqual(runtime.descriptor, 3)
                seals = fcntl.fcntl(runtime.descriptor, fcntl.F_GET_SEALS)
                required_seals = (
                    fcntl.F_SEAL_WRITE
                    | fcntl.F_SEAL_SHRINK
                    | fcntl.F_SEAL_GROW
                    | fcntl.F_SEAL_SEAL
                )
                self.assertEqual(seals & required_seals, required_seals)
            finally:
                os.close(runtime.descriptor)

    def test_rejects_bad_or_uppercase_expected_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(path)
            for invalid in ("0" * 63, "A" * 64):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(
                        ConfirmatoryRuntimeError,
                        "64 lowercase",
                    ):
                        validate_runtime_archive(path, invalid)
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "expected SHA-256",
            ):
                validate_runtime_archive(path, "0" * 64)

    def test_rejects_non_zip_artifact_with_matching_outer_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            path.write_bytes(b"not a zip-based PAR")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "valid ZIP-based PAR",
            ):
                validate_runtime_archive(path, expected)

    def test_rejects_relative_symlink_and_nonregular_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.par"
            expected = self._archive(path)
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "absolute",
                ):
                    validate_runtime_archive(Path("runtime.par"), expected)
            finally:
                os.chdir(old_cwd)

            link = root / "runtime-link.par"
            link.symlink_to(path)
            with self.assertRaisesRegex(ConfirmatoryRuntimeError, "symlink"):
                validate_runtime_archive(link, expected)
            with self.assertRaisesRegex(ConfirmatoryRuntimeError, "regular file"):
                validate_runtime_archive(root, expected)

    def test_rejects_tampered_source_even_with_matching_outer_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(path, tamper="models/module.py")
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "differs from the embedded manifest",
            ):
                validate_runtime_archive(path, expected)

    def test_rejects_tampered_config_even_with_matching_outer_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(
                path,
                tamper="configs/iclr_confirmatory/cell.yaml",
            )
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "differs from the embedded manifest",
            ):
                validate_runtime_archive(path, expected)

    def test_rejects_missing_extra_and_duplicate_sources(self) -> None:
        cases = {
            "missing": {"omit": "rl/module.py"},
            "missing_config": {
                "omit": "configs/iclr_confirmatory/cell.yaml",
            },
            "extra": {"extra": "models/extra.py"},
            "duplicate": {"duplicate": "utils/module.py"},
        }
        for name, options in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.par"
                expected = self._archive(path, **options)
                with self.assertRaises(ConfirmatoryRuntimeError):
                    validate_runtime_archive(path, expected)

    def test_rejects_behavior_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(path, bytecode="models/module.pyc")
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "bytecode",
            ):
                validate_runtime_archive(path, expected)

    def test_rejects_root_aliases_and_noncanonical_directory_members(self) -> None:
        for member_name in (".", "./", "/", "//", "foo//"):
            with (
                self.subTest(member_name=member_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "runtime.par"
                self._archive(path)
                with ZipFile(path, "a") as archive:
                    archive.writestr(member_name, b"")
                expected = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "unsafe member path",
                ):
                    validate_runtime_archive(path, expected)

    def test_rejects_canonical_member_path_collisions(self) -> None:
        cases = (
            (("alias", b"file"), ("alias/", b"")),
            (("prefix", b"file"), ("prefix/child", b"child")),
            (("prefix/child", b"child"), ("prefix", b"file")),
        )
        for members in cases:
            with (
                self.subTest(members=members),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "runtime.par"
                self._archive(path)
                with ZipFile(path, "a") as archive:
                    for member_name, content in members:
                        archive.writestr(member_name, content)
                expected = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "aliases|collision",
                ):
                    validate_runtime_archive(path, expected)

    def test_accepts_canonical_directory_with_child_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            self._archive(path)
            with ZipFile(path, "a") as archive:
                archive.writestr("canonical/", b"")
                archive.writestr("canonical/child", b"child")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            runtime = validate_runtime_archive(path, expected)
            os.close(runtime.descriptor)

    def test_identity_check_detects_replaced_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.par"
            path.write_bytes(b"original")
            path_before = path.lstat()
            with path.open("rb") as handle:
                descriptor_before = os.fstat(handle.fileno())
                replacement = root / "replacement.par"
                replacement.write_bytes(b"replacement")
                replacement.replace(path)
                descriptor_after = os.fstat(handle.fileno())
            with self.assertRaisesRegex(
                ConfirmatoryRuntimeError,
                "identity, size, or timestamps",
            ):
                _assert_runtime_unchanged(
                    path,
                    path_before,
                    descriptor_before,
                    descriptor_after,
                )

    def test_launch_uses_sealed_copy_after_source_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.par"
            expected = self._archive(path)
            runtime = validate_runtime_archive(path, expected)
            descriptor = runtime.descriptor
            replacement = root / "replacement.par"
            replacement.write_bytes(b"unverified replacement")
            replacement.replace(path)
            with mock.patch(
                "confirmatory_runtime_launcher.subprocess.Popen"
            ) as popen:
                popen.return_value.wait.return_value = 0
                self.assertEqual(
                    launch_verified_runtime(runtime, ["--confirmatory"]),
                    0,
                )
            popen.assert_called_once()
            self.assertEqual(
                popen.call_args.kwargs["executable"],
                f"/proc/self/fd/{descriptor}",
            )

    def test_sealed_copy_rejects_in_place_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(path)
            runtime = validate_runtime_archive(path, expected)
            try:
                with self.assertRaises(OSError):
                    os.write(runtime.descriptor, b"tamper")
            finally:
                os.close(runtime.descriptor)

    def test_exec_uses_exact_path_arguments_and_private_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            expected = self._archive(path)
            runtime = validate_runtime_archive(path, expected)
            descriptor = runtime.descriptor
            with mock.patch(
                "confirmatory_runtime_launcher.subprocess.Popen"
            ) as popen:
                popen.return_value.wait.return_value = 0
                self.assertEqual(
                    launch_verified_runtime(
                        runtime,
                        ["--confirmatory", "--config", "cell.yaml"],
                        environ={
                            "KEEP": "yes",
                            "LD_PRELOAD": "/tmp/untrusted.so",
                            "PAR_MAIN_OVERRIDE": "untrusted.module",
                            "PYTHONWARNINGS": "error::untrusted.Warning",
                            "PYTHONHASHSEED": "17",
                            VERIFIED_RUNTIME_PATH_ENV: "stale",
                        },
                    ),
                    0,
                )
            popen.assert_called_once()
            argv = popen.call_args.args[0]
            executable = popen.call_args.kwargs["executable"]
            environment = popen.call_args.kwargs["env"]
            descriptor_path = f"/proc/self/fd/{descriptor}"
            private_unpack_descriptor = int(environment[PRIVATE_UNPACK_FD_ENV])
            private_unpack_path = f"/proc/self/fd/{private_unpack_descriptor}"
            self.assertEqual(executable, descriptor_path)
            self.assertEqual(
                argv,
                [
                    descriptor_path,
                    "--confirmatory",
                    "--config",
                    "cell.yaml",
                ],
            )
            self.assertEqual(environment["KEEP"], "yes")
            self.assertNotIn("LD_PRELOAD", environment)
            self.assertNotIn("PAR_MAIN_OVERRIDE", environment)
            self.assertNotIn("PYTHONWARNINGS", environment)
            self.assertEqual(environment["PYTHONHASHSEED"], "17")
            self.assertEqual(environment[VERIFIED_RUNTIME_PATH_ENV], descriptor_path)
            self.assertEqual(environment[VERIFIED_RUNTIME_SHA256_ENV], expected)
            self.assertEqual(environment[VERIFIED_RUNTIME_FD_ENV], str(descriptor))
            self.assertEqual(environment[PAR_FILENAME_ENV], descriptor_path)
            self.assertEqual(
                environment["FB_PAR_UNPACK_BASEDIR"],
                private_unpack_path,
            )
            self.assertEqual(environment[PRIVATE_UNPACK_BASE_ENV], private_unpack_path)
            self.assertEqual(
                popen.call_args.kwargs["pass_fds"],
                (descriptor, private_unpack_descriptor),
            )
            self.assertFalse(Path(private_unpack_path).exists())

    def test_unpack_descriptor_survives_path_replacement_at_exec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.par"
            expected = self._archive(path)
            runtime = validate_runtime_archive(path, expected)
            private_unpack = root / "private-unpack"
            private_unpack.mkdir(mode=0o700)
            moved_unpack = root / "moved-unpack"
            process = mock.Mock()
            process.wait.return_value = 0

            def replace_path_at_exec(
                _argv: list[str],
                **kwargs: object,
            ) -> mock.Mock:
                environment = kwargs["env"]
                assert isinstance(environment, dict)
                unpack_descriptor = int(environment[PRIVATE_UNPACK_FD_ENV])
                descriptor_status = os.fstat(unpack_descriptor)
                private_unpack.rename(moved_unpack)
                private_unpack.mkdir(mode=0o700)
                moved_status = moved_unpack.stat()
                replacement_status = private_unpack.stat()
                self.assertEqual(
                    (descriptor_status.st_dev, descriptor_status.st_ino),
                    (moved_status.st_dev, moved_status.st_ino),
                )
                self.assertNotEqual(
                    (descriptor_status.st_dev, descriptor_status.st_ino),
                    (replacement_status.st_dev, replacement_status.st_ino),
                )
                descriptor_path = environment["FB_PAR_UNPACK_BASEDIR"]
                descriptor_path_status = os.stat(descriptor_path)
                self.assertEqual(
                    (
                        descriptor_path_status.st_dev,
                        descriptor_path_status.st_ino,
                    ),
                    (moved_status.st_dev, moved_status.st_ino),
                )
                return process

            with (
                mock.patch(
                    "confirmatory_runtime_launcher.tempfile.mkdtemp",
                    return_value=str(private_unpack),
                ),
                mock.patch(
                    "confirmatory_runtime_launcher.subprocess.Popen",
                    side_effect=replace_path_at_exec,
                ),
            ):
                self.assertEqual(
                    launch_verified_runtime(runtime, ["--confirmatory"]),
                    0,
                )
            self.assertTrue(private_unpack.is_dir())
            self.assertTrue(moved_unpack.is_dir())

    def test_exec_requires_confirmatory_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            path.write_bytes(b"runtime")
            descriptor = os.open(path, os.O_RDONLY)
            runtime = VerifiedRuntime(
                path=path,
                sha256="a" * 64,
                descriptor=descriptor,
            )
            with mock.patch(
                "confirmatory_runtime_launcher.subprocess.Popen"
            ) as popen:
                with self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "--confirmatory",
                ):
                    launch_verified_runtime(runtime, [])
            popen.assert_not_called()

    def test_exec_rejects_unsealed_runtime_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.par"
            path.write_bytes(b"runtime")
            descriptor = os.open(path, os.O_RDONLY)
            runtime = VerifiedRuntime(
                path=path,
                sha256="a" * 64,
                descriptor=descriptor,
            )
            with mock.patch(
                "confirmatory_runtime_launcher.subprocess.Popen"
            ) as popen:
                with self.assertRaises(ConfirmatoryRuntimeError):
                    launch_verified_runtime(runtime, ["--confirmatory"])
            popen.assert_not_called()

    def test_child_start_failure_removes_private_unpack_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runtime.par"
            expected = self._archive(path)
            runtime = validate_runtime_archive(path, expected)
            private_unpack = root / "private-unpack"
            private_unpack.mkdir(mode=0o700)
            with (
                mock.patch(
                    "confirmatory_runtime_launcher.tempfile.mkdtemp",
                    return_value=str(private_unpack),
                ),
                mock.patch(
                    "confirmatory_runtime_launcher.subprocess.Popen",
                    side_effect=OSError("cannot start child"),
                ),
                self.assertRaisesRegex(
                    ConfirmatoryRuntimeError,
                    "could not be executed",
                ),
            ):
                launch_verified_runtime(runtime, ["--confirmatory"])
            self.assertFalse(private_unpack.exists())


if __name__ == "__main__":
    unittest.main()
