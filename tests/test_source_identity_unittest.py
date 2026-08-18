"""Tests for the source manifest embedded in confirmatory binaries."""

import copy
import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from phase4_runtime_profile import (
    authorize_phase4_training_source,
    Phase4RuntimeProfileError,
)
from utils.source_identity import (
    assert_runtime_archive_sources_match_manifest,
    behavior_source_relative_paths,
    behavior_source_relative_paths_from_inventory,
    build_producer_source_manifest,
    inventory_schema_version,
    registered_config_directories,
    required_additional_sources,
    required_root_sources,
    SOURCE_MANIFEST_RELATIVE_PATH,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    SourceIdentityError,
    SUPPORTED_SOURCE_MANIFEST_SCHEMA_VERSIONS,
    validate_producer_source_manifest,
)


def _write_source_tree(root: Path) -> None:
    for relative_path in (
        "confirmatory_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_checkpoint_allowlist.py",
        "policy_improvement_smoke_checkpoint.py",
        "policy_improvement_smoke_runtime.py",
        "puzzle_dataset.py",
        "runtime_archive_preflight.py",
        "scripts/policy_improvement_registry.py",
        "scripts/policy_improvement_schema.py",
        "scripts/policy_improvement_populations.py",
        "scripts/policy_improvement_v2_registry.py",
        "scripts/policy_improvement_v2_schema.py",
        "upi_trm_train.py",
    ):
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"# {relative_path}\n", encoding="ascii")
    for directory in ("dataset", "evaluators", "models", "rl", "utils"):
        destination = root / directory / "module.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"# {directory}\n", encoding="ascii")
    config_dir = root / "configs" / "iclr_confirmatory"
    config_dir.mkdir(parents=True)
    (config_dir / "cell.yaml").write_text("gamma: 0.9\n", encoding="ascii")
    (config_dir / "registry.json").write_text("{}\n", encoding="ascii")
    policy_config_dir = root / "configs" / "policy_improvement_v1"
    policy_config_dir.mkdir(parents=True)
    (policy_config_dir / "protocol.json").write_text("{}\n", encoding="ascii")
    (policy_config_dir / "method.yaml").write_text("gamma: 0.9\n", encoding="ascii")
    policy_v2_config_dir = root / "configs" / "policy_improvement_v2"
    policy_v2_config_dir.mkdir(parents=True)
    (policy_v2_config_dir / "protocol.json").write_text("{}\n", encoding="ascii")
    (policy_v2_config_dir / "method.yaml").write_text("gamma: 0.9\n", encoding="ascii")
    policy_v2_amendments = policy_v2_config_dir / "amendments"
    policy_v2_amendments.mkdir()
    (policy_v2_amendments / "theory_bridge_v2.json").write_text(
        "{}\n", encoding="ascii"
    )

    phase4_config_dir = root / "configs" / "phase4_2x2_norm_ablation"
    phase4_config_dir.mkdir(parents=True)
    for condition in ("nc_nv", "nc_yv", "yc_nv", "yc_yv"):
        (phase4_config_dir / f"{condition}.yaml").write_text(
            "gamma: 0.9\n",
            encoding="ascii",
        )


def _write_manifest(root: Path, schema_version: int) -> None:
    sources = {
        relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        for relative_path in behavior_source_relative_paths(
            root,
            schema_version=schema_version,
        )
    }
    manifest_path = root / SOURCE_MANIFEST_RELATIVE_PATH
    manifest_path.write_text(
        json.dumps(
            {
                "source_manifest_schema_version": schema_version,
                "sources": sources,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _commit_source_tree(root: Path) -> str:
    commands = (
        ("init", "-q"),
        ("config", "user.email", "source-identity-test@example.com"),
        ("config", "user.name", "Source Identity Test"),
        ("add", "."),
        ("commit", "-qm", "source identity fixture"),
    )
    for arguments in commands:
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


class TestSourceIdentity(unittest.TestCase):
    @staticmethod
    def _unicode_path_member(raw_name: str, effective_name: str) -> ZipInfo:
        encoded_raw_name = raw_name.encode("ascii")
        encoded_effective_name = effective_name.encode("utf-8")
        info = ZipInfo(raw_name)
        info.extra = (
            struct.pack(
                "<HHBL",
                0x7075,
                5 + len(encoded_effective_name),
                1,
                zlib.crc32(encoded_raw_name),
            )
            + encoded_effective_name
        )
        return info

    def _source_tree(self, root: Path) -> None:
        _write_source_tree(root)

    def _runtime_archive(
        self,
        archive_path: Path,
        source_root: Path,
        manifest: dict[str, object],
        *,
        tampered_path: str | None = None,
        omitted_path: str | None = None,
        extra_path: str | None = None,
    ) -> None:
        sources = manifest["sources"]
        assert isinstance(sources, dict)
        with ZipFile(archive_path, "w") as archive:
            for relative_path in sources:
                if not relative_path.endswith(".py") or relative_path == omitted_path:
                    continue
                source_bytes = (source_root / relative_path).read_bytes()
                if relative_path == tampered_path:
                    source_bytes += b"# tampered\n"
                archive.writestr(relative_path, source_bytes)
            if extra_path is not None:
                archive.writestr(extra_path, b"# unregistered\n")

    def test_manifest_is_ordered_and_detects_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source_tree(root)
            manifest = build_producer_source_manifest(root)
            validated = validate_producer_source_manifest(manifest)
            self.assertEqual(validated, manifest)
            self.assertEqual(
                list(manifest["sources"]),
                behavior_source_relative_paths(root),
            )
            repository_files = [
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
            ]
            self.assertEqual(
                list(manifest["sources"]),
                behavior_source_relative_paths_from_inventory(repository_files),
            )
            self.assertNotIn(SOURCE_MANIFEST_RELATIVE_PATH, manifest["sources"])

            before = copy.deepcopy(manifest)
            (root / "rl" / "module.py").write_text("# changed\n", encoding="ascii")
            self.assertNotEqual(build_producer_source_manifest(root), before)

    def test_manifest_rejects_unsafe_paths_and_bad_hashes(self) -> None:
        invalid = {
            "source_manifest_schema_version": 1,
            "sources": {"../outside.py": "a" * 64},
        }
        with self.assertRaises(SourceIdentityError):
            validate_producer_source_manifest(invalid)
        invalid["sources"] = {"rl/module.py": "A" * 64}
        with self.assertRaises(SourceIdentityError):
            validate_producer_source_manifest(invalid)

    def test_manifest_schema_requires_exact_non_bool_integer(self) -> None:
        base: dict[str, object] = {
            "source_manifest_schema_version": 1,
            "sources": {"rl/module.py": "a" * 64},
        }
        for invalid in (True, 1.0, "1"):
            with self.subTest(invalid=invalid):
                manifest = copy.deepcopy(base)
                manifest["source_manifest_schema_version"] = invalid
                with self.assertRaisesRegex(
                    SourceIdentityError,
                    "source manifest schema",
                ):
                    validate_producer_source_manifest(manifest)

    def test_runtime_archive_sources_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source_tree(root)
            manifest = build_producer_source_manifest(root)
            archive_path = root / "runtime.par"
            self._runtime_archive(archive_path, root, manifest)

            with ZipFile(archive_path, "r") as archive:
                assert_runtime_archive_sources_match_manifest(archive, manifest)

    def test_runtime_archive_rejects_tampered_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source_tree(root)
            manifest = build_producer_source_manifest(root)
            archive_path = root / "runtime.par"
            self._runtime_archive(
                archive_path,
                root,
                manifest,
                tampered_path="rl/module.py",
            )

            with ZipFile(archive_path, "r") as archive:
                with self.assertRaisesRegex(
                    SourceIdentityError,
                    "source bytes differ",
                ):
                    assert_runtime_archive_sources_match_manifest(archive, manifest)

    def test_runtime_archive_rejects_missing_or_extra_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source_tree(root)
            manifest = build_producer_source_manifest(root)
            archive_path = root / "runtime.par"

            for options in (
                {"omitted_path": "rl/module.py"},
                {"extra_path": "rl/unregistered.py"},
            ):
                self._runtime_archive(archive_path, root, manifest, **options)
                with ZipFile(archive_path, "r") as archive:
                    with self.assertRaisesRegex(
                        SourceIdentityError,
                        "inventory differs",
                    ):
                        assert_runtime_archive_sources_match_manifest(
                            archive,
                            manifest,
                        )

    def test_runtime_archive_rejects_unverified_behavior_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source_tree(root)
            manifest = build_producer_source_manifest(root)
            archive_path = root / "runtime.par"
            self._runtime_archive(archive_path, root, manifest)
            with ZipFile(archive_path, "a") as archive:
                archive.writestr(
                    "rl/__pycache__/module.cpython-312.pyc",
                    b"unverified bytecode",
                )

            with ZipFile(archive_path, "r") as archive:
                with self.assertRaisesRegex(
                    SourceIdentityError,
                    "unverified behavior bytecode",
                ):
                    assert_runtime_archive_sources_match_manifest(
                        archive,
                        manifest,
                    )

    def test_runtime_archive_rejects_name_mismatch_on_nonbehavior_member(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source_tree(root)
            manifest = build_producer_source_manifest(root)
            archive_path = root / "runtime.par"
            self._runtime_archive(archive_path, root, manifest)
            with ZipFile(archive_path, "a") as archive:
                archive.writestr(
                    self._unicode_path_member(
                        "metadata.raw",
                        "metadata.effective",
                    ),
                    b"metadata",
                )

            with ZipFile(archive_path, "r") as archive:
                with self.assertRaisesRegex(
                    SourceIdentityError,
                    "raw and effective member names differ",
                ):
                    assert_runtime_archive_sources_match_manifest(
                        archive,
                        manifest,
                    )


class VersionedProducerInventoryTest(unittest.TestCase):
    """Historical producers must stay authenticatable; HEAD must not regress."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_versions_are_declared_consistently_across_modules(self) -> None:
        """phase4_runtime_profile duplicates the map; pin them together."""

        import confirmatory_runtime_launcher as launcher
        import phase4_runtime_profile as profile

        self.assertEqual(
            profile.PRODUCER_SOURCE_MANIFEST_SCHEMA_VERSION,
            SOURCE_MANIFEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            launcher.SUPPORTED_SOURCE_MANIFEST_SCHEMA_VERSIONS,
            SUPPORTED_SOURCE_MANIFEST_SCHEMA_VERSIONS,
        )
        for version in SUPPORTED_SOURCE_MANIFEST_SCHEMA_VERSIONS:
            with self.subTest(version=version):
                expected = tuple(sorted(required_root_sources(version)))
                self.assertEqual(
                    tuple(sorted(profile.producer_root_sources(version))), expected
                )
                self.assertEqual(
                    tuple(sorted(launcher._ROOT_SOURCES_BY_VERSION[version])), expected
                )
                self.assertEqual(
                    tuple(sorted(profile._producer_additional_sources(version))),
                    tuple(sorted(required_additional_sources(version))),
                )
                self.assertEqual(
                    tuple(sorted(profile._producer_config_directories(version))),
                    tuple(sorted(registered_config_directories(version))),
                )

    def test_versioned_inventory_additions_are_exact(self) -> None:
        self.assertIn(
            "policy_improvement_checkpoint_allowlist.py", required_root_sources(2)
        )
        self.assertNotIn(
            "policy_improvement_checkpoint_allowlist.py", required_root_sources(1)
        )
        self.assertIn(
            "scripts/policy_improvement_v2_schema.py",
            required_additional_sources(3),
        )
        self.assertNotIn(
            "scripts/policy_improvement_v2_schema.py",
            required_additional_sources(2),
        )
        self.assertIn(
            "configs/policy_improvement_v2",
            registered_config_directories(3),
        )
        self.assertNotIn(
            "configs/policy_improvement_v2",
            registered_config_directories(2),
        )
        with self.assertRaises(SourceIdentityError):
            required_root_sources(4)

    def test_historical_v1_checkout_still_enumerates(self) -> None:
        """A clean pre-allowlist producer checkout authenticates."""

        _write_source_tree(self.root)
        (self.root / "policy_improvement_checkpoint_allowlist.py").unlink()
        paths = behavior_source_relative_paths(self.root, schema_version=1)
        self.assertNotIn("policy_improvement_checkpoint_allowlist.py", paths)
        self.assertIn("upi_trm_train.py", paths)
        # Requesting the current version of that same tree fails closed.
        with self.assertRaises(SourceIdentityError):
            behavior_source_relative_paths(self.root, schema_version=2)

    def test_head_requires_every_current_source(self) -> None:
        _write_source_tree(self.root)
        paths = behavior_source_relative_paths(self.root)
        self.assertIn("policy_improvement_checkpoint_allowlist.py", paths)
        self.assertIn("scripts/policy_improvement_v2_schema.py", paths)
        self.assertIn(
            "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
            paths,
        )

    def test_deleting_the_allowlist_from_head_fails_closed(self) -> None:
        _write_source_tree(self.root)
        (self.root / "policy_improvement_checkpoint_allowlist.py").unlink()
        with self.assertRaisesRegex(SourceIdentityError, "missing source"):
            behavior_source_relative_paths(self.root)

    def test_inventory_version_is_derived_and_downgrade_is_rejected(self) -> None:
        _write_source_tree(self.root)
        inventory = [
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file()
        ]
        self.assertEqual(inventory_schema_version(inventory), 3)
        # A v1 manifest may not be presented for a tree that satisfies v2.
        with self.assertRaisesRegex(SourceIdentityError, "declares schema"):
            behavior_source_relative_paths_from_inventory(inventory, schema_version=1)
        with self.assertRaisesRegex(SourceIdentityError, "declares schema"):
            behavior_source_relative_paths_from_inventory(inventory, schema_version=2)
        self.assertTrue(
            behavior_source_relative_paths_from_inventory(inventory, schema_version=3)
        )

        v2_inventory = [
            item
            for item in inventory
            if item
            not in set(required_additional_sources(3))
            - set(required_additional_sources(2))
        ]
        self.assertEqual(inventory_schema_version(v2_inventory), 2)
        self.assertTrue(
            behavior_source_relative_paths_from_inventory(
                v2_inventory, schema_version=2
            )
        )
        with self.assertRaises(SourceIdentityError):
            behavior_source_relative_paths_from_inventory(
                v2_inventory, schema_version=3
            )

        older = [
            item
            for item in v2_inventory
            if item != "policy_improvement_checkpoint_allowlist.py"
        ]
        self.assertEqual(inventory_schema_version(older), 1)
        self.assertTrue(
            behavior_source_relative_paths_from_inventory(older, schema_version=1)
        )
        with self.assertRaises(SourceIdentityError):
            behavior_source_relative_paths_from_inventory(older, schema_version=2)

    def test_phase4_authorizer_accepts_clean_historical_v1_checkout(self) -> None:
        _write_source_tree(self.root)
        (self.root / "policy_improvement_checkpoint_allowlist.py").unlink()
        _write_manifest(self.root, 1)
        commit = _commit_source_tree(self.root)

        authorized = authorize_phase4_training_source(self.root, commit)
        self.assertEqual(authorized.git_commit, commit)

    def test_phase4_authorizer_accepts_clean_current_v2_checkout(self) -> None:
        _write_source_tree(self.root)
        for relative_path in set(required_additional_sources(3)) - set(
            required_additional_sources(2)
        ):
            (self.root / relative_path).unlink()
        for path in (self.root / "configs/policy_improvement_v2").rglob("*"):
            if path.is_file():
                path.unlink()
        (self.root / "configs/policy_improvement_v2/amendments").rmdir()
        (self.root / "configs/policy_improvement_v2").rmdir()
        _write_manifest(self.root, 2)
        commit = _commit_source_tree(self.root)

        authorized = authorize_phase4_training_source(self.root, commit)
        self.assertEqual(authorized.git_commit, commit)

    def test_phase4_authorizer_accepts_clean_current_v3_checkout(self) -> None:
        _write_source_tree(self.root)
        _write_manifest(self.root, 3)
        commit = _commit_source_tree(self.root)

        authorized = authorize_phase4_training_source(self.root, commit)
        self.assertEqual(authorized.git_commit, commit)

    def test_phase4_authorizer_rejects_downgraded_manifest_for_v3_checkout(
        self,
    ) -> None:
        _write_source_tree(self.root)
        for schema_version in (1, 2):
            with self.subTest(schema_version=schema_version):
                _write_manifest(self.root, schema_version)
                commit = _commit_source_tree(self.root)
                with self.assertRaisesRegex(
                    Phase4RuntimeProfileError,
                    f"declares schema {schema_version}.*satisfies schema 3",
                ):
                    authorize_phase4_training_source(self.root, commit)
                if schema_version == 1:
                    (self.root / SOURCE_MANIFEST_RELATIVE_PATH).unlink()


if __name__ == "__main__":
    unittest.main()
