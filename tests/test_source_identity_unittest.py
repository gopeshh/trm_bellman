"""Tests for the source manifest embedded in confirmatory binaries."""

import copy
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from utils.source_identity import (
    SOURCE_MANIFEST_RELATIVE_PATH,
    SourceIdentityError,
    assert_runtime_archive_sources_match_manifest,
    behavior_source_relative_paths,
    build_producer_source_manifest,
    validate_producer_source_manifest,
)


class TestSourceIdentity(unittest.TestCase):
    def _source_tree(self, root: Path) -> None:
        for relative_path in ("upi_trm_train.py", "puzzle_dataset.py"):
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


if __name__ == "__main__":
    unittest.main()
