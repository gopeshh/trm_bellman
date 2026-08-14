#!/usr/bin/env python3

import hashlib
import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from scripts import phase4_figure_publication as publication
from scripts.phase4_figure_publication import (
    PHASE4_FIGURE_OUTPUTS,
    Phase4FigurePublicationError,
    Phase4FigurePublicationIdentity,
    phase4_figure_generation_id,
    publish_phase4_generation,
    resolve_current_phase4_generation,
    verify_runtime_archive_sha256,
)
from utils.run_identity import canonical_json_bytes


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(tag: str) -> Phase4FigurePublicationIdentity:
    return Phase4FigurePublicationIdentity(
        summary_schema_version=4,
        summary_experiment="Phase4_2x2_norm_ablation",
        summary_sha256=_sha256(f"summary:{tag}"),
        summary_identity_sha256=_sha256(f"summary-identity:{tag}"),
        evaluator_git_commit="a" * 40,
        evaluator_source_manifest_sha256=_sha256("evaluator-source"),
        evaluator_runtime_artifact_sha256=_sha256("evaluator-runtime"),
        producer_git_commit="b" * 40,
        producer_source_manifest_sha256=_sha256("producer-source"),
        training_runtime_artifact_sha256=_sha256("training-runtime"),
        figure_git_commit="a" * 40,
        figure_source_manifest_sha256=_sha256("figure-source"),
        figure_runtime_artifact_sha256=_sha256("figure-runtime"),
        checkpoint_design_identity_sha256=_sha256("checkpoint-design"),
        checkpoint_run_count=12,
        diagnostic_dataset_name="sudoku-4x4-trivial",
        diagnostic_dataset_sha256=_sha256("diagnostic-dataset"),
        diagnostic_ordered_states_sha256=_sha256("ordered-states"),
        diagnostic_state_count=100,
    )


def _write_outputs(stage: Path, tag: str) -> None:
    for name in sorted(PHASE4_FIGURE_OUTPUTS):
        (stage / name).write_bytes(f"{tag}:{name}".encode("ascii"))


class Phase4FigurePublicationTest(unittest.TestCase):
    def _publish_old(self, owner: Path, root: Path):
        identity = _identity("old")
        return publish_phase4_generation(
            owner,
            root,
            identity,
            lambda stage: _write_outputs(stage, "old"),
            lambda: identity,
            generated_at="2026-08-14T00:00:00Z",
        )

    def _assert_pointer_complete(
        self,
        owner: Path,
        root: Path,
        allowed_tags: set[str],
    ) -> str:
        current = resolve_current_phase4_generation(owner, root)
        tags = set()
        for name in PHASE4_FIGURE_OUTPUTS:
            payload = (current.generation_path / name).read_text()
            tag, recorded_name = payload.split(":", 1)
            self.assertEqual(recorded_name, name)
            tags.add(tag)
        self.assertEqual(len(tags), 1)
        tag = tags.pop()
        self.assertIn(tag, allowed_tags)
        self.assertFalse(
            list((root / "generations").glob(".phase4-generation.stage.*"))
        )
        self.assertFalse(list(root.glob(".CURRENT.*")))
        return tag

    def test_successful_publication_and_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            identity = _identity("new")

            result = publish_phase4_generation(
                owner,
                root,
                identity,
                lambda stage: _write_outputs(stage, "new"),
                lambda: identity,
                generated_at="2026-08-14T00:00:00Z",
            )

            self.assertEqual(result.generation_id, phase4_figure_generation_id(identity))
            self.assertEqual(self._assert_pointer_complete(owner, root, {"new"}), "new")
            manifest = json.loads(
                (result.generation_path / "MANIFEST.json").read_text()
            )
            self.assertEqual(manifest["generation_id"], result.generation_id)
            self.assertEqual(len(manifest["outputs"]), 4)
            self.assertEqual(
                manifest["authorized_training_runtime_sha256"],
                identity.training_runtime_artifact_sha256,
            )

    def test_missing_extra_and_symlink_staged_files_are_rejected(self) -> None:
        cases = ("missing", "extra", "symlink", "hardlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                identity = _identity(case)

                def generate(stage: Path) -> None:
                    _write_outputs(stage, case)
                    if case == "missing":
                        (stage / sorted(PHASE4_FIGURE_OUTPUTS)[0]).unlink()
                    elif case == "extra":
                        (stage / "unexpected.txt").write_text("extra")
                    elif case == "symlink":
                        victim = sorted(PHASE4_FIGURE_OUTPUTS)[0]
                        (stage / victim).unlink()
                        (stage / victim).symlink_to(owner / "outside")
                    else:
                        first, second = sorted(PHASE4_FIGURE_OUTPUTS)[:2]
                        (stage / second).unlink()
                        os.link(stage / first, stage / second)

                with self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        generate,
                        lambda: identity,
                    )
                self.assertFalse((root / "CURRENT.json").exists())
                self.assertFalse(
                    list((root / "generations").glob(".phase4-generation.stage.*"))
                )

    def test_historical_nonlayout_content_is_rejected_without_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            root.mkdir()
            legacy = root / "fig_phase4_bar_comparison.pdf"
            legacy.write_bytes(b"historical")
            identity = _identity("new")

            with self.assertRaisesRegex(
                Phase4FigurePublicationError, "historical or unowned"
            ):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: identity,
                )

            self.assertEqual(legacy.read_bytes(), b"historical")
            self.assertEqual(set(path.name for path in root.iterdir()), {legacy.name})

    def test_existing_generation_and_generation_symlink_are_rejected(self) -> None:
        for case in ("directory", "symlink"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                generations = root / "generations"
                generations.mkdir(parents=True)
                identity = _identity("new")
                destination = generations / phase4_figure_generation_id(identity)
                if case == "directory":
                    destination.mkdir()
                else:
                    outside = owner / "outside"
                    outside.mkdir()
                    destination.symlink_to(outside, target_is_directory=True)

                with self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )

    def test_generation_and_pointer_path_escapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir) / "owner"
            owner.mkdir()
            with self.assertRaisesRegex(Phase4FigurePublicationError, "escapes"):
                publish_phase4_generation(
                    owner,
                    owner / ".." / "escape",
                    _identity("new"),
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: _identity("new"),
                )

            owner_link = Path(temp_dir) / "owner-link"
            owner_link.symlink_to(owner, target_is_directory=True)
            with self.assertRaisesRegex(
                Phase4FigurePublicationError, "owner root must not be a symlink"
            ):
                publish_phase4_generation(
                    owner_link,
                    owner_link / "publication",
                    _identity("new"),
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: _identity("new"),
                )

            root = owner / "publication"
            root.mkdir()
            outside = owner / "outside"
            outside.mkdir()
            (root / "generations").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(Phase4FigurePublicationError):
                publish_phase4_generation(
                    owner,
                    root,
                    _identity("new"),
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: _identity("new"),
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            (root / "generations").mkdir(parents=True)
            malicious = {
                "schema_version": 1,
                "generation_id": "../outside",
                "manifest_sha256": _sha256("manifest"),
            }
            (root / "CURRENT.json").write_bytes(
                canonical_json_bytes(malicious) + b"\n"
            )
            with self.assertRaises(Phase4FigurePublicationError):
                resolve_current_phase4_generation(owner, root)

    def test_resolver_is_read_only_when_publication_root_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "missing-publication"
            with self.assertRaises(Phase4FigurePublicationError):
                resolve_current_phase4_generation(owner, root)
            self.assertFalse(root.exists())

    def test_cleanup_does_not_remove_replaced_staging_or_pointer_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            identity = _identity("stage-replacement")
            replacement_paths = []

            def replace_stage(stage: Path) -> None:
                moved = stage.with_name(f"{stage.name}.moved")
                stage.rename(moved)
                stage.mkdir()
                replacement_paths.extend([stage, moved])
                raise RuntimeError("replaced staging path")

            with self.assertRaisesRegex(RuntimeError, "replaced staging"):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    replace_stage,
                    lambda: identity,
                )
            self.assertTrue(all(path.is_dir() for path in replacement_paths))

        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            self._publish_old(owner, root)
            identity = _identity("pointer-replacement")
            original_fsync = publication._fsync_fd
            replacement = []

            def replace_pointer_temp(descriptor: int, label: str) -> None:
                if label == "file:current-generation pointer":
                    pointer_path = next(root.glob(".CURRENT.*.tmp"))
                    moved = pointer_path.with_name(f"{pointer_path.name}.moved")
                    pointer_path.rename(moved)
                    pointer_path.write_bytes(b"replacement")
                    replacement.extend([pointer_path, moved])
                    raise OSError("replaced pointer temp")
                original_fsync(descriptor, label)

            with mock.patch.object(
                publication,
                "_fsync_fd",
                side_effect=replace_pointer_temp,
            ), self.assertRaises(Phase4FigurePublicationError):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: identity,
                )
            self.assertEqual(replacement[0].read_bytes(), b"replacement")
            self.assertTrue(replacement[1].is_file())
            for path in replacement:
                path.unlink()
            self.assertEqual(self._assert_pointer_complete(owner, root, {"old"}), "old")

    def test_post_rename_inode_replacement_or_output_change_keeps_old_pointer(self) -> None:
        for case in ("directory", "output"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                self._publish_old(owner, root)
                identity = _identity(f"post-rename:{case}")
                original_rename = publication._rename_directory_no_replace

                def tamper_after_rename(
                    parent: Path,
                    source_name: str,
                    destination_name: str,
                ) -> None:
                    original_rename(parent, source_name, destination_name)
                    destination = parent / destination_name
                    if case == "directory":
                        moved = destination.with_name(f"{destination.name}.moved")
                        destination.rename(moved)
                        destination.mkdir()
                    else:
                        name = sorted(PHASE4_FIGURE_OUTPUTS)[0]
                        (destination / name).write_bytes(b"tampered")

                with mock.patch.object(
                    publication,
                    "_rename_directory_no_replace",
                    side_effect=tamper_after_rename,
                ), self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )
                self.assertEqual(
                    self._assert_pointer_complete(owner, root, {"old"}),
                    "old",
                )

    def test_pointer_temp_swap_before_validation_keeps_old_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            self._publish_old(owner, root)
            identity = _identity("pointer-prevalidation-swap")
            original_verify = publication._verify_pointer_temp
            swapped_paths = []

            def swap_before_verify(path: Path, inode, expected: bytes) -> None:
                moved = path.with_name(f"{path.name}.moved")
                path.rename(moved)
                path.write_bytes(b"replacement")
                swapped_paths.extend([path, moved])
                original_verify(path, inode, expected)

            with mock.patch.object(
                publication,
                "_verify_pointer_temp",
                side_effect=swap_before_verify,
            ), self.assertRaises(Phase4FigurePublicationError):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: identity,
                )
            for path in swapped_paths:
                path.unlink()
            self.assertEqual(self._assert_pointer_complete(owner, root, {"old"}), "old")

    def test_pointer_swap_during_replace_restores_old_pointer(self) -> None:
        for replacement_kind in ("invalid", "valid-copy"):
            with self.subTest(replacement_kind=replacement_kind), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                self._publish_old(owner, root)
                identity = _identity(f"pointer-replace-swap:{replacement_kind}")
                moved_paths = []

                def replace_with_other_inode(
                    source: Path,
                    destination: Path,
                ) -> None:
                    payload = source.read_bytes()
                    moved = source.with_name(f"{source.name}.moved")
                    source.rename(moved)
                    source.write_bytes(
                        b"not-json"
                        if replacement_kind == "invalid"
                        else payload
                    )
                    os.replace(source, destination)
                    moved_paths.append(moved)

                with mock.patch.object(
                    publication,
                    "_replace_pointer",
                    side_effect=replace_with_other_inode,
                ), self.assertRaisesRegex(
                    Phase4FigurePublicationError,
                    "prior pointer was restored",
                ):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )
                self.assertEqual(
                    resolve_current_phase4_generation(owner, root).generation_id,
                    phase4_figure_generation_id(_identity("old")),
                )
                for path in moved_paths:
                    path.unlink()
                self.assertEqual(
                    self._assert_pointer_complete(owner, root, {"old"}),
                    "old",
                )

    def test_first_publication_fsyncs_each_new_path_parent(self) -> None:
        labels = (
            "directory:publication owner after child creation",
            "directory:publication root parent after child creation",
        )
        for failure_label in labels:
            with self.subTest(label=failure_label), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "parent" / "publication"
                identity = _identity(failure_label)
                original_fsync = publication._fsync_fd

                def fail_selected(descriptor: int, label: str) -> None:
                    if label == failure_label:
                        raise OSError(f"injected fsync failure: {label}")
                    original_fsync(descriptor, label)

                with mock.patch.object(
                    publication,
                    "_fsync_fd",
                    side_effect=fail_selected,
                ), self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )
                self.assertFalse((root / "CURRENT.json").exists())
                if (root / "generations").exists():
                    self.assertFalse(
                        list((root / "generations").glob(".phase4-generation.stage.*"))
                    )

                observed_labels = []

                def record_retry_fsync(descriptor: int, label: str) -> None:
                    observed_labels.append(label)
                    original_fsync(descriptor, label)

                retry_identity = _identity(f"retry:{failure_label}")
                with mock.patch.object(
                    publication,
                    "_fsync_fd",
                    side_effect=record_retry_fsync,
                ):
                    publish_phase4_generation(
                        owner,
                        root,
                        retry_identity,
                        lambda stage: _write_outputs(stage, "retry"),
                        lambda: retry_identity,
                    )
                self.assertIn(
                    "directory:publication owner after child creation",
                    observed_labels,
                )
                self.assertIn(
                    "directory:publication root parent after child creation",
                    observed_labels,
                )
                self.assertEqual(
                    self._assert_pointer_complete(owner, root, {"retry"}),
                    "retry",
                )

    def test_concurrently_visible_path_components_are_refsynced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "parent" / "publication"
            root.mkdir(parents=True)
            identity = _identity("concurrently-visible")
            observed_labels = []
            original_fsync = publication._fsync_fd

            def record_fsync(descriptor: int, label: str) -> None:
                observed_labels.append(label)
                original_fsync(descriptor, label)

            with mock.patch.object(
                publication,
                "_fsync_fd",
                side_effect=record_fsync,
            ):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "visible"),
                    lambda: identity,
                )
            self.assertIn(
                "directory:publication owner after child creation",
                observed_labels,
            )
            self.assertIn(
                "directory:publication root parent after child creation",
                observed_labels,
            )
            self.assertIn(
                "directory:publication root after generations validation",
                observed_labels,
            )
            self.assertEqual(
                self._assert_pointer_complete(owner, root, {"visible"}),
                "visible",
            )

    def test_retry_refsyncs_visible_generations_after_failed_parent_flush(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            identity = _identity("generations-retry")
            failure_label = (
                "directory:publication root after generations validation"
            )
            original_fsync = publication._fsync_fd
            failed_once = False
            observed_retry = False

            def fail_once(descriptor: int, label: str) -> None:
                nonlocal failed_once, observed_retry
                if label == failure_label:
                    if not failed_once:
                        failed_once = True
                        raise OSError(f"injected fsync failure: {label}")
                    observed_retry = True
                original_fsync(descriptor, label)

            with mock.patch.object(
                publication,
                "_fsync_fd",
                side_effect=fail_once,
            ):
                with self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )
                published = publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: identity,
                )

            self.assertTrue(failed_once)
            self.assertTrue(observed_retry)
            self.assertEqual(
                resolve_current_phase4_generation(owner, root).generation_id,
                published.generation_id,
            )

    def test_runtime_archive_requires_absolute_stable_regular_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "runtime.par"
            archive.write_bytes(b"runtime")
            digest = hashlib.sha256(b"runtime").hexdigest()
            self.assertEqual(
                verify_runtime_archive_sha256(archive, digest, "runtime"),
                digest,
            )
            with self.assertRaisesRegex(Phase4FigurePublicationError, "absolute"):
                verify_runtime_archive_sha256("runtime.par", digest, "runtime")
            with self.assertRaisesRegex(Phase4FigurePublicationError, "differ"):
                verify_runtime_archive_sha256(archive, "0" * 64, "runtime")
            link = root / "runtime-link.par"
            link.symlink_to(archive)
            with self.assertRaises(Phase4FigurePublicationError):
                verify_runtime_archive_sha256(link, digest, "runtime")

    def test_identity_change_after_generation_leaves_old_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            self._publish_old(owner, root)
            identity = _identity("new")

            with self.assertRaisesRegex(Phase4FigurePublicationError, "identity changed"):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: replace(identity, summary_sha256=_sha256("changed")),
                )

            self.assertEqual(self._assert_pointer_complete(owner, root, {"old"}), "old")

    def test_writer_failure_after_each_output_never_changes_pointer(self) -> None:
        names = sorted(PHASE4_FIGURE_OUTPUTS)
        for failure_name in names:
            with self.subTest(failure_name=failure_name), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                self._publish_old(owner, root)
                identity = _identity(f"new:{failure_name}")

                def generate(stage: Path) -> None:
                    for name in names:
                        (stage / name).write_bytes(f"new:{name}".encode())
                        if name == failure_name:
                            raise RuntimeError(f"injected writer failure: {name}")

                with self.assertRaisesRegex(RuntimeError, "injected writer"):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        generate,
                        lambda: identity,
                    )
                self.assertEqual(
                    self._assert_pointer_complete(owner, root, {"old"}),
                    "old",
                )

    def _run_fsync_failure(self, failure_label: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            self._publish_old(owner, root)
            identity = _identity(f"new:{failure_label}")
            original_fsync = publication._fsync_fd

            def fail_selected(descriptor: int, label: str) -> None:
                if label == failure_label:
                    raise OSError(f"injected fsync failure: {label}")
                original_fsync(descriptor, label)

            with mock.patch.object(
                publication,
                "_fsync_fd",
                side_effect=fail_selected,
            ), self.assertRaises(Phase4FigurePublicationError):
                publish_phase4_generation(
                    owner,
                    root,
                    identity,
                    lambda stage: _write_outputs(stage, "new"),
                    lambda: identity,
                )
            return self._assert_pointer_complete(owner, root, {"old", "new"})

    def test_each_file_and_directory_fsync_failure_preserves_complete_pointer(self) -> None:
        labels = [
            *(f"file:staged output {name}" for name in sorted(PHASE4_FIGURE_OUTPUTS)),
            "file:generation manifest",
            "directory:staging generation",
            "directory:staging generation final",
            "directory:generations",
            "directory:publication root after generation",
            "file:current-generation pointer",
            "directory:publication root after pointer",
        ]
        for label in labels:
            with self.subTest(label=label):
                observed = self._run_fsync_failure(label)
                if label == "directory:publication root after pointer":
                    self.assertEqual(observed, "new")
                else:
                    self.assertEqual(observed, "old")

    def test_manifest_and_pointer_write_failures_preserve_complete_pointer(self) -> None:
        for target in ("generation manifest", "current-generation pointer"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                self._publish_old(owner, root)
                identity = _identity(f"new:{target}")
                original_write = publication._write_all

                def fail_selected(descriptor: int, payload: bytes, label: str) -> None:
                    if label == target:
                        raise OSError(f"injected write failure: {label}")
                    original_write(descriptor, payload, label)

                with mock.patch.object(
                    publication,
                    "_write_all",
                    side_effect=fail_selected,
                ), self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )
                self.assertEqual(
                    self._assert_pointer_complete(owner, root, {"old"}),
                    "old",
                )

    def test_generation_rename_and_pointer_replace_failures_preserve_old_pointer(self) -> None:
        for target in ("generation", "pointer"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp_dir:
                owner = Path(temp_dir)
                root = owner / "publication"
                self._publish_old(owner, root)
                identity = _identity(f"new:{target}")
                patch_target = (
                    "_rename_directory_no_replace"
                    if target == "generation"
                    else "_replace_pointer"
                )
                with mock.patch.object(
                    publication,
                    patch_target,
                    side_effect=Phase4FigurePublicationError(
                        f"injected {target} failure"
                    ),
                ), self.assertRaises(Phase4FigurePublicationError):
                    publish_phase4_generation(
                        owner,
                        root,
                        identity,
                        lambda stage: _write_outputs(stage, "new"),
                        lambda: identity,
                    )
                self.assertEqual(
                    self._assert_pointer_complete(owner, root, {"old"}),
                    "old",
                )

    def test_concurrent_same_generation_has_one_winner_and_no_mixed_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            identity = _identity("same")
            barrier = threading.Barrier(2)
            results = []
            errors = []

            def worker() -> None:
                def generate(stage: Path) -> None:
                    _write_outputs(stage, "same")
                    barrier.wait(timeout=5)

                try:
                    results.append(
                        publish_phase4_generation(
                            owner,
                            root,
                            identity,
                            generate,
                            lambda: identity,
                            generated_at="2026-08-14T00:00:00Z",
                        )
                    )
                except Exception as error:
                    errors.append(error)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(results), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], Phase4FigurePublicationError)
            self.assertEqual(self._assert_pointer_complete(owner, root, {"same"}), "same")

    def test_concurrent_different_generations_leave_one_complete_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = Path(temp_dir)
            root = owner / "publication"
            barrier = threading.Barrier(2)
            results = []
            errors = []

            def worker(tag: str) -> None:
                identity = _identity(tag)

                def generate(stage: Path) -> None:
                    _write_outputs(stage, tag)
                    barrier.wait(timeout=5)

                try:
                    results.append(
                        publish_phase4_generation(
                            owner,
                            root,
                            identity,
                            generate,
                            lambda: identity,
                            generated_at="2026-08-14T00:00:00Z",
                        )
                    )
                except Exception as error:
                    errors.append(error)

            threads = [
                threading.Thread(target=worker, args=(tag,))
                for tag in ("first", "second")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(
                self._assert_pointer_complete(owner, root, {"first", "second"})
                in {"first", "second"},
                True,
            )
            self.assertTrue(all(result.generation_path.is_dir() for result in results))


if __name__ == "__main__":
    unittest.main()
