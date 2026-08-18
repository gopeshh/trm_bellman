#!/usr/bin/env fbpython
"""Tests for authenticated policy-improvement dataset attribution."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import dataset.build_policy_improvement_4x4 as builder


_ATTESTATION = {
    "git_commit": "a" * 40,
    "launcher_sha256": "b" * 64,
    "runtime_sha256": "c" * 64,
    "source_manifest_sha256": "d" * 64,
}


class PolicyDatasetBuilderTest(unittest.TestCase):
    def test_publication_never_replaces_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            output = root / "output"
            stage.mkdir()
            output.mkdir()
            (stage / "new").write_text("new", encoding="ascii")
            (output / "old").write_text("old", encoding="ascii")
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "already exists",
            ):
                builder._publish_directory_no_replace(stage, output)
            self.assertEqual((stage / "new").read_text(encoding="ascii"), "new")
            self.assertEqual((output / "old").read_text(encoding="ascii"), "old")

    def test_build_and_verify_require_same_external_producer(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "DEFAULT_SPLITS",
            tiny_splits,
        ):
            output = Path(directory) / "corpus"
            built = builder.build_dataset(
                output,
                owner_root=Path(directory),
                producer_attestation=_ATTESTATION,
            )
            self.assertEqual(built["producer_source"], _ATTESTATION)
            self.assertEqual(
                set(built["split_ordered_record_sha256"]),
                {"train", "validation", "test"},
            )
            self.assertTrue(
                all(
                    len(value) == 64
                    for value in built["split_ordered_record_sha256"].values()
                )
            )
            self.assertEqual(
                builder.verify_dataset(
                    output,
                    owner_root=Path(directory),
                    expected_producer=_ATTESTATION,
                ),
                built,
            )
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="ascii")
            )
            build_config = json.loads(
                (output / "build_config.json").read_text(encoding="ascii")
            )
            self.assertEqual(manifest["producer_source"], _ATTESTATION)
            self.assertEqual(build_config["producer_source"], _ATTESTATION)

            wrong = dict(_ATTESTATION)
            wrong["runtime_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "differs",
            ):
                builder.verify_dataset(
                    output,
                    owner_root=Path(directory),
                    expected_producer=wrong,
                )

    def test_stage0_verification_never_opens_test_content(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                builder,
                "DEFAULT_SPLITS",
                tiny_splits,
            ),
        ):
            owner = Path(directory)
            output = owner / "corpus"
            built = builder.build_dataset(
                output,
                owner_root=owner,
                producer_attestation=_ATTESTATION,
            )
            opened: list[Path] = []
            real_open = Path.open

            def tracking_open(path: Path, *args: object, **kwargs: object):
                opened.append(path)
                if output / "test" == path or output / "test" in path.parents:
                    raise AssertionError(f"held-out test path was opened: {path}")
                return real_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", tracking_open):
                verified = builder.verify_dataset(
                    output,
                    owner_root=owner,
                    expected_producer=_ATTESTATION,
                    verify_test_content=False,
                )
            self.assertEqual(verified, built)
            self.assertTrue(any(output / "train" in path.parents for path in opened))
            self.assertTrue(
                any(output / "validation" in path.parents for path in opened)
            )
            self.assertFalse(any(output / "test" in path.parents for path in opened))

            test_array = output / "test/all__inputs.npy"
            test_array.write_bytes(test_array.read_bytes() + b"tampered")
            builder.verify_dataset(
                output,
                owner_root=owner,
                expected_producer=_ATTESTATION,
                verify_test_content=False,
            )
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "hash differs",
            ):
                builder.verify_dataset(
                    output,
                    owner_root=owner,
                    expected_producer=_ATTESTATION,
                    verify_test_content=True,
                )

    def test_stage0_verification_still_hashes_train_and_validation_content(
        self,
    ) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        for split in ("train", "validation"):
            with (
                self.subTest(split=split),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(builder, "DEFAULT_SPLITS", tiny_splits),
            ):
                owner = Path(directory)
                output = owner / "corpus"
                builder.build_dataset(
                    output,
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
                target = output / split / "all__inputs.npy"
                target.write_bytes(target.read_bytes() + b"tampered")
                with self.assertRaisesRegex(
                    builder.PolicyImprovementDatasetError,
                    "hash differs",
                ):
                    builder.verify_dataset(
                        output,
                        owner_root=owner,
                        expected_producer=_ATTESTATION,
                        verify_test_content=False,
                    )

    def test_explicit_train_only_verification_opens_no_validation_or_test_content(
        self,
    ) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(builder, "DEFAULT_SPLITS", tiny_splits),
        ):
            owner = Path(directory)
            output = owner / "corpus"
            built = builder.build_dataset(
                output,
                owner_root=owner,
                producer_attestation=_ATTESTATION,
            )
            opened: list[Path] = []
            real_open = Path.open

            def tracking_open(path: Path, *args: object, **kwargs: object):
                opened.append(path)
                if any(
                    output / split == path or output / split in path.parents
                    for split in ("validation", "test")
                ):
                    raise AssertionError(f"non-training content was opened: {path}")
                return real_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", tracking_open):
                verified = builder.verify_dataset(
                    output,
                    owner_root=owner,
                    expected_producer=_ATTESTATION,
                    verify_content_splits={"train"},
                )
            self.assertEqual(verified, built)
            self.assertTrue(any(output / "train" in path.parents for path in opened))
            self.assertFalse(
                any(
                    output / split in path.parents
                    for split in ("validation", "test")
                    for path in opened
                )
            )

            for split in ("validation", "test"):
                target = output / split / "all__inputs.npy"
                target.write_bytes(target.read_bytes() + b"tampered")
            builder.verify_dataset(
                output,
                owner_root=owner,
                expected_producer=_ATTESTATION,
                verify_content_splits={"train"},
            )

    def test_owner_root_is_private_and_output_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            owner = Path(directory)
            outside = owner.parent / "outside-policy-corpus"
            os.chmod(owner, 0o755)
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "mode-0700",
            ):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            os.chmod(owner, 0o700)
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "direct child",
            ):
                builder.build_dataset(
                    outside,
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            private = owner / "private"
            private.mkdir(mode=0o700)
            alias = owner / "private-alias"
            alias.symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "alias or symlink",
            ):
                builder.build_dataset(
                    alias / "corpus",
                    owner_root=alias,
                    producer_attestation=_ATTESTATION,
                )

    def test_stage_swap_is_rejected_and_cleanup_preserves_replacement(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        real_verify = builder.verify_dataset
        swapped: list[Path] = []

        def verify_then_swap(
            root: str | Path,
            *,
            owner_root: str | Path,
            expected_producer: Mapping[str, object],
        ) -> dict[str, object]:
            result = real_verify(
                root,
                owner_root=owner_root,
                expected_producer=expected_producer,
            )
            path = Path(root)
            if path.name.startswith(".corpus.staging-") and not swapped:
                original = path.with_name(f"{path.name}.original")
                path.rename(original)
                path.mkdir(mode=0o700)
                (path / "replacement").write_text("keep", encoding="ascii")
                swapped.extend((path, original))
            return result

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "DEFAULT_SPLITS",
            tiny_splits,
        ), mock.patch.object(
            builder,
            "verify_dataset",
            side_effect=verify_then_swap,
        ):
            owner = Path(directory)
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "staging directory changed",
            ):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            replacement, original = swapped
            self.assertEqual(
                (replacement / "replacement").read_text(encoding="ascii"),
                "keep",
            )
            self.assertTrue((original / "MANIFEST.json").is_file())
            self.assertFalse((owner / "corpus").exists())

    def test_owner_swap_before_publish_is_rejected(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        real_verify = builder.verify_dataset
        swapped = False

        def verify_then_swap_owner(
            root: str | Path,
            *,
            owner_root: str | Path,
            expected_producer: Mapping[str, object],
        ) -> dict[str, object]:
            nonlocal swapped
            result = real_verify(
                root,
                owner_root=owner_root,
                expected_producer=expected_producer,
            )
            path = Path(root)
            if path.name.startswith(".corpus.staging-") and not swapped:
                owner = Path(owner_root)
                owner.rename(owner.with_name("owner-original"))
                owner.mkdir(mode=0o700)
                swapped = True
            return result

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "DEFAULT_SPLITS",
            tiny_splits,
        ), mock.patch.object(
            builder,
            "verify_dataset",
            side_effect=verify_then_swap_owner,
        ):
            outer = Path(directory)
            owner = outer / "owner"
            owner.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "owner root changed",
            ):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            self.assertEqual(list(owner.iterdir()), [])
            original = outer / "owner-original"
            self.assertFalse(
                any(
                    path.name.startswith(".corpus.staging-")
                    for path in original.iterdir()
                )
            )
            self.assertFalse((original / "corpus").exists())

    def test_failure_before_rename_removes_only_exact_staging_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "_write_bytes",
            side_effect=OSError("injected write failure"),
        ):
            owner = Path(directory)
            with self.assertRaisesRegex(OSError, "injected write failure"):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            self.assertEqual(list(owner.iterdir()), [])

    def test_rename_failure_removes_staging_and_never_publishes(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "DEFAULT_SPLITS",
            tiny_splits,
        ), mock.patch.object(
            builder,
            "_rename_directory_no_replace",
            side_effect=OSError("injected rename failure"),
        ):
            owner = Path(directory)
            with self.assertRaisesRegex(OSError, "injected rename failure"):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            self.assertEqual(list(owner.iterdir()), [])

    def test_staging_parent_fsync_failure_cleans_exact_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "_fsync_directory_descriptor",
            side_effect=OSError("injected staging parent fsync failure"),
        ):
            owner = Path(directory)
            with self.assertRaisesRegex(OSError, "staging parent fsync failure"):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            self.assertEqual(list(owner.iterdir()), [])

    def test_parent_fsync_failure_leaves_complete_retryable_dataset(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        real_fsync = builder._fsync_directory_descriptor
        calls = 0

        def fail_after_publication(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected parent fsync failure")
            real_fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "DEFAULT_SPLITS",
            tiny_splits,
        ):
            owner = Path(directory)
            output = owner / "corpus"
            with mock.patch.object(
                builder,
                "_fsync_directory_descriptor",
                side_effect=fail_after_publication,
            ), self.assertRaisesRegex(OSError, "parent fsync failure"):
                builder.build_dataset(
                    output,
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            self.assertTrue((output / "MANIFEST.json").is_file())
            retried = builder.build_dataset(
                output,
                owner_root=owner,
                producer_attestation=_ATTESTATION,
            )
            self.assertEqual(retried["total_records"], 3)
            self.assertFalse(
                any(
                    path.name.startswith(".corpus.staging-") for path in owner.iterdir()
                )
            )

    def test_published_path_must_name_the_exact_staged_inode(self) -> None:
        tiny_splits = (
            ("train", 1, 26081401),
            ("validation", 1, 26081402),
            ("test", 1, 26081403),
        )
        real_rename = builder._rename_directory_no_replace

        def rename_then_replace(
            source_descriptor: int,
            source_name: str,
            destination_descriptor: int,
            destination_name: str,
        ) -> None:
            real_rename(
                source_descriptor,
                source_name,
                destination_descriptor,
                destination_name,
            )
            os.rename(
                destination_name,
                f".{destination_name}.published-original",
                src_dir_fd=destination_descriptor,
                dst_dir_fd=destination_descriptor,
            )
            os.mkdir(destination_name, mode=0o700, dir_fd=destination_descriptor)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            builder,
            "DEFAULT_SPLITS",
            tiny_splits,
        ), mock.patch.object(
            builder,
            "_rename_directory_no_replace",
            side_effect=rename_then_replace,
        ):
            owner = Path(directory)
            with self.assertRaisesRegex(
                builder.PolicyImprovementDatasetError,
                "differs from the staged directory",
            ):
                builder.build_dataset(
                    owner / "corpus",
                    owner_root=owner,
                    producer_attestation=_ATTESTATION,
                )
            self.assertEqual(list((owner / "corpus").iterdir()), [])
            self.assertTrue(
                (owner / ".corpus.published-original" / "MANIFEST.json").is_file()
            )

    def test_attestation_inventory_and_cli_are_fail_closed(self) -> None:
        missing = dict(_ATTESTATION)
        missing.pop("launcher_sha256")
        extra = {**_ATTESTATION, "caller_assertion": "trusted"}
        uppercase = {**_ATTESTATION, "runtime_sha256": "A" * 64}
        for value in (missing, extra, uppercase):
            with self.subTest(value=value), self.assertRaises(
                builder.PolicyImprovementDatasetError
            ):
                builder._producer_attestation(value)
        with self.assertRaisesRegex(
            builder.PolicyImprovementDatasetError,
            "authenticated launcher",
        ):
            builder.main(
                [
                    "build",
                    "--owner-root",
                    "/tmp",
                    "--output-root",
                    "/tmp/not-created",
                ]
            )


if __name__ == "__main__":
    unittest.main()
