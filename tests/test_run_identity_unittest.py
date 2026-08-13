from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from utils.run_identity import (
    RUN_IDENTITY_SCHEMA_VERSION,
    RunIdentityError,
    assert_git_files_match_head,
    assert_matching_run_identity,
    build_checkpoint_lineage,
    build_run_identity,
    canonical_json_bytes,
    canonical_json_sha256,
    discover_clean_git_source,
    file_sha256,
    git_files_at_head,
    run_identity_sha256,
    validate_checkpoint_lineage,
    validate_run_identity,
    validate_upi_effective_config,
)


class RunIdentityTest(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    def _clean_repository(self, root: Path) -> str:
        self._git(root, "init", "--quiet")
        self._git(root, "config", "user.email", "test@example.invalid")
        self._git(root, "config", "user.name", "Run Identity Test")
        (root / "tracked.txt").write_text("version one\n", encoding="ascii")
        self._git(root, "add", "tracked.txt")
        self._git(root, "commit", "--quiet", "-m", "initial")
        return self._git(root, "rev-parse", "HEAD")

    def _identity(
        self,
        root: Path,
        *,
        initialization_kind: str = "random",
        initialization_artifact_sha256: str | None = None,
    ) -> dict:
        return build_run_identity(
            run_id="confirmatory.seed7",
            training_seed=7,
            git_lookup_root=root,
            effective_config={
                "effective_config_schema_version": 1,
                "algorithm": "upi_trm",
                "training_protocol": "fixed_base_exact",
                "backbone": "trm",
                "rl_config": {
                    "gamma": 0.9,
                    "K": 3,
                    "latent_projection_mode": "enabled",
                    "latent_ball_radius": 10.0,
                },
                "model_config": {
                    "hidden_size": 64,
                    "rl_latent_projection_mode": "enabled",
                    "rl_latent_ball_radius": 10.0,
                },
                "execution_device": "cpu",
                "runtime_fingerprint_sha256": "a" * 64,
                "dataset": {
                    "train_split": "train",
                    "eval_split": "heldout",
                    "train_record_count": 1,
                    "eval_record_count": 1,
                },
                "budget": {
                    "outer_train_steps": 10,
                    "environment_interactions": 100,
                },
                "schedule": {
                    "log_outer_interval": 1,
                    "eval_outer_interval": 2,
                    "save_outer_interval": 5,
                    "log_environment_interval": 10,
                    "eval_environment_interval": 20,
                    "save_environment_interval": 50,
                },
                "evaluation": {
                    "episode_count": 1,
                    "seed": 1729,
                    "pool_size": 1,
                },
                "puzzle_embedding_optimizer": {
                    "learning_rate": 0.01,
                    "weight_decay": 0.1,
                },
                "imitation": {"enabled": False, "epochs": 0},
                "external_logging": "disabled",
                "debug_checks": False,
                "config_source_sha256s": [],
            },
            dataset_provenance={
                "provenance_schema_version": 1,
                "ordered_records": {"train": ["a"], "eval": ["b"]},
            },
            initialization_kind=initialization_kind,
            initialization_artifact_sha256=initialization_artifact_sha256,
        )

    def test_canonical_json_is_order_stable_and_strict(self) -> None:
        left = {"z": [1, True, None], "a": {"b": 2.5, "a": "x"}}
        right = {"a": {"a": "x", "b": 2.5}, "z": [1, True, None]}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(canonical_json_sha256(left), canonical_json_sha256(right))
        self.assertEqual(
            json.loads(canonical_json_bytes(left).decode("ascii")),
            right,
        )

        invalid_values = (
            {"value": math.nan},
            {"value": math.inf},
            {1: "non-string key"},
            {"value": (1, 2)},
            {"value": b"bytes"},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(RunIdentityError):
                    canonical_json_bytes(value)

    def test_build_binds_clean_commit_without_persisting_lookup_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = self._clean_repository(root)
            identity = self._identity(root)

            self.assertEqual(identity["run_identity_schema_version"], 1)
            self.assertEqual(identity["producer"]["git_commit"], commit)
            self.assertIs(identity["producer"]["git_clean"], True)
            self.assertEqual(identity["training_seed"], 7)
            self.assertEqual(
                identity["effective_config_sha256"],
                canonical_json_sha256(identity["effective_config"]),
            )
            encoded = canonical_json_bytes(identity)
            self.assertNotIn(str(root).encode("ascii"), encoded)
            self.assertEqual(
                run_identity_sha256(identity), hashlib.sha256(encoded).hexdigest()
            )

    def test_schema_two_effective_config_binds_seed_data_and_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            provenance = {
                "provenance_schema_version": 1,
                "ordered_records": {"train": ["a"], "eval": ["b"]},
            }
            legacy = self._identity(root)
            effective = copy.deepcopy(legacy["effective_config"])
            effective["effective_config_schema_version"] = 3
            effective["registration"] = {
                "cell": "C2_UPI_TRM",
                "tier": "confirmatory",
                "run_id": "confirmatory.seed7",
                "training_seed": 7,
                "attempt_index": 0,
                "registry_sha256": "f" * 64,
            }
            effective["dataset_provenance_sha256"] = canonical_json_sha256(
                provenance
            )
            effective["initialization"] = {
                "kind": "random",
                "artifact_sha256": None,
            }

            identity = build_run_identity(
                run_id="confirmatory.seed7",
                training_seed=7,
                git_lookup_root=root,
                effective_config=effective,
                dataset_provenance=provenance,
                initialization_kind="random",
                initialization_artifact_sha256=None,
            )
            self.assertEqual(
                identity["effective_config"]["registration"]["cell"],
                "C2_UPI_TRM",
            )
            self.assertEqual(
                identity["effective_config"]["registration"]["attempt_index"],
                0,
            )

            historical_schema_two = copy.deepcopy(effective)
            historical_schema_two["effective_config_schema_version"] = 2
            historical_schema_two["registration"].pop("attempt_index")
            historical_identity = build_run_identity(
                run_id="confirmatory.seed7",
                training_seed=7,
                git_lookup_root=root,
                effective_config=historical_schema_two,
                dataset_provenance=provenance,
                initialization_kind="random",
                initialization_artifact_sha256=None,
            )
            self.assertEqual(
                historical_identity["effective_config"][
                    "effective_config_schema_version"
                ],
                2,
            )

            current_schema_four = copy.deepcopy(effective)
            current_schema_four["effective_config_schema_version"] = 4
            current_schema_four["runtime_artifact_sha256"] = "e" * 64
            current_identity = build_run_identity(
                run_id="confirmatory.seed7",
                training_seed=7,
                git_lookup_root=root,
                effective_config=current_schema_four,
                dataset_provenance=provenance,
                initialization_kind="random",
                initialization_artifact_sha256=None,
            )
            self.assertEqual(
                current_identity["effective_config"][
                    "runtime_artifact_sha256"
                ],
                "e" * 64,
            )

            for invalid in (None, "E" * 64):
                with self.subTest(runtime_artifact_sha256=invalid):
                    invalid_schema_four = copy.deepcopy(current_schema_four)
                    if invalid is None:
                        invalid_schema_four.pop("runtime_artifact_sha256")
                    else:
                        invalid_schema_four["runtime_artifact_sha256"] = invalid
                    with self.assertRaises(RunIdentityError):
                        validate_upi_effective_config(invalid_schema_four)

            wrong_seed = copy.deepcopy(effective)
            wrong_seed["registration"]["training_seed"] = 8
            with self.assertRaisesRegex(RunIdentityError, "seed differs"):
                build_run_identity(
                    run_id="confirmatory.seed7",
                    training_seed=7,
                    git_lookup_root=root,
                    effective_config=wrong_seed,
                    dataset_provenance=provenance,
                    initialization_kind="random",
                    initialization_artifact_sha256=None,
                )

            invalid_attempt = copy.deepcopy(effective)
            invalid_attempt["registration"]["attempt_index"] = -1
            with self.assertRaisesRegex(RunIdentityError, "attempt_index"):
                build_run_identity(
                    run_id="confirmatory.seed7",
                    training_seed=7,
                    git_lookup_root=root,
                    effective_config=invalid_attempt,
                    dataset_provenance=provenance,
                    initialization_kind="random",
                    initialization_artifact_sha256=None,
                )

    def test_git_discovery_rejects_untracked_and_tracked_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            self.assertTrue(discover_clean_git_source(root)["git_clean"])

            (root / "untracked.txt").write_text("new\n", encoding="ascii")
            with self.assertRaisesRegex(RunIdentityError, "not clean"):
                discover_clean_git_source(root)
            (root / "untracked.txt").unlink()

            (root / "tracked.txt").write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(RunIdentityError, "not clean"):
                discover_clean_git_source(root)

    def test_head_verification_rejects_unsafe_index_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            assert_git_files_match_head(root, ["tracked.txt"])
            self._git(root, "update-index", "--assume-unchanged", "tracked.txt")
            try:
                with self.assertRaisesRegex(RunIdentityError, "unsafe index flags"):
                    assert_git_files_match_head(root, ["tracked.txt"])
            finally:
                self._git(
                    root,
                    "update-index",
                    "--no-assume-unchanged",
                    "tracked.txt",
                )

    def test_git_files_at_head_uses_committed_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            (root / "ignored.py").write_text("ignored\n", encoding="ascii")

            self.assertEqual(
                git_files_at_head(root, ["tracked.txt"]),
                ["tracked.txt"],
            )

    def test_initialization_artifact_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            artifact = root.parent / f"{root.name}.weights"
            artifact_path = str(artifact)
            artifact.write_bytes(b"model weights")
            try:
                digest = file_sha256(artifact)
                identity = self._identity(
                    root,
                    initialization_kind="weights_checkpoint",
                    initialization_artifact_sha256=digest,
                )
            finally:
                artifact.unlink(missing_ok=True)

        self.assertEqual(identity["initialization"]["artifact_sha256"], digest)
        self.assertNotIn(artifact_path, canonical_json_bytes(identity).decode("ascii"))

    def test_validation_rejects_ambiguous_or_inconsistent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            base = self._identity(root)

        cases: list[tuple[str, dict, str]] = []

        unknown = copy.deepcopy(base)
        unknown["unexpected"] = True
        cases.append(("unknown", unknown, "unknown fields"))

        bad_run_id = copy.deepcopy(base)
        bad_run_id["run_id"] = "../escape"
        cases.append(("run_id", bad_run_id, "run_id"))

        bool_seed = copy.deepcopy(base)
        bool_seed["training_seed"] = True
        cases.append(("bool_seed", bool_seed, "training_seed"))

        negative_seed = copy.deepcopy(base)
        negative_seed["training_seed"] = -1
        cases.append(("negative_seed", negative_seed, "training_seed"))

        bad_commit = copy.deepcopy(base)
        bad_commit["producer"]["git_commit"] = "A" * 40
        cases.append(("commit", bad_commit, "git_commit"))

        dirty = copy.deepcopy(base)
        dirty["producer"]["git_clean"] = False
        cases.append(("dirty", dirty, "git_clean"))

        config_mismatch = copy.deepcopy(base)
        config_mismatch["effective_config"]["rl_config"]["K"] = 5
        cases.append(("config_hash", config_mismatch, "does not match"))

        incomplete_effective_config = copy.deepcopy(base)
        incomplete_effective_config["effective_config"].pop("schedule")
        cases.append(
            (
                "incomplete_effective_config",
                incomplete_effective_config,
                "missing fields",
            )
        )

        random_with_artifact = copy.deepcopy(base)
        random_with_artifact["initialization"]["artifact_sha256"] = "a" * 64
        cases.append(("random_artifact", random_with_artifact, "must not name"))

        checkpoint_without_artifact = copy.deepcopy(base)
        checkpoint_without_artifact["initialization"]["kind"] = "weights_checkpoint"
        cases.append(("missing_artifact", checkpoint_without_artifact, "64 lowercase"))

        fabricated_kind = copy.deepcopy(base)
        fabricated_kind["initialization"] = {
            "kind": "fabricated",
            "artifact_sha256": "a" * 64,
        }
        cases.append(("fabricated_kind", fabricated_kind, "random.*weights_checkpoint"))

        for name, identity, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RunIdentityError, message):
                    validate_run_identity(identity)

    def test_schema_versions_require_exact_non_bool_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            base = self._identity(root)

        root_lineage = build_checkpoint_lineage(
            parent_checkpoint_sha256=None,
            parent_checkpoint_step=None,
            parent_environment_steps=None,
        )
        for invalid in (True, 1.0, "1"):
            with self.subTest(boundary="effective_config", invalid=invalid):
                effective_config = copy.deepcopy(base["effective_config"])
                effective_config["effective_config_schema_version"] = invalid
                with self.assertRaises(RunIdentityError):
                    validate_upi_effective_config(effective_config)

            with self.subTest(boundary="run_identity", invalid=invalid):
                identity = copy.deepcopy(base)
                identity["run_identity_schema_version"] = invalid
                with self.assertRaises(RunIdentityError):
                    validate_run_identity(identity)

            with self.subTest(boundary="checkpoint_lineage", invalid=invalid):
                lineage = copy.deepcopy(root_lineage)
                lineage["checkpoint_lineage_schema_version"] = invalid
                with self.assertRaises(RunIdentityError):
                    validate_checkpoint_lineage(lineage)

    def test_exact_comparison_rejects_each_valid_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._clean_repository(root)
            saved = self._identity(root)

        self.assertEqual(assert_matching_run_identity(saved, saved), saved)

        changed_seed = copy.deepcopy(saved)
        changed_seed["training_seed"] = 8
        with self.assertRaisesRegex(RunIdentityError, "training_seed"):
            assert_matching_run_identity(saved, changed_seed)

        changed_config = copy.deepcopy(saved)
        changed_config["effective_config"]["rl_config"]["K"] = 5
        changed_config["effective_config_sha256"] = canonical_json_sha256(
            changed_config["effective_config"]
        )
        with self.assertRaisesRegex(RunIdentityError, "effective_config"):
            assert_matching_run_identity(saved, changed_config)

        changed_dataset = copy.deepcopy(saved)
        changed_dataset["dataset_provenance_sha256"] = "f" * 64
        with self.assertRaisesRegex(RunIdentityError, "dataset_provenance_sha256"):
            assert_matching_run_identity(saved, changed_dataset)

    def test_non_repository_and_unborn_repository_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RunIdentityError):
                discover_clean_git_source(root)
            self._git(root, "init", "--quiet")
            with self.assertRaises(RunIdentityError):
                discover_clean_git_source(root)

    def test_checkpoint_lineage_is_complete_and_content_addressed(self) -> None:
        root = build_checkpoint_lineage(
            parent_checkpoint_sha256=None,
            parent_checkpoint_step=None,
            parent_environment_steps=None,
        )
        self.assertIsNone(root["parent_checkpoint_sha256"])

        child = build_checkpoint_lineage(
            parent_checkpoint_sha256="a" * 64,
            parent_checkpoint_step=3,
            parent_environment_steps=17,
        )
        self.assertEqual(validate_checkpoint_lineage(child), child)

        partial = copy.deepcopy(child)
        partial["parent_checkpoint_step"] = None
        with self.assertRaisesRegex(RunIdentityError, "all null or all set"):
            validate_checkpoint_lineage(partial)
if __name__ == "__main__":
    unittest.main()
