import csv
import copy
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.aggregate_exp1_results import aggregate_radius_sweep
from scripts.aggregate_hard4x4_trusted_baselines import (
    _parse_final_success_rate,
    _write_summary_md,
)
from utils.dataset_provenance import (
    DatasetProvenanceError,
    assert_matching_dataset_provenance,
    build_dataset_provenance,
    dataset_source_build_metadata,
    ordered_pool_sha256,
    ordered_record_sha256,
    sample_sha256,
    validate_dataset_provenance,
)


class TestResultProvenance(unittest.TestCase):
    @staticmethod
    def _checkpoint_provenance():
        train_records = [
            sample_sha256([1, 2], [2, 1]),
            sample_sha256([3, 4], [4, 3]),
        ]
        eval_records = [sample_sha256([5, 6], [6, 5])]
        return build_dataset_provenance(
            builder_name="dataset.build_4x4_sudoku",
            builder_version=2,
            generation_seed=1729,
            train_record_sha256s=train_records,
            eval_record_sha256s=eval_records,
            train_split="train",
            eval_split="test",
            environment_config={"gamma": 0.99, "max_edits": 8},
            action_mask_config={
                "disable_constraint_masking": False,
                "stop_action_id": 96,
            },
        )

    def test_upi_parser_requires_training_and_fixed_pool_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "[DATASET] eval_split=test eval_samples=50 "
                f"eval_pool_sha256={'0' * 64}\n"
                "[step 20000] eval_success_rate=0.500\n"
            )
            with self.assertRaisesRegex(RuntimeError, "held-out"):
                _parse_final_success_rate(log_path)

            log_path.write_text(
                "[DATASET] train_split=train train_samples=32\n"
                "[DATASET] eval_split=test eval_samples=1 "
                f"eval_pool_sha256={'0' * 64}\n"
                "[step 20000] eval_success_rate=1.000\n"
            )
            with self.assertRaisesRegex(RuntimeError, "50-instance"):
                _parse_final_success_rate(log_path)

    def test_ordered_pool_hash_depends_on_order(self):
        inputs = [[1, 2], [3, 4]]
        solutions = [[2, 1], [4, 3]]
        forward = ordered_pool_sha256(inputs, solutions, 2)
        reverse = ordered_pool_sha256(inputs[::-1], solutions[::-1], 2)
        self.assertNotEqual(forward, reverse)

    def test_checkpoint_provenance_rejects_tampered_record_digest(self):
        provenance = self._checkpoint_provenance()
        provenance["ordered_records"]["train"]["ordered_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            DatasetProvenanceError,
            "does not match its ordered record list",
        ):
            validate_dataset_provenance(provenance)

    def test_checkpoint_provenance_compares_every_resume_dimension(self):
        expected = self._checkpoint_provenance()
        mutations = {
            "builder version": lambda value: value["dataset_builder"].__setitem__(
                "version", 3
            ),
            "generation seed": lambda value: value.__setitem__(
                "generation_seed", 1730
            ),
            "environment": lambda value: value["environment_config"].__setitem__(
                "gamma", 0.95
            ),
            "action mask": lambda value: value["action_mask_config"].__setitem__(
                "disable_constraint_masking", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                saved = copy.deepcopy(expected)
                mutate(saved)
                with self.assertRaisesRegex(
                    DatasetProvenanceError,
                    "Dataset provenance mismatch",
                ):
                    assert_matching_dataset_provenance(saved, expected)

        reordered = copy.deepcopy(expected)
        records = reordered["ordered_records"]["train"]["record_sha256s"]
        records.reverse()
        reordered["ordered_records"]["train"]["ordered_sha256"] = (
            ordered_record_sha256(records)
        )
        with self.assertRaisesRegex(
            DatasetProvenanceError,
            r"record_sha256s\[0\]",
        ):
            assert_matching_dataset_provenance(reordered, expected)

    def test_checkpoint_provenance_rejects_schema_change(self):
        provenance = self._checkpoint_provenance()
        provenance["provenance_schema_version"] = 999
        with self.assertRaisesRegex(
            DatasetProvenanceError,
            "schema version 999",
        ):
            assert_matching_dataset_provenance(
                provenance,
                self._checkpoint_provenance(),
            )

    def test_dataset_source_builder_version_and_seed_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "anonymous-dataset"
            root.mkdir()
            config_path = root / "build_config.json"
            config_path.write_text(
                '{"builder":"dataset.build_4x4_sudoku",'
                '"build_schema_version":2,"seed":31415,'
                '"producer_git_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                '"splits":{"test":{"seed":31417},'
                '"train":{"seed":31415}}}\n'
            )
            metadata = dataset_source_build_metadata([str(root)])

        self.assertEqual(metadata[0]["source_name"], "anonymous-dataset")
        self.assertEqual(metadata[0]["builder_name"], "dataset.build_4x4_sudoku")
        self.assertEqual(metadata[0]["builder_version"], 2)
        self.assertEqual(metadata[0]["generation_seed"], 31415)
        self.assertEqual(
            metadata[0]["split_generation_seeds"],
            {"test": 31417, "train": 31415},
        )
        self.assertEqual(metadata[0]["producer_git_commit"], "a" * 40)
        build_config_sha256 = metadata[0]["build_config_sha256"]
        self.assertIsNotNone(build_config_sha256)
        assert build_config_sha256 is not None
        self.assertEqual(len(build_config_sha256), 64)

    def test_legacy_source_metadata_shape_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "legacy-dataset"
            root.mkdir()
            (root / "build_config.json").write_text(
                '{"builder":"dataset.build_4x4_sudoku",'
                '"build_schema_version":1,"seed":42}\n'
            )
            metadata = dataset_source_build_metadata([str(root)])[0]

        self.assertEqual(
            set(metadata),
            {
                "source_name",
                "builder_name",
                "builder_version",
                "generation_seed",
                "build_config_sha256",
            },
        )

    def test_radius_aggregation_fixes_evaluation_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("model_a", "model_b"):
                csv_path = root / "seed41" / "R10" / f"{model}_b0_per_state.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["n1", "n2", "delta_V"])
                    writer.writeheader()
                    writer.writerow({"n1": 2, "n2": 4, "delta_V": 100.0})
                    writer.writerow({"n1": 2, "n2": 8, "delta_V": 2.0})

            summary = aggregate_radius_sweep(
                str(root),
                seeds=[41],
                radii=[10.0],
                out_dir=str(root / "out"),
                n_train=2,
                n_eval=8,
            )
            self.assertEqual(summary["model_a"][10.0]["delta_V"]["mean"], 2.0)

    def test_summary_writer_keeps_budget_units_separate(self):
        methods = {
            "upi_trm": {
                "train_steps": 20_000,
                "budget_unit": "outer_updates",
                "num_seeds": 1,
                "per_seed_success_rates": [0.5],
                "success_rate_ci_95": [0.5, 0.5],
            },
            "sb3_ppo": {
                "display_name": "SB3 PPO",
                "train_steps": 20_000,
                "budget_unit": "environment_steps",
                "num_seeds": 1,
                "per_seed_success_rates": [0.0],
                "success_rate_ci_95": [0.0, 0.0],
                "mean_return_mean": 0.0,
                "mean_return_std": 0.0,
                "invalid_action_rate_mean": 0.0,
                "invalid_action_rate_std": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = _write_summary_md(
                Path(tmp), methods, ["upi_trm", "sb3_ppo"]
            )
            text = output.read_text()
        self.assertIn("20k outer updates", text)
        self.assertIn("20k env steps", text)
        self.assertIn("No between-method gap interval", text)

    def test_repository_zip_builder_creates_valid_portable_archive(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            file_list = tmp_path / "files.txt"
            file_list.write_text("README.md\nAUDIT_REPORT.md\n")
            output_zip = tmp_path / "bundle.zip"
            env = dict(os.environ)
            env["UPI_TRM_ARCHIVE_FILE_LIST"] = str(file_list)
            env["UPI_TRM_ARCHIVE_OUTPUT"] = str(output_zip)
            subprocess.run(
                ["bash", "scripts/build_artifact_zip.sh"],
                cwd=project_root,
                env=env,
                check=True,
            )
            with zipfile.ZipFile(output_zip) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(
                    set(archive.namelist()),
                    {"README.md", "AUDIT_REPORT.md", "artifact/SHA256SUMS"},
                )
            subprocess.run(
                ["sha256sum", "-c", output_zip.name + ".sha256"],
                cwd=tmp_path,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
