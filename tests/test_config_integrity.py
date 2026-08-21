import hashlib
import json
import unittest
from pathlib import Path

import yaml

from rl.config import RLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return parsed


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return parsed


def _model_dump(config: RLConfig) -> dict:
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    return dict(config.dict())


def _main_default_rl_config() -> RLConfig:
    """Mirror the argparse defaults applied before main layers YAML files."""

    return RLConfig(
        batch_size=32,
        num_train_steps=200,
        rollout_episodes_per_step=1,
        max_edits=8,
        log_interval=10,
        eval_interval=50,
        eval_num_episodes=50,
        eval_seed=1729,
        use_tqdm=False,
        debug_checks=False,
    )


def _layer_rl_configs(config_dir: Path, names: list[str]) -> tuple[RLConfig, dict]:
    merged = _model_dump(_main_default_rl_config())
    for name in names:
        merged = {**merged, **_load_yaml(config_dir / name)}
    return RLConfig(**merged), merged


def _changed_fields(left: RLConfig, right: RLConfig) -> set[str]:
    left_values = _model_dump(left)
    right_values = _model_dump(right)
    return {
        key
        for key in set(left_values).union(right_values)
        if left_values.get(key) != right_values.get(key)
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_rl_config_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.yaml")
        if "pretrain" not in path.parts
    )


class TestConfigIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = PROJECT_ROOT
        cls.config_dir = cls.root_dir / "configs"
        cls.rl_config_paths = _iter_rl_config_paths(cls.config_dir)
        cls.confirmatory_dir = cls.config_dir / "iclr_confirmatory"
        cls.run_matrix = _load_json(cls.confirmatory_dir / "run_matrix.json")

    def test_project_root_is_repo_relative(self):
        self.assertTrue((self.root_dir / "README.md").exists())

    def test_zero_edit_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            RLConfig(max_edits=0)

    def test_rl_config_schema_exposes_single_disable_value_head_norm_field(self):
        matching_fields = [
            field_name
            for field_name in RLConfig.model_fields
            if field_name == "disable_value_head_norm"
        ]
        self.assertEqual(matching_fields, ["disable_value_head_norm"])

    def test_all_rl_configs_parse(self):
        self.assertGreater(len(self.rl_config_paths), 0)
        for config_path in self.rl_config_paths:
            cfg = _load_yaml(config_path)
            parsed = RLConfig(**cfg)
            self.assertIsInstance(parsed, RLConfig, f"Failed to parse {config_path}")

    def test_projection_disabled_configs_use_explicit_identity_mode(self):
        for config_path in self.rl_config_paths:
            cfg = _load_yaml(config_path)
            self.assertNotIn(
                cfg.get("latent_ball_radius"),
                (0, 0.0),
                f"{config_path} must not use R=0 to disable projection",
            )
            if cfg.get("latent_projection_mode") == "disabled":
                self.assertIsNone(
                    cfg.get("latent_ball_radius"),
                    f"{config_path} disabled projection must have no radius",
                )

    def test_exact_mixture_flag_does_not_relabel_legacy_training(self):
        parsed = RLConfig(
            theory_exact_mixture=True,
            exact_k_step_targets=True,
            exact_baseline_summation=True,
            enable_contraction=False,
        )

        self.assertEqual(parsed.training_protocol, "legacy")
        self.assertFalse(parsed.is_theory_exact())

    def test_phase4_value_head_norm_matches_filename(self):
        phase4_dir = self.config_dir / "phase4_2x2_norm_ablation"
        for config_path in sorted(phase4_dir.glob("*.yaml")):
            cfg = _load_yaml(config_path)
            if config_path.stem.endswith("_nv"):
                self.assertTrue(
                    cfg.get("disable_value_head_norm", False),
                    f"{config_path.name} should disable value-head norm",
                )
            elif config_path.stem.endswith("_yv"):
                self.assertFalse(
                    cfg.get("disable_value_head_norm", True),
                    f"{config_path.name} should keep value-head norm enabled",
                )
            else:
                self.fail(f"Unexpected phase4 config naming: {config_path.name}")

    def test_confirmatory_matrix_is_registered_and_references_real_configs(self):
        matrix = self.run_matrix
        self.assertEqual(matrix["artifact_schema_version"], 1)
        self.assertEqual(matrix["status"], "registered_not_authorized")
        self.assertEqual(
            matrix["run_id_templates"],
            {
                "confirmatory": "{cell_lower}-seed{seed}",
                "debug": "debug-{cell_lower}-seed{seed}",
            },
        )
        self.assertNotIn("effective_config_sha256", matrix)
        self.assertEqual(
            set(matrix["bridge_cells"]),
            {"B0_I00", "Bz_I10", "Bd_I01", "Bb", "I11"},
        )
        self.assertIn("Bt", matrix["non_executable_cells"])
        self.assertIn(
            "both paths use the frozen target network",
            matrix["non_executable_cells"]["Bt"],
        )
        self.assertEqual(set(matrix["matched_cells"]), {"UPI_TRM", "TRM_PPO"})

        referenced_layers = [
            layer
            for group in (matrix["bridge_cells"], matrix["matched_cells"])
            for layers in group.values()
            for layer in layers
        ]
        self.assertGreater(len(referenced_layers), 0)
        for layer in referenced_layers:
            with self.subTest(layer=layer):
                self.assertIsInstance(layer, str)
                self.assertEqual(Path(layer).name, layer)
                self.assertEqual(Path(layer).suffix, ".yaml")
                self.assertTrue((self.confirmatory_dir / layer).is_file())

        for cell, layers in {
            **matrix["bridge_cells"],
            **matrix["matched_cells"],
        }.items():
            with self.subTest(cell=cell):
                parsed, _ = _layer_rl_configs(self.confirmatory_dir, layers)
                self.assertIsInstance(parsed, RLConfig)

        output_root = matrix["runtime_output_root"]
        self.assertEqual(
            output_root,
            {
                "environment_variable": "UPI_TRM_EVIDENCE_ROOT",
                "relative_path": "iclr-confirmatory-v1",
            },
        )
        self.assertNotIn("/" + "home/", json.dumps(output_root))
        self.assertNotIn("/tmp/", json.dumps(output_root))

        self.assertEqual(
            matrix["initialization"],
            {
                "mode": "random_paired_by_training_seed",
                "load_checkpoint_allowed": False,
            },
        )

        debug = matrix["debug_runs"]
        self.assertEqual(debug["status"], "debug_only_excluded_from_confirmatory")
        self.assertEqual(debug["seeds"], [9001, 9002, 9003])
        self.assertEqual(
            debug["evaluation_split"],
            matrix["dataset"]["validation_split"],
        )
        self.assertEqual(debug["config_layers"], ["debug_smoke.yaml"])
        debug_config, _ = _layer_rl_configs(
            self.confirmatory_dir,
            debug["config_layers"],
        )
        self.assertEqual(
            debug_config.eval_num_episodes,
            matrix["dataset"]["validation_count"],
        )
        debug_budget = debug["environment_interactions"]
        self.assertEqual(debug_budget, 80)
        debug_intervals = [
            debug["log_environment_interval"],
            debug["evaluation_environment_interval"],
            debug["save_environment_interval"],
        ]
        self.assertEqual(debug_intervals, [debug_budget] * len(debug_intervals))
        self.assertEqual(debug["reported_environment_checkpoints"], [debug_budget])
        self.assertEqual(debug["save_outer_interval"], 0)
        for interval in debug_intervals:
            self.assertEqual(debug_budget % interval, 0)
        confirmatory_layers = set(referenced_layers)
        self.assertTrue(set(debug["config_layers"]).isdisjoint(confirmatory_layers))

    def test_confirmatory_bridge_cells_change_only_registered_factors(self):
        cells = {
            name: _layer_rl_configs(self.confirmatory_dir, layers)[0]
            for name, layers in self.run_matrix["bridge_cells"].items()
        }
        base = cells["B0_I00"]
        expected_changes = {
            "Bz_I10": {"episodic_latent"},
            "Bd_I01": {"evaluation_policy_mode"},
            "Bb": {"exact_baseline_summation"},
            "I11": {"episodic_latent", "evaluation_policy_mode"},
        }
        for cell, expected in expected_changes.items():
            with self.subTest(cell=cell):
                self.assertEqual(_changed_fields(base, cells[cell]), expected)

        self.assertFalse(base.episodic_latent)
        self.assertTrue(cells["Bz_I10"].episodic_latent)
        self.assertFalse(base.theory_exact_mixture)
        self.assertFalse(cells["Bd_I01"].theory_exact_mixture)
        self.assertTrue(base.capture_preinterpolation_policy_pair)
        self.assertEqual(base.evaluation_policy_mode, "stochastic_deployed")
        self.assertEqual(
            cells["Bd_I01"].evaluation_policy_mode,
            "preinterpolation_exact_mixture",
        )
        self.assertFalse(base.exact_k_step_targets)
        self.assertFalse(base.exact_baseline_summation)
        self.assertTrue(cells["Bb"].exact_baseline_summation)

        self.assertEqual(base.K, 1)

    def test_confirmatory_c2_cells_and_exact_schedule_are_coherent(self):
        matrix = self.run_matrix
        upi, upi_raw = _layer_rl_configs(
            self.confirmatory_dir,
            matrix["matched_cells"]["UPI_TRM"],
        )
        ppo, ppo_raw = _layer_rl_configs(
            self.confirmatory_dir,
            matrix["matched_cells"]["TRM_PPO"],
        )

        self.assertEqual(upi.training_protocol, "fixed_base_exact")
        self.assertTrue(upi.is_fixed_base_proposal_exact())
        self.assertFalse(upi.episodic_latent)
        self.assertTrue(upi.exact_k_step_targets)
        self.assertTrue(upi.exact_baseline_summation)
        self.assertTrue(upi.theory_exact_mixture)
        self.assertFalse(upi.distill_mixture_policy)
        self.assertEqual(upi.policy_epsilon, 0.0)
        self.assertEqual(upi_raw["training_protocol"], "fixed_base_exact")

        self.assertEqual(ppo_raw["algorithm"], "ppo")
        self.assertTrue(ppo.episodic_latent)
        self.assertEqual(matrix["architecture_cli"]["backbone"], "trm")
        for field in (
            "gamma",
            "inner_unroll_n",
            "max_edits",
            "task_name",
            "solved_threshold",
            "use_constraint_checker",
            "use_progress_checker",
            "use_feasibility_checker",
            "feasibility_violation_weight",
            "feasibility_zerocand_weight",
            "batch_size",
            "rollout_episodes_per_step",
            "policy_lr",
            "value_lr",
            "entropy_coef",
            "lr_schedule",
            "enable_contraction",
            "latent_projection_mode",
            "latent_ball_radius",
            "disable_constraint_masking",
            "reward_shaping",
            "solve_terminal_reward",
            "fail_terminal_reward",
            "stop_action_mode",
            "stop_action_penalty",
            "C_max",
            "eval_num_episodes",
            "eval_seed",
        ):
            with self.subTest(shared_field=field):
                self.assertEqual(getattr(upi, field), getattr(ppo, field))

        dataset = matrix["dataset"]
        self.assertEqual(upi.eval_num_episodes, dataset["test_count"])
        self.assertEqual(ppo.eval_num_episodes, dataset["test_count"])
        self.assertEqual(
            set(matrix["matched_cells"]),
            {"UPI_TRM", "TRM_PPO"},
        )
        self.assertTrue(
            set(matrix["matched_cells"]).issubset(matrix["deployment_estimands"])
        )
        self.assertIn("Bd_I01", matrix["deployment_estimands"])
        self.assertIn("I11", matrix["deployment_estimands"])

        confirmatory_seeds = matrix["confirmatory_seeds"]
        debug_seeds = matrix["debug_runs"]["seeds"]
        self.assertEqual(len(confirmatory_seeds), len(set(confirmatory_seeds)))
        self.assertEqual(len(debug_seeds), len(set(debug_seeds)))
        self.assertEqual(debug_seeds, [9001, 9002, 9003])
        self.assertFalse(set(confirmatory_seeds).intersection(debug_seeds))

        budget = matrix["environment_interactions"]
        self.assertEqual(matrix["save_outer_interval"], 0)
        self.assertIsInstance(budget, int)
        self.assertGreater(budget, 0)
        interval_keys = (
            "log_environment_interval",
            "evaluation_environment_interval",
            "save_environment_interval",
        )
        ppo_num_steps = int(ppo_raw["ppo_num_steps"])
        intervals = [matrix[key] for key in interval_keys]
        for key, interval in zip(interval_keys, intervals):
            with self.subTest(interval=key):
                self.assertIsInstance(interval, int)
                self.assertGreater(interval, 0)
                self.assertEqual(budget % interval, 0)
                self.assertEqual(interval % ppo_num_steps, 0)
        self.assertEqual(budget % ppo_num_steps, 0)
        for key in interval_keys:
            with self.subTest(debug_interval=key):
                self.assertEqual(
                    matrix["debug_runs"][key] % ppo_num_steps,
                    0,
                )

        checkpoints = matrix["reported_environment_checkpoints"]
        self.assertEqual(checkpoints, sorted(set(checkpoints)))
        self.assertEqual(checkpoints[-1], budget)
        for checkpoint in checkpoints:
            with self.subTest(checkpoint=checkpoint):
                self.assertGreater(checkpoint, 0)
                self.assertLessEqual(checkpoint, budget)
                for interval in intervals:
                    self.assertEqual(checkpoint % interval, 0)

    def test_confirmatory_train_and_test_manifests_match_materialized_dataset(self):
        matrix_dataset = self.run_matrix["dataset"]
        registry = _load_json(self.confirmatory_dir / "dataset.json")
        dataset_root = self.root_dir / matrix_dataset["root"]
        self.assertEqual(matrix_dataset["root"], registry["dataset_root"])
        self.assertTrue(dataset_root.is_dir())

        corpus = _load_json(dataset_root / "corpus_manifest.json")
        build_config = _load_json(dataset_root / "build_config.json")
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                split_name = matrix_dataset[f"{split}_split"]
                count = matrix_dataset[f"{split}_count"]
                manifest_sha256 = matrix_dataset[f"{split}_manifest_sha256"]
                self.assertEqual(split_name, split)

                manifest_path = dataset_root / "manifests" / f"{split}.json"
                manifest = _load_json(manifest_path)
                split_metadata = _load_json(dataset_root / split / "dataset.json")
                self.assertEqual(manifest_sha256, _file_sha256(manifest_path))
                self.assertEqual(
                    manifest_sha256,
                    registry["splits"][split]["manifest_sha256"],
                )
                self.assertEqual(
                    manifest_sha256,
                    corpus["split_manifests"][split]["sha256"],
                )
                self.assertEqual(
                    corpus["split_manifests"][split]["path"],
                    f"manifests/{split}.json",
                )

                self.assertEqual(count, registry["splits"][split]["count"])
                self.assertEqual(count, build_config["splits"][split]["count"])
                self.assertEqual(count, manifest["generated_count"])
                self.assertEqual(count, len(manifest["input_sha256s"]))
                self.assertEqual(count, len(manifest["record_sha256s"]))
                self.assertEqual(count, split_metadata["total_puzzles"])
                self.assertEqual(count, split_metadata["total_groups"])
                self.assertEqual(count, split_metadata["num_puzzle_identifiers"])


if __name__ == "__main__":
    unittest.main()
