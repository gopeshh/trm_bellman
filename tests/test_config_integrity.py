import unittest
from pathlib import Path

import yaml

from rl.config import RLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


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

    def test_project_root_is_repo_relative(self):
        self.assertTrue((self.root_dir / "README.md").exists())

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

    def test_stability_experiments_disable_value_head_norm(self):
        stability_dirs = [
            self.config_dir / "exp2_contraction_sweep",
            self.config_dir / "exp3_projection_ablation",
            self.config_dir / "table3_hard_controlled",
        ]
        stability_files = list((self.config_dir / "ablations").glob("*no_vhead_norm*.yaml"))

        for config_path in [
            *(path for base_dir in stability_dirs for path in sorted(base_dir.glob("*.yaml"))),
            *sorted(stability_files),
        ]:
            cfg = _load_yaml(config_path)
            self.assertTrue(
                cfg.get("disable_value_head_norm", False),
                f"{config_path.name} must set disable_value_head_norm=true",
            )

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


if __name__ == "__main__":
    unittest.main()
