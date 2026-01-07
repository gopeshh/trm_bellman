
import unittest
import yaml
import os
from pathlib import Path

# Get the actual project root (not the Buck2 link-tree)
PROJECT_ROOT = Path("/home/buiksat/trm_bellman")

class TestConfigIntegrity(unittest.TestCase):
    def setUp(self):
        # Use absolute path to project root
        self.root_dir = PROJECT_ROOT
        self.config_dir = self.root_dir / "configs"
        self.ablation_dir = self.config_dir / "ablations"

    def load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def test_ablation_checkers_are_correct(self):
        """Verify all ablation configs use progress checker."""
        for config_file in self.ablation_dir.glob("ablation_*.yaml"):
            cfg = self.load_yaml(config_file)

            # Check for critical fix
            self.assertTrue(cfg.get("use_progress_checker", False),
                            f"{config_file.name} missing use_progress_checker=True")
            self.assertFalse(cfg.get("use_constraint_checker", True),
                             f"{config_file.name} has use_constraint_checker=True")

    def test_ablation_logic_consistency(self):
        """Verify ablation logic (e.g. no_contraction actually disables contraction)."""
        
        # 1. No Contraction
        cfg = self.load_yaml(self.ablation_dir / "ablation_no_contraction.yaml")
        self.assertFalse(cfg.get("enable_contraction", True), "ablation_no_contraction should disable contraction")
        
        # 2. No Projection
        cfg = self.load_yaml(self.ablation_dir / "ablation_no_projection.yaml")
        self.assertEqual(cfg.get("latent_ball_radius", 1.0), 0.0, "ablation_no_projection should set radius to 0")
        
        # 3. No Exact Baseline
        cfg = self.load_yaml(self.ablation_dir / "ablation_no_exact_baseline.yaml")
        self.assertFalse(cfg.get("exact_baseline_summation", True), "ablation_no_exact_baseline should disable exact baseline")

    def test_max_edits_consistency(self):
        """Verify max_edits is consistent across ablations (should be 20)."""
        for config_file in self.ablation_dir.glob("ablation_*.yaml"):
            cfg = self.load_yaml(config_file)
            self.assertEqual(cfg.get("max_edits"), 20, 
                             f"{config_file.name} has wrong max_edits (expected 20)")

if __name__ == '__main__':
    unittest.main()

