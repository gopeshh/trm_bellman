#!/usr/bin/env python3
"""
Verify that all generated configs are valid and align with paper theory.

This script checks:
1. YAML syntax is valid
2. All required RLConfig fields are present
3. Theory alignment validation passes
4. Configs match their intended ablation purpose
"""

import sys
from pathlib import Path
import yaml

# Add parent directory to path to import RLConfig
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.config import RLConfig


def verify_config(config_path: Path) -> tuple[bool, list[str]]:
    """
    Verify a single config file.

    Returns:
        (success, issues) tuple
    """
    issues = []

    # 1. Check file exists
    if not config_path.exists():
        issues.append(f"File not found: {config_path}")
        return False, issues

    # 2. Load YAML
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        issues.append(f"YAML parsing error: {e}")
        return False, issues

    # 3. Validate as RLConfig
    try:
        config = RLConfig(**config_dict)
    except Exception as e:
        issues.append(f"RLConfig validation error: {e}")
        return False, issues

    # 4. Run theory alignment validation
    validation_result = config.validate_theory_alignment(warn=False)

    if not validation_result["theory_aligned"] and "theory_exact" in config_path.name:
        # Theory-exact configs should pass validation
        issues.extend(validation_result["issues"])

    return len(issues) == 0, issues


def main():
    print("="*70)
    print("Config Validation Report")
    print("="*70)
    print()

    # Configs to verify
    config_paths = [
        Path("configs/rl_sudoku_shaped_theory_exact.yaml"),
        Path("configs/rl_sudoku_sparse_theory_exact.yaml"),
    ]

    # Add ablation configs
    ablations_dir = Path("configs/ablations")
    if ablations_dir.exists():
        config_paths.extend(sorted(ablations_dir.glob("ablation_*.yaml")))

    total = 0
    passed = 0
    failed = 0

    for config_path in config_paths:
        total += 1
        success, issues = verify_config(config_path)

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {config_path.name}")

        if success:
            passed += 1
            # Load config to show key settings
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            config = RLConfig(**config_dict)

            # Show key theory flags
            flags = []
            if config.reward_shaping:
                flags.append(f"shaped(fail_r={config.fail_terminal_reward})")
            else:
                flags.append("sparse")

            if config.exact_baseline_summation:
                flags.append("exact_baseline")
            if config.enable_contraction:
                flags.append(f"L_z={config.target_Lz}")
            if config.theory_exact_mixture:
                flags.append(f"α={config.mixture_alpha}")

            print(f"  Settings: {', '.join(flags)}")
        else:
            failed += 1
            for issue in issues:
                print(f"  ⚠️  {issue}")
        print()

    print("="*70)
    print(f"Summary: {passed}/{total} configs passed, {failed} failed")
    print("="*70)

    # Specific checks for theory-exact configs
    print("\nTheory-Exact Config Verification:")
    print("-" * 70)

    theory_exact_configs = [
        "configs/rl_sudoku_shaped_theory_exact.yaml",
    ]

    for config_path_str in theory_exact_configs:
        config_path = Path(config_path_str)
        if not config_path.exists():
            print(f"❌ {config_path.name}: NOT FOUND")
            continue

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = RLConfig(**config_dict)

        is_exact = config.is_theory_exact()
        status = "✅" if is_exact else "❌"
        print(f"{status} {config_path.name}: is_theory_exact() = {is_exact}")

        # Check specific requirements for shaped theory-exact
        checks = {
            "reward_shaping": config.reward_shaping,
            "fail_terminal_reward == -10.0": config.fail_terminal_reward == -10.0,
            "exact_baseline_summation": config.exact_baseline_summation,
            "theory_exact_mixture": config.theory_exact_mixture,
            "enable_contraction": config.enable_contraction,
            "target_Lz < 1.0": config.target_Lz < 1.0,
            "latent_ball_radius > 0": config.latent_ball_radius > 0,
            "NOT distill_mixture_policy": not config.distill_mixture_policy,
        }

        for check_name, check_passed in checks.items():
            status = "  ✅" if check_passed else "  ❌"
            print(f"{status} {check_name}")

    print()

    # Ablation verification
    print("\nAblation Config Verification:")
    print("-" * 70)

    expected_ablations = {
        "ablation_no_contraction.yaml": {
            "enable_contraction": False,
        },
        "ablation_no_exact_baseline.yaml": {
            "exact_baseline_summation": False,
        },
        "ablation_no_conservative_mixture.yaml": {
            "mixture_alpha": 1.0,
        },
        "ablation_no_projection.yaml": {
            "latent_ball_radius": 0.0,
        },
        "ablation_no_theory_exact_mixture.yaml": {
            "theory_exact_mixture": False,
        },
        "ablation_no_theory_features.yaml": {
            "enable_contraction": False,
            "exact_baseline_summation": False,
            "mixture_alpha": 1.0,
            "latent_ball_radius": 0.0,
            "theory_exact_mixture": False,
        },
    }

    for ablation_name, expected_settings in expected_ablations.items():
        config_path = Path("configs/ablations") / ablation_name
        if not config_path.exists():
            print(f"❌ {ablation_name}: NOT FOUND")
            continue

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = RLConfig(**config_dict)

        all_correct = True
        for setting_name, expected_value in expected_settings.items():
            actual_value = getattr(config, setting_name)
            if actual_value != expected_value:
                all_correct = False
                print(f"❌ {ablation_name}: {setting_name} = {actual_value} (expected {expected_value})")

        if all_correct:
            print(f"✅ {ablation_name}: All settings correct")

    print()
    print("="*70)
    print("Validation complete!")
    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
