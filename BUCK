load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

python_library(
    name = "dataset",
    srcs = glob(["dataset/*.py"]),
    base_module = "",
    deps = [
        ":utils",
        "fbsource//third-party/pypi/huggingface-hub:huggingface-hub",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/tqdm:tqdm",
    ],
)

python_library(
    name = "models",
    srcs = glob(["models/*.py", "models/**/*.py"]),
    base_module = "",
    deps = [
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/einops:einops",
    ],
)

python_library(
    name = "utils",
    srcs = glob(
        ["utils/*.py"],
        exclude = ["utils/evaluation_artifacts.py"],
    ) + ["utils/evaluation_artifacts.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_library(
    name = "rl",
    srcs = glob(["rl/*.py", "rl/**/*.py"]),
    base_module = "",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
    ],
)

python_library(
    name = "replay_theory_diagnostics",
    srcs = [
        "rl/replay.py",
        "rl/theory_diagnostics.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_library(
    name = "evaluators",
    srcs = glob(["evaluators/*.py"]),
    base_module = "",
    deps = [
        ":dataset",
        ":models",
        ":utils",
        ":rl",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numba:numba",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_library(
    name = "script_eval_unroll_sensitivity_lib",
    srcs = ["scripts/eval/unroll_sensitivity.py"],
    base_module = "",
    deps = [
        ":models",
        ":utils",
        ":rl",
        ":puzzle_dataset_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_library(
    name = "entrypoints",
    srcs = glob(["entrypoints/*.py"]),
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_library(
    name = "puzzle_dataset_lib",
    srcs = ["puzzle_dataset.py"],
    base_module = "",
    deps = [
        ":dataset",
        ":models",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
    ],
)

python_library(
    name = "confirmatory_runtime_launcher_lib",
    srcs = ["confirmatory_runtime_launcher.py"],
    base_module = "",
)

python_binary(
    name = "confirmatory_runtime_launcher",
    srcs = ["confirmatory_runtime_launcher.py"],
    base_module = "",
    main_module = "confirmatory_runtime_launcher",
)

python_library(
    name = "runtime_archive_preflight",
    srcs = ["runtime_archive_preflight.py"],
    base_module = "",
)

python_library(
    name = "upi_trm_train_lib",
    srcs = ["upi_trm_train.py"],
    base_module = "",
    resources = glob([
        "configs/iclr_confirmatory/*.json",
        "configs/iclr_confirmatory/*.yaml",
    ]),
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":runtime_archive_preflight",
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":dataset",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/tqdm:tqdm",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/omegaconf:omegaconf",
        "fbsource//third-party/pypi/einops:einops",
        "fbsource//third-party/pypi/coolname:coolname",
    ],
)

python_binary(
    name = "upi_trm_train",
    srcs = ["upi_trm_train.py"],
    base_module = "",
    compile = False,
    keep_gpu_sections = True,
    main_module = "upi_trm_train",
    resources = glob([
        "configs/iclr_confirmatory/*.json",
        "configs/iclr_confirmatory/*.yaml",
    ]),
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":runtime_archive_preflight",
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":dataset",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/tqdm:tqdm",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/omegaconf:omegaconf",
        "fbsource//third-party/pypi/einops:einops",
        "fbsource//third-party/pypi/coolname:coolname",
    ],
)

python_binary(
    name = "cleanrl_runner",
    srcs = [],
    base_module = "",
    keep_gpu_sections = True,
    main_module = "rl.cleanrl.cleanrl_runner",
    deps = [
        ":puzzle_dataset_lib",
        ":rl",
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "imitation_train",
    srcs = ["imitation_train.py"],
    base_module = "",
    main_module = "imitation_train",
    deps = [
        ":entrypoints",
    ],
)

python_binary(
    name = "build_4x4_sudoku",
    srcs = ["dataset/build_4x4_sudoku.py"],
    base_module = "",
    main_module = "dataset.build_4x4_sudoku",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "build_iclr_confirmatory_4x4",
    srcs = [
        "dataset/build_4x4_sudoku.py",
        "dataset/build_iclr_confirmatory_4x4.py",
    ],
    base_module = "",
    main_module = "dataset.build_iclr_confirmatory_4x4",
    deps = [
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "build_4x4_trivial",
    srcs = ["dataset/build_4x4_trivial.py"],
    base_module = "",
    main_module = "dataset.build_4x4_trivial",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "inspect_4x4_dataset",
    srcs = ["scripts/inspect_4x4_dataset.py"],
    base_module = "",
    main_module = "scripts.inspect_4x4_dataset",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "eval_random_baseline",
    srcs = ["scripts/eval_random_baseline.py"],
    base_module = "",
    main_module = "scripts.eval_random_baseline",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "diagnose_contraction_lipschitz",
    srcs = ["scripts/diagnose_contraction_lipschitz.py"],
    base_module = "",
    main_module = "scripts.diagnose_contraction_lipschitz",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "diagnose_contraction_components",
    srcs = ["scripts/diagnose_contraction_components.py"],
    base_module = "",
    main_module = "scripts.diagnose_contraction_components",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "diagnose_contraction_fix",
    srcs = ["scripts/diagnose_contraction_fix.py"],
    base_module = "",
    main_module = "scripts.diagnose_contraction_fix",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "eval_theorem_facing_ordinal_check",
    srcs = ["scripts/eval_theorem_facing_ordinal_check.py"],
    base_module = "",
    main_module = "scripts.eval_theorem_facing_ordinal_check",
    deps = [
        ":models",
        ":utils",
        ":rl",
        ":puzzle_dataset_lib",
        ":script_eval_unroll_sensitivity_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "persistent_checkpoint_diagnostics",
    srcs = ["scripts/persistent_checkpoint_diagnostics.py"],
    base_module = "",
    main_module = "scripts.persistent_checkpoint_diagnostics",
    keep_gpu_sections = True,
    deps = [
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "materialize_hard4x4_closure_batch",
    srcs = ["scripts/materialize_hard4x4_closure_batch.py"],
    base_module = "",
    main_module = "scripts.materialize_hard4x4_closure_batch",
    deps = [
        ":models",
        ":puzzle_dataset_lib",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "episodic_z_hard_suite_diagnostics",
    srcs = [
        "scripts/episodic_z_hard_suite_diagnostics.py",
        "scripts/eval_theorem_facing_ordinal_check.py",
    ],
    base_module = "",
    main_module = "scripts.episodic_z_hard_suite_diagnostics",
    deps = [
        ":models",
        ":utils",
        ":rl",
        ":puzzle_dataset_lib",
        ":script_eval_unroll_sensitivity_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "run_exp1_finite_r_primary",
    srcs = [
        "scripts/run_exp1_finite_r_primary.py",
        "scripts/eval_theorem_facing_ordinal_check.py",
    ],
    base_module = "",
    main_module = "scripts.run_exp1_finite_r_primary",
    deps = [
        ":models",
        ":utils",
        ":rl",
        ":puzzle_dataset_lib",
        ":script_eval_unroll_sensitivity_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "analyze_exp1_finite_r_sweep",
    srcs = ["scripts/analyze_exp1_finite_r_sweep.py"],
    base_module = "",
    main_module = "scripts.analyze_exp1_finite_r_sweep",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "build_9x9_sudoku",
    srcs = [
        "dataset/build_easy_sudoku.py",
    ],
    base_module = "",
    main_module = "dataset.build_easy_sudoku",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/tqdm:tqdm",
    ],
)

load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

python_unittest(
    name = "test_run_identity",
    srcs = [
        "tests/__init__.py",
        "tests/test_compute_accounting_unittest.py",
        "tests/test_confirmatory_runtime_launcher_unittest.py",
        "tests/test_evaluation_artifacts_unittest.py",
        "tests/test_run_identity_unittest.py",
        "tests/test_source_identity_unittest.py",
    ],
    base_module = "",
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":utils",
    ],
)

python_unittest(
    name = "test_upi_trm_trainer_smoke",
    srcs = [
        "tests/__init__.py",
        "tests/test_upi_trm_trainer_smoke_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_algorithm2_boundary_contract",
    srcs = [
        "tests/__init__.py",
        "tests/test_algorithm2_boundary_contract_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":puzzle_dataset_lib",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_unittest(
    name = "test_cpi_mixture_policy_smoke",
    srcs = [
        "tests/__init__.py",
        "tests/test_cpi_mixture_policy_smoke_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_upi_trm_logging_smoke",
    srcs = [
        "tests/__init__.py",
        "tests/test_upi_trm_logging_smoke_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_rl_k_step_targets",
    srcs = [
        "tests/__init__.py",
        "tests/test_rl_k_step_targets_unittest.py",
    ],
    base_module = "",
    deps = [
        ":rl",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_theory_exact_components",
    srcs = [
        "tests/__init__.py",
        "tests/test_theory_exact_components_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_trm_latent_unroll",
    srcs = [
        "tests/__init__.py",
        "tests/test_trm_latent_unroll_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_plan_edit_env",
    srcs = [
        "tests/__init__.py",
        "tests/test_plan_edit_env.py",
        "tests/test_plan_edit_env_unittest.py",
    ],
    base_module = "",
    deps = [
        ":rl",
        ":utils",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_rl_plan_evaluator_smoke",
    srcs = [
        "tests/__init__.py",
        "tests/test_rl_plan_evaluator_smoke_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_edit_policy_head",
    srcs = [
        "tests/__init__.py",
        "tests/test_edit_policy_head_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_trm_rl_heads",
    srcs = [
        "tests/__init__.py",
        "tests/test_trm_rl_heads_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_lipschitz_spectral_norm",
    srcs = [
        "tests/__init__.py",
        "tests/test_lipschitz_spectral_norm_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_theory_metrics",
    srcs = [
        "tests/__init__.py",
        "tests/test_theory_metrics_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_gae",
    srcs = [
        "tests/__init__.py",
        "tests/test_gae_unittest.py",
    ],
    base_module = "",
    deps = [
        ":rl",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_plan_edit_env_reward_shaping",
    srcs = [
        "tests/__init__.py",
        "tests/test_plan_edit_env_reward_shaping_unittest.py",
    ],
    base_module = "",
    deps = [
        ":rl",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_rl_k_step_value_update_trainer",
    srcs = [
        "tests/__init__.py",
        "tests/test_rl_k_step_value_update_trainer_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":evaluators",
        ":puzzle_dataset_lib",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_z_init_encoder",
    srcs = [
        "tests/__init__.py",
        "tests/test_z_init_encoder_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_refactored_modules",
    srcs = [
        "tests/__init__.py",
        "tests/test_refactored_modules_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_diagnostic_postprocess",
    srcs = [
        "tests/__init__.py",
        "tests/test_diagnostic_postprocess_unittest.py",
        "scripts/postprocess_episodic_z_hard_suite.py",
    ],
    base_module = "",
)

python_unittest(
    name = "test_result_provenance",
    srcs = [
        "tests/__init__.py",
        "tests/test_result_provenance_unittest.py",
        "scripts/aggregate_exp1_results.py",
        "scripts/aggregate_hard4x4_trusted_baselines.py",
    ],
    base_module = "",
    resources = [
        "README.md",
        "AUDIT_REPORT.md",
        "scripts/build_artifact_zip.sh",
    ],
    deps = [
        ":utils",
    ],
)

python_unittest(
    name = "test_dataset_builders",
    srcs = [
        "dataset/build_iclr_confirmatory_4x4.py",
        "tests/__init__.py",
        "tests/test_dataset_builders_unittest.py",
    ],
    base_module = "",
    resources = glob([
        "data/iclr-confirmatory-sudoku4x4-v1/**",
    ]),
    deps = [
        ":dataset",
    ],
)

python_unittest(
    name = "test_cleanrl_regressions",
    srcs = [
        "tests/__init__.py",
        "tests/test_cleanrl_regressions_unittest.py",
        "tests/test_cleanrl_runner_unittest.py",
    ],
    base_module = "",
    deps = [
        ":external_baselines_lib",
        ":puzzle_dataset_lib",
        ":rl",
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_unittest(
    name = "test_baselines",
    srcs = [
        "tests/__init__.py",
        "tests/test_baselines_unittest.py",
        "tests/test_baseline_selection.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_rl_algos_mock",
    srcs = [
        "tests/__init__.py",
        "tests/test_rl_algos_mock.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_undo_and_sequences",
    srcs = [
        "tests/__init__.py",
        "tests/test_undo_and_sequences_unittest.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_optimized_exact_baseline",
    srcs = [
        "tests/__init__.py",
        "tests/test_optimized_exact_baseline.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_unittest(
    name = "test_augmented_replay_diagnostics",
    srcs = [
        "tests/__init__.py",
        "tests/test_augmented_replay_diagnostics_unittest.py",
    ],
    base_module = "",
    deps = [
        ":replay_theory_diagnostics",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_unittest(
    name = "test_persistent_checkpoint_diagnostics",
    srcs = [
        "tests/__init__.py",
        "tests/test_persistent_checkpoint_diagnostics_unittest.py",
        "scripts/episodic_z_hard_suite_diagnostics.py",
        "scripts/eval_theorem_facing_ordinal_check.py",
        "scripts/persistent_checkpoint_diagnostics.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":script_eval_unroll_sensitivity_lib",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_sudoku_checkers",
    srcs = [
        "tests/__init__.py",
        "tests/test_sudoku_checkers.py",
    ],
    base_module = "",
    deps = [
        ":rl",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_config_integrity",
    srcs = [
        "tests/__init__.py",
        "tests/test_config_integrity.py",
    ],
    base_module = "",
    resources = ["README.md"] + glob([
        "configs/**/*.json",
        "configs/**/*.yaml",
        "data/iclr-confirmatory-sudoku4x4-v1/**/*.json",
    ]),
    deps = [
        ":rl",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_convergence_smoke",
    srcs = [
        "tests/__init__.py",
        "tests/test_convergence_smoke.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "sanity_check_feasibility",
    srcs = ["scripts/sanity_check_feasibility.py"],
    base_module = "",
    main_module = "scripts.sanity_check_feasibility",
    deps = [
        ":rl",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "test_solved_termination",
    srcs = ["scripts/test_solved_termination.py"],
    base_module = "",
    main_module = "scripts.test_solved_termination",
    deps = [
        ":rl",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "parse_feasibility_logs",
    srcs = ["scripts/parse_feasibility_logs.py"],
    base_module = "",
    main_module = "scripts.parse_feasibility_logs",
    deps = [],
)

python_binary(
    name = "plot_feasibility_curves",
    srcs = ["scripts/plot_feasibility_curves.py"],
    base_module = "",
    main_module = "scripts.plot_feasibility_curves",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "plot_6to8empties_paper_style",
    srcs = ["scripts/plot_6to8empties_paper_style.py"],
    base_module = "",
    main_module = "scripts.plot_6to8empties_paper_style",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "plot_contraction_sgd_tradeoff",
    srcs = ["scripts/plot_contraction_sgd_tradeoff.py"],
    base_module = "",
    main_module = "scripts.plot_contraction_sgd_tradeoff",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "diagnose_latent_collapse",
    srcs = ["scripts/diagnostics/diagnose_latent_collapse.py"],
    base_module = "",
    main_module = "scripts.diagnostics.diagnose_latent_collapse",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "diagnose_latent_collapse_v2",
    srcs = ["scripts/diagnostics/diagnose_latent_collapse_v2.py"],
    base_module = "",
    main_module = "scripts.diagnostics.diagnose_latent_collapse_v2",
    deps = [
        ":models",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "run_contraction_collapse_isolation_2x2",
    srcs = ["scripts/diagnostics/run_contraction_collapse_isolation_2x2.py"],
    base_module = "",
    main_module = "scripts.diagnostics.run_contraction_collapse_isolation_2x2",
    deps = [
        ":models",
        ":rl",
        ":utils",
        ":puzzle_dataset_lib",
        ":upi_trm_train_lib",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/omegaconf:omegaconf",
        "fbsource//third-party/pypi/tqdm:tqdm",
    ],
)

python_binary(
    name = "plot_zonly_contraction_preview",
    srcs = ["scripts/plot_zonly_contraction_preview.py"],
    base_module = "",
    main_module = "scripts.plot_zonly_contraction_preview",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/pandas:pandas",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

# ICML Phase 1: Unroll Sensitivity Evaluation
python_library(
    name = "eval_unroll_sensitivity_lib",
    srcs = [
        "scripts/eval_unroll_sensitivity.py",
        "scripts/eval/__init__.py",
        "scripts/eval/unroll_sensitivity.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "eval_unroll_sensitivity",
    srcs = ["scripts/eval_unroll_sensitivity.py"],
    base_module = "",
    main_module = "scripts.eval_unroll_sensitivity",
    deps = [
        ":eval_unroll_sensitivity_lib",
    ],
)

python_unittest(
    name = "test_unroll_sensitivity",
    srcs = [
        "tests/__init__.py",
        "tests/test_unroll_sensitivity_unittest.py",
        "scripts/eval_unroll_sensitivity.py",
        "scripts/eval/unroll_sensitivity.py",
    ],
    base_module = "",
    deps = [
        ":eval_unroll_sensitivity_lib",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

# ICML Phase 1: Plotting scripts
python_binary(
    name = "plot_exp1_unroll_sensitivity",
    srcs = ["scripts/plot_exp1_unroll_sensitivity.py"],
    base_module = "",
    main_module = "scripts.plot_exp1_unroll_sensitivity",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "plot_exp1_radius_sweep",
    srcs = ["scripts/plot_exp1_radius_sweep.py"],
    base_module = "",
    main_module = "scripts.plot_exp1_radius_sweep",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

# ICML Phase 1 v4.1: Post-processing with fixes
python_binary(
    name = "postprocess_exp1_v4_1",
    srcs = ["scripts/postprocess_exp1_v4_1.py"],
    base_module = "",
    main_module = "scripts.postprocess_exp1_v4_1",
    deps = [],
)

python_binary(
    name = "plot_exp1_v4_1",
    srcs = ["scripts/plot_exp1_v4_1.py"],
    base_module = "",
    main_module = "scripts.plot_exp1_v4_1",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

# Paper-ready figures and tables
python_binary(
    name = "make_paper_figures_exp1",
    srcs = ["scripts/make_paper_figures_exp1.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp1",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "make_paper_figures_exp1_split",
    srcs = ["scripts/make_paper_figures_exp1_split.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp1_split",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "make_paper_figures_exp1_final",
    srcs = ["scripts/make_paper_figures_exp1_final.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp1_final",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "audit_exp1_paper_ready",
    srcs = ["scripts/audit_exp1_paper_ready.py"],
    base_module = "",
    main_module = "scripts.audit_exp1_paper_ready",
    deps = [],
)

# ============================================================================
# Experiment 2: Contraction Sweep
# ============================================================================

python_binary(
    name = "run_exp2_contraction_sweep",
    srcs = ["scripts/run_exp2_contraction_sweep.py"],
    base_module = "",
    main_module = "scripts.run_exp2_contraction_sweep",
    deps = [],
)

python_binary(
    name = "make_paper_figures_exp2",
    srcs = ["scripts/make_paper_figures_exp2.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp2",
    deps = [
        ":eval_unroll_sensitivity_lib",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "audit_exp2_paper_ready",
    srcs = ["scripts/audit_exp2_paper_ready.py"],
    base_module = "",
    main_module = "scripts.audit_exp2_paper_ready",
    deps = [],
)

python_binary(
    name = "diagnose_contraction_saturation",
    srcs = ["scripts/diagnose_contraction_saturation.py"],
    base_module = "",
    main_module = "scripts.diagnose_contraction_saturation",
    deps = [
        ":eval_unroll_sensitivity_lib",
        ":models",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "eval_exp2c_lite",
    srcs = ["scripts/eval_exp2c_lite.py"],
    base_module = "",
    main_module = "scripts.eval_exp2c_lite",
    deps = [
        ":eval_unroll_sensitivity_lib",
        ":models",
        ":rl",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "inspect_checkpoint",
    srcs = ["scripts/inspect_checkpoint.py"],
    base_module = "",
    main_module = "scripts.inspect_checkpoint",
    deps = ["fbsource//third-party/pypi/torch:torch"],
)

python_binary(
    name = "reevaluate_upi_baseline_interface",
    srcs = [
        "puzzle_dataset.py",
        "scripts/reevaluate_upi_baseline_interface.py",
    ],
    base_module = "",
    main_module = "scripts.reevaluate_upi_baseline_interface",
    deps = [
        ":eval_unroll_sensitivity_lib",
        ":puzzle_dataset_lib",
        ":rl",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "make_paper_figures_exp2_final",
    srcs = ["scripts/make_paper_figures_exp2_final.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp2_final",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "audit_exp2_final_paper_ready",
    srcs = ["scripts/audit_exp2_final_paper_ready.py"],
    base_module = "",
    main_module = "scripts.audit_exp2_final_paper_ready",
)

python_binary(
    name = "exp1_lipschitz_diag",
    srcs = ["scripts/exp1_lipschitz_diag.py"],
    base_module = "",
    main_module = "scripts.exp1_lipschitz_diag",
    deps = [
        ":models",
        ":rl",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "exp1_value_head_lipschitz",
    srcs = ["scripts/exp1_value_head_lipschitz.py"],
    base_module = "",
    main_module = "scripts.exp1_value_head_lipschitz",
    deps = [
        ":eval_unroll_sensitivity_lib",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_binary(
    name = "audit_exp1_lipschitz_diag",
    srcs = ["scripts/audit_exp1_lipschitz_diag.py"],
    base_module = "",
    main_module = "scripts.audit_exp1_lipschitz_diag",
)

# ============================================================================
# Experiment 3: Projection Ablation
# ============================================================================

python_binary(
    name = "eval_exp3_projection_ablation",
    srcs = ["scripts/eval_exp3_projection_ablation.py"],
    base_module = "",
    main_module = "scripts.eval_exp3_projection_ablation",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

# ============================================================================
# Experiment 4: Projection-free Contraction Dial
# ============================================================================

python_binary(
    name = "exp4_range_test",
    srcs = ["scripts/exp4_range_test.py"],
    base_module = "",
    main_module = "scripts.exp4_range_test",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/scipy:scipy",
    ],
)

python_binary(
    name = "exp4_final",
    srcs = ["scripts/exp4_final.py"],
    base_module = "",
    main_module = "scripts.exp4_final",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "audit_exp4_range_test",
    srcs = ["scripts/audit_exp4_range_test.py"],
    base_module = "",
    main_module = "scripts.audit_exp4_range_test",
)

python_binary(
    name = "audit_exp4_final_paper_ready",
    srcs = ["scripts/audit_exp4_final_paper_ready.py"],
    base_module = "",
    main_module = "scripts.audit_exp4_final_paper_ready",
)

python_binary(
    name = "exp4_final_v2",
    srcs = ["scripts/exp4_final_v2.py"],
    base_module = "",
    main_module = "scripts.exp4_final_v2",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "generate_exp4_figures",
    srcs = ["scripts/generate_exp4_figures.py"],
    base_module = "",
    main_module = "scripts.generate_exp4_figures",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/scipy:scipy",
    ],
)

python_binary(
    name = "audit_exp4_final_v2",
    srcs = ["scripts/audit_exp4_final_v2.py"],
    base_module = "",
    main_module = "scripts.audit_exp4_final_v2",
)

# ============================================================================
# Experiment 5: Stability–Expressivity Tradeoff Curve
# ============================================================================

python_binary(
    name = "exp5_tradeoff_curve",
    srcs = ["scripts/exp5_tradeoff_curve.py"],
    base_module = "",
    main_module = "scripts.exp5_tradeoff_curve",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "generate_exp5_figures",
    srcs = ["scripts/generate_exp5_figures.py"],
    base_module = "",
    main_module = "scripts.generate_exp5_figures",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "audit_exp5_tradeoff_curve",
    srcs = ["scripts/audit_exp5_tradeoff_curve.py"],
    base_module = "",
    main_module = "scripts.audit_exp5_tradeoff_curve",
)

python_binary(
    name = "diagnose_success_discrepancy",
    srcs = ["scripts/diagnose_success_discrepancy.py"],
    base_module = "",
    main_module = "scripts.diagnose_success_discrepancy",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "diagnose_success_rate_discrepancy",
    srcs = ["scripts/diagnose_success_rate_discrepancy.py"],
    base_module = "",
    main_module = "scripts.diagnose_success_rate_discrepancy",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "phase5_centering_alpha",
    srcs = ["scripts/phase5_centering_alpha.py"],
    base_module = "",
    main_module = "scripts.phase5_centering_alpha",
    deps = [
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

# ============================================================================
# Phase 4: 2×2 Norm Ablation (Multi-seed)
# ============================================================================

python_library(
    name = "phase4_result_schema",
    srcs = ["scripts/phase4_result_schema.py"],
    base_module = "",
)

python_library(
    name = "phase4_diagnostic_inputs",
    srcs = ["scripts/phase4_diagnostic_inputs.py"],
    base_module = "",
    deps = [
        ":phase4_result_schema",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_library(
    name = "phase4_source",
    srcs = ["scripts/phase4_source.py"],
    base_module = "",
    deps = [
        ":phase4_runtime_profile",
        ":utils",
    ],
)

python_library(
    name = "phase4_runtime_profile",
    srcs = ["phase4_runtime_profile.py"],
    base_module = "",
)

python_library(
    name = "phase4_runtime_entrypoint",
    srcs = ["phase4_runtime_entrypoint.py"],
    base_module = "",
    deps = [
        ":runtime_archive_preflight",
    ],
)

python_library(
    name = "phase4_runtime_launcher_lib",
    srcs = ["phase4_runtime_launcher.py"],
    base_module = "",
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":phase4_runtime_profile",
    ],
)

python_binary(
    name = "phase4_runtime_launcher",
    srcs = ["phase4_runtime_launcher.py"],
    base_module = "",
    main_module = "phase4_runtime_launcher",
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":phase4_runtime_profile",
    ],
)

python_library(
    name = "phase4_checkpoint",
    srcs = ["scripts/phase4_checkpoint.py"],
    base_module = "",
    deps = [
        ":models",
        ":phase4_result_schema",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "run_phase4_training",
    srcs = ["scripts/run_phase4_training.py"],
    base_module = "",
    main_module = "scripts.run_phase4_training",
    deps = [
        ":phase4_result_schema",
        ":phase4_source",
    ],
)

python_binary(
    name = "eval_phase4_2x2_norm_ablation",
    srcs = [
        "phase4_runtime_entrypoint.py",
        "scripts/eval_phase4_2x2_norm_ablation.py",
    ],
    base_module = "",
    compile = False,
    main_module = "phase4_runtime_entrypoint",
    deps = [
        ":models",
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_result_schema",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":runtime_archive_preflight",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_binary(
    name = "make_paper_figures_phase4",
    srcs = [
        "phase4_runtime_entrypoint.py",
        "scripts/make_paper_figures_phase4.py",
    ],
    base_module = "",
    compile = False,
    main_module = "phase4_runtime_entrypoint",
    deps = [
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_result_schema",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":runtime_archive_preflight",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "audit_phase4_paper_ready",
    srcs = [
        "phase4_runtime_entrypoint.py",
        "scripts/audit_phase4_paper_ready.py",
    ],
    base_module = "",
    compile = False,
    main_module = "phase4_runtime_entrypoint",
    deps = [
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_result_schema",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":runtime_archive_preflight",
        ":models",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_phase4_reporting",
    srcs = [
        "tests/__init__.py",
        "tests/test_phase4_reporting_unittest.py",
        "scripts/audit_phase4_paper_ready.py",
        "scripts/eval_phase4_2x2_norm_ablation.py",
        "scripts/make_paper_figures_phase4.py",
        "scripts/phase4_diagnostic_inputs.py",
        "scripts/run_phase4_training.py",
    ],
    base_module = "",
    deps = [
        ":models",
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_result_schema",
        ":phase4_runtime_launcher_lib",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":runtime_archive_preflight",
        ":rl",
        ":utils",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_phase4_runtime_launcher",
    srcs = [
        "tests/__init__.py",
        "tests/test_phase4_runtime_launcher_unittest.py",
    ],
    base_module = "",
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":phase4_runtime_launcher_lib",
        ":phase4_runtime_profile",
        ":runtime_archive_preflight",
    ],
)

python_binary(
    name = "plot_table3_baselines",
    srcs = ["scripts/plot_table3_baselines.py"],
    base_module = "",
    main_module = "scripts.plot_table3_baselines",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "plot_table3_hard",
    srcs = ["scripts/plot_table3_hard.py"],
    base_module = "",
    main_module = "scripts.plot_table3_hard",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "plot_table3_hard_controlled",
    srcs = ["scripts/plot_table3_hard_controlled.py"],
    base_module = "",
    main_module = "scripts.plot_table3_hard_controlled",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

python_binary(
    name = "generate_figure2",
    srcs = ["scripts/generate_figure2.py"],
    base_module = "",
    main_module = "scripts.generate_figure2",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/matplotlib:matplotlib",
    ],
)

# ============================================================================
# 9x9 Sudoku Dataset Generation and Experiments
# ============================================================================

python_binary(
    name = "gen_sudoku9x9",
    srcs = ["scripts/gen_sudoku9x9.py"],
    base_module = "",
    main_module = "scripts.gen_sudoku9x9",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "run_experiments_parallel",
    srcs = ["scripts/run_experiments_parallel.py"],
    base_module = "",
    main_module = "scripts.run_experiments_parallel",
    deps = [
        "fbsource//third-party/pypi/pyyaml:pyyaml",
    ],
)

python_unittest(
    name = "test_constraint_aware_masking",
    srcs = [
        "tests/__init__.py",
        "tests/test_constraint_aware_masking_unittest.py",
    ],
    base_module = "",
    deps = [
        ":rl",
        "//caffe2:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_binary(
    name = "plot_9x9_training_progress",
    srcs = ["scripts/plot_9x9_training_progress.py"],
    base_module = "",
    main_module = "scripts.plot_9x9_training_progress",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "plot_9x9_success_vs_steps",
    srcs = ["scripts/plot_9x9_success_vs_steps.py"],
    base_module = "",
    main_module = "scripts.plot_9x9_success_vs_steps",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "plot_9x9_mean_score_vs_steps",
    srcs = ["scripts/plot_9x9_mean_score_vs_steps.py"],
    base_module = "",
    main_module = "scripts.plot_9x9_mean_score_vs_steps",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_library(
    name = "external_baselines_lib",
    srcs = glob(["external_baselines/*.py"]),
    base_module = "",
    deps = [
        ":utils",
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "run_baseline",
    srcs = ["run_baseline.py"],
    base_module = "",
    main_module = "run_baseline",
    deps = [
        ":external_baselines_lib",
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/shimmy:shimmy",
        "fbsource//third-party/pypi/stable-baselines3:stable-baselines3",
    ],
)
