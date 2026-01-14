load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

python_library(
    name = "dataset",
    srcs = glob(["dataset/*.py"]),
    base_module = "",
)

python_library(
    name = "models",
    srcs = glob(["models/*.py", "models/**/*.py"]),
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/einops:einops",
    ],
)

python_library(
    name = "utils",
    srcs = glob(["utils/*.py"]),
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
    name = "evaluators",
    srcs = glob(["evaluators/*.py"]),
    base_module = "",
    deps = [
        ":models",
        ":utils",
        ":rl",
        "fbsource//third-party/pypi/torch:torch",
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
    name = "upi_trm_train_lib",
    srcs = ["upi_trm_train.py"],
    base_module = "",
    deps = [
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
    main_module = "upi_trm_train",
    deps = [
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
    name = "imitation_train",
    srcs = ["imitation_train.py"],
    base_module = "",
    main_module = "imitation_train",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/numpy:numpy",
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
        ":rl",
        ":utils",
        ":evaluators",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/pytest:pytest",
    ],
)

python_unittest(
    name = "test_baselines",
    srcs = [
        "tests/__init__.py",
        "tests/test_baselines_unittest.py",
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
    deps = [
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
