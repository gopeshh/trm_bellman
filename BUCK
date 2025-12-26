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
