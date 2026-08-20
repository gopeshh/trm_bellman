load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

python_library(
    name = "dataset",
    srcs = glob(["dataset/*.py"]),
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/huggingface-hub:huggingface-hub",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/tqdm:tqdm",
        ":phase4_runtime_profile",
        ":utils",
    ],
)

python_library(
    name = "models",
    srcs = glob(["models/*.py", "models/**/*.py"]),
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/einops:einops",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/torch:torch",
        ":utils",
    ],
)

python_library(
    name = "utils",
    srcs = glob(
        ["utils/*.py"],
        exclude = ["utils/evaluation_artifacts.py"],
    )
    + ["utils/evaluation_artifacts.py"],
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
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
        "fbsource//third-party/pypi/numba:numba",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":dataset",
        ":models",
        ":rl",
        ":utils",
    ],
)

python_library(
    name = "script_eval_unroll_sensitivity_lib",
    srcs = ["scripts/eval/unroll_sensitivity.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
    ],
)

python_library(
    name = "entrypoints",
    srcs = glob(["entrypoints/*.py"]),
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
    ],
)

python_library(
    name = "puzzle_dataset_lib",
    srcs = ["puzzle_dataset.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/torch:torch",
        ":dataset",
        ":models",
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
    name = "policy_improvement_checkpoint_allowlist",
    srcs = ["policy_improvement_checkpoint_allowlist.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":rl",
    ],
)

python_library(
    name = "policy_improvement_sealed_evidence",
    srcs = ["policy_improvement_sealed_evidence.py"],
    base_module = "",
)

python_library(
    name = "policy_improvement_schema",
    srcs = ["scripts/policy_improvement_schema.py"],
    base_module = "",
    deps = [
        ":policy_improvement_v2_schema",
    ],
)

python_library(
    name = "policy_improvement_populations",
    srcs = ["scripts/policy_improvement_populations.py"],
    base_module = "",
)

python_library(
    name = "policy_improvement_v2_schema",
    srcs = ["scripts/policy_improvement_v2_schema.py"],
    base_module = "",
    deps = [
        ":policy_improvement_populations",
    ],
)

python_library(
    name = "policy_improvement_theory_schema",
    srcs = ["scripts/policy_improvement_theory_schema.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_theory_schema_v2",
    srcs = ["scripts/policy_improvement_theory_schema_v2.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_registry",
    srcs = ["scripts/policy_improvement_registry.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
        ":policy_improvement_v2_registry",
    ],
)

python_library(
    name = "policy_improvement_v2_registry",
    srcs = ["scripts/policy_improvement_v2_registry.py"],
    base_module = "",
    deps = [
        ":policy_improvement_populations",
        ":policy_improvement_v2_schema",
    ],
)

python_library(
    name = "policy_improvement_runtime_authorization",
    srcs = ["scripts/policy_improvement_runtime_authorization.py"],
    base_module = "",
    typing = True,
    deps = [
        ":confirmatory_runtime_launcher_lib",
        ":phase4_runtime_profile",
        ":policy_improvement_populations",
        ":policy_improvement_schema",
        ":policy_improvement_v2_registry",
        ":policy_improvement_v2_schema",
    ],
)

python_binary(
    name = "generate_policy_improvement_runtime_authorization",
    srcs = [],
    base_module = "",
    main_module = "scripts.policy_improvement_runtime_authorization",
    deps = [
        ":policy_improvement_runtime_authorization",
    ],
)

python_library(
    name = "policy_improvement_evidence",
    srcs = ["scripts/policy_improvement_evidence.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
    ],
)

python_library(
    name = "policy_improvement_full_runtime",
    srcs = ["scripts/policy_improvement_full_runtime.py"],
    base_module = "",
    deps = [
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_theory_schema",
    ],
)

python_library(
    name = "policy_improvement_non_smoke_checkpoint",
    srcs = ["policy_improvement_non_smoke_checkpoint.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_smoke_checkpoint",
        ":rl",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_full_backend",
    srcs = ["policy_improvement_full_backend.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":policy_improvement_full_runtime",
        ":policy_improvement_non_smoke_checkpoint",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_runtime",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_full_entrypoint",
    srcs = ["policy_improvement_full_entrypoint.py"],
    base_module = "",
    deps = [
        ":runtime_archive_preflight",
    ],
)

python_binary(
    name = "policy_improvement_full",
    srcs = ["policy_improvement_full_entrypoint.py"],
    base_module = "",
    compile = False,
    keep_gpu_sections = True,
    main_module = "policy_improvement_full_entrypoint",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
        "configs/policy_improvement_v1/amendments/*.json",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        ":policy_improvement_full_backend",
        ":policy_improvement_full_runtime",
        ":policy_improvement_non_smoke_checkpoint",
        ":policy_improvement_theory_schema",
        ":runtime_archive_preflight",
    ],
)

python_library(
    name = "policy_improvement_theory_bridge_lib",
    srcs = ["scripts/policy_improvement_theory_bridge.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
        ":policy_improvement_theory_schema",
    ],
)

python_library(
    name = "policy_improvement_theory_bridge_v2_lib",
    srcs = ["scripts/policy_improvement_theory_bridge_v2.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
        ":policy_improvement_theory_schema_v2",
    ],
)

python_library(
    name = "policy_improvement_theory_backend_v2",
    srcs = ["scripts/policy_improvement_theory_backend_v2.py"],
    base_module = "",
    deps = [
        ":policy_improvement_checkpoint_validator",
        ":policy_improvement_evidence",
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_runtime",
        ":policy_improvement_theory_bridge_v2_lib",
        ":policy_improvement_theory_schema_v2",
        ":policy_improvement_v2_schema",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_theory_backend",
    srcs = ["scripts/policy_improvement_theory_backend.py"],
    base_module = "",
    deps = [
        ":policy_improvement_full_backend",
        ":policy_improvement_schema",
        ":policy_improvement_theory_bridge_lib",
    ],
)

python_library(
    name = "policy_improvement_theory_bridge_entrypoint",
    srcs = ["policy_improvement_theory_bridge_entrypoint.py"],
    base_module = "",
    deps = [
        ":runtime_archive_preflight",
    ],
)

python_binary(
    name = "policy_improvement_theory_bridge",
    srcs = ["policy_improvement_theory_bridge_entrypoint.py"],
    base_module = "",
    compile = False,
    keep_gpu_sections = True,
    main_module = "policy_improvement_theory_bridge_entrypoint",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
        "configs/policy_improvement_v1/amendments/*.json",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        ":policy_improvement_full_backend",
        ":policy_improvement_theory_backend",
        ":policy_improvement_theory_backend_v2",
        ":policy_improvement_theory_bridge_lib",
        ":policy_improvement_theory_bridge_v2_lib",
        ":policy_improvement_theory_schema",
        ":policy_improvement_theory_schema_v2",
        ":runtime_archive_preflight",
    ],
)

python_library(
    name = "policy_improvement_test_open",
    srcs = ["scripts/policy_improvement_test_open.py"],
    base_module = "",
    deps = [
        ":policy_improvement_registry",
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_test_open_cli",
    srcs = ["scripts/policy_improvement_test_open_cli.py"],
    base_module = "",
    deps = [
        ":policy_improvement_audit_lib",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_test_open",
    ],
)

python_library(
    name = "policy_improvement_audit_lib",
    srcs = ["scripts/policy_improvement_audit.py"],
    base_module = "",
    deps = [
        ":policy_improvement_checkpoint_validator",
        ":policy_improvement_evidence",
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_test_open",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_checkpoint_validator",
    srcs = ["policy_improvement_checkpoint_validator.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_full_backend",
        ":policy_improvement_full_runtime",
        ":policy_improvement_non_smoke_checkpoint",
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_checkpoint",
        ":policy_improvement_smoke_runtime",
        ":upi_trm_train_lib",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_consumer_entrypoint",
    srcs = ["policy_improvement_consumer_entrypoint.py"],
    base_module = "",
    deps = [
        ":runtime_archive_preflight",
    ],
)

python_binary(
    name = "policy_improvement_audit",
    srcs = [
        "policy_improvement_consumer_entrypoint.py",
        "scripts/policy_improvement_audit.py",
    ],
    base_module = "",
    compile = False,
    keep_gpu_sections = True,
    main_module = "policy_improvement_consumer_entrypoint",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
        "configs/policy_improvement_v1/amendments/*.json",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        ":policy_improvement_checkpoint_validator",
        ":policy_improvement_evidence",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_test_open",
        ":policy_improvement_test_open_cli",
        ":runtime_archive_preflight",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_statistics",
    srcs = ["scripts/policy_improvement_statistics.py"],
    base_module = "",
    deps = [
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_analysis_lib",
    srcs = ["scripts/policy_improvement_analysis.py"],
    base_module = "",
    deps = [
        ":policy_improvement_audit_lib",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_statistics",
    ],
)

python_binary(
    name = "policy_improvement_analysis",
    srcs = [
        "policy_improvement_consumer_entrypoint.py",
        "scripts/policy_improvement_analysis.py",
    ],
    base_module = "",
    compile = False,
    keep_gpu_sections = True,
    main_module = "policy_improvement_consumer_entrypoint",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
        "configs/policy_improvement_v1/amendments/*.json",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        ":policy_improvement_audit_lib",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_statistics",
        ":runtime_archive_preflight",
    ],
)

python_library(
    name = "policy_dataset_builder_impl",
    srcs = [
        "dataset/__init__.py",
        "dataset/build_4x4_sudoku.py",
        "dataset/build_iclr_confirmatory_4x4.py",
        "dataset/build_policy_improvement_4x4.py",
        "utils/__init__.py",
        "utils/dataset_provenance.py",
        "utils/run_identity.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        ":phase4_runtime_profile",
    ],
)

python_library(
    name = "policy_dataset_builder_entrypoint",
    srcs = ["policy_dataset_builder_entrypoint.py"],
    base_module = "",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
    ]),
    deps = [
        ":policy_dataset_builder_impl",
        ":runtime_archive_preflight",
    ],
)

python_binary(
    name = "policy_dataset_builder",
    srcs = ["policy_dataset_builder_entrypoint.py"],
    base_module = "",
    compile = False,
    main_module = "policy_dataset_builder_entrypoint",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
    ]),
    deps = [
        ":policy_dataset_builder_impl",
        ":runtime_archive_preflight",
    ],
)

python_library(
    name = "policy_improvement_smoke_checkpoint",
    srcs = ["policy_improvement_smoke_checkpoint.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_checkpoint_allowlist",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_smoke_runtime",
    srcs = ["policy_improvement_smoke_runtime.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":dataset",
        ":models",
        ":phase4_runtime_profile",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_populations",
        ":policy_improvement_smoke_checkpoint",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
    ],
)

python_library(
    name = "policy_improvement_smoke_plan_lib",
    srcs = ["scripts/policy_improvement_smoke_plan.py"],
    base_module = "",
    deps = [
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
    ],
)

python_binary(
    name = "policy_improvement_smoke_plan",
    srcs = [],
    base_module = "",
    main_module = "scripts.policy_improvement_smoke_plan",
    deps = [
        ":policy_improvement_smoke_plan_lib",
    ],
)

python_library(
    name = "upi_trm_train_lib",
    srcs = ["upi_trm_train.py"],
    base_module = "",
    resources = glob([
        "configs/iclr_confirmatory/*.json",
        "configs/iclr_confirmatory/*.yaml",
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        "fbsource//third-party/pypi/coolname:coolname",
        "fbsource//third-party/pypi/einops:einops",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/omegaconf:omegaconf",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/tqdm:tqdm",
        ":confirmatory_runtime_launcher_lib",
        ":dataset",
        ":evaluators",
        ":models",
        ":policy_improvement_checkpoint_allowlist",
        ":policy_improvement_smoke_runtime",
        ":puzzle_dataset_lib",
        ":rl",
        ":runtime_archive_preflight",
        ":utils",
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
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        "fbsource//third-party/pypi/coolname:coolname",
        "fbsource//third-party/pypi/einops:einops",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/omegaconf:omegaconf",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/tqdm:tqdm",
        ":confirmatory_runtime_launcher_lib",
        ":dataset",
        ":evaluators",
        ":models",
        ":policy_improvement_checkpoint_allowlist",
        ":policy_improvement_smoke_runtime",
        ":puzzle_dataset_lib",
        ":rl",
        ":runtime_archive_preflight",
        ":utils",
    ],
)

python_binary(
    name = "cleanrl_runner",
    srcs = [],
    base_module = "",
    keep_gpu_sections = True,
    main_module = "rl.cleanrl.cleanrl_runner",
    deps = [
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":puzzle_dataset_lib",
        ":rl",
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
        "fbsource//third-party/pypi/numpy:numpy",
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
    ],
)

python_binary(
    name = "diagnose_contraction_components",
    srcs = ["scripts/diagnose_contraction_components.py"],
    base_module = "",
    main_module = "scripts.diagnose_contraction_components",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
    ],
)

python_binary(
    name = "diagnose_contraction_fix",
    srcs = ["scripts/diagnose_contraction_fix.py"],
    base_module = "",
    main_module = "scripts.diagnose_contraction_fix",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
    ],
)

python_binary(
    name = "eval_theorem_facing_ordinal_check",
    srcs = ["scripts/eval_theorem_facing_ordinal_check.py"],
    base_module = "",
    main_module = "scripts.eval_theorem_facing_ordinal_check",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":script_eval_unroll_sensitivity_lib",
        ":utils",
    ],
)

python_binary(
    name = "persistent_checkpoint_diagnostics",
    srcs = ["scripts/persistent_checkpoint_diagnostics.py"],
    base_module = "",
    keep_gpu_sections = True,
    main_module = "scripts.persistent_checkpoint_diagnostics",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
    ],
)

python_binary(
    name = "materialize_hard4x4_closure_batch",
    srcs = ["scripts/materialize_hard4x4_closure_batch.py"],
    base_module = "",
    main_module = "scripts.materialize_hard4x4_closure_batch",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        ":models",
        ":puzzle_dataset_lib",
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":script_eval_unroll_sensitivity_lib",
        ":utils",
    ],
)

python_binary(
    name = "run_exp1_finite_r_primary",
    srcs = [
        "scripts/eval_theorem_facing_ordinal_check.py",
        "scripts/run_exp1_finite_r_primary.py",
    ],
    base_module = "",
    main_module = "scripts.run_exp1_finite_r_primary",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":script_eval_unroll_sensitivity_lib",
        ":utils",
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

python_unittest(
    name = "test_policy_improvement_v1",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_v1_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":dataset",
        ":models",
        ":policy_improvement_analysis_lib",
        ":policy_improvement_audit_lib",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_plan_lib",
        ":policy_improvement_statistics",
        ":policy_improvement_test_open",
        ":policy_improvement_test_open_cli",
        ":rl",
    ],
)

python_unittest(
    name = "test_policy_improvement_v2",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_v2_unittest.py",
    ],
    base_module = "",
    resources = [
        "BUCK",
        "configs/policy_improvement_v1/fixed_base_exact_episodic.yaml",
        "configs/policy_improvement_v1/fixed_base_exact_persistent.yaml",
        "configs/policy_improvement_v1/legacy_parameter_interpolation.yaml",
        "configs/policy_improvement_v1/matched_ppo.yaml",
        "configs/policy_improvement_v1/protocol.json",
        "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
        "configs/policy_improvement_v2/fixed_base_exact_episodic.yaml",
        "configs/policy_improvement_v2/fixed_base_exact_persistent.yaml",
        "configs/policy_improvement_v2/legacy_parameter_interpolation.yaml",
        "configs/policy_improvement_v2/matched_ppo.yaml",
        "configs/policy_improvement_v2/populations.json",
        "configs/policy_improvement_v2/protocol.json",
        "configs/policy_improvement_v2/registry.json",
    ],
    deps = [
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_smoke_plan_lib",
        ":policy_improvement_v2_registry",
        ":policy_improvement_v2_schema",
    ],
)

python_unittest(
    name = "test_policy_improvement_runtime_authorization",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_runtime_authorization_unittest.py",
    ],
    base_module = "",
    typing = True,
    resources = [
        "configs/policy_improvement_v2/fixed_base_exact_episodic.yaml",
        "configs/policy_improvement_v2/fixed_base_exact_persistent.yaml",
        "configs/policy_improvement_v2/legacy_parameter_interpolation.yaml",
        "configs/policy_improvement_v2/matched_ppo.yaml",
        "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
        "configs/policy_improvement_v2/populations.json",
        "configs/policy_improvement_v2/protocol.json",
        "configs/policy_improvement_v2/registry.json",
    ],
    deps = [
        ":phase4_runtime_profile",
        ":policy_improvement_runtime_authorization",
        ":policy_improvement_schema",
    ],
)

python_unittest(
    name = "test_policy_improvement_audit",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_audit_unittest.py",
    ],
    base_module = "",
    deps = [
        ":policy_improvement_audit_lib",
        ":policy_improvement_schema",
    ],
)

python_unittest(
    name = "test_policy_improvement_checkpoint_allowlist",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_checkpoint_allowlist_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_checkpoint_allowlist",
        ":rl",
    ],
)

python_unittest(
    name = "test_policy_improvement_sealed_evidence",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_sealed_evidence_unittest.py",
    ],
    base_module = "",
    resources = ["policy_improvement_consumer_entrypoint.py"],
    deps = [
        ":policy_improvement_audit_lib",
        ":policy_improvement_evidence",
        ":policy_improvement_full_runtime",
        ":policy_improvement_sealed_evidence",
    ],
)

python_unittest(
    name = "test_policy_improvement_evidence",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_evidence_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_checkpoint_allowlist",
        ":policy_improvement_evidence",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_test_open",
    ],
)

python_unittest(
    name = "test_policy_improvement_full_runtime",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_full_runtime_unittest.py",
    ],
    base_module = "",
    deps = [
        ":policy_improvement_full_runtime",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
    ],
)

python_unittest(
    name = "test_policy_improvement_full_backend",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_full_backend_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":policy_improvement_checkpoint_validator",
        ":policy_improvement_full_backend",
        ":policy_improvement_full_runtime",
        ":policy_improvement_non_smoke_checkpoint",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_checkpoint",
        ":rl",
        ":utils",
    ],
)

python_unittest(
    name = "test_policy_improvement_theory_bridge",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_theory_bridge_unittest.py",
    ],
    base_module = "",
    deps = [
        ":policy_improvement_full_backend",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_theory_backend",
        ":policy_improvement_theory_bridge_lib",
        ":policy_improvement_theory_schema",
    ],
)

python_unittest(
    name = "test_policy_improvement_theory_bridge_v2",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_theory_bridge_v2_unittest.py",
    ],
    base_module = "",
    deps = [
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_theory_backend_v2",
        ":policy_improvement_theory_bridge_v2_lib",
        ":policy_improvement_theory_schema_v2",
    ],
)

python_unittest(
    name = "test_policy_dataset_builder",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_dataset_builder_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        ":policy_dataset_builder_impl",
    ],
)

python_unittest(
    name = "test_policy_improvement_smoke_checkpoint",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_smoke_checkpoint_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":policy_improvement_smoke_checkpoint",
        ":rl",
        ":utils",
    ],
)

python_unittest(
    name = "test_policy_improvement_smoke_runtime",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_smoke_runtime_unittest.py",
    ],
    base_module = "",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
    ]),
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_smoke_runtime",
    ],
)

python_unittest(
    name = "test_policy_improvement_checkpoint_validator",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_checkpoint_validator_unittest.py",
    ],
    base_module = "",
    resources = glob([
        "configs/policy_improvement_v1/*.json",
        "configs/policy_improvement_v1/*.yaml",
    ]),
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_checkpoint_validator",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_checkpoint",
        ":policy_improvement_smoke_runtime",
        ":utils",
    ],
)

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
        ":phase4_runtime_profile",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
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
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":rl",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":rl",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":rl",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":evaluators",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
    ],
)

python_unittest(
    name = "test_diagnostic_postprocess",
    srcs = [
        "scripts/postprocess_episodic_z_hard_suite.py",
        "tests/__init__.py",
        "tests/test_diagnostic_postprocess_unittest.py",
    ],
    base_module = "",
)

python_unittest(
    name = "test_result_provenance",
    srcs = [
        "scripts/aggregate_exp1_results.py",
        "scripts/aggregate_hard4x4_trusted_baselines.py",
        "tests/__init__.py",
        "tests/test_result_provenance_unittest.py",
    ],
    base_module = "",
    resources = [
        "AUDIT_REPORT.md",
        "README.md",
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
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":external_baselines_lib",
        ":puzzle_dataset_lib",
        ":rl",
    ],
)

python_unittest(
    name = "test_baselines",
    srcs = [
        "tests/__init__.py",
        "tests/test_baseline_selection.py",
        "tests/test_baselines_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/pytest:pytest",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/torch:torch",
        ":replay_theory_diagnostics",
        ":utils",
    ],
)

python_unittest(
    name = "test_persistent_checkpoint_diagnostics",
    srcs = [
        "scripts/episodic_z_hard_suite_diagnostics.py",
        "scripts/eval_theorem_facing_ordinal_check.py",
        "scripts/persistent_checkpoint_diagnostics.py",
        "tests/__init__.py",
        "tests/test_persistent_checkpoint_diagnostics_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":script_eval_unroll_sensitivity_lib",
        ":utils",
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
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":rl",
        ":upi_trm_train_lib",
    ],
)

python_unittest(
    name = "test_config_integrity",
    srcs = [
        "tests/__init__.py",
        "tests/test_config_integrity.py",
    ],
    base_module = "",
    resources = ["README.md"]
    + glob([
        "configs/**/*.json",
        "configs/**/*.yaml",
        "data/iclr-confirmatory-sudoku4x4-v1/**/*.json",
    ]),
    deps = [
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        ":rl",
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
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
    ],
)

python_binary(
    name = "sanity_check_feasibility",
    srcs = ["scripts/sanity_check_feasibility.py"],
    base_module = "",
    main_module = "scripts.sanity_check_feasibility",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":rl",
        ":upi_trm_train_lib",
    ],
)

python_binary(
    name = "test_solved_termination",
    srcs = ["scripts/test_solved_termination.py"],
    base_module = "",
    main_module = "scripts.test_solved_termination",
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":rl",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
    ],
)

python_binary(
    name = "diagnose_latent_collapse_v2",
    srcs = ["scripts/diagnostics/diagnose_latent_collapse_v2.py"],
    base_module = "",
    main_module = "scripts.diagnostics.diagnose_latent_collapse_v2",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":utils",
    ],
)

python_binary(
    name = "run_contraction_collapse_isolation_2x2",
    srcs = ["scripts/diagnostics/run_contraction_collapse_isolation_2x2.py"],
    base_module = "",
    main_module = "scripts.diagnostics.run_contraction_collapse_isolation_2x2",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/omegaconf:omegaconf",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        "fbsource//third-party/pypi/tqdm:tqdm",
        ":models",
        ":puzzle_dataset_lib",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
    ],
)

python_binary(
    name = "plot_zonly_contraction_preview",
    srcs = ["scripts/plot_zonly_contraction_preview.py"],
    base_module = "",
    main_module = "scripts.plot_zonly_contraction_preview",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pandas:pandas",
    ],
)

# ICML Phase 1: Unroll Sensitivity Evaluation
python_library(
    name = "eval_unroll_sensitivity_lib",
    srcs = [
        "scripts/eval/__init__.py",
        "scripts/eval/unroll_sensitivity.py",
        "scripts/eval_unroll_sensitivity.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
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
        "scripts/eval/unroll_sensitivity.py",
        "scripts/eval_unroll_sensitivity.py",
        "tests/__init__.py",
        "tests/test_unroll_sensitivity_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":eval_unroll_sensitivity_lib",
    ],
)

# ICML Phase 1: Plotting scripts
python_binary(
    name = "plot_exp1_unroll_sensitivity",
    srcs = ["scripts/plot_exp1_unroll_sensitivity.py"],
    base_module = "",
    main_module = "scripts.plot_exp1_unroll_sensitivity",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "plot_exp1_radius_sweep",
    srcs = ["scripts/plot_exp1_radius_sweep.py"],
    base_module = "",
    main_module = "scripts.plot_exp1_radius_sweep",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
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
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

# Paper-ready figures and tables
python_binary(
    name = "make_paper_figures_exp1",
    srcs = ["scripts/make_paper_figures_exp1.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp1",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "make_paper_figures_exp1_split",
    srcs = ["scripts/make_paper_figures_exp1_split.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp1_split",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "make_paper_figures_exp1_final",
    srcs = ["scripts/make_paper_figures_exp1_final.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp1_final",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
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
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":eval_unroll_sensitivity_lib",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":eval_unroll_sensitivity_lib",
        ":models",
    ],
)

python_binary(
    name = "eval_exp2c_lite",
    srcs = ["scripts/eval_exp2c_lite.py"],
    base_module = "",
    main_module = "scripts.eval_exp2c_lite",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":eval_unroll_sensitivity_lib",
        ":models",
        ":rl",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":eval_unroll_sensitivity_lib",
        ":puzzle_dataset_lib",
        ":rl",
    ],
)

python_binary(
    name = "make_paper_figures_exp2_final",
    srcs = ["scripts/make_paper_figures_exp2_final.py"],
    base_module = "",
    main_module = "scripts.make_paper_figures_exp2_final",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
    ],
)

python_binary(
    name = "exp1_value_head_lipschitz",
    srcs = ["scripts/exp1_value_head_lipschitz.py"],
    base_module = "",
    main_module = "scripts.exp1_value_head_lipschitz",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/torch:torch",
        ":eval_unroll_sensitivity_lib",
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
    ],
)

python_binary(
    name = "exp4_final",
    srcs = ["scripts/exp4_final.py"],
    base_module = "",
    main_module = "scripts.exp4_final",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
    ],
)

python_binary(
    name = "generate_exp4_figures",
    srcs = ["scripts/generate_exp4_figures.py"],
    base_module = "",
    main_module = "scripts.generate_exp4_figures",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
    ],
)

python_binary(
    name = "generate_exp5_figures",
    srcs = ["scripts/generate_exp5_figures.py"],
    base_module = "",
    main_module = "scripts.generate_exp5_figures",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
    ],
)

python_binary(
    name = "diagnose_success_rate_discrepancy",
    srcs = ["scripts/diagnose_success_rate_discrepancy.py"],
    base_module = "",
    main_module = "scripts.diagnose_success_rate_discrepancy",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
    ],
)

python_binary(
    name = "phase5_centering_alpha",
    srcs = ["scripts/phase5_centering_alpha.py"],
    base_module = "",
    main_module = "scripts.phase5_centering_alpha",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":rl",
        ":utils",
    ],
)

# ============================================================================
# Exact finite-MDP numerical sanity suite (no learned checkpoints)
# ============================================================================

python_library(
    name = "exact_finite_mdp_sanity_lib",
    srcs = ["scripts/exact_finite_mdp_sanity.py"],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "exact_finite_mdp_sanity",
    srcs = [],
    base_module = "",
    main_module = "scripts.exact_finite_mdp_sanity",
    deps = [
        ":exact_finite_mdp_sanity_lib",
    ],
)

python_unittest(
    name = "test_exact_finite_mdp_sanity",
    srcs = [
        "tests/__init__.py",
        "tests/test_exact_finite_mdp_sanity_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        ":exact_finite_mdp_sanity_lib",
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
        "fbsource//third-party/pypi/numpy:numpy",
        ":phase4_result_schema",
        ":utils",
    ],
)

python_library(
    name = "phase4_figure_publication",
    srcs = ["scripts/phase4_figure_publication.py"],
    base_module = "",
    deps = [
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":phase4_result_schema",
        ":puzzle_dataset_lib",
        ":rl",
        ":utils",
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
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/scipy:scipy",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_result_schema",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":rl",
        ":runtime_archive_preflight",
        ":utils",
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
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_figure_publication",
        ":phase4_result_schema",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":runtime_archive_preflight",
        ":utils",
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
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_result_schema",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":rl",
        ":runtime_archive_preflight",
        ":utils",
    ],
)

python_unittest(
    name = "test_phase4_reporting",
    srcs = [
        "scripts/audit_phase4_paper_ready.py",
        "scripts/eval_phase4_2x2_norm_ablation.py",
        "scripts/make_paper_figures_phase4.py",
        "scripts/phase4_diagnostic_inputs.py",
        "scripts/run_phase4_training.py",
        "tests/__init__.py",
        "tests/test_phase4_reporting_unittest.py",
    ],
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pyyaml:pyyaml",
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":phase4_checkpoint",
        ":phase4_diagnostic_inputs",
        ":phase4_figure_publication",
        ":phase4_result_schema",
        ":phase4_runtime_launcher_lib",
        ":phase4_runtime_profile",
        ":phase4_source",
        ":rl",
        ":runtime_archive_preflight",
        ":utils",
    ],
)

python_unittest(
    name = "test_phase4_figure_publication",
    srcs = [
        "tests/__init__.py",
        "tests/test_phase4_figure_publication_unittest.py",
    ],
    base_module = "",
    deps = [
        ":phase4_figure_publication",
        ":utils",
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
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "plot_table3_hard",
    srcs = ["scripts/plot_table3_hard.py"],
    base_module = "",
    main_module = "scripts.plot_table3_hard",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "plot_table3_hard_controlled",
    srcs = ["scripts/plot_table3_hard_controlled.py"],
    base_module = "",
    main_module = "scripts.plot_table3_hard_controlled",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_binary(
    name = "generate_figure2",
    srcs = ["scripts/generate_figure2.py"],
    base_module = "",
    main_module = "scripts.generate_figure2",
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
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
        "fbsource//third-party/pypi/pytest:pytest",
        ":rl",
        "//caffe2:torch",
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
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/numpy:numpy",
        ":utils",
    ],
)

python_binary(
    name = "run_baseline",
    srcs = ["run_baseline.py"],
    base_module = "",
    main_module = "run_baseline",
    deps = [
        "fbsource//third-party/pypi/gym:gym",
        "fbsource//third-party/pypi/gymnasium:gymnasium",
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/shimmy:shimmy",
        "fbsource//third-party/pypi/stable-baselines3:stable-baselines3",
        ":external_baselines_lib",
    ],
)
