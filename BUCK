load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")
load("@fbcode_macros//build_defs:python_unittest.bzl", "python_unittest")

python_library(
    name = "dataset",
    srcs = glob(["dataset/*.py"]),
    base_module = "",
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/pydantic:pydantic",
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
    name = "evaluators",
    srcs = glob(["evaluators/*.py"]),
    base_module = "",
    deps = [
        ":rl",
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
        ":policy_improvement_populations",
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
        # Same call-time import gap as the Stage 0 binaries. This tool reaches
        # validate_v2_registry_document today rather than generate_registry,
        # so bind them before it ever takes the other path.
        ":policy_improvement_theory_schema",
        ":policy_improvement_theory_schema_v2",
    ],
)

python_library(
    name = "policy_improvement_base_policy_lib",
    srcs = ["scripts/policy_improvement_base_policy.py"],
    base_module = "",
    typing = True,
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":phase4_runtime_profile",
        ":policy_improvement_smoke_runtime",
        ":policy_improvement_v2_registry",
        ":policy_improvement_v2_schema",
        ":rl",
        ":upi_trm_train_lib",
        ":utils",
    ],
)

python_binary(
    name = "policy_improvement_base_policy",
    srcs = [],
    base_module = "",
    compile = False,
    keep_gpu_sections = True,
    main_module = "scripts.policy_improvement_base_policy",
    resources = glob([
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        ":policy_improvement_base_policy_lib",
        # generate_registry imports both theory schemas at call time.
        ":policy_improvement_theory_schema",
        ":policy_improvement_theory_schema_v2",
    ],
)

python_unittest(
    name = "test_policy_improvement_base_policy",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_base_policy_unittest.py",
    ],
    base_module = "",
    typing = True,
    resources = glob([
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
        "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1/MANIFEST.json",
        "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1/manifests/*.json",
    ]),
    deps = [
        ":policy_improvement_base_policy_lib",
        ":policy_improvement_v2_registry",
        ":policy_improvement_v2_schema",
        ":rl",
    ],
)

python_library(
    name = "policy_improvement_base_policy_restore",
    srcs = ["scripts/policy_improvement_base_policy_restore.py"],
    base_module = "",
    typing = True,
    deps = [
        ":policy_improvement_v2_schema",
        ":rl",
    ],
)

python_unittest(
    name = "test_policy_improvement_base_policy_restore",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_base_policy_restore_unittest.py",
    ],
    base_module = "",
    typing = True,
    resources = glob([
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":models",
        ":phase4_runtime_launcher_lib",
        ":policy_improvement_base_policy_restore",
        ":policy_improvement_smoke_runtime",
        ":policy_improvement_v2_schema",
        ":rl",
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
        ":policy_improvement_base_policy_restore",
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_theory_schema",
        ":policy_improvement_v2_schema",
    ],
)

python_library(
    name = "policy_improvement_throughput",
    srcs = ["scripts/policy_improvement_throughput.py"],
    base_module = "",
    typing = True,
    deps = [
        ":policy_improvement_populations",
        ":policy_improvement_v2_registry",
        ":policy_improvement_v2_schema",
        ":utils",
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
        ":policy_improvement_base_policy_restore",
        ":policy_improvement_full_runtime",
        ":policy_improvement_non_smoke_checkpoint",
        ":policy_improvement_schema",
        ":policy_improvement_sealed_evidence",
        ":policy_improvement_smoke_runtime",
        ":policy_improvement_v2_schema",
        ":policy_improvement_throughput",
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
        "configs/policy_improvement_exp1b/*.json",
        "configs/policy_improvement_exp1b/amendments/*.json",
    ]),
    deps = [
        # Experiment 1B Stage A is imported dynamically off the launcher-owned
        # marker, so the PAR must declare it directly. It deliberately does not
        # reach the Stage B theory backend: that module is selected by the exact
        # source profile, and the full profile does not carry it.
        ":policy_improvement_exp1b_runtime",
        ":policy_improvement_full_backend",
        ":policy_improvement_full_runtime",
        ":policy_improvement_non_smoke_checkpoint",
        ":policy_improvement_theory_schema",
        # policy_improvement_full_backend imports this at call time on the
        # theory-identity path. The full source profile declares it too.
        ":policy_improvement_theory_schema_v2",
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
        ":policy_improvement_audit_lib",
        ":policy_improvement_checkpoint_validator",
        ":policy_improvement_evidence",
        ":policy_improvement_full_runtime",
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
        "configs/policy_improvement_exp1b/*.json",
        "configs/policy_improvement_exp1b/amendments/*.json",
    ]),
    deps = [
        # Stage B restores a sealed checkpoint through the sanctioned allowlist
        # loader, which `open_exp1b_sealed_evaluation_session` imports at call
        # time so the full PAR does not pick it up.
        ":policy_improvement_checkpoint_allowlist",
        # The publication gate runs the independent consumer before any result
        # is written, so the evaluator PAR carries it.
        ":policy_improvement_exp1b_auditor",
        ":policy_improvement_exp1b_runtime",
        ":policy_improvement_exp1b_theory_backend",
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
        ":policy_improvement_populations",
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
        ":policy_improvement_theory_schema_v2",
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
        ":policy_improvement_theory_schema_v2",
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
        # generate_registry -> validate_v2_amendment_history imports both theory
        # schemas at call time to break an import cycle, so the library graph
        # cannot express them. Bind them at the binary instead.
        ":policy_improvement_theory_schema",
        ":policy_improvement_theory_schema_v2",
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
        # phase4_runtime_profile enumerates evaluators/ as producer source, so
        # the checked-in producer manifest and every training-source profile
        # already declare these two modules. Without this dep the PARs ship
        # without them and fail source authentication.
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
        # Keep in step with :upi_trm_train_lib. The producer manifest declares
        # evaluators/ as producer source, so the standalone training PAR must
        # carry it too.
        ":evaluators",
        ":models",
        ":policy_improvement_checkpoint_allowlist",
        ":policy_improvement_smoke_runtime",
        # Stage 0 calls generate_registry, which imports both theory schemas at
        # call time to break an import cycle. Bind them at the binary. Neither
        # is a producer source, so the embedded producer-manifest inventory is
        # unchanged. Do not move these to :upi_trm_train_lib: that would pull
        # policy_improvement_theory_schema_v2 into policy_improvement_full,
        # whose source profile does not list it.
        ":policy_improvement_theory_schema",
        ":policy_improvement_theory_schema_v2",
        ":puzzle_dataset_lib",
        ":rl",
        ":runtime_archive_preflight",
        ":utils",
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
        ":policy_improvement_analysis_lib",
        ":policy_improvement_populations",
        ":policy_improvement_registry",
        ":policy_improvement_schema",
        ":policy_improvement_smoke_plan_lib",
        ":policy_improvement_test_open",
        ":policy_improvement_test_open_cli",
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
    name = "test_policy_improvement_launcher_identity",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_launcher_identity_unittest.py",
    ],
    base_module = "",
    typing = True,
    # The real Buck-built launcher, not a synthetic archive. This is the only
    # test that can detect the checked-in launcher identity drifting away from
    # what Buck actually produces.
    resources = {
        "BUCK": "BUCK",
        ":phase4_runtime_launcher": "phase4_runtime_launcher.par",
    },
    deps = [
        ":phase4_runtime_profile",
        ":policy_improvement_runtime_authorization",
    ],
)

python_unittest(
    name = "test_policy_improvement_audit",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_audit_unittest.py",
    ],
    base_module = "",
    resources = glob([
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/amendments/*.json",
    ]),
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
    name = "test_policy_improvement_throughput",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_throughput_unittest.py",
    ],
    base_module = "",
    typing = True,
    resources = [
        "configs/policy_improvement_v2/fixed_base_exact_episodic.yaml",
        "configs/policy_improvement_v2/fixed_base_exact_persistent.yaml",
        "configs/policy_improvement_v2/legacy_parameter_interpolation.yaml",
        "configs/policy_improvement_v2/matched_ppo.yaml",
        "configs/policy_improvement_v2/populations.json",
        "configs/policy_improvement_v2/protocol.json",
        "configs/policy_improvement_v2/registry.json",
    ],
    deps = [
        "fbsource//third-party/pypi/torch:torch",
        ":policy_improvement_throughput",
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
        ":policy_improvement_audit_lib",
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
    name = "source_identity",
    srcs = ["utils/source_identity.py"],
    base_module = "",
    typing = True,
)

python_binary(
    name = "generate_producer_source_manifest",
    srcs = ["scripts/generate_producer_source_manifest.py"],
    base_module = "",
    main_module = "scripts.generate_producer_source_manifest",
    deps = [
        ":source_identity",
    ],
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
    compile = False,
    # The launcher authenticates a runtime PAR before any behavior import. Its
    # authorization boundary pins the exact executable closure, so nothing may
    # run inside this PAR that the pin does not name. The fbcode macro layer
    # adds fbcode//python/imports_monitor by default and installs it as a
    # startup function, which would put import-time telemetry in front of
    # authentication. Opt out here rather than widening the pinned closure.
    imports_monitor = False,
    main_module = "phase4_runtime_launcher",
    resources = [
        "configs/iclr_confirmatory/producer_source_manifest.json",
    ],
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

# --- Experiment 1B reduced study --------------------------------------------
#
# Standard-library only, except the Stage A/B Torch work which lives in
# `:policy_improvement_full_backend`. Dependency edges here mirror the module
# imports exactly, because the exact source profiles are compared against the
# built PAR's selected sources: a transitive edge that pulls a
# `scripts/policy_improvement_*.py` module into a PAR whose profile omits it
# fails archive validation.

python_library(
    name = "policy_improvement_exp1_diagnostics",
    srcs = ["scripts/policy_improvement_exp1_diagnostics.py"],
    base_module = "",
)

python_library(
    name = "policy_improvement_exp1b_bootstrap",
    srcs = ["scripts/policy_improvement_exp1b_bootstrap.py"],
    base_module = "",
)

python_library(
    name = "policy_improvement_exp1b_schema",
    srcs = ["scripts/policy_improvement_exp1b_schema.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1b_bootstrap",
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_exp1b_evidence",
    srcs = ["scripts/policy_improvement_exp1b_evidence.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_exp1b_bridge",
    srcs = ["scripts/policy_improvement_exp1b_bridge.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1_diagnostics",
        ":policy_improvement_exp1b_evidence",
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_exp1b_aggregate",
    srcs = ["scripts/policy_improvement_exp1b_aggregate.py"],
    base_module = "",
    # The auditor is NOT a dependency: `validated_exp1b_document` imports it at
    # call time so the full PAR, which never publishes, does not carry it. The
    # theory-bridge PAR declares it directly instead.
    deps = [
        ":policy_improvement_exp1_diagnostics",
        ":policy_improvement_exp1b_bootstrap",
        ":policy_improvement_exp1b_bridge",
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_exp1b_session",
    srcs = ["scripts/policy_improvement_exp1b_session.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1b_schema",
    ],
)

python_library(
    name = "policy_improvement_exp1b_theory_backend",
    srcs = ["scripts/policy_improvement_exp1b_theory_backend.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1b_evidence",
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_schema",
    ],
)

python_library(
    name = "policy_improvement_exp1b_auditor",
    srcs = ["scripts/policy_improvement_exp1b_auditor.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1b_bootstrap",
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_schema",
    ],
)

# Independent Experiment 1B audit consumer. Its own binary on purpose: a
# producer PAR must not be able to run it, and it must not be able to reach a
# producer's Torch stack. Reads a published result plus four registered
# documents; no dataset, no checkpoint, no evidence payload.
python_binary(
    name = "policy_improvement_exp1b_auditor_bin",
    base_module = "",
    compile = False,
    main_module = "scripts.policy_improvement_exp1b_auditor",
    resources = glob([
        "configs/policy_improvement_exp1b/*.json",
        "configs/policy_improvement_exp1b/amendments/*.json",
    ]) + [
        # The parent population registry. The census-ordering rederivation is
        # not optional any more, so the binary that advertises it has to ship
        # the document it rederives from.
        "configs/policy_improvement_v2/populations.json",
    ],
    deps = [
        ":phase4_runtime_profile",
        ":policy_improvement_exp1b_auditor",
    ],
)

python_library(
    name = "policy_improvement_exp1b_runtime",
    srcs = ["scripts/policy_improvement_exp1b_runtime.py"],
    base_module = "",
    deps = [
        ":policy_improvement_exp1b_aggregate",
        ":policy_improvement_exp1b_bridge",
        ":policy_improvement_exp1b_evidence",
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_exp1b_session",
        ":policy_improvement_schema",
    ],
)

python_unittest(
    name = "test_policy_improvement_exp1_diagnostics",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_exp1_diagnostics_unittest.py",
    ],
    base_module = "",
    typing = True,
    deps = [
        ":policy_improvement_exp1_diagnostics",
        # The oracle half of this suite compares the finite-reference bound
        # against the v2 theory bridge's own arithmetic, so the test module
        # imports it at module scope. Without this edge all 46 tests die at
        # import in a hermetic runfiles tree; a standalone checkout hides it
        # because the whole repository is already on sys.path.
        ":policy_improvement_theory_bridge_v2_lib",
    ],
)

python_unittest(
    name = "test_policy_improvement_exp1b",
    srcs = [
        "tests/__init__.py",
        "tests/test_policy_improvement_exp1b_unittest.py",
    ],
    base_module = "",
    typing = True,
    # The wiring tests read these files by path, so every one of them must be
    # declared or the hermetic runfiles tree will not contain it.
    resources = glob([
        "configs/policy_improvement_exp1b/*.json",
        "configs/policy_improvement_exp1b/amendments/*.json",
        "configs/policy_improvement_v2/*.json",
        "configs/policy_improvement_v2/*.yaml",
        "configs/policy_improvement_v2/amendments/*.json",
    ]) + [
        "BUCK",
        "phase4_runtime_launcher.py",
        "phase4_runtime_profile.py",
        "policy_improvement_full_backend.py",
        "policy_improvement_full_entrypoint.py",
        "policy_improvement_theory_bridge_entrypoint.py",
        "scripts/policy_improvement_exp1b_aggregate.py",
        "scripts/policy_improvement_exp1b_auditor.py",
        "scripts/policy_improvement_exp1b_bootstrap.py",
        "scripts/policy_improvement_exp1b_bridge.py",
        "scripts/policy_improvement_exp1b_evidence.py",
        "scripts/policy_improvement_exp1b_runtime.py",
        "scripts/policy_improvement_exp1b_schema.py",
        "scripts/policy_improvement_exp1b_session.py",
        "scripts/policy_improvement_exp1b_theory_backend.py",
        "scripts/policy_improvement_schema.py",
        "scripts/policy_improvement_theory_backend_v2.py",
    ],
    # Imported directly by the test module. Declaring these files as resources
    # does not make them importable; only a dependency does.
    deps = [
        # Five Torch-gated classes hold eleven methods between them, and each
        # class self-skips on `importlib.util.find_spec("torch")`. Without this
        # edge the PAR has no Torch and all eleven skip while the suite still
        # reports OK. Four of them then execute the real serialization path:
        # `Exp1bTorchFourModuleRoundTripTest` seals through
        # `seal_exp1b_training_checkpoint`, reloads through
        # `load_data_only_checkpoint`, and restores all four module states into
        # fresh modules. The other three that remain skipped are unimplemented
        # placeholders, not capability skips -- see the handoff.
        "fbsource//third-party/pypi/torch:torch",
        ":confirmatory_runtime_launcher_lib",
        ":phase4_runtime_launcher_lib",
        ":phase4_runtime_profile",
        # `Exp1bBaseArtifactBufferTest` imports this module directly. Declaring
        # a file as a resource puts it in the runfiles tree; only a dependency
        # makes it importable.
        ":policy_improvement_base_policy_restore",
        # `Exp1bTorchSealedCheckpointTest` and
        # `Exp1bTorchFourModuleRoundTripTest` import the backend directly, and
        # the round trip calls its serializer. It is also a resource above, for
        # the tests that read it as source; that is not an import edge.
        ":policy_improvement_full_backend",
        ":policy_improvement_exp1_diagnostics",
        ":policy_improvement_exp1b_aggregate",
        ":policy_improvement_exp1b_auditor",
        ":policy_improvement_exp1b_bootstrap",
        ":policy_improvement_exp1b_bridge",
        ":policy_improvement_exp1b_evidence",
        ":policy_improvement_exp1b_runtime",
        ":policy_improvement_exp1b_schema",
        ":policy_improvement_exp1b_session",
        ":policy_improvement_exp1b_theory_backend",
        ":policy_improvement_schema",
        ":policy_improvement_theory_bridge_v2_lib",
    ],
)
