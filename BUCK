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
