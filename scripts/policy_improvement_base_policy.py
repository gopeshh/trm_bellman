#!/usr/bin/env fbpython
"""Produce one authenticated train-only Sudoku base policy for protocol v2.

Experiment 0. Every Stage 1 row is blocked because the v2 protocol registers
``base_policy_artifact`` as unavailable. This producer materializes the single
competent base policy that persistent, episodic, parameter-interpolation, and
matched-PPO runs will all share unchanged.

Three properties make the artifact usable as a shared base:

* Train only. The producer reads the registered 1,024-record train split and
  nothing else. It never resolves, opens, hashes, or materializes validation or
  test content, and it performs no evaluation at all.
* Budget selected. The training procedure is frozen before launch and the
  final-budget checkpoint is the artifact. No checkpoint is chosen by observed
  accuracy or solve rate, and imitation early stopping is disabled.
* Method neutral. Oracle imitation does not optimize any policy-improvement
  objective. Every configuration value the producer consumes is asserted
  identical across all four registered method configs, so no method is
  favored, and the architecture comes from the protocol.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


PRODUCER_SCHEMA_NAME = "policy_improvement_base_policy_producer_v2"
PRODUCER_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Frozen training procedure. Every value here is fixed before launch and is
# covered by TRAINING_PROCEDURE, whose canonical digest becomes the artifact's
# training_procedure_sha256.
# ---------------------------------------------------------------------------

# Distinct from every registered protocol seed: smoke 1257297357, pilot
# 784831257 / 2087907586 / 4056782312, and the eight confirmatory seeds.
BASE_POLICY_SEED = 1904261137
IMITATION_EPOCHS = 40
# The oracle is deterministic over a fixed corpus, so repeating collection only
# duplicates identical demonstrations and multiplies epoch cost without adding
# information. One pass, then epochs supply the reshuffling.
DEMONSTRATION_REPEATS = 1
IMITATION_LEARNING_RATE = 0.003
IMITATION_BATCH_SIZE = 64
# Oracle imitation only. Running any policy-improvement objective here would
# favor one of the four methods under comparison.
RL_ENVIRONMENT_INTERACTIONS = 0
# Three of the four registered methods, including both exact methods, use the
# terminal STOP contract. The oracle never emits STOP, so this fixes the action
# mask rather than the demonstrated behavior.
STOP_ACTION_MODE = "terminal"
EARLY_STOP_ACCURACY = None
# Disposable pre-flight budget. It is deliberately NOT part of the frozen
# procedure: the health check runs in its own process, saves nothing, and never
# touches the artifact lineage.
HEALTH_CHECK_EPOCHS = 1
TRAIN_SPLIT = "train"
TRAIN_RECORD_COUNT = 1024

TRAINING_PROCEDURE: dict[str, Any] = {
    "schema_name": "policy_improvement_base_policy_procedure_v2",
    "schema_version": 1,
    "kind": "oracle_imitation_pretraining",
    "objective": "cross_entropy_to_oracle_cell_fill_action",
    "seed": BASE_POLICY_SEED,
    "imitation_epochs": IMITATION_EPOCHS,
    "demonstration_repeats": DEMONSTRATION_REPEATS,
    "demonstration_episodes_per_repeat": TRAIN_RECORD_COUNT,
    "imitation_learning_rate": IMITATION_LEARNING_RATE,
    "imitation_batch_size": IMITATION_BATCH_SIZE,
    "imitation_optimizer": "adam",
    "early_stop_accuracy": EARLY_STOP_ACCURACY,
    "rl_environment_interactions": RL_ENVIRONMENT_INTERACTIONS,
    "stop_action_mode": STOP_ACTION_MODE,
    "training_split": TRAIN_SPLIT,
    "training_record_count": TRAIN_RECORD_COUNT,
    "checkpoint_selection": "final_budget_only",
    "validation_data_used": False,
    "test_data_used": False,
    "shared_across_persistent_and_episodic": True,
}

# Configuration the producer reads from the registered method configs. Every
# one must be identical across all four, which is what makes the base neutral.
SHARED_CONFIG_FIELDS = (
    "C_max",
    "batch_size",
    "disable_constraint_masking",
    "enable_contraction",
    "fail_terminal_reward",
    "gamma",
    "inner_unroll_n",
    "latent_ball_radius",
    "latent_projection_mode",
    "max_edits",
    "reward_shaping",
    "solve_terminal_reward",
    "solved_threshold",
    "target_Lv",
    "target_Lz",
    "task_name",
)
REGISTERED_METHOD_CONFIGS = (
    "fixed_base_exact_persistent",
    "fixed_base_exact_episodic",
    "legacy_parameter_interpolation",
    "matched_ppo",
)


class BasePolicyProducerError(RuntimeError):
    """Raised when the train-only base policy cannot be produced cleanly."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_procedure_sha256() -> str:
    return canonical_sha256(TRAINING_PROCEDURE)


def neutral_shared_config(
    configs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Return the configuration values every registered method agrees on.

    Any disagreement is a method-specific value and must not reach the shared
    base policy, so it fails closed rather than silently picking one method.
    """

    missing = sorted(set(REGISTERED_METHOD_CONFIGS) - set(configs))
    if missing:
        raise BasePolicyProducerError(
            f"Registered method configs are missing {missing!r}."
        )
    shared: dict[str, object] = {}
    for field in SHARED_CONFIG_FIELDS:
        values = {}
        for method in REGISTERED_METHOD_CONFIGS:
            if field not in configs[method]:
                raise BasePolicyProducerError(
                    f"Registered config {method!r} does not define {field!r}."
                )
            values[method] = configs[method][field]
        distinct = {canonical_json_bytes(value) for value in values.values()}
        if len(distinct) != 1:
            raise BasePolicyProducerError(
                f"Configuration {field!r} differs across registered methods; "
                "a shared base policy cannot adopt a method-specific value."
            )
        shared[field] = values[REGISTERED_METHOD_CONFIGS[0]]
    return shared


def _assert_absent(path: Path, *, label: str) -> None:
    """Guard that held-out content is never even addressed."""

    if path.exists():
        raise BasePolicyProducerError(
            f"Refusing to run: the producer addressed {label} at {path}."
        )


def authenticate_train_split(dataset_root: Path) -> dict[str, Any]:
    """Authenticate the registered train split without touching held-out data."""

    if not dataset_root.is_absolute() or os.path.realpath(dataset_root) != str(
        dataset_root
    ):
        raise BasePolicyProducerError(
            "Dataset root must be an absolute canonical path."
        )
    manifest_path = dataset_root / "MANIFEST.json"
    train_manifest_path = dataset_root / "manifests/train.json"
    for path in (dataset_root, manifest_path, train_manifest_path):
        if stat.S_ISLNK(path.lstat().st_mode):
            raise BasePolicyProducerError(f"Dataset path {path} is a symlink.")
    train_manifest = json.loads(train_manifest_path.read_text(encoding="ascii"))
    if train_manifest["generated_count"] != TRAIN_RECORD_COUNT:
        raise BasePolicyProducerError(
            "Registered train split does not hold the registered record count."
        )
    records = train_manifest["record_sha256s"]
    if len(records) != TRAIN_RECORD_COUNT or len(set(records)) != TRAIN_RECORD_COUNT:
        raise BasePolicyProducerError("Train record identities are not unique.")
    return {
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "train_ordered_record_sha256": train_manifest["ordered_record_sha256"],
        "train_record_count": int(train_manifest["generated_count"]),
        "validation_content_opened": False,
        "test_content_opened": False,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--health-check",
        action="store_true",
        help=(
            "Run a short disposable timing and health segment, report it, and "
            "exit without saving. The artifact run is a separate single "
            "uninterrupted lineage, because restarting imitation would reset "
            "its optimizer and make the artifact depend on this flag."
        ),
    )
    parser.add_argument("--print-procedure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.print_procedure:
        sys.stdout.buffer.write(
            canonical_json_bytes(
                {
                    "training_procedure": TRAINING_PROCEDURE,
                    "training_procedure_sha256": training_procedure_sha256(),
                }
            )
            + b"\n"
        )
        return 0

    # Imported here so --print-procedure stays a standard-library operation.
    import torch

    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    from phase4_runtime_profile import authorize_phase4_training_source
    from policy_improvement_smoke_runtime import build_protocol_v2_model_config
    from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
    from rl.persistent_diagnostic_checkpoint import state_dict_sha256
    from scripts.policy_improvement_v2_registry import load_v2_base_configs
    from scripts.policy_improvement_v2_schema import (
        load_strict_json,
        validate_v2_protocol,
    )
    from utils.seeding import set_global_seed
    import upi_trm_train as training_module

    started = time.time()
    project_root = Path(arguments.project_root).resolve(strict=True)
    dataset_root = Path(arguments.dataset_root).resolve(strict=True)
    output_root = Path(arguments.output_root)

    # Producer identity. This also proves the checkout is clean at the exact
    # commit and yields the producer source-manifest digest.
    authorized = authorize_phase4_training_source(
        str(project_root), arguments.expected_git_commit
    )

    _assert_absent(output_root / "TEST_OPEN", label="a test-open record")
    dataset_identity = authenticate_train_split(dataset_root)

    protocol = validate_v2_protocol(
        load_strict_json(project_root / "configs/policy_improvement_v2/protocol.json")
    )
    base_configs = load_v2_base_configs(protocol, project_root)
    shared = neutral_shared_config(base_configs)

    log = functools.partial(print, file=sys.stderr, flush=True)
    log("[Base policy] frozen procedure")
    log(f"  seed                  {BASE_POLICY_SEED}")
    log(f"  imitation epochs      {IMITATION_EPOCHS}")
    log(f"  demonstration repeats {DEMONSTRATION_REPEATS}")
    log(f"  RL interactions       {RL_ENVIRONMENT_INTERACTIONS}")
    log(f"  procedure sha256      {training_procedure_sha256()}")

    set_global_seed(BASE_POLICY_SEED)

    dataset, seq_len, vocab_size, num_identifiers = (
        training_module.build_dataset_from_paths(
            dataset_paths=[str(dataset_root)],
            pool_size=TRAIN_RECORD_COUNT,
            split=TRAIN_SPLIT,
        )
    )
    if len(dataset) != TRAIN_RECORD_COUNT:
        raise BasePolicyProducerError(
            "Materialized train pool differs from the registered record count."
        )
    training_module._validate_materialized_split_manifest(
        dataset_root=str(dataset_root),
        split=TRAIN_SPLIT,
        registered_sha256=dataset_identity["train_manifest_sha256"],
        dataset=dataset,
    )

    rl_config = training_module.RLConfig(
        **{
            "max_edits": int(shared["max_edits"]),
            "gamma": float(shared["gamma"]),
            "reward_shaping": bool(shared["reward_shaping"]),
            "task_name": str(shared["task_name"]),
            "batch_size": int(shared["batch_size"]),
            "inner_unroll_n": int(shared["inner_unroll_n"]),
            "enable_contraction": bool(shared["enable_contraction"]),
            "target_Lz": float(shared["target_Lz"]),
            "target_Lv": float(shared["target_Lv"]),
            "latent_projection_mode": str(shared["latent_projection_mode"]),
            "latent_ball_radius": float(shared["latent_ball_radius"]),
            "C_max": float(shared["C_max"]),
            "disable_constraint_masking": bool(shared["disable_constraint_masking"]),
            "fail_terminal_reward": float(shared["fail_terminal_reward"]),
            "solve_terminal_reward": float(shared["solve_terminal_reward"]),
            "solved_threshold": shared["solved_threshold"],
            "stop_action_mode": STOP_ACTION_MODE,
            "use_tqdm": False,
        }
    )

    resolution = training_module.resolve_checker_from_dataset(
        rl_cfg=rl_config, dataset=dataset, seq_len=seq_len
    )
    checker_fn = resolution.checker_fn
    task = None
    if resolution.checker_kind in {"solution", "constraint", "progress", "feasibility"}:
        from rl.task_config import get_task_config

        task = get_task_config(
            "sudoku",
            disable_constraint_masking=rl_config.disable_constraint_masking,
        )
    env_config = PlanEditEnvConfig(
        max_edits=rl_config.max_edits,
        gamma=rl_config.gamma,
        reward_shaping=rl_config.reward_shaping,
        vocab_size=vocab_size,
        solved_threshold=rl_config.solved_threshold,
        task_type=rl_config.task_name,
        stop_action_mode=rl_config.stop_action_mode,
        stop_action_penalty=rl_config.stop_action_penalty,
        fail_terminal_reward=rl_config.fail_terminal_reward,
        solve_terminal_reward=rl_config.solve_terminal_reward,
        C_max=rl_config.C_max,
        disable_constraint_masking=rl_config.disable_constraint_masking,
    )
    env = PlanEditEnv(
        dataset=dataset, checker=checker_fn, config=env_config, task_config=task
    )
    action_count = seq_len * vocab_size + 1
    env.set_stop_action_id(action_count - 1)

    model_config = build_protocol_v2_model_config(
        architecture=protocol["architecture"],
        rl_config=rl_config,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_identifiers=num_identifiers,
        action_count=action_count,
    )
    # Two distinct digests, deliberately not interchangeable. The amendment
    # binds the protocol's registered architecture block; the artifact payload
    # binds the full model configuration built from it. Conflating them makes
    # the amendment fail validate_v2_amendment_history at Stage 1 load.
    registered_architecture_sha256 = canonical_sha256(protocol["architecture"])
    model_config_sha256 = canonical_sha256(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRecursiveReasoningModel_ACTV1(model_config).to(device)
    initialization_sha256 = state_dict_sha256(model.state_dict())

    baseline = training_module.select_baseline_from_configs(None, [])
    trainer = training_module.build_trainer(
        model=model,
        env=env,
        rl_cfg=rl_config,
        device=device,
        baseline_selection=baseline,
        cli_baseline=None,
        verbose=False,
    )
    if not hasattr(trainer, "imitation_pretrain"):
        raise BasePolicyProducerError(
            "The base-policy producer requires the default UPI-TRM trainer."
        )

    if arguments.health_check:
        health_started = time.time()
        with redirect_stdout(sys.stderr):
            stats = trainer.imitation_pretrain(
                dataset=dataset,
                checker=checker_fn,
                num_epochs=HEALTH_CHECK_EPOCHS,
                batch_size=IMITATION_BATCH_SIZE,
                log_interval=1,
                imitation_lr=IMITATION_LEARNING_RATE,
                demonstration_episodes=TRAIN_RECORD_COUNT,
                demonstration_repeats=DEMONSTRATION_REPEATS,
                early_stop_accuracy=EARLY_STOP_ACCURACY,
            )
        elapsed = time.time() - health_started
        report = {
            "schema_name": "policy_improvement_base_policy_health_v2",
            "status": "complete",
            "artifact_saved": False,
            "device": str(device),
            "health_epochs": HEALTH_CHECK_EPOCHS,
            "seconds": round(elapsed, 3),
            "seconds_per_epoch": round(elapsed / HEALTH_CHECK_EPOCHS, 3),
            "predicted_budget_seconds": round(
                elapsed / HEALTH_CHECK_EPOCHS * IMITATION_EPOCHS, 1
            ),
            "imitation_loss": float(stats["imitation_loss"]),
            "imitation_accuracy": float(stats["imitation_accuracy"]),
            "training_procedure_sha256": training_procedure_sha256(),
            "architecture_sha256": model_config_sha256,
            "initialization_sha256": initialization_sha256,
            "wall_time_seconds": round(time.time() - started, 3),
        }
        sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
        return 0

    # One uninterrupted lineage to the frozen budget. Imitation is not resumed
    # or restarted, so the artifact is a function of the frozen procedure alone.
    with redirect_stdout(sys.stderr):
        final_stats = trainer.imitation_pretrain(
            dataset=dataset,
            checker=checker_fn,
            num_epochs=IMITATION_EPOCHS,
            batch_size=IMITATION_BATCH_SIZE,
            log_interval=1,
            imitation_lr=IMITATION_LEARNING_RATE,
            demonstration_episodes=TRAIN_RECORD_COUNT,
            demonstration_repeats=DEMONSTRATION_REPEATS,
            early_stop_accuracy=EARLY_STOP_ACCURACY,
        )

    # The final-budget state is the artifact. Nothing selects among snapshots.
    model_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    model_state_sha256 = state_dict_sha256(model_state)
    payload = {
        "schema_name": PRODUCER_SCHEMA_NAME,
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "initialization_kind": "train_only_pretrained",
        "model_config": model_config,
        "architecture_sha256": model_config_sha256,
        "model_state": model_state,
        "model_state_sha256": model_state_sha256,
        "training_procedure": TRAINING_PROCEDURE,
        "training_procedure_sha256": training_procedure_sha256(),
        "producer_git_commit": authorized.git_commit,
        "producer_source_manifest_sha256": authorized.source_manifest_sha256,
        "dataset_identity": dataset_identity,
        "initialization_sha256": initialization_sha256,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "base_policy.pt"
    if checkpoint_path.exists():
        raise BasePolicyProducerError("Base-policy checkpoint already exists.")
    torch.save(payload, checkpoint_path)
    os.chmod(checkpoint_path, 0o400)
    checkpoint_status = checkpoint_path.lstat()

    document = {
        "schema_name": PRODUCER_SCHEMA_NAME,
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "status": "complete",
        "initialization_kind": "train_only_pretrained",
        "architecture_sha256": registered_architecture_sha256,
        "model_config_sha256": model_config_sha256,
        "model_state_sha256": model_state_sha256,
        "initialization_sha256": initialization_sha256,
        "producer_git_commit": authorized.git_commit,
        "producer_source_manifest_sha256": authorized.source_manifest_sha256,
        "training_dataset_manifest_sha256": dataset_identity[
            "dataset_manifest_sha256"
        ],
        "train_manifest_sha256": dataset_identity["train_manifest_sha256"],
        "training_split": TRAIN_SPLIT,
        "training_split_ordered_record_sha256": dataset_identity[
            "train_ordered_record_sha256"
        ],
        "training_procedure": TRAINING_PROCEDURE,
        "training_procedure_sha256": training_procedure_sha256(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_status.st_size,
        "model_config": model_config,
        "neutral_shared_config": shared,
        "device": str(device),
        "final_imitation_loss": float(final_stats["imitation_loss"]),
        "final_imitation_accuracy": float(final_stats["imitation_accuracy"]),
        "wall_time_seconds": round(time.time() - started, 3),
        "shared_across_persistent_and_episodic": True,
        "not_selected_by_validation_or_test": True,
        "validation_data_opened": False,
        "test_data_opened": False,
    }
    identity_path = output_root / "base_policy_identity.json"
    identity_path.write_bytes(canonical_json_bytes(document) + b"\n")
    os.chmod(identity_path, 0o400)

    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
