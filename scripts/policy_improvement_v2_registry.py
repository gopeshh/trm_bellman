#!/usr/bin/env fbpython
"""Generate the deterministic 139-row policy-improvement v2 registry."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.policy_improvement_populations import (
    canonical_json_bytes as population_json_bytes,
    load_strict_json as load_population_json,
    validate_v2_populations,
)
from scripts.policy_improvement_v2_schema import (
    EXACT_METHOD_IDS,
    METHOD_IDS,
    PROTOCOL_ID,
    REGISTRY_ROW_SCHEMA_NAME,
    REGISTRY_ROW_SCHEMA_VERSION,
    REGISTRY_SCHEMA_NAME,
    REGISTRY_SCHEMA_VERSION,
    PolicyImprovementV2SchemaError,
    canonical_json_bytes,
    effective_config_sha256,
    load_strict_json,
    sha256_json,
    validate_v2_protocol,
    validate_v2_registry_row,
)


EXPECTED_PHASE_COUNTS = {
    "stage0_smoke": 4,
    "stage1_screen": 24,
    "stage1_alpha": 9,
    "stage1_baseline_readiness": 6,
    "stage2_confirmatory": 32,
    "stage3_ablation": 64,
}
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_FLOAT = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]*)\.[0-9]+|(?:0|[1-9][0-9]*)[eE][+-]?[0-9]+)$"
)


def load_flat_registered_yaml(path: str | Path) -> dict[str, object]:
    """Parse the deliberately flat method configurations without PyYAML."""

    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyImprovementV2SchemaError(
            "Registered method config could not be read as ASCII."
        ) from exc
    result: dict[str, object] = {}
    for line_number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if raw[:1].isspace() or text.count(":") != 1:
            raise PolicyImprovementV2SchemaError(
                f"Registered config line {line_number} is not flat canonical YAML."
            )
        key, scalar = (part.strip() for part in text.split(":", 1))
        if not key or key in result:
            raise PolicyImprovementV2SchemaError(
                f"Registered config line {line_number} has a duplicate/empty key."
            )
        if scalar == "null":
            value: object = None
        elif scalar == "true":
            value = True
        elif scalar == "false":
            value = False
        elif _INTEGER.fullmatch(scalar) is not None:
            value = int(scalar)
        elif _FLOAT.fullmatch(scalar) is not None:
            value = float(scalar)
        elif not scalar or scalar.startswith(("[", "{", "&", "*", "!", "|", ">")):
            raise PolicyImprovementV2SchemaError(
                f"Registered config line {line_number} uses unsupported YAML."
            )
        else:
            value = scalar
        result[key] = value
    if not result:
        raise PolicyImprovementV2SchemaError("Registered method config is empty.")
    canonical_json_bytes(result)
    return result


def load_v2_base_configs(
    protocol_value: object, project_root: str | Path
) -> dict[str, Mapping[str, object]]:
    protocol = validate_v2_protocol(protocol_value)
    root = Path(project_root).resolve(strict=True)
    configs: dict[str, Mapping[str, object]] = {}
    for method in protocol["methods"]:
        path = (root / str(method["config_path"])).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PolicyImprovementV2SchemaError(
                "Registered method config escaped the project root."
            ) from exc
        raw_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_digest != method["config_sha256"]:
            raise PolicyImprovementV2SchemaError(
                f"Raw config bytes for {method['id']!r} differ from the protocol."
            )
        parsed = load_flat_registered_yaml(path)
        canonical_digest = hashlib.sha256(canonical_json_bytes(parsed)).hexdigest()
        if canonical_digest != method["canonical_config_sha256"]:
            raise PolicyImprovementV2SchemaError(
                f"Canonical config for {method['id']!r} differs from the protocol."
            )
        configs[str(method["id"])] = parsed
    return configs


def _factor_applicability(
    method_id: str | None, *, ablation: bool = False, baseline: bool = False
) -> dict[str, bool]:
    if baseline:
        return {"n": True, "K": False, "alpha": False, "ablation": False}
    if method_id == "matched_ppo":
        return {"n": True, "K": False, "alpha": False, "ablation": ablation}
    return {"n": True, "K": True, "alpha": True, "ablation": ablation}


def _checked_base_config(
    method_id: str,
    protocol: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    if method_id not in base_configs:
        raise PolicyImprovementV2SchemaError(
            f"Missing canonical base config for {method_id!r}."
        )
    base = base_configs[method_id]
    registration = next(item for item in protocol["methods"] if item["id"] == method_id)
    if sha256_json(base) != registration["canonical_config_sha256"]:
        raise PolicyImprovementV2SchemaError(
            f"Canonical base config for {method_id!r} differs from the protocol."
        )
    return base


def _concrete_config_fields(
    method_id: str,
    *,
    n: int,
    K: int,
    alpha: float,
    protocol: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    base = _checked_base_config(method_id, protocol, base_configs)
    override: dict[str, object] = {"inner_unroll_n": n}
    if method_id != "matched_ppo":
        override.update({"K": K, "mixture_alpha": alpha})
    return {
        "base_method_id": method_id,
        "config_override": override,
        "base_config_canonical_sha256": sha256_json(base),
        "expected_effective_config_sha256": effective_config_sha256(base, override),
    }


def _row(
    *,
    row_kind: str,
    phase: str,
    tier: str,
    run_id: str,
    method_id: str | None,
    base_method_id: str | None,
    seed: int,
    evaluation_population: str,
    evaluation_split: str,
    n: int | None,
    K: int | None,
    alpha: float | None,
    ablation_variant: str | None,
    selection_rule: str | None,
    continuation_rule: str | None,
    checkpoints: Sequence[int],
    factor_applicability: Mapping[str, bool],
    scientific_selection: bool,
    paper_evidence_eligible: bool,
    config_override: Mapping[str, object] | None = None,
    base_config_canonical_sha256: str | None = None,
    expected_effective_config_sha256: str | None = None,
) -> dict[str, Any]:
    return validate_v2_registry_row(
        {
            "schema_name": REGISTRY_ROW_SCHEMA_NAME,
            "schema_version": REGISTRY_ROW_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "row_kind": row_kind,
            "phase": phase,
            "tier": tier,
            "run_id": run_id,
            "method_id": method_id,
            "base_method_id": base_method_id,
            "seed": seed,
            "evaluation_population": evaluation_population,
            "evaluation_split": evaluation_split,
            "n": n,
            "K": K,
            "alpha": alpha,
            "ablation_variant": ablation_variant,
            "selection_rule": selection_rule,
            "continuation_rule": continuation_rule,
            "checkpoint_environment_interactions": list(checkpoints),
            "factor_applicability": dict(factor_applicability),
            "scientific_selection": scientific_selection,
            "paper_evidence_eligible": paper_evidence_eligible,
            "config_override": (
                None if config_override is None else dict(config_override)
            ),
            "base_config_canonical_sha256": base_config_canonical_sha256,
            "expected_effective_config_sha256": expected_effective_config_sha256,
        }
    )


def generate_v2_registry(
    protocol_value: object,
    populations_value: object,
    *,
    base_configs: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Return the immutable pre-outcome v2 registry without running a cell."""

    protocol = validate_v2_protocol(protocol_value)
    populations = validate_v2_populations(populations_value)
    population_digest = hashlib.sha256(population_json_bytes(populations)).hexdigest()
    if population_digest != protocol["population_registry"]["sha256"]:
        raise PolicyImprovementV2SchemaError(
            "Population document differs from the protocol binding."
        )
    for split in ("train", "validation"):
        registration = protocol["dataset"]["splits"][split]["ordered_record_sha256"]
        if (
            registration.get("status") != "available"
            or registration.get("value")
            != populations["split_ordered_record_sha256"][split]
        ):
            raise PolicyImprovementV2SchemaError(
                f"Protocol {split} identity differs from the population document."
            )
    seeds = protocol["seeds"]
    grid = protocol["grid"]
    rows: list[dict[str, Any]] = []

    stage0 = grid["stage0"]
    for method_id in METHOD_IDS:
        config = _concrete_config_fields(
            method_id,
            n=2,
            K=1,
            alpha=0.1,
            protocol=protocol,
            base_configs=base_configs,
        )
        rows.append(
            _row(
                row_kind="concrete",
                phase="stage0_smoke",
                tier="smoke",
                run_id=f"s0-{method_id}-s{seeds['smoke'][0]}",
                method_id=method_id,
                seed=int(seeds["smoke"][0]),
                evaluation_population="stage0_smoke",
                evaluation_split="train",
                n=int(stage0["n"]),
                K=int(stage0["K"]),
                alpha=float(stage0["alpha"]),
                ablation_variant=None,
                selection_rule=None,
                continuation_rule="strict_prepare_resume_v2",
                checkpoints=[16, 32],
                factor_applicability=_factor_applicability(method_id),
                scientific_selection=False,
                paper_evidence_eligible=False,
                **config,
            )
        )

    screen = grid["stage1_screen"]
    for method_id in EXACT_METHOD_IDS:
        for n in screen["n_values"]:
            for K in screen["K_values"]:
                for seed in seeds["pilot"]:
                    config = _concrete_config_fields(
                        method_id,
                        n=int(n),
                        K=int(K),
                        alpha=float(screen["alpha"]),
                        protocol=protocol,
                        base_configs=base_configs,
                    )
                    rows.append(
                        _row(
                            row_kind="concrete",
                            phase="stage1_screen",
                            tier="pilot",
                            run_id=f"s1-{method_id}-n{n}-k{K}-s{seed}",
                            method_id=method_id,
                            seed=int(seed),
                            evaluation_population="validation_select",
                            evaluation_split="validation",
                            n=int(n),
                            K=int(K),
                            alpha=float(screen["alpha"]),
                            ablation_variant=None,
                            selection_rule=None,
                            continuation_rule=str(screen["continuation_rule"]),
                            checkpoints=screen["checkpoint_environment_interactions"],
                            factor_applicability=_factor_applicability(method_id),
                            scientific_selection=True,
                            paper_evidence_eligible=False,
                            **config,
                        )
                    )

    alpha = grid["stage1_alpha"]
    for alpha_index, alpha_value in enumerate(alpha["alpha_values"]):
        for seed in seeds["pilot"]:
            rows.append(
                _row(
                    row_kind="selection_template",
                    phase="stage1_alpha",
                    tier="pilot",
                    run_id=f"s1a-a{alpha_index}-s{seed}",
                    method_id=None,
                    base_method_id=None,
                    seed=int(seed),
                    evaluation_population="validation_select",
                    evaluation_split="validation",
                    n=None,
                    K=None,
                    alpha=float(alpha_value),
                    ablation_variant=None,
                    selection_rule=str(alpha["selection_rule"]),
                    continuation_rule=None,
                    checkpoints=[80000],
                    factor_applicability={
                        "n": True,
                        "K": True,
                        "alpha": True,
                        "ablation": False,
                    },
                    scientific_selection=True,
                    paper_evidence_eligible=False,
                )
            )

    baselines = grid["stage1_baseline_readiness"]
    for method_id in baselines["method_ids"]:
        for seed in seeds["pilot"]:
            rows.append(
                _row(
                    row_kind="selection_template",
                    phase="stage1_baseline_readiness",
                    tier="pilot",
                    run_id=f"s1b-{method_id}-s{seed}",
                    method_id=str(method_id),
                    base_method_id=None,
                    seed=int(seed),
                    evaluation_population="validation_select",
                    evaluation_split="validation",
                    n=None,
                    K=None,
                    alpha=None,
                    ablation_variant=None,
                    selection_rule=str(baselines["materialization_rule"]),
                    continuation_rule=None,
                    checkpoints=[80000],
                    factor_applicability=_factor_applicability(
                        str(method_id), baseline=True
                    ),
                    scientific_selection=False,
                    paper_evidence_eligible=False,
                )
            )

    stage2 = grid["stage2"]
    for method_id in METHOD_IDS:
        for seed in seeds["confirmatory"]:
            rows.append(
                _row(
                    row_kind="selection_template",
                    phase="stage2_confirmatory",
                    tier="confirmatory",
                    run_id=f"s2-{method_id}-s{seed}",
                    method_id=method_id,
                    base_method_id=None,
                    seed=int(seed),
                    evaluation_population="confirmatory_test",
                    evaluation_split="test",
                    n=None,
                    K=None,
                    alpha=None,
                    ablation_variant=None,
                    selection_rule=str(stage2["selection_rule"]),
                    continuation_rule=None,
                    checkpoints=[80000],
                    factor_applicability=_factor_applicability(method_id),
                    scientific_selection=False,
                    paper_evidence_eligible=True,
                )
            )

    stage3 = grid["stage3"]
    for variant in stage3["training_variants"]:
        for seed in seeds["confirmatory"]:
            rows.append(
                _row(
                    row_kind="selection_template",
                    phase="stage3_ablation",
                    tier="ablation",
                    run_id=f"s3-{variant}-s{seed}",
                    method_id=None,
                    base_method_id=None,
                    seed=int(seed),
                    evaluation_population="confirmatory_test",
                    evaluation_split="test",
                    n=None,
                    K=None,
                    alpha=None,
                    ablation_variant=str(variant),
                    selection_rule=str(stage3["selection_rule"]),
                    continuation_rule=None,
                    checkpoints=[80000],
                    factor_applicability={
                        "n": True,
                        "K": True,
                        "alpha": True,
                        "ablation": True,
                    },
                    scientific_selection=False,
                    paper_evidence_eligible=True,
                )
            )

    run_ids = [str(row["run_id"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise PolicyImprovementV2SchemaError("Registry run IDs are not unique.")
    phase_counts = dict(Counter(str(row["phase"]) for row in rows))
    if phase_counts != EXPECTED_PHASE_COUNTS:
        raise PolicyImprovementV2SchemaError(
            f"Registry phase counts differ: {phase_counts!r}."
        )
    registry: dict[str, Any] = {
        "schema_name": REGISTRY_SCHEMA_NAME,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_schema_name": protocol["schema_name"],
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": sha256_json(protocol),
        "population_registry_schema_name": populations["schema_name"],
        "population_registry_schema_version": populations["schema_version"],
        "population_registry_sha256": population_digest,
        "amendment_history_sha256": hashlib.sha256(
            canonical_json_bytes([])
        ).hexdigest(),
        "counts": {
            "smoke_rows": 4,
            "full_rows": 135,
            "total_rows": 139,
            **EXPECTED_PHASE_COUNTS,
        },
        "rows": rows,
    }
    canonical_json_bytes(registry)
    return registry


def validate_v2_registry_document(
    value: object,
    protocol_value: object,
    populations_value: object,
    *,
    base_configs: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    expected = generate_v2_registry(
        protocol_value,
        populations_value,
        base_configs=base_configs,
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise PolicyImprovementV2SchemaError(
            "Registry differs from deterministic v2 generation."
        )
    return expected


def registry_sha256(registry: object) -> str:
    return hashlib.sha256(canonical_json_bytes(registry)).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--populations", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    protocol = load_strict_json(arguments.protocol)
    populations = load_population_json(arguments.populations)
    configs = load_v2_base_configs(protocol, arguments.project_root)
    registry = generate_v2_registry(
        protocol,
        populations,
        base_configs=configs,
    )
    payload = canonical_json_bytes(registry)
    if arguments.output is None:
        print(payload.decode("ascii"))
    else:
        _write_new(Path(arguments.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
