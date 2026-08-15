#!/usr/bin/env fbpython
"""Generate the deterministic ``policy_improvement_v1`` run registry.

Selection-dependent rows are templates.  They cannot be made concrete until
the registered validation-only selection has been frozen by a protocol
amendment.  Keeping them in the registry still fixes their count, seeds, and
allowed selection rule before any test result is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.policy_improvement_schema import (
    METHOD_IDS,
    REGISTRY_ROW_SCHEMA_VERSION,
    SCHEMA_NAME,
    PolicyImprovementSchemaError,
    amendment_history_sha256,
    canonical_json_bytes,
    effective_config_sha256,
    load_strict_json,
    stage3_method_id,
    validate_amendment_history,
    validate_protocol,
    validate_registry_row,
)


REGISTRY_SCHEMA_VERSION = 3


def derive_seed(namespace: str, tier: str, index: int) -> int:
    """Derive one unsigned 32-bit seed from an explicit namespace."""

    if not namespace or not namespace.isascii() or ":" in namespace:
        raise PolicyImprovementSchemaError("Seed namespace is not canonical.")
    if not tier or not tier.isascii() or ":" in tier:
        raise PolicyImprovementSchemaError("Seed tier is not canonical.")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PolicyImprovementSchemaError("Seed index must be nonnegative.")
    digest = hashlib.sha256(f"{namespace}:{tier}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _row(
    *,
    protocol_id: str,
    row_kind: str,
    phase: str,
    tier: str,
    run_id: str,
    method_id: str | None,
    base_method_id: str | None,
    seed: int,
    evaluation_split: str,
    n: int | None,
    K: int | None,
    alpha: float | None,
    ablation_variant: str | None,
    selection_rule: str | None,
    factor_applicability: Mapping[str, bool],
    paper_evidence_eligible: bool,
    config_override: Mapping[str, object] | None = None,
    base_config_canonical_sha256: str | None = None,
    expected_effective_config_sha256: str | None = None,
) -> dict[str, Any]:
    return validate_registry_row(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": REGISTRY_ROW_SCHEMA_VERSION,
            "protocol_id": protocol_id,
            "row_kind": row_kind,
            "phase": phase,
            "tier": tier,
            "run_id": run_id,
            "method_id": method_id,
            "base_method_id": base_method_id,
            "seed": seed,
            "evaluation_split": evaluation_split,
            "n": n,
            "K": K,
            "alpha": alpha,
            "ablation_variant": ablation_variant,
            "selection_rule": selection_rule,
            "factor_applicability": dict(factor_applicability),
            "paper_evidence_eligible": paper_evidence_eligible,
            "config_override": (
                None if config_override is None else dict(config_override)
            ),
            "base_config_canonical_sha256": base_config_canonical_sha256,
            "expected_effective_config_sha256": expected_effective_config_sha256,
        }
    )


def _factor_applicability(method_id: str, *, ablation: bool = False) -> dict[str, bool]:
    return {
        "n": True,
        "K": method_id != "matched_ppo",
        "alpha": method_id != "matched_ppo",
        "ablation": ablation,
    }


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
    override: dict[str, object] = {"inner_unroll_n": int(n)}
    if method_id != "matched_ppo":
        override.update({"K": int(K), "mixture_alpha": float(alpha)})
    return {
        "base_method_id": method_id,
        "config_override": override,
        "base_config_canonical_sha256": hashlib.sha256(
            canonical_json_bytes(base)
        ).hexdigest(),
        "expected_effective_config_sha256": effective_config_sha256(base, override),
    }


def _generate_base_registry(
    protocol_value: object,
    base_configs: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Return the complete registered plan without executing a run."""

    protocol = validate_protocol(protocol_value)
    protocol_id = str(protocol["protocol_id"])
    namespace = str(protocol["seeds"]["namespace"])
    expected_seeds = {
        "smoke": [derive_seed(namespace, "smoke", 0)],
        "pilot": [derive_seed(namespace, "pilot", index) for index in range(3)],
        "confirmatory": [
            derive_seed(namespace, "confirmatory", index) for index in range(8)
        ],
    }
    for tier, expected in expected_seeds.items():
        if protocol["seeds"][tier] != expected:
            raise PolicyImprovementSchemaError(
                f"Registered {tier} seeds differ from their deterministic derivation."
            )

    grid = protocol["grid"]
    rows: list[dict[str, Any]] = []

    smoke_grid = grid["stage0"]
    smoke_seed = expected_seeds["smoke"][0]
    for method_id in METHOD_IDS:
        config_fields = _concrete_config_fields(
            method_id,
            n=int(smoke_grid["n"]),
            K=int(smoke_grid["K"]),
            alpha=float(smoke_grid["alpha"]),
            protocol=protocol,
            base_configs=base_configs,
        )
        rows.append(
            _row(
                protocol_id=protocol_id,
                row_kind="concrete",
                phase="stage0_smoke",
                tier="smoke",
                run_id=f"s0-{method_id}-s{smoke_seed}",
                method_id=method_id,
                seed=smoke_seed,
                evaluation_split="validation",
                n=int(smoke_grid["n"]),
                K=int(smoke_grid["K"]),
                alpha=float(smoke_grid["alpha"]),
                ablation_variant=None,
                selection_rule=None,
                factor_applicability=_factor_applicability(method_id),
                paper_evidence_eligible=False,
                **config_fields,
            )
        )

    screen = grid["stage1_screen"]
    for method_id in METHOD_IDS:
        for n in screen["n_values"]:
            for K in screen["K_values"]:
                for seed in expected_seeds["pilot"]:
                    config_fields = _concrete_config_fields(
                        method_id,
                        n=int(n),
                        K=int(K),
                        alpha=float(screen["alpha"]),
                        protocol=protocol,
                        base_configs=base_configs,
                    )
                    rows.append(
                        _row(
                            protocol_id=protocol_id,
                            row_kind="concrete",
                            phase="stage1_screen",
                            tier="pilot",
                            run_id=f"s1-{method_id}-n{n}-k{K}-s{seed}",
                            method_id=method_id,
                            seed=seed,
                            evaluation_split="validation",
                            n=int(n),
                            K=int(K),
                            alpha=float(screen["alpha"]),
                            ablation_variant=None,
                            selection_rule=None,
                            factor_applicability=_factor_applicability(method_id),
                            paper_evidence_eligible=False,
                            **config_fields,
                        )
                    )

    alpha_grid = grid["stage1_alpha"]
    alpha_rule = str(alpha_grid["selection_rule"])
    for alpha_index, alpha in enumerate(alpha_grid["alpha_values"]):
        for seed in expected_seeds["pilot"]:
            rows.append(
                _row(
                    protocol_id=protocol_id,
                    row_kind="selection_template",
                    phase="stage1_alpha",
                    tier="pilot",
                    run_id=f"s1a-a{alpha_index}-s{seed}",
                    method_id=None,
                    base_method_id=None,
                    seed=seed,
                    evaluation_split="validation",
                    n=None,
                    K=None,
                    alpha=float(alpha),
                    ablation_variant=None,
                    selection_rule=alpha_rule,
                    factor_applicability={
                        "n": True,
                        "K": True,
                        "alpha": True,
                        "ablation": False,
                    },
                    paper_evidence_eligible=False,
                )
            )

    stage2 = grid["stage2"]
    stage2_rule = str(stage2["selection_rule"])
    for method_id in METHOD_IDS:
        for seed in expected_seeds["confirmatory"]:
            rows.append(
                _row(
                    protocol_id=protocol_id,
                    row_kind="selection_template",
                    phase="stage2_confirmatory",
                    tier="confirmatory",
                    run_id=f"s2-{method_id}-s{seed}",
                    method_id=method_id,
                    base_method_id=None,
                    seed=seed,
                    evaluation_split="test",
                    n=None,
                    K=None,
                    alpha=None,
                    ablation_variant=None,
                    selection_rule=stage2_rule,
                    factor_applicability=_factor_applicability(method_id),
                    paper_evidence_eligible=True,
                )
            )

    stage3 = grid["stage3"]
    stage3_rule = str(stage3["selection_rule"])
    for variant in stage3["training_variants"]:
        for seed in expected_seeds["confirmatory"]:
            rows.append(
                _row(
                    protocol_id=protocol_id,
                    row_kind="selection_template",
                    phase="stage3_ablation",
                    tier="ablation",
                    run_id=f"s3-{variant}-s{seed}",
                    method_id=None,
                    base_method_id=None,
                    seed=seed,
                    evaluation_split="test",
                    n=None,
                    K=None,
                    alpha=None,
                    ablation_variant=str(variant),
                    selection_rule=stage3_rule,
                    factor_applicability={
                        "n": True,
                        "K": True,
                        "alpha": True,
                        "ablation": True,
                    },
                    paper_evidence_eligible=True,
                )
            )

    run_ids = [str(row["run_id"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise PolicyImprovementSchemaError("Registry run IDs are not unique.")
    phase_counts = Counter(str(row["phase"]) for row in rows)
    expected_counts = {
        "stage0_smoke": 4,
        "stage1_screen": 48,
        "stage1_alpha": 9,
        "stage2_confirmatory": 32,
        "stage3_ablation": 64,
    }
    if dict(phase_counts) != expected_counts:
        raise PolicyImprovementSchemaError(
            f"Registry phase counts differ: {dict(phase_counts)!r}."
        )
    document: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "amendment_history_sha256": amendment_history_sha256([]),
        "counts": {
            "smoke_rows": 4,
            "full_rows": 153,
            "total_rows": 157,
            **expected_counts,
        },
        "rows": rows,
    }
    canonical_json_bytes(document)
    return document


def _alpha_token(value: float) -> str:
    token = format(value, ".15g")
    if "e" in token or "E" in token or token.startswith("-"):
        raise PolicyImprovementSchemaError("Alpha cannot be encoded canonically.")
    return token.replace(".", "p")


def _materialize_screen_selection(
    registry: Mapping[str, Any],
    selection: Mapping[str, Any],
    protocol: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    selected = selection["selected_exact"]
    rows: list[dict[str, Any]] = []
    for raw in registry["rows"]:
        row = dict(raw)
        if row["phase"] == "stage1_alpha":
            seed = int(row["seed"])
            row["method_id"] = selected["method_id"]
            row["n"] = selected["n"]
            row["K"] = selected["K"]
            row["run_id"] = (
                f"s1a-{selected['method_id']}-n{selected['n']}-k{selected['K']}-"
                f"a{_alpha_token(float(row['alpha']))}-s{seed}"
            )
            row["row_kind"] = "concrete"
            row["selection_rule"] = None
            row["factor_applicability"] = _factor_applicability(str(row["method_id"]))
            row.update(
                _concrete_config_fields(
                    str(row["method_id"]),
                    n=int(row["n"]),
                    K=int(row["K"]),
                    alpha=float(row["alpha"]),
                    protocol=protocol,
                    base_configs=base_configs,
                )
            )
            row = validate_registry_row(row)
        rows.append(row)
    result = dict(registry)
    result["rows"] = rows
    return result


def _checked_base_config(
    method_id: str,
    protocol: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    if method_id not in base_configs:
        raise PolicyImprovementSchemaError(
            f"Missing canonical base config for {method_id!r}."
        )
    base = base_configs[method_id]
    digest = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
    registration = next(
        method for method in protocol["methods"] if method["id"] == method_id
    )
    if digest != registration["canonical_config_sha256"]:
        raise PolicyImprovementSchemaError(
            f"Canonical base config for {method_id!r} differs from the protocol."
        )
    return base


def _materialize_final_selection(
    registry: Mapping[str, Any],
    selection: Mapping[str, Any],
    protocol: Mapping[str, Any],
    base_configs: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    selected = selection["selected_exact"]
    variants = {item["variant"]: item for item in selection["stage3_variants"]}
    rows: list[dict[str, Any]] = []
    for raw in registry["rows"]:
        row = dict(raw)
        phase = str(row["phase"])
        seed = int(row["seed"])
        if phase == "stage2_confirmatory":
            row["n"] = selected["n"]
            row["K"] = selected["K"]
            row["alpha"] = selected["alpha"]
            row["run_id"] = (
                f"s2-{row['method_id']}-n{selected['n']}-k{selected['K']}-"
                f"a{_alpha_token(float(selected['alpha']))}-s{seed}"
            )
            row["row_kind"] = "concrete"
            row["selection_rule"] = None
            row["factor_applicability"] = _factor_applicability(str(row["method_id"]))
            row.update(
                _concrete_config_fields(
                    str(row["method_id"]),
                    n=int(row["n"]),
                    K=int(row["K"]),
                    alpha=float(row["alpha"]),
                    protocol=protocol,
                    base_configs=base_configs,
                )
            )
            row = validate_registry_row(row)
        elif phase == "stage3_ablation":
            variant = variants[str(row["ablation_variant"])]
            method_id = str(variant["method_id"])
            base_method_id = str(variant["base_method_id"])
            if method_id != stage3_method_id(str(variant["variant"]), base_method_id):
                raise PolicyImprovementSchemaError(
                    "Stage 3 selection method identity is inconsistent."
                )
            row["method_id"] = method_id
            row["base_method_id"] = base_method_id
            row["n"] = variant["n"]
            row["K"] = variant["K"]
            row["alpha"] = variant["alpha"]
            row["run_id"] = (
                f"s3-{variant['variant']}-{base_method_id}-n{variant['n']}-"
                f"k{variant['K']}-a{_alpha_token(float(variant['alpha']))}-s{seed}"
            )
            row["row_kind"] = "concrete"
            row["selection_rule"] = None
            row["factor_applicability"] = _factor_applicability(
                method_id, ablation=True
            )
            full_override = {
                "inner_unroll_n": int(variant["n"]),
                "K": int(variant["K"]),
                "mixture_alpha": float(variant["alpha"]),
                **dict(variant["override_payload"]),
            }
            base = _checked_base_config(base_method_id, protocol, base_configs)
            row["config_override"] = full_override
            row["base_config_canonical_sha256"] = hashlib.sha256(
                canonical_json_bytes(base)
            ).hexdigest()
            row["expected_effective_config_sha256"] = effective_config_sha256(
                base, full_override
            )
            row = validate_registry_row(row)
        rows.append(row)
    result = dict(registry)
    result["rows"] = rows
    return result


def generate_registry(
    protocol_value: object,
    amendment_history: Sequence[object] | None = None,
    *,
    base_configs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Generate the exact registry for one immutable amendment-history prefix."""

    protocol = validate_protocol(protocol_value)
    if base_configs is None:
        base_configs = load_registered_base_configs(
            protocol,
            Path(__file__).resolve().parents[1],
        )
    history = validate_amendment_history(
        [] if amendment_history is None else amendment_history,
        protocol=protocol,
    )
    registry = _generate_base_registry(protocol, base_configs)
    applied: list[dict[str, Any]] = []
    for index, amendment in enumerate(history):
        if amendment["source_registry_sha256"] != registry_sha256(registry):
            raise PolicyImprovementSchemaError(
                "Amendment does not bind the exact prior registry generation."
            )
        if index == 1:
            registry = _materialize_screen_selection(
                registry, amendment, protocol, base_configs
            )
        elif index == 2:
            registry = _materialize_final_selection(
                registry, amendment, protocol, base_configs
            )
        applied.append(amendment)
        registry["amendment_history_sha256"] = amendment_history_sha256(applied)
    run_ids = [str(row["run_id"]) for row in registry["rows"]]
    if len(run_ids) != len(set(run_ids)):
        raise PolicyImprovementSchemaError(
            "Materialized registry run IDs are not unique."
        )
    canonical_json_bytes(registry)
    return registry


def validate_registry_document(
    value: object,
    protocol_value: object,
    amendment_history: Sequence[object] | None = None,
    *,
    base_configs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Reject any registry that is not exact regeneration output."""

    if not isinstance(value, Mapping):
        raise PolicyImprovementSchemaError("Registry document must be an object.")
    expected = generate_registry(
        protocol_value, amendment_history, base_configs=base_configs
    )
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise PolicyImprovementSchemaError(
            "Registry bytes do not match deterministic regeneration."
        )
    return expected


def registry_sha256(registry: object) -> str:
    return hashlib.sha256(canonical_json_bytes(registry)).hexdigest()


_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_FLOAT = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]*)\.[0-9]+|(?:0|[1-9][0-9]*)[eE][+-]?[0-9]+)$"
)


def load_flat_registered_yaml(path: str | Path) -> dict[str, object]:
    """Parse the deliberately flat registered method configs without imports."""

    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyImprovementSchemaError(
            "Registered method config could not be read as ASCII."
        ) from exc
    result: dict[str, object] = {}
    for line_number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if raw[:1].isspace() or text.count(":") != 1:
            raise PolicyImprovementSchemaError(
                f"Registered config line {line_number} is not flat canonical YAML."
            )
        key, scalar = (part.strip() for part in text.split(":", 1))
        if not key or key in result:
            raise PolicyImprovementSchemaError(
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
            raise PolicyImprovementSchemaError(
                f"Registered config line {line_number} uses unsupported YAML."
            )
        else:
            value = scalar
        result[key] = value
    if not result:
        raise PolicyImprovementSchemaError("Registered method config is empty.")
    canonical_json_bytes(result)
    return result


def load_registered_base_configs(
    protocol_value: object, project_root: str | Path
) -> dict[str, Mapping[str, object]]:
    protocol = validate_protocol(protocol_value)
    root = Path(project_root).resolve(strict=True)
    configs: dict[str, Mapping[str, object]] = {}
    for method in protocol["methods"]:
        path = (root / str(method["config_path"])).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PolicyImprovementSchemaError(
                "Registered method config escaped the project root."
            ) from exc
        raw_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_digest != method["config_sha256"]:
            raise PolicyImprovementSchemaError(
                f"Raw config bytes for {method['id']!r} differ from the protocol."
            )
        parsed = load_flat_registered_yaml(path)
        canonical_digest = hashlib.sha256(canonical_json_bytes(parsed)).hexdigest()
        if canonical_digest != method["canonical_config_sha256"]:
            raise PolicyImprovementSchemaError(
                f"Canonical config for {method['id']!r} differs from the protocol."
            )
        configs[str(method["id"])] = parsed
    return configs


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
    parser.add_argument("--amendment", action="append", default=[])
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    protocol = load_strict_json(arguments.protocol)
    amendments = [load_strict_json(path) for path in arguments.amendment]
    base_configs = load_registered_base_configs(protocol, arguments.project_root)
    registry = generate_registry(protocol, amendments, base_configs=base_configs)
    payload = canonical_json_bytes(registry)
    if arguments.output is None:
        print(payload.decode("ascii"))
    else:
        _write_new(Path(arguments.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
