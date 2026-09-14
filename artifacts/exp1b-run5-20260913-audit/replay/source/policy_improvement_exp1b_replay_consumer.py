#!/usr/bin/env fbpython
"""Independent consumer for Experiment 1B per-state replay exports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA_NAME: str = "policy_improvement_exp1b_replay_recomputation_v1"
REPLAY_SCHEMA_NAME: str = "policy_improvement_exp1b_per_state_replay_v1"
FLOAT_ENCODING: str = "float_hex_v1"
TENSOR_ENCODING: str = "tensor_bytes_base64_v1"
CONSUMER_TARGET: str = "fbcode//buiksat_trm:policy_improvement_exp1b_replay_consumer"
BUILD_MODE: str = "@fbcode//mode/opt"
SEEDS: tuple[int, ...] = (
    2081976412,
    781025396,
    1148619853,
    913535362,
    2143253519,
    2279379379,
    402312153,
    2736405725,
)
GAMMA: float = 0.99
ALPHA: float = 0.1
DEPTH_N: int = 2
DEPTH_M: int = 8
STATE_COUNT: int = 128
BOOTSTRAP_SCHEME: str = "exp1b_paired_seed_counter_sha256_v1"
BOOTSTRAP_NAMESPACE: str = "upi-trm-exp1b-seed-bootstrap-v1"
BOOTSTRAP_SEED: int = 3246702300714487323
REPLICATES: int = 10000
LOWER_PROBABILITY: float = 0.025
UPPER_PROBABILITY: float = 0.975


class ReplayMismatch(RuntimeError):
    """The first mismatch under the frozen traversal and comparison rules."""


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayMismatch(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(
        path.read_text(encoding="ascii"), object_pairs_hook=_pairs_no_duplicates
    )
    if not isinstance(parsed, dict):
        raise ReplayMismatch(f"{path} is not a JSON object")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _float(value: object, path: str) -> float:
    if not isinstance(value, str):
        raise ReplayMismatch(f"{path}: expected canonical binary64 hex text")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ReplayMismatch(f"{path}: invalid binary64 hex text") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ReplayMismatch(f"{path}: nonfinite or noncanonical binary64 text")
    return result


def _published_float(value: object, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
    ):
        raise ReplayMismatch(f"{path}: expected a finite published binary64 float")
    return value


def _bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _ordered_bits(value: float) -> int:
    bits = _bits(value)
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_bits(left) - _ordered_bits(right))


def _difference(left: float, right: float) -> dict[str, object]:
    absolute = abs(left - right)
    scale = max(abs(left), abs(right))
    relative = 0.0 if scale == 0.0 else absolute / scale
    return {
        "absolute": absolute,
        "relative": relative,
        "ulp": _ulp_distance(left, right),
    }


def _require(condition: bool, path: str, detail: str) -> None:
    if not condition:
        raise ReplayMismatch(f"{path}: {detail}")


def _same_float(left: float, right: float, path: str) -> None:
    _require(_bits(left) == _bits(right), path, f"{left.hex()} != {right.hex()}")


@dataclass
class DifferenceMaximum:
    absolute: float = -1.0
    absolute_path: str = ""
    relative: float = -1.0
    relative_path: str = ""
    ulp: int = -1
    ulp_path: str = ""
    comparisons: int = 0

    def observe(self, left: float, right: float, path: str) -> None:
        difference = _difference(left, right)
        self.comparisons += 1
        if difference["absolute"] > self.absolute:
            self.absolute = float(difference["absolute"])
            self.absolute_path = path
        if difference["relative"] > self.relative:
            self.relative = float(difference["relative"])
            self.relative_path = path
        if difference["ulp"] > self.ulp:
            self.ulp = int(difference["ulp"])
            self.ulp_path = path

    def document(self) -> dict[str, object]:
        return {
            "comparisons": self.comparisons,
            "maximum_absolute_difference": self.absolute,
            "maximum_absolute_difference_path": self.absolute_path,
            "maximum_relative_difference": self.relative,
            "maximum_relative_difference_path": self.relative_path,
            "maximum_ulp_difference": self.ulp,
            "maximum_ulp_difference_path": self.ulp_path,
        }


def _decode_identity(value: object) -> object:
    if isinstance(value, str) and (value.startswith("0x") or value.startswith("-0x")):
        try:
            decoded = float.fromhex(value)
        except ValueError:
            return value
        if decoded.hex() == value:
            return decoded
    if isinstance(value, dict):
        return {key: _decode_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_identity(item) for item in value]
    return value


def _verify_tensor_tree(identity: object, raw: object, path: str) -> None:
    if isinstance(identity, dict) and set(identity) == {"dtype", "shape", "sha256"}:
        _require(isinstance(raw, dict), path, "tensor raw value is not an object")
        _require(raw.get("kind") == "tensor", path, "raw value is not a tensor")
        _require(
            raw.get("encoding") == TENSOR_ENCODING, path, "tensor encoding differs"
        )
        _require(raw.get("dtype") == identity["dtype"], path, "tensor dtype differs")
        _require(raw.get("shape") == identity["shape"], path, "tensor shape differs")
        encoded = raw.get("data_base64")
        _require(isinstance(encoded, str), path, "tensor bytes are missing")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ReplayMismatch(
                f"{path}: tensor bytes are not canonical base64"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        _require(
            digest == identity["sha256"] == raw.get("sha256"),
            path,
            "tensor digest differs",
        )
        return
    if isinstance(identity, dict) and set(identity) == {"dataclass", "fields"}:
        _require(isinstance(raw, dict), path, "dataclass raw value is not an object")
        _verify_tensor_tree(identity["fields"], raw, path)
        return
    if isinstance(identity, dict):
        _require(
            isinstance(raw, dict) and set(raw) == set(identity),
            path,
            "mapping shape differs",
        )
        for key in sorted(identity):
            _verify_tensor_tree(identity[key], raw[key], f"{path}.{key}")
        return
    if isinstance(identity, list):
        _require(
            isinstance(raw, list) and len(raw) == len(identity),
            path,
            "sequence shape differs",
        )
        for index, (item, other) in enumerate(zip(identity, raw)):
            _verify_tensor_tree(item, other, f"{path}[{index}]")
        return


def _exact_mean(values: list[float]) -> float:
    return float(sum((Fraction(value) for value in values), Fraction(0)) / len(values))


def _exact_interpolate(low: float, high: float, weight: float) -> float:
    low_n, low_d = low.as_integer_ratio()
    high_n, high_d = high.as_integer_ratio()
    weight_n, weight_d = weight.as_integer_ratio()
    numerator = low_n * (weight_d - weight_n) * high_d + high_n * weight_n * low_d
    denominator = low_d * high_d * weight_d
    return numerator / denominator


def _type7(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return _exact_interpolate(ordered[lower], ordered[upper], position - lower)


def _bootstrap(gaps: list[float]) -> dict[str, object]:
    prefix = (
        BOOTSTRAP_NAMESPACE.encode("ascii") + b"\0" + BOOTSTRAP_SEED.to_bytes(8, "big")
    )
    plan_digest = hashlib.sha256()
    plan_digest.update(f"{BOOTSTRAP_SCHEME}:plan\n".encode("ascii"))
    replicate_means = []
    for replicate in range(REPLICATES):
        replicate_field = replicate.to_bytes(4, "big")
        row = []
        for draw in range(len(SEEDS)):
            digest = hashlib.sha256(
                prefix + replicate_field + draw.to_bytes(1, "big")
            ).digest()
            row.append(digest[0] & 0b111)
        plan_digest.update(bytes(row))
        plan_digest.update(b"\n")
        replicate_means.append(_exact_mean([gaps[index] for index in row]))
    vector_digest = hashlib.sha256()
    vector_digest.update(f"{BOOTSTRAP_SCHEME}\n".encode("ascii"))
    vector_hex = []
    for value in replicate_means:
        encoded = value.hex()
        vector_hex.append(encoded)
        vector_digest.update(f"{encoded}\n".encode("ascii"))
    ordered = sorted(replicate_means)
    return {
        "observed_mean": _exact_mean(gaps),
        "interval_lower": _type7(ordered, LOWER_PROBABILITY),
        "interval_upper": _type7(ordered, UPPER_PROBABILITY),
        "replicate_means_hex": vector_hex,
        "replicate_vector_sha256": vector_digest.hexdigest(),
        "resample_plan_sha256": plan_digest.hexdigest(),
    }


def _max_first(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    best = rows[0]
    for row in rows[1:]:
        if float(row[field]) > float(best[field]):
            best = row
    return {
        "value": float(best[field]),
        "state_id": best["state_id"],
        "record_index": best["record_index"],
    }


def _secondary_value(block: dict[str, Any], name: str, path: str) -> float:
    return _float(block[name], f"{path}.{name}")


def _audit_seed(
    document: dict[str, Any],
    payload: dict[str, Any],
    result_row: dict[str, Any],
    position: int,
) -> tuple[float, dict[str, Any]]:
    path = f"seed[{position}]"
    _require(
        document.get("schema_name") == REPLAY_SCHEMA_NAME, path, "replay schema differs"
    )
    _require(
        document.get("float_encoding") == FLOAT_ENCODING, path, "float encoding differs"
    )
    replay = document.get("replay")
    _require(isinstance(replay, dict), path, "missing replay object")
    _require(replay.get("seed_position") == position, path, "seed position differs")
    _require(replay.get("seed") == SEEDS[position], path, "registered seed differs")
    _require(replay.get("state_count") == STATE_COUNT, path, "state count differs")
    before = replay.get("model_state_identities_before")
    after = replay.get("model_state_identities_after")
    _require(
        before == after and replay.get("model_state_unchanged") is True,
        path,
        "model state changed",
    )
    _require(
        _canonical_json_sha256(before) == replay.get("checkpoint_model_state_sha256"),
        path,
        "folded model-state identity differs from checkpoint",
    )

    states = replay.get("states")
    _require(
        isinstance(states, list) and len(states) == STATE_COUNT,
        path,
        "state inventory differs",
    )
    exact_differences = DifferenceMaximum()
    mixture_differences = DifferenceMaximum()
    advantage_differences = DifferenceMaximum()
    derived_rows = []
    target_rows = []
    centering_rows = []
    identity_rows = []
    mismatch_rows = []
    mass_rows = []
    latent_digests = []
    law_digests = []
    seen_states = set()
    seen_record_indices = set()

    for offset, state in enumerate(states):
        state_path = f"{path}.states[{offset}]"
        record_index = state.get("record_index")
        _require(
            isinstance(record_index, int)
            and not isinstance(record_index, bool)
            and record_index >= 0
            and record_index not in seen_record_indices,
            state_path,
            "record index is invalid or repeated",
        )
        seen_record_indices.add(record_index)
        state_id = state.get("state_id")
        _require(
            isinstance(state_id, str) and state_id not in seen_states,
            state_path,
            "state identity repeats",
        )
        seen_states.add(state_id)
        mask = state.get("action_mask")
        _require(
            isinstance(mask, list)
            and any(mask)
            and all(isinstance(v, bool) for v in mask),
            state_path,
            "action mask differs",
        )
        base = [
            _float(value, f"{state_path}.base_probabilities")
            for value in state["base_probabilities"]
        ]
        candidate = [
            _float(value, f"{state_path}.candidate_probabilities")
            for value in state["candidate_probabilities"]
        ]
        deployed = [
            _float(value, f"{state_path}.deployed_probabilities")
            for value in state["deployed_probabilities"]
        ]
        action_n = [
            _float(value, f"{state_path}.action_values_n")
            for value in state["action_values_n"]
        ]
        action_m = [
            _float(value, f"{state_path}.action_values_m")
            for value in state["action_values_m"]
        ]
        _require(
            len(mask)
            == len(base)
            == len(candidate)
            == len(deployed)
            == len(action_n)
            == len(action_m),
            state_path,
            "action vector lengths differ",
        )
        endpoint_n = _float(state["endpoint_value_n"], f"{state_path}.endpoint_value_n")
        endpoint_m = _float(state["endpoint_value_m"], f"{state_path}.endpoint_value_m")
        operator_n = sum(
            p * q for p, q, allowed in zip(base, action_n, mask) if allowed
        )
        operator_m = sum(
            p * q for p, q, allowed in zip(base, action_m, mask) if allowed
        )
        residual_n = endpoint_n - operator_n
        residual_m = endpoint_m - operator_m
        discrepancy = abs(endpoint_n - endpoint_m)
        diagnostic = state["diagnostic"]
        for name, computed in (
            ("endpoint_value_n", endpoint_n),
            ("endpoint_value_m", endpoint_m),
            ("base_operator_n", operator_n),
            ("base_operator_m", operator_m),
            ("signed_residual_n", residual_n),
            ("signed_residual_m", residual_m),
            ("absolute_residual_n", abs(residual_n)),
            ("absolute_residual_m", abs(residual_m)),
            ("endpoint_discrepancy", discrepancy),
        ):
            recorded = _float(diagnostic[name], f"{state_path}.diagnostic.{name}")
            exact_differences.observe(computed, recorded, f"{state_path}.{name}")
            _same_float(computed, recorded, f"{state_path}.{name}")
        derived_rows.append(
            {
                "state_id": state_id,
                "record_index": record_index,
                "absolute_residual_n": abs(residual_n),
                "absolute_residual_m": abs(residual_m),
                "endpoint_discrepancy": discrepancy,
            }
        )

        reconstructed = [
            (1.0 - ALPHA) * left + ALPHA * right for left, right in zip(base, candidate)
        ]
        identity_tv = 0.5 * math.fsum(
            abs(left - right) for left, right in zip(reconstructed, deployed)
        )
        mismatch_tv = 0.5 * math.fsum(
            abs(left - right) for left, right in zip(candidate, base)
        )
        for index, (left, right) in enumerate(zip(reconstructed, deployed)):
            mixture_differences.observe(left, right, f"{state_path}.mixture[{index}]")

        clipping_kind = state.get("clipping_kind")
        _require(
            clipping_kind == "clip_then_exact_recenter",
            state_path,
            "clipping kind differs",
        )
        clip_value = _float(state["clip_value"], f"{state_path}.clip_value")
        constructed_baseline = math.fsum(
            p * q for p, q, allowed in zip(base, action_n, mask) if allowed
        )
        raw_advantages = [
            q - constructed_baseline if allowed else 0.0
            for q, allowed in zip(action_n, mask)
        ]
        clipped = [
            max(-clip_value, min(clip_value, value)) if allowed else 0.0
            for value, allowed in zip(raw_advantages, mask)
        ]
        clipped_mean = math.fsum(
            p * value for p, value, allowed in zip(base, clipped, mask) if allowed
        )
        constructed = [
            value - clipped_mean if allowed else 0.0
            for value, allowed in zip(clipped, mask)
        ]
        exported_constructed = [
            _float(value, f"{state_path}.constructed_advantages")
            for value in state["constructed_advantages"]
        ]
        trainer = [
            _float(value, f"{state_path}.trainer_advantages")
            for value in state["trainer_advantages"]
        ]
        for index, (computed, recorded) in enumerate(
            zip(constructed, exported_constructed)
        ):
            exact_differences.observe(
                computed, recorded, f"{state_path}.constructed_advantages[{index}]"
            )
            _same_float(
                computed, recorded, f"{state_path}.constructed_advantages[{index}]"
            )
        for index, (left, right) in enumerate(zip(constructed, trainer)):
            if mask[index]:
                advantage_differences.observe(
                    left, right, f"{state_path}.advantages[{index}]"
                )
        constructed_centering = abs(
            math.fsum(
                p * value
                for p, value, allowed in zip(base, constructed, mask)
                if allowed
            )
        )
        trainer_defect = abs(
            math.fsum(
                p * value for p, value, allowed in zip(base, trainer, mask) if allowed
            )
        )
        parity_error = max(
            (
                abs(left - right)
                for left, right, allowed in zip(constructed, trainer, mask)
                if allowed
            ),
            default=0.0,
        )
        target_value = _float(state["target_value_n"], f"{state_path}.target_value_n")
        target_lag = abs(endpoint_n - target_value)
        normalization_mass_error = _float(
            state["normalization_mass_error"], f"{state_path}.normalization_mass_error"
        )
        target_rows.append(
            {"state_id": state_id, "record_index": record_index, "value": target_lag}
        )
        centering_rows.append(
            {
                "state_id": state_id,
                "record_index": record_index,
                "constructed": constructed_centering,
                "defect": trainer_defect,
                "parity": parity_error,
            }
        )
        identity_rows.append(
            {"state_id": state_id, "record_index": record_index, "value": identity_tv}
        )
        mismatch_rows.append(
            {"state_id": state_id, "record_index": record_index, "value": mismatch_tv}
        )
        mass_rows.append(
            {
                "state_id": state_id,
                "record_index": record_index,
                "value": normalization_mass_error,
            }
        )

        for latent_name in (
            "initial_latent",
            "base_next_latent",
            "deployed_next_latent",
        ):
            latent = state[latent_name]
            _verify_tensor_tree(
                latent["identity"], latent["raw"], f"{state_path}.{latent_name}"
            )
        deployed_identity = _decode_identity(state["deployed_next_latent"]["identity"])
        latent_digests.append(_canonical_json_sha256(deployed_identity))
        law_digests.append(
            _canonical_json_sha256(
                {
                    "state_id": state_id,
                    "action_mask": mask,
                    "base": base,
                    "candidate": candidate,
                    "deployed": deployed,
                }
            )
        )

        actions = state.get("actions")
        _require(
            isinstance(actions, list) and len(actions) == len(mask),
            state_path,
            "transition inventory differs",
        )
        for action_index, action in enumerate(actions):
            action_path = f"{state_path}.actions[{action_index}]"
            _require(
                action.get("action_index") == action_index
                and action.get("allowed") is mask[action_index],
                action_path,
                "action identity differs",
            )
            _same_float(
                _float(action["action_value_n"], action_path),
                action_n[action_index],
                action_path + ".action_value_n",
            )
            _same_float(
                _float(action["action_value_m"], action_path),
                action_m[action_index],
                action_path + ".action_value_m",
            )
            if not mask[action_index]:
                _require(
                    action.get("successor_identity") is None,
                    action_path,
                    "masked action has a successor",
                )
                continue
            identity_input = _decode_identity(action["successor_identity_input"])
            _require(
                _canonical_json_sha256(identity_input) == action["successor_identity"],
                action_path,
                "successor identity differs",
            )
            _verify_tensor_tree(
                action["successor_identity_input"]["x"],
                action["successor_x"],
                action_path + ".successor_x",
            )
            _verify_tensor_tree(
                action["successor_identity_input"]["plan"],
                action["successor_plan"],
                action_path + ".successor_plan",
            )
            initial_clock = action["initial_clock"]
            successor_clock = action["successor_clock"]
            _require(
                successor_clock["step_count"] == initial_clock["step_count"] + 1,
                action_path,
                "step clock did not advance once",
            )
            if initial_clock["remaining_edits"] is not None:
                _require(
                    successor_clock["remaining_edits"]
                    == max(initial_clock["remaining_edits"] - 1, 0),
                    action_path,
                    "remaining-edits clock differs",
                )
            reward = _float(action["reward"], action_path + ".reward")
            if action["terminal"] is True:
                _same_float(
                    reward, action_n[action_index], action_path + ".terminal_q_n"
                )
                _same_float(
                    reward, action_m[action_index], action_path + ".terminal_q_m"
                )

    maximum_discrepancy = _max_first(derived_rows, "endpoint_discrepancy")
    maximum_n = _max_first(derived_rows, "absolute_residual_n")
    maximum_m = _max_first(derived_rows, "absolute_residual_m")
    denominator = 1.0 - GAMMA
    direct = float(maximum_n["value"]) / denominator
    finite_reference = (
        float(maximum_discrepancy["value"]) + float(maximum_m["value"]) / denominator
    )
    gap = direct - finite_reference
    recomputed = {
        "state_count": STATE_COUNT,
        "maximum_endpoint_discrepancy": float(maximum_discrepancy["value"]),
        "maximum_absolute_residual_n": float(maximum_n["value"]),
        "maximum_absolute_residual_m": float(maximum_m["value"]),
        "direct_residual_bound_n": direct,
        "finite_reference_bound": finite_reference,
        "signed_gap": gap,
        "discrepancy_witness_state_id": maximum_discrepancy["state_id"],
        "residual_n_witness_state_id": maximum_n["state_id"],
        "residual_m_witness_state_id": maximum_m["state_id"],
    }
    producer_summary = replay["producer_summary"]
    for name, value in recomputed.items():
        if isinstance(value, float):
            recorded = _float(producer_summary[name], f"{path}.producer_summary.{name}")
            exact_differences.observe(value, recorded, f"{path}.summary.{name}")
            _same_float(value, recorded, f"{path}.producer_summary.{name}")
            _same_float(
                value,
                _published_float(payload[name], f"payload[{position}].{name}"),
                f"payload[{position}].{name}",
            )
            _same_float(
                value,
                _published_float(
                    result_row[name], f"result.seed_rows[{position}].{name}"
                ),
                f"result.seed_rows[{position}].{name}",
            )
        else:
            _require(
                producer_summary[name] == value,
                f"{path}.producer_summary.{name}",
                "value differs",
            )
            _require(
                payload[name] == value, f"payload[{position}].{name}", "value differs"
            )
            _require(
                result_row[name] == value,
                f"result.seed_rows[{position}].{name}",
                "value differs",
            )

    secondary = replay["secondary_diagnostics"]
    target_max = _max_first(target_rows, "value")
    identity_max = _max_first(identity_rows, "value")
    mismatch_max = _max_first(mismatch_rows, "value")
    mass_max = _max_first(mass_rows, "value")
    constructed_max = _max_first(
        [
            {
                "state_id": row["state_id"],
                "record_index": row["record_index"],
                "value": row["constructed"],
            }
            for row in centering_rows
        ],
        "value",
    )
    defect_max = _max_first(
        [
            {
                "state_id": row["state_id"],
                "record_index": row["record_index"],
                "value": row["defect"],
            }
            for row in centering_rows
        ],
        "value",
    )
    parity_max = _max_first(
        [
            {
                "state_id": row["state_id"],
                "record_index": row["record_index"],
                "value": row["parity"],
            }
            for row in centering_rows
        ],
        "value",
    )
    secondary_checks = (
        ("target_lag", "maximum_absolute_target_lag", target_max["value"]),
        (
            "centering_parity",
            "constructed_centering_roundoff",
            constructed_max["value"],
        ),
        (
            "centering_parity",
            "training_estimator_centering_defect",
            defect_max["value"],
        ),
        (
            "centering_parity",
            "training_estimator_parity_max_abs_error",
            parity_max["value"],
        ),
        ("mixture_identity", "maximum_identity_total_variation", identity_max["value"]),
        (
            "deployment_mismatch",
            "maximum_candidate_base_total_variation",
            mismatch_max["value"],
        ),
        ("deployment_mismatch", "maximum_normalization_mass_error", mass_max["value"]),
    )
    for block_name, field_name, value in secondary_checks:
        recorded = _secondary_value(
            secondary[block_name], field_name, f"{path}.secondary.{block_name}"
        )
        exact_differences.observe(
            float(value), recorded, f"{path}.secondary.{block_name}.{field_name}"
        )
        _same_float(
            float(value), recorded, f"{path}.secondary.{block_name}.{field_name}"
        )
    for block_name, field_name, maximum in (
        ("target_lag", "target_lag_witness_state_id", target_max),
        ("centering_parity", "centering_witness_state_id", constructed_max),
        ("centering_parity", "centering_defect_witness_state_id", defect_max),
        ("centering_parity", "parity_witness_state_id", parity_max),
        ("mixture_identity", "identity_witness_state_id", identity_max),
        ("deployment_mismatch", "mismatch_witness_state_id", mismatch_max),
    ):
        _require(
            secondary[block_name][field_name] == maximum["state_id"],
            f"{path}.secondary.{block_name}.{field_name}",
            "witness differs",
        )
    _require(
        secondary["persistent_state"]["carried_successor_latent_sha256"]
        == _canonical_json_sha256(latent_digests),
        f"{path}.secondary.persistent_state",
        "latent identity digest differs",
    )
    _require(
        secondary["persistent_state"]["action_probabilities_sha256"]
        == _canonical_json_sha256(law_digests),
        f"{path}.secondary.persistent_state",
        "policy identity digest differs",
    )

    return gap, {
        "seed_position": position,
        "seed": SEEDS[position],
        "published_gap": _published_float(
            payload["signed_gap"], f"payload[{position}].signed_gap"
        ),
        "recomputed_gap": gap,
        "gap_difference": _difference(
            gap,
            _published_float(payload["signed_gap"], f"payload[{position}].signed_gap"),
        ),
        "maxima": recomputed,
        "exact_recomputation_differences": exact_differences.document(),
        "raw_mixture_differences": mixture_differences.document(),
        "raw_trainer_constructed_advantage_differences": advantage_differences.document(),
    }


def _lossless_report(value: object) -> object:
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {key: _lossless_report(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_lossless_report(item) for item in value]
    return value


def _install(path: Path, document: dict[str, Any]) -> str:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise ReplayMismatch(f"consumer output already exists: {path}")
    payload = _canonical_json_bytes(_lossless_report(document)) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    with path.with_suffix(path.suffix + ".sha256").open(
        "x", encoding="ascii"
    ) as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--published-result", type=Path, required=True)
    parser.add_argument("--payload-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-identity", type=Path, required=True)
    parser.add_argument("--binary-path", type=Path, required=True)
    arguments = parser.parse_args()
    build_identity = _read_json(arguments.build_identity)
    binary_sha256 = _file_sha256(arguments.binary_path)
    _require(
        build_identity["consumer"]["sha256"] == binary_sha256,
        "build.consumer.sha256",
        "binary differs",
    )
    _require(
        build_identity["consumer"]["target"] == CONSUMER_TARGET,
        "build.consumer.target",
        "target differs",
    )
    _require(
        build_identity["consumer"]["build_mode"] == BUILD_MODE,
        "build.consumer.build_mode",
        "mode differs",
    )

    result = _read_json(arguments.published_result)
    seed_rows = result.get("seed_rows")
    _require(
        isinstance(seed_rows, list) and len(seed_rows) == len(SEEDS),
        "result.seed_rows",
        "inventory differs",
    )
    gaps = []
    reports = []
    producer_build_identities = []
    residual_witnesses = []
    for position in range(len(SEEDS)):
        replay_path = arguments.replay_output / f"seed-{position}.json"
        sidecar = replay_path.with_suffix(replay_path.suffix + ".sha256")
        _require(
            sidecar.read_text(encoding="ascii").strip() == _file_sha256(replay_path),
            f"seed[{position}].sha256",
            "sidecar differs",
        )
        replay_document = _read_json(replay_path)
        producer_build_identities.append(replay_document["build_identity_sha256"])
        payload = _read_json(arguments.payload_directory / f"seed-{position}.json")
        gap, report = _audit_seed(
            replay_document, payload, seed_rows[position], position
        )
        gaps.append(gap)
        reports.append(report)
        residual_witnesses.append(
            (
                report["maxima"]["residual_n_witness_state_id"],
                report["maxima"]["residual_m_witness_state_id"],
            )
        )
        _same_float(
            gap,
            _published_float(
                result["signed_gaps"][position], f"result.signed_gaps[{position}]"
            ),
            f"result.signed_gaps[{position}]",
        )
    _require(
        len(set(producer_build_identities)) == 1,
        "build.producer",
        "seed exports used different producer builds",
    )

    mean = _exact_mean(gaps)
    published_mean = _published_float(
        result["mean_signed_gap"], "result.mean_signed_gap"
    )
    _same_float(mean, published_mean, "result.mean_signed_gap")
    bootstrap = _bootstrap(gaps)
    interval = result["interval"]
    _require(
        bootstrap["replicate_means_hex"] == interval["replicate_means_hex"],
        "result.interval.replicate_means_hex",
        "vector differs",
    )
    for name in ("replicate_vector_sha256", "resample_plan_sha256"):
        _require(
            bootstrap[name] == interval[name],
            f"result.interval.{name}",
            "digest differs",
        )
    for name in ("interval_lower", "interval_upper"):
        _same_float(
            float(bootstrap[name]),
            _published_float(interval[name], f"result.interval.{name}"),
            f"result.interval.{name}",
        )
    _require(
        len({item for pair in residual_witnesses for item in pair}) == 1,
        "shared_residual_witness",
        "residual-maximizing state is not shared across all seeds and depths",
    )
    shared_witness = residual_witnesses[0][0]
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "status": "agree",
        "first_divergent_quantity": None,
        "comparison_rules": "comparison-rules.md",
        "consumer_build_identity": build_identity["consumer"],
        "producer_build_identity_sha256": producer_build_identities[0],
        "seeds": reports,
        "mean_signed_gap": {
            "published": published_mean,
            "recomputed": mean,
            "difference": _difference(mean, published_mean),
        },
        "frozen_bootstrap_interval": {
            "published_lower": interval["interval_lower"],
            "recomputed_lower": bootstrap["interval_lower"],
            "published_upper": interval["interval_upper"],
            "recomputed_upper": bootstrap["interval_upper"],
            "replicate_vector_sha256": bootstrap["replicate_vector_sha256"],
            "resample_plan_sha256": bootstrap["resample_plan_sha256"],
        },
        "shared_residual_maximizing_state": {
            "state_id": shared_witness,
            "verified_for_all_seeds_and_both_depths": True,
        },
    }
    digest = _install(arguments.output.resolve(), report)
    print(f"exp1b-replay-consumer: AGREES mean={mean!r} sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReplayMismatch, ValueError, KeyError, TypeError) as exc:
        print(f"exp1b-replay-consumer: DIVERGED: {exc}", file=sys.stderr)
        raise SystemExit(2)
