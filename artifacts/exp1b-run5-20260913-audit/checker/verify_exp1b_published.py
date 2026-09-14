#!/usr/bin/env python3
"""Independently verify the published Experiment 1B run5 audit bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


SEED_COUNT = 8
STATE_COUNT = 128
EXPECTED_GAMMA = 0.99
EXPECTED_HORIZON = 1
EXPECTED_DEPTH_N = 2
EXPECTED_DEPTH_M = 8
EXPECTED_BOOTSTRAP_REPLICATES = 10_000
EXPECTED_LOWER_PROBABILITY = 0.025
EXPECTED_UPPER_PROBABILITY = 0.975
EXPECTED_PARITY_ABSOLUTE_TOLERANCE = 1e-6
EXPECTED_PARITY_RELATIVE_TOLERANCE = 4 * 2**-23
EXPECTED_ADVANTAGE_CLIP = 10.0
EXPECTED_PARITY_BOUND = 5.76837158203125e-6
LOWER_HEX_64 = set("0123456789abcdef")
SUMMARY_NAME = "exp1b_independent_checks.json"


class VerificationError(RuntimeError):
    """Raised on the first failed verification check."""


# Vendored from utils/run_identity.py at source revision
# 2a14eb656bd983c37b60e92852083524348511d8. Keeping this encoder here makes
# the checker independent of the repository and its Python import graph.
def _canonical_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VerificationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise VerificationError(f"{path} has a non-string key")
        return {
            key: _canonical_json_value(value[key], path=f"{path}.{key}")
            for key in sorted(keys)
        }
    if isinstance(value, list):
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise VerificationError(
        f"{path} contains unsupported JSON value type {type(value).__name__}"
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict JSON using the run's content-addressing convention."""

    canonical = _canonical_json_value(value, path="value")
    return json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _reject_json_constant(value: str) -> object:
    raise VerificationError(f"JSON contains non-finite value {value}")


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise VerificationError(f"top-level JSON value in {path} is not an object or list")
    return value


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f"{path} is not an object")
    return value


def _array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise VerificationError(f"{path} is not an array")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{path} is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise VerificationError(f"{path} is not finite")
    return result


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"{path} is not an integer")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise VerificationError(f"{path} is not a string")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_digest(value: object, path: str) -> str:
    digest = _string(value, path)
    if len(digest) != 64 or any(character not in LOWER_HEX_64 for character in digest):
        raise VerificationError(f"{path} is not a lowercase SHA-256 digest")
    return digest


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _same_value(expected: object, observed: object) -> bool:
    if isinstance(expected, float) and isinstance(observed, float):
        return _float_bits(expected) == _float_bits(observed)
    if type(expected) is not type(observed):
        return False
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            return False
        return all(_same_value(expected[key], observed[key]) for key in expected)
    if isinstance(expected, list):
        return len(expected) == len(observed) and all(
            _same_value(left, right) for left, right in zip(expected, observed)
        )
    return expected == observed


def _lossless(value: object) -> object:
    if isinstance(value, float):
        return {"decimal": repr(value), "float_hex": value.hex()}
    if isinstance(value, dict):
        return {key: _lossless(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_lossless(item) for item in value]
    return value


def _display(value: object) -> str:
    return json.dumps(_lossless(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class ChainLink:
    link: str
    source: str
    convention: str
    expected: str
    observed: str

    def table_row(self) -> dict[str, str]:
        return {
            "link": self.link,
            "expected": self.expected,
            "observed": self.observed,
            "source": self.source,
            "convention": self.convention,
            "status": "MATCH" if self.expected == self.observed else "MISMATCH",
        }


class CheckRecorder:
    def __init__(self, summary_path: Path) -> None:
        self._summary_path = summary_path
        self._checks: list[dict[str, object]] = []

    def check(
        self,
        name: str,
        expected: object,
        observed: object,
        *,
        passed: bool | None = None,
    ) -> None:
        matches = _same_value(expected, observed) if passed is None else passed
        record = {
            "name": name,
            "status": "pass" if matches else "fail",
            "expected": _lossless(expected),
            "observed": _lossless(observed),
        }
        self._checks.append(record)
        print(
            f"{'PASS' if matches else 'FAIL'} {name}: "
            f"expected={_display(expected)} observed={_display(observed)}",
            flush=True,
        )
        if not matches:
            raise VerificationError(name)

    def add_unexpected_failure(self, message: str) -> None:
        if self._checks and self._checks[-1]["status"] == "fail":
            return
        self._checks.append(
            {
                "name": "checker execution",
                "status": "fail",
                "expected": "all inputs readable and well-formed",
                "observed": message,
            }
        )
        print(
            "FAIL checker execution: "
            f"expected={_display('all inputs readable and well-formed')} "
            f"observed={_display(message)}",
            flush=True,
        )

    def write_summary(self, status: str, first_failure: str | None) -> None:
        document = {
            "schema_name": "exp1b_independent_checks_v1",
            "schema_version": 1,
            "status": status,
            "first_failure": first_failure,
            "checks": self._checks,
        }
        payload = (
            json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = self._summary_path.with_name(
            f".{self._summary_path.name}.partial-{os.getpid()}"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._summary_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _type7(values: list[float], probability: float) -> float:
    if not values:
        raise VerificationError("bootstrap replicate vector is empty")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    fraction = coordinate - lower
    if lower == len(ordered) - 1:
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def _load_inputs(root: Path) -> dict[str, object]:
    result_dir = root / "result"
    study_dir = root / "study_documents" / "configs" / "policy_improvement_exp1b"
    signed_dir = root / "signed_chain"
    return {
        "result": _object(_read_json(result_dir / "exp1b_result.json"), "result"),
        "result_sidecar": (result_dir / "exp1b_result.sha256")
        .read_text(encoding="ascii")
        .strip(),
        "provenance": _object(_read_json(result_dir / "provenance.json"), "provenance"),
        "chain_table": _array(_read_json(root / "chain_table.json"), "chain_table"),
        "protocol": _object(_read_json(study_dir / "protocol.json"), "protocol"),
        "registry": _object(_read_json(study_dir / "registry.json"), "registry"),
        "amendment": _object(
            _read_json(study_dir / "amendments" / "reduced_study_exp1b.json"),
            "amendment",
        ),
        "base_policy_amendment": _object(
            _read_json(signed_dir / "base_policy_amendment_exp0_20260913_run5.json"),
            "base_policy_amendment",
        ),
        "admission": _object(
            _read_json(signed_dir / "exp1b_execution_admission_20260913_run5.json"),
            "admission",
        ),
        "authorization": _object(
            _read_json(signed_dir / "exp1b_runtime_authorization_20260913_run5.json"),
            "authorization",
        ),
        "manifests": [
            _object(
                _read_json(
                    root
                    / "recovered_stage_a_manifests"
                    / f"seed-{position}"
                    / "run_manifest.json"
                ),
                f"manifest[{position}]",
            )
            for position in range(SEED_COUNT)
        ],
        "payloads": [
            _object(
                _read_json(root / "stage_b_payloads" / f"seed-{position}.json"),
                f"payload[{position}]",
            )
            for position in range(SEED_COUNT)
        ],
    }


def _chain_links(root: Path, inputs: dict[str, object]) -> list[ChainLink]:
    result = _object(inputs["result"], "result")
    provenance = _object(inputs["provenance"], "provenance")
    protocol = _object(inputs["protocol"], "protocol")
    registry = _object(inputs["registry"], "registry")
    amendment = _object(inputs["amendment"], "amendment")
    base_amendment = _object(inputs["base_policy_amendment"], "base_policy_amendment")
    admission = _object(inputs["admission"], "admission")
    authorization = _object(inputs["authorization"], "authorization")
    manifests = _array(inputs["manifests"], "manifests")
    payloads = _array(inputs["payloads"], "payloads")

    authorization_digest = _canonical_sha256(authorization)
    base_amendment_digest = _canonical_sha256(base_amendment)
    admission_digest = _canonical_sha256(admission)
    provenance_digest = _canonical_sha256(provenance)
    result_digest = _canonical_sha256(result)
    full_authorization = _object(
        _object(
            _object(admission["runtime_authorization"], "admission.runtime_authorization")[
                "value"
            ],
            "admission.runtime_authorization.value",
        )["policy-improvement-full"],
        "admission.runtime_authorization.value.policy-improvement-full",
    )
    bridge_authorization = _object(
        _object(
            _object(admission["runtime_authorization"], "admission.runtime_authorization")[
                "value"
            ],
            "admission.runtime_authorization.value",
        )["policy-improvement-theory-bridge"],
        "admission.runtime_authorization.value.policy-improvement-theory-bridge",
    )
    admission_base = _object(
        _object(admission["base_policy_artifact"], "admission.base_policy_artifact")[
            "value"
        ],
        "admission.base_policy_artifact.value",
    )

    links = [
        ChainLink(
            "exp1b protocol",
            "configs/policy_improvement_exp1b/protocol.json",
            "canonical",
            _validate_digest(admission["protocol_sha256"], "admission.protocol_sha256"),
            _canonical_sha256(protocol),
        ),
        ChainLink(
            "exp1b registry",
            "configs/policy_improvement_exp1b/registry.json",
            "canonical",
            _validate_digest(admission["registry_sha256"], "admission.registry_sha256"),
            _canonical_sha256(registry),
        ),
        ChainLink(
            "exp1b amendment",
            "configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json",
            "canonical",
            _validate_digest(
                admission["prior_amendment_sha256"], "admission.prior_amendment_sha256"
            ),
            _canonical_sha256(amendment),
        ),
        ChainLink(
            "bp amendment -> auth",
            "signed/base_policy_amendment",
            "canonical",
            authorization_digest,
            _validate_digest(
                base_amendment["runtime_authorization_sha256"],
                "base_policy_amendment.runtime_authorization_sha256",
            ),
        ),
        ChainLink(
            "admission[full] -> auth",
            "signed/execution_admission",
            "canonical",
            authorization_digest,
            _validate_digest(
                full_authorization["authorization_sha256"],
                "admission.runtime_authorization.value.policy-improvement-full.authorization_sha256",
            ),
        ),
        ChainLink(
            "admission[bridge] -> auth",
            "signed/execution_admission",
            "canonical",
            authorization_digest,
            _validate_digest(
                bridge_authorization["authorization_sha256"],
                "admission.runtime_authorization.value.policy-improvement-theory-bridge.authorization_sha256",
            ),
        ),
        ChainLink(
            "admission -> bp amendment",
            "signed/execution_admission",
            "canonical",
            base_amendment_digest,
            _validate_digest(
                admission_base["amendment_sha256"],
                "admission.base_policy_artifact.value.amendment_sha256",
            ),
        ),
        ChainLink(
            "admission -> exp1b amendment",
            "signed/execution_admission",
            "canonical",
            _canonical_sha256(amendment),
            _validate_digest(
                admission["prior_amendment_sha256"], "admission.prior_amendment_sha256"
            ),
        ),
        ChainLink(
            "provenance -> admission",
            "result/provenance.json",
            "canonical",
            admission_digest,
            _validate_digest(provenance["admission_sha256"], "provenance.admission_sha256"),
        ),
        ChainLink(
            "result -> provenance",
            "result/exp1b_result.json",
            "canonical",
            provenance_digest,
            _validate_digest(result["provenance_sha256"], "result.provenance_sha256"),
        ),
        ChainLink(
            "published result digest",
            "result/exp1b_result.sha256",
            "canonical",
            _validate_digest(inputs["result_sidecar"], "result sidecar"),
            result_digest,
        ),
    ]

    manifest_digests = _array(
        provenance["run_manifest_sha256s"], "provenance.run_manifest_sha256s"
    )
    payload_digests = _array(result["payload_digests"], "result.payload_digests")
    for position in range(SEED_COUNT):
        manifest = _object(manifests[position], f"manifest[{position}]")
        payload = _object(payloads[position], f"payload[{position}]")
        links.extend(
            [
                ChainLink(
                    f"provenance -> manifest[{position}]",
                    f"checkpoints/seed-{position}/run_manifest.json",
                    "canonical",
                    _validate_digest(
                        manifest_digests[position],
                        f"provenance.run_manifest_sha256s[{position}]",
                    ),
                    _canonical_sha256(manifest),
                ),
                ChainLink(
                    f"manifest[{position}] -> checkpoint",
                    f"checkpoints/seed-{position}/checkpoint.pt",
                    "file bytes",
                    _validate_digest(
                        manifest["checkpoint_sha256"],
                        f"manifest[{position}].checkpoint_sha256",
                    ),
                    _file_sha256(
                        root
                        / "stage_a_checkpoints"
                        / f"seed-{position}"
                        / "checkpoint.pt"
                    ),
                ),
                ChainLink(
                    f"result -> payload[{position}]",
                    f"payloads/seed-{position}.json",
                    "canonical",
                    _validate_digest(
                        payload_digests[position],
                        f"result.payload_digests[{position}]",
                    ),
                    _canonical_sha256(payload),
                ),
            ]
        )
    return links


def _verify(root: Path, recorder: CheckRecorder) -> None:
    inputs = _load_inputs(root)
    result = _object(inputs["result"], "result")
    provenance = _object(inputs["provenance"], "provenance")
    manifests = _array(inputs["manifests"], "manifests")
    payloads = _array(inputs["payloads"], "payloads")

    sidecar = _validate_digest(inputs["result_sidecar"], "result sidecar")
    result_digest = _canonical_sha256(result)
    recorder.check("result canonical digest", sidecar, result_digest)

    links = _chain_links(root, inputs)
    table = _array(inputs["chain_table"], "chain_table")
    recorder.check("chain link count", len(links), len(table))
    for index, link in enumerate(links):
        recorded = _object(table[index], f"chain_table[{index}]")
        recorder.check(f"chain table row {index}", link.table_row(), recorded)
        recorder.check(f"chain link {index}: {link.link}", link.expected, link.observed)

    provenance_manifest_digests = _array(
        provenance["run_manifest_sha256s"], "provenance.run_manifest_sha256s"
    )
    recorder.check(
        "Stage A manifest digest inventory", SEED_COUNT, len(provenance_manifest_digests)
    )
    seed_ids = _array(provenance["seed_ids"], "provenance.seed_ids")
    recorder.check("registered seed inventory", SEED_COUNT, len(seed_ids))
    for position, manifest_value in enumerate(manifests):
        manifest = _object(manifest_value, f"manifest[{position}]")
        recorder.check(
            f"Stage A manifest digest {position}",
            _validate_digest(
                provenance_manifest_digests[position],
                f"provenance.run_manifest_sha256s[{position}]",
            ),
            _canonical_sha256(manifest),
        )
        recorder.check(
            f"Stage A manifest seed position {position}",
            position,
            _integer(manifest["seed_position"], f"manifest[{position}].seed_position"),
        )
        recorder.check(
            f"Stage A manifest seed {position}",
            _integer(seed_ids[position], f"provenance.seed_ids[{position}]"),
            _integer(manifest["seed"], f"manifest[{position}].seed"),
        )

    result_payload_digests = _array(result["payload_digests"], "result.payload_digests")
    recorder.check(
        "Stage B payload digest inventory", SEED_COUNT, len(result_payload_digests)
    )
    for position, payload_value in enumerate(payloads):
        payload = _object(payload_value, f"payload[{position}]")
        recorder.check(
            f"Stage B payload digest {position}",
            _validate_digest(
                result_payload_digests[position], f"result.payload_digests[{position}]"
            ),
            _canonical_sha256(payload),
        )

    recorder.check("result seed count", SEED_COUNT, _integer(result["seed_count"], "result.seed_count"))
    recorder.check("result gamma", EXPECTED_GAMMA, _number(result["gamma"], "result.gamma"))
    recorder.check(
        "result Bellman horizon",
        EXPECTED_HORIZON,
        _integer(result["bellman_horizon"], "result.bellman_horizon"),
    )
    recorder.check(
        "result deployed depth",
        EXPECTED_DEPTH_N,
        _integer(result["deployed_depth_n"], "result.deployed_depth_n"),
    )
    recorder.check(
        "result reference depth",
        EXPECTED_DEPTH_M,
        _integer(result["reference_depth_m"], "result.reference_depth_m"),
    )
    gamma = _number(result["gamma"], "result.gamma")
    signed_gaps = _array(result["signed_gaps"], "result.signed_gaps")
    seed_rows = _array(result["seed_rows"], "result.seed_rows")
    recorder.check("signed gap inventory", SEED_COUNT, len(signed_gaps))
    recorder.check("seed row inventory", SEED_COUNT, len(seed_rows))
    recomputed_gaps = []
    for position, payload_value in enumerate(payloads):
        payload = _object(payload_value, f"payload[{position}]")
        row = _object(seed_rows[position], f"result.seed_rows[{position}]")
        recorder.check(
            f"payload state count {position}",
            STATE_COUNT,
            _integer(payload["state_count"], f"payload[{position}].state_count"),
        )
        recorder.check(
            f"payload seed position {position}",
            position,
            _integer(payload["seed_position"], f"payload[{position}].seed_position"),
        )
        recorder.check(
            f"payload seed {position}",
            _integer(seed_ids[position], f"provenance.seed_ids[{position}]"),
            _integer(payload["seed"], f"payload[{position}].seed"),
        )
        residual_n = _number(
            payload["maximum_absolute_residual_n"],
            f"payload[{position}].maximum_absolute_residual_n",
        )
        residual_m = _number(
            payload["maximum_absolute_residual_m"],
            f"payload[{position}].maximum_absolute_residual_m",
        )
        discrepancy = _number(
            payload["maximum_endpoint_discrepancy"],
            f"payload[{position}].maximum_endpoint_discrepancy",
        )
        gap = residual_n / (1.0 - gamma) - (
            discrepancy + residual_m / (1.0 - gamma)
        )
        recomputed_gaps.append(gap)
        payload_gap = _number(payload["signed_gap"], f"payload[{position}].signed_gap")
        result_gap = _number(signed_gaps[position], f"result.signed_gaps[{position}]")
        row_gap = _number(row["signed_gap"], f"result.seed_rows[{position}].signed_gap")
        recorder.check(f"per-seed gap {position}: payload", payload_gap, gap)
        recorder.check(f"per-seed gap {position}: result vector", result_gap, gap)
        recorder.check(f"per-seed gap {position}: result row", row_gap, gap)

    exact_sum = sum((Fraction.from_float(value) for value in recomputed_gaps), Fraction())
    recomputed_mean = float(exact_sum / SEED_COUNT)
    recorder.check(
        "equal-weight mean signed gap",
        _number(result["mean_signed_gap"], "result.mean_signed_gap"),
        recomputed_mean,
    )

    interval = _object(result["interval"], "result.interval")
    recorder.check(
        "bootstrap interval convention",
        "hyndman_fan_type7_two_sided_percentile_v1",
        _string(interval["interval_convention"], "result.interval.interval_convention"),
    )
    recorder.check(
        "bootstrap replicate encoding",
        "float_hex_v1",
        _string(interval["replicate_encoding"], "result.interval.replicate_encoding"),
    )
    lower_probability = _number(
        interval["lower_probability"], "result.interval.lower_probability"
    )
    confidence = _number(interval["confidence_level"], "result.interval.confidence_level")
    upper_probability = lower_probability + confidence
    recorder.check("bootstrap lower probability", EXPECTED_LOWER_PROBABILITY, lower_probability)
    recorder.check("bootstrap upper probability", EXPECTED_UPPER_PROBABILITY, upper_probability)
    encoded_replicates = _array(
        interval["replicate_means_hex"], "result.interval.replicate_means_hex"
    )
    recorder.check(
        "bootstrap replicate count", EXPECTED_BOOTSTRAP_REPLICATES, len(encoded_replicates)
    )
    try:
        replicates = [
            float.fromhex(_string(value, f"result.interval.replicate_means_hex[{index}]"))
            for index, value in enumerate(encoded_replicates)
        ]
    except ValueError as exc:
        raise VerificationError(f"invalid bootstrap hex float: {exc}") from exc
    if not all(math.isfinite(value) for value in replicates):
        raise VerificationError("bootstrap replicate vector contains a non-finite value")
    lower = _type7(replicates, lower_probability)
    upper = _type7(replicates, upper_probability)
    recorder.check(
        "bootstrap lower endpoint",
        _number(interval["interval_lower"], "result.interval.interval_lower"),
        lower,
    )
    recorder.check(
        "bootstrap upper endpoint",
        _number(interval["interval_upper"], "result.interval.interval_upper"),
        upper,
    )

    parity_bound = (
        EXPECTED_PARITY_ABSOLUTE_TOLERANCE
        + EXPECTED_PARITY_RELATIVE_TOLERANCE * EXPECTED_ADVANTAGE_CLIP
    )
    recorder.check("parity bound arithmetic", EXPECTED_PARITY_BOUND, parity_bound)
    for position, row_value in enumerate(seed_rows):
        row = _object(row_value, f"result.seed_rows[{position}]")
        payload = _object(payloads[position], f"payload[{position}]")
        row_secondary = _object(
            row["secondary_diagnostics"], f"result.seed_rows[{position}].secondary_diagnostics"
        )
        payload_secondary = _object(
            payload["secondary_diagnostics"], f"payload[{position}].secondary_diagnostics"
        )
        row_parity = _object(
            row_secondary["centering_parity"],
            f"result.seed_rows[{position}].secondary_diagnostics.centering_parity",
        )
        payload_parity = _object(
            payload_secondary["centering_parity"],
            f"payload[{position}].secondary_diagnostics.centering_parity",
        )
        reported = _number(
            row_parity["training_estimator_parity_max_abs_error"],
            f"result.seed_rows[{position}].secondary_diagnostics.centering_parity.training_estimator_parity_max_abs_error",
        )
        payload_reported = _number(
            payload_parity["training_estimator_parity_max_abs_error"],
            f"payload[{position}].secondary_diagnostics.centering_parity.training_estimator_parity_max_abs_error",
        )
        recorder.check(
            f"reported parity consistency {position}", payload_reported, reported
        )
        recorder.check(
            f"reported parity bound {position}",
            f"<= {parity_bound!r} ({parity_bound.hex()})",
            reported,
            passed=reported <= parity_bound,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("artifact_directory", type=Path)
    arguments = parser.parse_args()
    recorder = CheckRecorder(Path(__file__).resolve().with_name(SUMMARY_NAME))
    first_failure = None
    try:
        root = arguments.artifact_directory.resolve(strict=True)
        if not root.is_dir():
            raise VerificationError(f"artifact path is not a directory: {root}")
        _verify(root, recorder)
    except VerificationError as exc:
        first_failure = str(exc)
        recorder.add_unexpected_failure(first_failure)
        recorder.write_summary("fail", first_failure)
        print(f"SUMMARY FAIL first_failure={first_failure}", flush=True)
        return 1
    except Exception as exc:
        first_failure = f"{type(exc).__name__}: {exc}"
        recorder.add_unexpected_failure(first_failure)
        recorder.write_summary("fail", first_failure)
        print(f"SUMMARY FAIL first_failure={first_failure}", flush=True)
        return 1
    recorder.write_summary("pass", None)
    print(f"SUMMARY PASS checks={len(recorder._checks)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
