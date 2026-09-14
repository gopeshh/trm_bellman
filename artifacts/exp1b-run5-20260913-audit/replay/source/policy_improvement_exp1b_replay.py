#!/usr/bin/env fbpython
"""Read-only, per-state replay of the sealed Experiment 1B Stage B octet."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from policy_improvement_full_backend import _tree_identity
from rl.persistent_diagnostic_checkpoint import state_dict_sha256
from scripts.policy_improvement_exp1_diagnostics import (
    paired_population_summary,
    paired_state_diagnostic,
    PairedEndpointObservation,
)
from scripts.policy_improvement_exp1b_auditor import audit_exp1b_result_document
from scripts.policy_improvement_exp1b_bridge import (
    census_from_population,
    Exp1bRuntimeAttestation,
)
from scripts.policy_improvement_exp1b_evidence import (
    authenticate_base_artifact,
    authenticate_sealed_octet_checkpoints,
    Exp1bEvidenceGeneration,
    stable_bytes,
)
from scripts.policy_improvement_exp1b_schema import (
    apply_exp1b_admission,
    attestation_matches_authorization,
    DEPLOYED_DEPTH_N,
    EVALUATION_POPULATION_ID,
    exp1b_document_sha256,
    exp1b_state_id_for,
    Exp1bRequestSchedule,
    Exp1bSchemaError,
    FULL_ROLE,
    GAMMA,
    REFERENCE_DEPTH_M,
    registry_rows_by_run_id,
    THEORY_BRIDGE_ROLE,
    UNITS,
    validate_exp1b_admission,
    validate_exp1b_amendment,
    validate_exp1b_protocol,
    validate_exp1b_registry,
    validate_substitute_provenance,
)
from scripts.policy_improvement_exp1b_theory_backend import Exp1bTheoryBackend
from scripts.policy_improvement_populations import validate_v2_populations
from scripts.policy_improvement_schema import (
    load_strict_json_bytes,
    runtime_authorization_sha256,
    validate_runtime_authorization,
)
from scripts.policy_improvement_v2_schema import (
    validate_base_policy_amendment,
    validate_v2_protocol,
)
from utils.run_identity import canonical_json_sha256


SCHEMA_NAME: str = "policy_improvement_exp1b_per_state_replay_v1"
FLOAT_ENCODING: str = "float_hex_v1"
TENSOR_ENCODING: str = "tensor_bytes_base64_v1"
REPLAY_TARGET: str = "fbcode//buiksat_trm:policy_improvement_exp1b_replay"
REPLAY_BUILD_MODE: str = "@fbcode//mode/opt"
EXPECTED_SOURCE_REVISION: str = "2a14eb656bd983c37b60e92852083524348511d8"


class ReplayError(RuntimeError):
    """Raised when the replay cannot preserve its read-only audit contract."""


@dataclass(frozen=True)
class ReplayInputs:
    protocol: Mapping[str, Any]
    effective_protocol: Mapping[str, Any]
    provenance: Mapping[str, Any]
    population_registry: Mapping[str, Any]
    result: Mapping[str, Any]
    payloads: tuple[Mapping[str, Any], ...]
    checkpoints: tuple[Any, ...]
    census: Any
    evaluation_split_manifest_sha256: str
    evaluator_attestation: Mapping[str, str]
    authentication_report: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, *, stable: bool) -> tuple[Mapping[str, Any], bytes]:
    if stable:
        payload, _digest, _size = stable_bytes(path, label=str(path))
    else:
        payload = path.read_bytes()
    parsed = load_strict_json_bytes(payload)
    if not isinstance(parsed, Mapping):
        raise ReplayError(f"{path} is not a JSON object.")
    return dict(parsed), payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _first_difference(left: object, right: object, path: str = "value") -> str | None:
    if isinstance(left, float) and isinstance(right, float):
        return None if left.hex() == right.hex() else path
    if type(left) is not type(right):
        return path
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return f"{path}.keys"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (item, other) in enumerate(zip(left, right)):
            difference = _first_difference(item, other, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else path


def _source_documents(project_root: Path) -> dict[str, tuple[Mapping[str, Any], bytes]]:
    paths = {
        "protocol": project_root / "configs/policy_improvement_exp1b/protocol.json",
        "registry": project_root / "configs/policy_improvement_exp1b/registry.json",
        "amendment": project_root
        / "configs/policy_improvement_exp1b/amendments/reduced_study_exp1b.json",
        "parent_protocol": project_root / "configs/policy_improvement_v2/protocol.json",
        "parent_registry": project_root / "configs/policy_improvement_v2/registry.json",
        "parent_populations": project_root
        / "configs/policy_improvement_v2/populations.json",
        "parent_theory_amendment": project_root
        / "configs/policy_improvement_v2/amendments/theory_bridge_v2.json",
    }
    return {name: _strict_json(path, stable=False) for name, path in paths.items()}


def _validate_schedule_and_payloads(
    *,
    generation: Exp1bEvidenceGeneration,
    provenance: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    schedule, _raw = _strict_json(generation.schedule_path, stable=True)
    expected_schedule_fields = {
        "schema_name",
        "schema_version",
        "route_id",
        "provenance_sha256",
        "generation_sha256",
        "slots",
    }
    _require(
        set(schedule) == expected_schedule_fields, "Schedule field inventory differs."
    )
    _require(
        schedule["provenance_sha256"] == exp1b_document_sha256(provenance),
        "Schedule provenance digest differs.",
    )
    _require(
        schedule["generation_sha256"] == provenance["generation_sha256"],
        "Schedule generation digest differs.",
    )
    slots = schedule["slots"]
    _require(
        isinstance(slots, list) and len(slots) == UNITS, "Schedule is not an octet."
    )
    descriptors = provenance["sealed_checkpoints"]
    result_payload_digests = result["payload_digests"]
    result_attempts = result["attempt_counts"]
    payloads = []
    refusal_messages = []
    probe = object.__new__(Exp1bRequestSchedule)
    probe._owner = "read-only-replay-probe"
    probe._payload_digests = {}
    probe._slots = []
    for position, (slot, descriptor) in enumerate(zip(slots, descriptors)):
        _require(
            isinstance(slot, Mapping), f"Schedule slot {position} is not an object."
        )
        for field in ("seed_position", "seed", "run_id", "checkpoint_sha256"):
            _require(
                slot[field] == descriptor[field],
                f"Schedule slot {position} {field} differs from provenance.",
            )
        _require(slot["state"] == "served", f"Schedule slot {position} is not served.")
        _require(
            slot["claim_owner"] is None, f"Schedule slot {position} retains an owner."
        )
        _require(
            isinstance(slot["attempts"], int) and slot["attempts"] >= 1,
            f"Schedule slot {position} has no completed attempt.",
        )
        _require(
            slot["attempts"] == result_attempts[position],
            f"Schedule slot {position} attempt count differs from the result.",
        )
        payload, _payload_raw = _strict_json(
            generation.payload_path(position), stable=True
        )
        payload_digest = exp1b_document_sha256(payload)
        _require(
            payload_digest
            == slot["payload_sha256"]
            == result_payload_digests[position],
            f"Payload {position} digest differs from schedule or result.",
        )
        for field in (
            "seed_position",
            "seed",
            "run_id",
            "checkpoint_sha256",
        ):
            _require(
                payload[field] == descriptor[field],
                f"Payload {position} {field} differs from provenance.",
            )
        _require(payload["state_count"] == 128, f"Payload {position} is not 128-state.")
        payloads.append(payload)
        probe._slots.append(SimpleNamespace(**dict(slot)))
        probe._payload_digests[position] = payload_digest

    schedule_identity = [
        {
            "seed_position": slot["seed_position"],
            "seed": slot["seed"],
            "run_id": slot["run_id"],
            "checkpoint_sha256": slot["checkpoint_sha256"],
        }
        for slot in slots
    ]
    schedule_sha256 = exp1b_document_sha256(schedule_identity)
    _require(
        schedule_sha256
        == provenance["request_schedule_sha256"]
        == result["schedule_sha256"],
        "Schedule identity digest differs from provenance or result.",
    )

    for position, descriptor in enumerate(descriptors):
        try:
            probe._validate_request(
                seed_position=position,
                seed=descriptor["seed"],
                run_id=descriptor["run_id"],
                checkpoint_sha256=descriptor["checkpoint_sha256"],
                evaluation_population=EVALUATION_POPULATION_ID,
                depths=(DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M),
            )
        except Exp1bSchemaError as exc:
            message = str(exc)
            _require(
                message
                == "Experiment 1B refuses a post-payload retry for a served seed.",
                f"Schedule slot {position} refused for an unexpected reason: {message}",
            )
            refusal_messages.append(message)
        else:
            raise ReplayError(
                f"Schedule slot {position} did not preserve post-payload retry refusal."
            )

    return tuple(payloads), {
        "schedule_sha256": schedule_sha256,
        "served_positions": list(range(UNITS)),
        "attempt_counts": list(result_attempts),
        "post_payload_retry_refusal": [True] * UNITS,
        "post_payload_retry_refusal_message": refusal_messages[0],
    }


def _authenticate_inputs(project_root: Path, snapshot_root: Path) -> ReplayInputs:
    revision = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(revision == EXPECTED_SOURCE_REVISION, "Source checkout revision differs.")
    dirty = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain"], text=True
    )
    _require(not dirty, "Source checkout is not clean at the required revision.")

    documents = _source_documents(project_root)
    protocol = documents["protocol"][0]
    registry = documents["registry"][0]
    amendment = documents["amendment"][0]
    parent_protocol = documents["parent_protocol"][0]
    parent_registry = documents["parent_registry"][0]
    populations = documents["parent_populations"][0]
    parent_amendment = documents["parent_theory_amendment"][0]
    validate_exp1b_protocol(protocol)
    protocol_sha256 = exp1b_document_sha256(protocol)
    validate_exp1b_registry(registry, protocol_sha256=protocol_sha256)
    registry_sha256 = exp1b_document_sha256(registry)
    validate_exp1b_amendment(
        amendment,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
    )
    amendment_sha256 = exp1b_document_sha256(amendment)
    validate_v2_protocol(parent_protocol)
    validate_v2_populations(populations)

    signed_root = snapshot_root / "signed"
    admission_document, _ = _strict_json(
        signed_root / "exp1b_execution_admission_20260913_run5.json", stable=True
    )
    admission = validate_exp1b_admission(
        admission_document,
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        prior_amendment_sha256=amendment_sha256,
    )
    effective_protocol = apply_exp1b_admission(protocol, admission)
    runtime_authorization_document, _ = _strict_json(
        signed_root / "exp1b_runtime_authorization_20260913_run5.json", stable=True
    )
    runtime_authorization = validate_runtime_authorization(
        runtime_authorization_document
    )
    runtime_authorization_digest = runtime_authorization_sha256(runtime_authorization)
    base_amendment_document, _ = _strict_json(
        signed_root / "base_policy_amendment_exp0_20260913_run5.json", stable=True
    )
    base_amendment = validate_base_policy_amendment(base_amendment_document)
    base_amendment_sha256 = exp1b_document_sha256(base_amendment_document)
    chain, _ = _strict_json(signed_root / "chain_run5.json", stable=True)

    generation_id = str(chain["evidence_generation"])
    generation_directory = snapshot_root / "evidence" / generation_id
    provenance, _ = _strict_json(generation_directory / "provenance.json", stable=True)
    checked_provenance = validate_substitute_provenance(
        provenance, registry_rows=registry_rows_by_run_id(registry)
    )
    result, result_bytes = _strict_json(
        generation_directory / "exp1b_result.json", stable=True
    )
    result_sidecar = (
        (generation_directory / "exp1b_result.sha256")
        .read_text(encoding="ascii")
        .strip()
    )
    audit_report = audit_exp1b_result_document(
        result_bytes,
        protocol=protocol,
        registry=registry,
        amendment=amendment,
        provenance=provenance,
        population_registry=populations,
        admission=admission_document,
    )
    _require(
        result_sidecar == audit_report.document_sha256,
        "Published result sidecar differs from the audited result.",
    )

    expected_generation_sha256 = exp1b_document_sha256(
        {
            "root": chain["evidence_root"],
            "generation_id": generation_id,
            "schedule": "schedule_state.json",
            "payloads": "payloads",
            "checkpoints": "checkpoints",
        }
    )
    _require(
        expected_generation_sha256 == checked_provenance["generation_sha256"],
        "Provenance generation identity differs from the sealed chain.",
    )
    generation = Exp1bEvidenceGeneration(
        root=snapshot_root / "evidence",
        generation_id=generation_id,
        directory=generation_directory,
        provenance_path=generation_directory / "provenance.json",
        schedule_path=generation_directory / "schedule_state.json",
        payload_directory=generation_directory / "payloads",
        lock_path=generation_directory / ".lock",
        checkpoint_directory=generation_directory / "checkpoints",
        generation_sha256=expected_generation_sha256,
    )

    parent = protocol["parent"]
    parent_digests = {
        "protocol_sha256": exp1b_document_sha256(parent_protocol),
        "registry_sha256": exp1b_document_sha256(parent_registry),
        "population_registry_sha256": exp1b_document_sha256(populations),
        "theory_amendment_sha256": exp1b_document_sha256(parent_amendment),
    }
    for field, digest in parent_digests.items():
        _require(parent[field] == digest, f"Parent {field} differs.")

    _require(
        runtime_authorization_digest
        == chain["authorization_sha256"]
        == admission.authorization_for(FULL_ROLE).runtime_authorization_sha256
        == admission.authorization_for(THEORY_BRIDGE_ROLE).runtime_authorization_sha256,
        "Runtime authorization digest differs across the signed chain.",
    )
    _require(
        base_amendment_sha256
        == chain["base_policy_amendment_sha256"]
        == admission.base_policy_amendment_sha256
        == checked_provenance["base_policy_artifact"]["amendment_sha256"],
        "Base-policy amendment digest differs across the signed chain.",
    )
    _require(
        admission.admission_sha256
        == chain["admission_sha256"]
        == checked_provenance["admission_sha256"],
        "Execution admission digest differs across the signed chain.",
    )
    _require(chain["commit"] == EXPECTED_SOURCE_REVISION, "Chain commit differs.")
    _require(
        chain["reduced_study_amendment_sha256"] == amendment_sha256,
        "Chain amendment differs.",
    )
    _require(
        runtime_authorization["protocol"]["sha256"] == parent_digests["protocol_sha256"]
        and runtime_authorization["registry"]["sha256"]
        == parent_digests["registry_sha256"]
        and runtime_authorization["amendments"][0]["sha256"]
        == parent_digests["theory_amendment_sha256"],
        "Runtime authorization does not bind the parent documents.",
    )
    role_rows = {row["role"]: row for row in runtime_authorization["roles"]}
    for role, par_name, chain_field in (
        (FULL_ROLE, "policy_improvement_full.par", "full_runtime_sha256"),
        (
            THEORY_BRIDGE_ROLE,
            "policy_improvement_theory_bridge.par",
            "bridge_runtime_sha256",
        ),
    ):
        admitted = admission.authorization_for(role)
        row = role_rows[role]
        par_sha256 = _file_sha256(snapshot_root / "pars" / par_name)
        _require(
            par_sha256
            == row["runtime_sha256"]
            == admitted.runtime_sha256
            == chain[chain_field],
            f"{role} runtime digest differs across the signed chain.",
        )
        _require(
            row["source_git_commit"]
            == admitted.source_git_commit
            == EXPECTED_SOURCE_REVISION,
            f"{role} source revision differs.",
        )
    launcher_sha256 = _file_sha256(snapshot_root / "pars/phase4_runtime_launcher.par")
    _require(
        launcher_sha256
        == chain["launcher_sha256"]
        == runtime_authorization["launcher_sha256"],
        "Launcher digest differs across the signed chain.",
    )
    _require(
        attestation_matches_authorization(
            checked_provenance["producer_attestation"],
            admission.authorization_for(FULL_ROLE),
        ),
        "Provenance producer attestation is not admitted.",
    )
    evaluator_attestation = result["evaluator_attestation"]
    _require(
        attestation_matches_authorization(
            evaluator_attestation,
            admission.authorization_for(THEORY_BRIDGE_ROLE),
        ),
        "Result evaluator attestation is not admitted.",
    )
    _require(
        base_amendment["base_policy_artifact"]["checkpoint_sha256"]
        == admission.base_policy_checkpoint_sha256,
        "Base-policy amendment artifact differs from admission.",
    )

    registered_rows = registry_rows_by_run_id(registry)
    effective_config_sha256 = str(
        effective_protocol["effective_config"]["effective_config_sha256"]
    )
    training_population = effective_protocol["training_population"]
    checkpoints = authenticate_sealed_octet_checkpoints(
        generation=generation,
        provenance=checked_provenance,
        registry_rows=registered_rows,
        effective_config_sha256=effective_config_sha256,
        train_ordered_record_sha256=str(training_population["ordered_record_sha256"]),
    )
    base_digest = authenticate_base_artifact(
        generation=generation, provenance=checked_provenance
    )
    _require(
        base_digest == admission.base_policy_checkpoint_sha256,
        "Sealed base artifact differs from admission.",
    )

    population = populations["populations"][EVALUATION_POPULATION_ID]
    census = census_from_population(population, state_id_for=exp1b_state_id_for)
    evaluation_population = effective_protocol["evaluation_population"]
    _require(
        census.ordered_record_sha256
        == evaluation_population["ordered_record_sha256"]
        == checked_provenance["ordered_population_sha256"],
        "Registered census record ordering differs.",
    )
    _require(
        census.binding_sha256
        == evaluation_population["binding_sha256"]
        == checked_provenance["population_binding_sha256"],
        "Registered census binding differs.",
    )
    _require(
        census.ordering_sha256() == result["census_ordering_sha256"],
        "Derived census member ordering differs from result.",
    )
    payloads, schedule_report = _validate_schedule_and_payloads(
        generation=generation, provenance=checked_provenance, result=result
    )
    validation_split = parent_protocol["dataset"]["splits"]["validation"]
    validation_manifest = validation_split["manifest_sha256"]
    _require(
        validation_manifest["status"] == "available",
        "Validation manifest is unavailable.",
    )

    return ReplayInputs(
        protocol=protocol,
        effective_protocol=effective_protocol,
        provenance=checked_provenance,
        population_registry=populations,
        result=result,
        payloads=payloads,
        checkpoints=checkpoints,
        census=census,
        evaluation_split_manifest_sha256=str(validation_manifest["value"]),
        evaluator_attestation={
            key: str(evaluator_attestation[key])
            for key in (
                "source_git_commit",
                "runtime_sha256",
                "launcher_sha256",
                "runtime_authorization_sha256",
            )
        },
        authentication_report={
            "source_revision": revision,
            "protocol_sha256": protocol_sha256,
            "registry_sha256": registry_sha256,
            "amendment_sha256": amendment_sha256,
            "admission_sha256": admission.admission_sha256,
            "runtime_authorization_sha256": runtime_authorization_digest,
            "base_policy_amendment_sha256": base_amendment_sha256,
            "provenance_sha256": exp1b_document_sha256(provenance),
            "published_result_sha256": audit_report.document_sha256,
            "parent_digests": parent_digests,
            "checkpoint_count": len(checkpoints),
            "census_state_count": len(census.members),
            **schedule_report,
        },
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()


def _lossless(value: object) -> object:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        payload = _tensor_bytes(tensor)
        return {
            "kind": "tensor",
            "encoding": TENSOR_ENCODING,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "data_base64": base64.b64encode(payload).decode("ascii"),
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReplayError("Replay output contains a nonfinite float.")
        return value.hex()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _lossless(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_lossless(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _lossless(getattr(value, item.name)) for item in fields(value)
        }
    if type(value).__module__.startswith("numpy"):
        if (
            hasattr(value, "tobytes")
            and hasattr(value, "dtype")
            and hasattr(value, "shape")
        ):
            payload = value.tobytes(order="C")
            return {
                "kind": "ndarray",
                "encoding": TENSOR_ENCODING,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "data_base64": base64.b64encode(payload).decode("ascii"),
            }
        if hasattr(value, "item"):
            return _lossless(value.item())
    raise ReplayError(f"Cannot encode replay value of type {type(value).__name__}.")


def _module_identities(session: Any) -> dict[str, str]:
    modules = {
        "model": session.model,
        "policy_model_old": session._trainer.policy_model_old,
        "policy_model_candidate": session._trainer.policy_model_candidate,
        "target_model": session._trainer.target_model,
    }
    return {
        name: state_dict_sha256(module.state_dict()) for name, module in modules.items()
    }


def _remaining_edits(x: object) -> int | None:
    if not isinstance(x, Mapping) or "remaining_edits" not in x:
        return None
    value = x["remaining_edits"]
    if torch.is_tensor(value):
        return int(value.reshape(()).item())
    return int(value)


def _transition_rows(
    session: Any, record: Any, values_by_depth: Mapping[int, Sequence[float]]
) -> list[dict[str, Any]]:
    assert record.action_mask is not None
    rows = []
    initial_clock = {
        "step_count": int(record.environment_state["step_count"]),
        "remaining_edits": _remaining_edits(record.x),
    }
    for action_index, allowed in enumerate(record.action_mask):
        row: dict[str, Any] = {
            "action_index": action_index,
            "allowed": allowed,
            "action_value_n": float(values_by_depth[DEPLOYED_DEPTH_N][action_index]),
            "action_value_m": float(values_by_depth[REFERENCE_DEPTH_M][action_index]),
            "initial_clock": initial_clock,
        }
        if not allowed:
            row.update(
                {
                    "reward": None,
                    "terminal": None,
                    "successor_identity": None,
                    "successor_clock": None,
                    "successor_x": None,
                    "successor_plan": None,
                    "transition_info": None,
                }
            )
            rows.append(row)
            continue
        environment = session._new_environment()
        environment.load_checkpoint_state(record.environment_state)
        (x_next, plan_next), reward, terminal, info = environment.step(action_index)
        successor_clock = {
            "step_count": int(environment.step_count),
            "remaining_edits": _remaining_edits(x_next),
        }
        identity_input = {
            "x": _tree_identity(x_next),
            "plan": _tree_identity(plan_next),
            "clock": successor_clock,
        }
        row.update(
            {
                "reward": float(reward),
                "terminal": bool(terminal),
                "successor_identity": canonical_json_sha256(identity_input),
                "successor_identity_input": identity_input,
                "successor_clock": successor_clock,
                "successor_x": x_next,
                "successor_plan": plan_next,
                "transition_info": info,
            }
        )
        rows.append(row)
    return rows


def _summary_mapping(summary: Any) -> dict[str, Any]:
    return {
        "state_count": summary.state_count,
        "maximum_endpoint_discrepancy": summary.maximum_endpoint_discrepancy.value,
        "maximum_absolute_residual_n": summary.maximum_absolute_residual_n.value,
        "maximum_absolute_residual_m": summary.maximum_absolute_residual_m.value,
        "direct_residual_bound_n": summary.direct_residual_bound_n,
        "finite_reference_bound": summary.finite_reference_bound,
        "signed_gap": summary.signed_gap,
        "discrepancy_witness_state_id": summary.maximum_endpoint_discrepancy.state_id,
        "residual_n_witness_state_id": summary.maximum_absolute_residual_n.state_id,
        "residual_m_witness_state_id": summary.maximum_absolute_residual_m.state_id,
    }


def _replay_seed(
    inputs: ReplayInputs, project_root: Path, position: int
) -> dict[str, Any]:
    descriptor = inputs.checkpoints[position]
    training_population = inputs.effective_protocol["training_population"]
    evaluation_population = inputs.effective_protocol["evaluation_population"]
    effective_config_sha256 = str(
        inputs.effective_protocol["effective_config"]["effective_config_sha256"]
    )
    backend = Exp1bTheoryBackend(
        runtime_attestation=inputs.evaluator_attestation,
        authenticated_checkpoints=inputs.checkpoints,
        effective_config_sha256=effective_config_sha256,
        census_ordering_sha256=inputs.census.ordering_sha256(),
        census=inputs.census.members,
        dataset_root=(
            project_root / str(training_population["dataset_root"])
        ).resolve(),
        evaluation_split=str(evaluation_population["split"]),
        evaluation_split_manifest_sha256=inputs.evaluation_split_manifest_sha256,
    )
    attestation = Exp1bRuntimeAttestation(**inputs.evaluator_attestation)
    evaluator = backend.prepare_exp1b_bridge(
        seed_position=position,
        runtime_attestation=attestation,
        effective_config_sha256=effective_config_sha256,
    )
    session = backend._sessions[position]
    before = _module_identities(session)
    _require(
        before == session.module_state_sha256s,
        "Restored pre-replay model identities differ.",
    )

    action_cache: dict[tuple[str, int], tuple[float, ...]] = {}
    trainer_cache: dict[str, tuple[tuple[float, ...], str, float | None]] = {}
    constructed_cache: dict[str, tuple[float, ...]] = {}
    original_action_values = session.action_values
    original_trainer_advantages = session._trainer_advantages
    original_constructed_advantages = session._constructed_advantages

    def cached_action_values(state_id: str, depth: int) -> tuple[float, ...]:
        key = (state_id, depth)
        if key not in action_cache:
            action_cache[key] = tuple(original_action_values(state_id, depth))
        return action_cache[key]

    def captured_trainer_advantages(
        record: Any,
    ) -> tuple[tuple[float, ...], str, float | None]:
        value = original_trainer_advantages(record)
        trainer_cache[record.state_id] = value
        return value

    def captured_constructed_advantages(
        record: Any,
        action_values: Sequence[float],
        *,
        clipping_kind: str,
        clip_value: float | None,
    ) -> tuple[float, ...]:
        value = tuple(
            original_constructed_advantages(
                record,
                action_values,
                clipping_kind=clipping_kind,
                clip_value=clip_value,
            )
        )
        constructed_cache[record.state_id] = value
        return value

    session.action_values = cached_action_values
    session._trainer_advantages = captured_trainer_advantages
    session._constructed_advantages = captured_constructed_advantages

    secondary = session.secondary_diagnostics()
    sealed_payload = inputs.payloads[position]
    difference = _first_difference(
        secondary, sealed_payload["secondary_diagnostics"], "secondary_diagnostics"
    )
    _require(difference is None, f"First divergent quantity: {difference}.")

    diagnostics = []
    state_rows = []
    for member in inputs.census.members:
        state_id = member.state_id
        endpoint_n = float(evaluator.endpoint_values(state_id, DEPLOYED_DEPTH_N))
        endpoint_m = float(evaluator.endpoint_values(state_id, REFERENCE_DEPTH_M))
        action_n = tuple(evaluator.action_values(state_id, DEPLOYED_DEPTH_N))
        action_m = tuple(evaluator.action_values(state_id, REFERENCE_DEPTH_M))
        mask = tuple(evaluator.action_mask(state_id))
        base = tuple(evaluator.base_probabilities(state_id))
        candidate = tuple(session.candidate_probabilities(state_id))
        deployed = tuple(session.deployed_probabilities(state_id))
        observation = PairedEndpointObservation(
            state_id=state_id,
            record_index=member.record_index,
            dataset_record_sha256=member.dataset_record_sha256,
            snapshot_id=descriptor.checkpoint_sha256,
            deployed_depth_n=DEPLOYED_DEPTH_N,
            reference_depth_m=REFERENCE_DEPTH_M,
            action_mask=mask,
            base_probabilities=base,
            endpoint_value_n=endpoint_n,
            endpoint_value_m=endpoint_m,
            action_values_n=action_n,
            action_values_m=action_m,
            gamma=GAMMA,
        )
        diagnostic = paired_state_diagnostic(observation)
        diagnostics.append(diagnostic)
        record = session._record(state_id)
        session._populate(record)
        trainer_advantages, clipping_kind, clip_value = trainer_cache[state_id]
        constructed_advantages = constructed_cache[state_id]
        target_value = float(
            session._target_value_at(
                record.x,
                record.plan,
                record.latent,
                DEPLOYED_DEPTH_N,
            )
        )
        reconstructed_mixture = tuple(
            (1.0 - float(session._rl_config.mixture_alpha)) * left
            + float(session._rl_config.mixture_alpha) * right
            for left, right in zip(base, candidate)
        )
        state_rows.append(
            {
                "state_id": state_id,
                "record_index": member.record_index,
                "dataset_record_sha256": member.dataset_record_sha256,
                "endpoint_value_n": endpoint_n,
                "endpoint_value_m": endpoint_m,
                "action_mask": mask,
                "base_probabilities": base,
                "candidate_probabilities": candidate,
                "deployed_probabilities": deployed,
                "reconstructed_mixture": reconstructed_mixture,
                "action_values_n": action_n,
                "action_values_m": action_m,
                "diagnostic": asdict(diagnostic),
                "trainer_advantages": trainer_advantages,
                "constructed_advantages": constructed_advantages,
                "clipping_kind": clipping_kind,
                "clip_value": clip_value,
                "target_value_n": target_value,
                "normalization_mass_error": float(record.normalization_mass_error),
                "initial_state": {
                    "identity": _tree_identity(
                        {
                            "x": record.x,
                            "plan": record.plan,
                            "environment": record.environment_state,
                        }
                    ),
                    "x": record.x,
                    "plan": record.plan,
                    "environment": record.environment_state,
                },
                "initial_latent": {
                    "identity": _tree_identity(record.latent),
                    "raw": record.latent,
                },
                "base_next_latent": {
                    "identity": _tree_identity(record.base_next_latent),
                    "raw": record.base_next_latent,
                },
                "deployed_next_latent": {
                    "identity": _tree_identity(record.deployed_next_latent),
                    "raw": record.deployed_next_latent,
                },
                "actions": _transition_rows(
                    session,
                    record,
                    {
                        DEPLOYED_DEPTH_N: action_n,
                        REFERENCE_DEPTH_M: action_m,
                    },
                ),
            }
        )

    summary = paired_population_summary(
        snapshot_id=descriptor.checkpoint_sha256,
        population_id=EVALUATION_POPULATION_ID,
        deployed_depth_n=DEPLOYED_DEPTH_N,
        reference_depth_m=REFERENCE_DEPTH_M,
        expected_population=inputs.census.members,
        state_diagnostics=diagnostics,
    )
    summary_mapping = _summary_mapping(summary)
    sealed_summary = {name: sealed_payload[name] for name in summary_mapping}
    difference = _first_difference(summary_mapping, sealed_summary, "seed_summary")
    _require(difference is None, f"First divergent quantity: {difference}.")
    after = _module_identities(session)
    _require(
        before == after, "First divergent quantity: model_state_identity_after_replay."
    )

    return {
        "seed_position": position,
        "seed": descriptor.seed,
        "run_id": descriptor.run_id,
        "checkpoint_sha256": descriptor.checkpoint_sha256,
        "checkpoint_size_bytes": descriptor.checkpoint_size_bytes,
        "checkpoint_model_state_sha256": descriptor.model_state_sha256,
        "census_ordering_sha256": inputs.census.ordering_sha256(),
        "state_count": len(state_rows),
        "model_state_identities_before": before,
        "model_state_identities_after": after,
        "model_state_unchanged": before == after,
        "producer_summary": summary_mapping,
        "sealed_payload_summary": sealed_summary,
        "secondary_diagnostics": secondary,
        "states": state_rows,
    }


def _install_output(path: Path, document: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise ReplayError(f"Replay output already exists: {path}")
    payload = (
        json.dumps(
            _lossless(document),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    with sidecar.open("x", encoding="ascii") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--seed-position", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-identity", type=Path, required=True)
    parser.add_argument("--binary-path", type=Path, required=True)
    arguments = parser.parse_args()
    if not 0 <= arguments.seed_position < UNITS:
        raise ReplayError("Seed position is outside the registered octet.")
    build_identity, build_identity_bytes = _strict_json(
        arguments.build_identity, stable=False
    )
    binary_sha256 = _file_sha256(arguments.binary_path)
    _require(
        build_identity["producer"]["sha256"] == binary_sha256,
        "Running replay binary differs from its recorded build identity.",
    )
    _require(
        build_identity["producer"]["target"] == REPLAY_TARGET
        and build_identity["producer"]["build_mode"] == REPLAY_BUILD_MODE,
        "Replay build target or mode differs from its recorded identity.",
    )

    inputs = _authenticate_inputs(
        arguments.project_root.resolve(), arguments.snapshot_root.resolve()
    )
    seed_output = _replay_seed(
        inputs, arguments.project_root.resolve(), arguments.seed_position
    )
    document = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "float_encoding": FLOAT_ENCODING,
        "tensor_encoding": TENSOR_ENCODING,
        "read_only": True,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "build_identity_sha256": hashlib.sha256(build_identity_bytes).hexdigest(),
        "build_identity": build_identity,
        "authentication": inputs.authentication_report,
        "replay": seed_output,
    }
    digest = _install_output(arguments.output.resolve(), document)
    print(
        f"exp1b-replay: seed_position={arguments.seed_position} "
        f"gap={seed_output['producer_summary']['signed_gap']!r} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReplayError, Exp1bSchemaError) as exc:
        print(f"exp1b-replay: REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1)
