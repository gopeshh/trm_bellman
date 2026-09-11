#!/usr/bin/env fbpython
"""Experiment 1B reduced-study namespace: documents, provenance, and schedule.

Experiment 1B is a reduced study registered in its own namespace. It does not
edit, extend, or reinterpret any policy-improvement-v2 document: it cites the
frozen v2 protocol, registry, and population registry **by canonical digest**
and fails closed if any of them moves. The v2 Stage 1-3 program is untouched.

What this module owns
---------------------
* the Experiment 1B protocol, registry, and amendment schemas;
* the D1 substitute provenance record, which is what authorises the
  Experiment 1B validation-bridge route in place of the Stage 1 ``V_select``
  binding the standing v2 gate demands;
* the eight-request schedule and its state machine.

The standing v2 gate
--------------------
``scripts/policy_improvement_theory_backend_v2.py`` refuses every
``validation_bridge`` request outright, pending "the canonical 30-request
schedule and its independently re-audited V_select selection provenance".
Experiment 1B does not lift, weaken, patch, or route around that refusal: the v2
route stays exactly as it is, and a v2 request still terminates there. Instead
D1 registers a **separate** route with a different, narrower authorisation:
exactly eight requests, one per registered seed, admissible only after all eight
budget-final checkpoints are sealed, over the fixed ordered 128-record census,
with no selection use of any kind. The substitute provenance below is the
document that has to be complete and internally consistent for that route to
open at all.

Fail-closed posture
-------------------
Every validator here rejects rather than repairs. Nothing has a default that
could stand in for a missing owner input. Two inputs remain registered
*unavailable* slots and block execution until supplied: the admitted
Experiment 0 base-policy artifact, and the runtime authorisation for the commit
containing this slice.

The bootstrap RNG namespace was the third such slot and is now resolved: the
owner approved the ASCII literal ``upi-trm-exp1b-seed-bootstrap-v1``. The
protocol carries it as an *available* slot whose value must equal
``BOOTSTRAP_RNG_NAMESPACE``, and ``interval_emission_allowed`` is now required
to be exactly ``true``. The two remaining slots gate execution independently, so
enabling interval emission does not open training or the bridge.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Any

from scripts.policy_improvement_exp1b_bootstrap import (
    BOOTSTRAP_RNG_NAMESPACE,
    BOOTSTRAP_SCHEME,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    INTERVAL_CONVENTION,
    LOWER_PROBABILITY,
    REPLICATES,
    UNITS,
    UPPER_PROBABILITY,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
)

__all__ = [
    "ACCESS_STATES",
    "ADMISSIBLE_SLOTS",
    "ADMISSION_SCHEMA_NAME",
    "ADMISSION_SCHEMA_VERSION",
    "AMENDMENT_SCHEMA_NAME",
    "AMENDMENT_SCHEMA_VERSION",
    "BRIDGE_ROUTE_ID",
    "DEPLOYED_DEPTH_N",
    "Exp1bRequestSchedule",
    "Exp1bScheduleStateError",
    "registry_rows_by_run_id",
    "Exp1bSchemaError",
    "EVALUATION_POPULATION_ID",
    "EXECUTION_ENVIRONMENT_VARIABLE",
    "EXECUTION_PURPOSE",
    "EXECUTION_REQUIRED_VALUE",
    "ADMISSION_ROLES",
    "Exp1bExecutionAdmission",
    "Exp1bRoleAuthorization",
    "FULL_ROLE",
    "role_attestation_document",
    "THEORY_BRIDGE_ROLE",
    "apply_exp1b_admission",
    "PARENT_PROTOCOL_ID",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA_NAME",
    "PROTOCOL_SCHEMA_VERSION",
    "REFERENCE_DEPTH_M",
    "REGISTERED_SEEDS",
    "REGISTRY_SCHEMA_NAME",
    "REGISTRY_SCHEMA_VERSION",
    "TERMINAL_ENVIRONMENT_INTERACTIONS",
    "TRAINING_RECORD_COUNT",
    "STATE_ID_NAMESPACE",
    "exp1b_document_sha256",
    "exp1b_state_id_for",
    "attestation_matches_authorization",
    "CENTERING_PARITY_TOLERANCE",
    "TRAINER_RECONSTRUCTION_CONTRACT",
    "MIXTURE_IDENTITY_TOLERANCE",
    "TARGET_LAG_KIND",
    "CENTERING_PARITY_KIND",
    "MIXTURE_IDENTITY_KIND",
    "DEPLOYMENT_MISMATCH_KIND",
    "PERSISTENT_STATE_KIND",
    "SECONDARY_DIAGNOSTIC_NAMES",
    "SECONDARY_DIAGNOSTIC_SCHEMA_NAME",
    "SECONDARY_DIAGNOSTIC_SCHEMA_VERSION",
    "validate_exp1b_admission",
    "validate_secondary_diagnostics",
    "validate_exp1b_amendment",
    "validate_exp1b_protocol",
    "validate_exp1b_registry",
    "validate_substitute_provenance",
]


PROTOCOL_ID = "policy-improvement-exp1b-20260909"
PROTOCOL_SCHEMA_NAME = "policy_improvement_exp1b_protocol_v1"
PROTOCOL_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_NAME = "policy_improvement_exp1b_registry_v1"
REGISTRY_SCHEMA_VERSION = 1
AMENDMENT_SCHEMA_NAME = "policy_improvement_exp1b_amendment_v1"
AMENDMENT_SCHEMA_VERSION = 1
#: Version 2: the four flat runtime fields became one role-tagged
#: ``producer_attestation``, and ``run_manifest_sha256s`` was added. A version-1
#: document cannot satisfy the version-2 inventory and must not be able to claim
#: it does.
PROVENANCE_SCHEMA_NAME = "policy_improvement_exp1b_substitute_provenance_v1"
PROVENANCE_SCHEMA_VERSION = 2

#: The parent namespace whose frozen documents Experiment 1B cites by digest.
PARENT_PROTOCOL_ID = "policy-improvement-v2-20260818"
PARENT_PROTOCOL_SCHEMA_NAME = "policy_improvement_protocol_v2"
PARENT_PROTOCOL_SCHEMA_VERSION = 2

BRIDGE_ROUTE_ID = "exp1b_sealed_octet_validation_bridge_v1"
METHOD_ID = "fixed_base_exact_persistent"
EVALUATION_POPULATION_ID = "validation_bridge"
EVALUATION_SPLIT = "validation"
EVALUATION_RECORD_COUNT = 128
TRAINING_POPULATION_ID = "train_full"
TRAINING_SPLIT = "train"
TRAINING_RECORD_COUNT = 1024
#: The dataset the parent namespace registers, cited verbatim. Experiment 1B
#: builds no dataset and selects no records.
#: The regenerated corpus. Content-identical to the v1/v2 registration --
#: all six content digests and all 128 validation_bridge record digests were
#: verified equal after publication -- but a distinct entry, because
#: MANIFEST.json embeds the producer attestation and the original corpus no
#: longer exists. See configs/policy_improvement_exp1b/REGENERATION.md.
DATASET_NAME = "policy-improvement-hard-4x4-v1-regen-20260911"
DATASET_ROOT = "data/policy-improvement-v1-owner/policy-improvement-hard-4x4-v1-regen-20260911"
PARENT_CONFIG_DIRECTORY = "policy_improvement_v2"
TERMINAL_ENVIRONMENT_INTERACTIONS = 10000
DEPLOYED_DEPTH_N = 2
REFERENCE_DEPTH_M = 8
BELLMAN_HORIZON_K = 1
GAMMA = 0.99
MIXTURE_ALPHA = 0.1

#: The eight pre-outcome values frozen in the parent protocol's
#: ``seeds.confirmatory``, cited verbatim. Experiment 1B selects no seed.
REGISTERED_SEEDS = (
    2081976412,
    781025396,
    1148619853,
    913535362,
    2143253519,
    2279379379,
    402312153,
    2736405725,
)
SEED_DERIVATION = "explicit_preoutcome_values_v2"
SEED_NAMESPACE = "upi-trm-policy-improvement-v2"

#: Ordered lifecycle of the Experiment 1B bridge route. The route may only
#: open in ``sealed_octet_complete``; every request advances no state until the
#: whole octet has been served, and ``payload_emitted`` is terminal.
ACCESS_STATES = (
    "training_incomplete",
    "sealed_octet_complete",
    "bridge_open",
    "payload_emitted",
    "closed",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: Inputs the owner has not supplied. Each is a registered slot, not a default.
#: ``bootstrap_rng_namespace_literal_not_supplied`` was retired when the owner
#: approved the literal; a document still carrying it is now rejected.
UNAVAILABLE_REASONS = frozenset(
    {
        "experiment0_base_policy_artifact_not_admitted",
        "runtime_authorization_not_minted_for_slice_commit",
    }
)


class Exp1bSchemaError(ValueError):
    """Raised when an Experiment 1B document violates its registered schema."""


#: Namespace for the census state identifier. Distinct from every v2 namespace,
#: so an identifier minted for another population cannot collide into this one.
STATE_ID_NAMESPACE = "upi-trm-exp1b-validation-bridge-state-v1"


def exp1b_state_id_for(record_index: int, dataset_record_sha256: str) -> str:
    """Derive one census state identifier from its registered record identity.

    Both the position *and* the record digest go into the identifier. Position
    alone would let a reordered census reuse identifiers; digest alone would
    collapse two identical records into one state and silently shorten the
    population.

    Lives here rather than in the Stage B backend because it is a property of
    the census, and because the Stage A runtime needs it without pulling the
    Torch-restoring Stage B module into the full PAR.
    """

    if isinstance(record_index, bool) or not isinstance(record_index, int):
        raise Exp1bSchemaError("Census record index must be an integer.")
    if record_index < 0:
        raise Exp1bSchemaError("Census record index must be nonnegative.")
    _sha256(dataset_record_sha256, path="census record digest")
    digest = hashlib.sha256()
    digest.update(STATE_ID_NAMESPACE.encode("ascii"))
    digest.update(b"\x00")
    digest.update(str(record_index).encode("ascii"))
    digest.update(b"\x00")
    digest.update(dataset_record_sha256.encode("ascii"))
    return f"exp1b-state-{digest.hexdigest()}"


def exp1b_document_sha256(value: object) -> str:
    """Canonical-JSON SHA-256, matching the parent namespace's convention."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Exp1bSchemaError(f"{path} must be an object.")
    return dict(value)


def _exact_fields(value: object, expected: set[str], *, path: str) -> dict[str, Any]:
    result = _mapping(value, path=path)
    if set(result) != expected:
        raise Exp1bSchemaError(
            f"{path} field inventory differs: {sorted(set(result) ^ expected)!r}."
        )
    return result


def _string(
    value: object,
    *,
    path: str,
    choices: frozenset[str] | set[str] | None = None,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise Exp1bSchemaError(f"{path} must be nonempty ASCII.")
    if choices is not None and value not in choices:
        raise Exp1bSchemaError(f"{path} is not registered.")
    if identifier and _IDENTIFIER.fullmatch(value) is None:
        raise Exp1bSchemaError(f"{path} is not an identifier.")
    return value


def _sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Exp1bSchemaError(f"{path} is not a SHA-256 digest.")
    return value


def _git_commit(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise Exp1bSchemaError(f"{path} is not a Git commit.")
    return value


def _timestamp(value: object, *, path: str) -> str:
    text = _string(value, path=path)
    if _UTC_TIMESTAMP.fullmatch(text) is None:
        raise Exp1bSchemaError(f"{path} is not a UTC timestamp.")
    return text


def _integer(value: object, *, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Exp1bSchemaError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise Exp1bSchemaError(f"{path} is below its minimum.")
    return value


def _depth(value: object, *, path: str) -> int:
    """A nonnegative integer unroll depth. Rejects bool and float lookalikes."""

    return _integer(value, path=path, minimum=0)


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Exp1bSchemaError(f"{path} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise Exp1bSchemaError(f"{path} must be finite.")
    return result


def _sequence(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise Exp1bSchemaError(f"{path} must be an array.")
    return list(value)


def _false(value: object, *, path: str) -> bool:
    if value is not False:
        raise Exp1bSchemaError(f"{path} must be exactly false.")
    return False


def _true(value: object, *, path: str) -> bool:
    if value is not True:
        raise Exp1bSchemaError(f"{path} must be exactly true.")
    return True


def _namespace_slot(value: object, *, path: str) -> dict[str, Any]:
    """The resolved RNG namespace slot: available, and exactly the frozen value.

    Registered as ``{status, value}`` rather than a bare string so the document
    records that this was an owner decision, and so a regression to
    ``{status, reason}`` is rejected by field inventory rather than silently
    read as absent.
    """

    slot = _exact_fields(value, {"status", "value"}, path=path)
    if slot["status"] != "available":
        raise Exp1bSchemaError(f"{path} must register the approved namespace.")
    literal = _string(slot["value"], path=f"{path}.value")
    if literal != BOOTSTRAP_RNG_NAMESPACE:
        raise Exp1bSchemaError(
            f"{path}.value is not the owner-approved bootstrap RNG namespace."
        )
    return slot


def _unavailable_slot(value: object, *, path: str) -> dict[str, Any]:
    """A registered {status, reason} slot for an input the owner owes.

    ``status`` may only be ``unavailable``. This module never writes an
    ``available`` slot from inferred data; admission is a runtime input.
    """

    slot = _exact_fields(value, {"status", "reason"}, path=path)
    if slot["status"] != "unavailable":
        raise Exp1bSchemaError(f"{path} may only register an unavailable input.")
    _string(
        slot["reason"],
        path=f"{path}.reason",
        choices=UNAVAILABLE_REASONS,
        identifier=True,
    )
    return slot


#: The exact reduced-study deltas applied on top of the registered v2 method
#: configuration. Two kinds of change, and no others:
#:
#: * ``num_train_steps``/``log_interval`` drop from 80,000 to the registered
#:   10,000 terminal environment interactions. 10,000 is already a registered v2
#:   checkpoint boundary, so this is a shorter prefix of the same schedule, not a
#:   new budget.
#: * in-training evaluation is switched off. Not a scientific preference: the
#:   train-only access contract refuses every non-train load, so a run with
#:   ``eval_interval > 0`` would abort at the first evaluation tick.
#:
#: Everything else -- gamma, K, inner_unroll_n, the learning rates, the mixture,
#: the reward shaping -- is inherited byte-for-byte from the v2 template.
EFFECTIVE_CONFIG_OVERRIDES: dict[str, Any] = {
    "eval_interval": 0,
    "eval_num_episodes": 0,
    "log_interval": TERMINAL_ENVIRONMENT_INTERACTIONS,
    "num_train_steps": TERMINAL_ENVIRONMENT_INTERACTIONS,
    "track_theory_metrics": False,
    "use_tqdm": False,
}
EFFECTIVE_CONFIG_SCHEMA_NAME = "policy_improvement_exp1b_effective_config_v1"
EFFECTIVE_CONFIG_SCHEMA_VERSION = 1
EFFECTIVE_CONFIG_DERIVATION = (
    "v2_registered_method_config_with_reduced_interaction_budget_and_"
    "no_in_training_evaluation"
)
_EFFECTIVE_CONFIG_IDENTITY_FIELDS = (
    "base_canonical_config_sha256",
    "base_config_path",
    "base_config_sha256",
    "method_id",
    "overrides",
    "schema_name",
    "schema_version",
)


def effective_config_sha256(block: Mapping[str, Any]) -> str:
    """Identity of the reduced-study effective configuration.

    Deliberately a digest over *inputs*, not over a materialised RLConfig: the
    merged configuration is a deterministic function of the base template bytes
    and the explicit override mapping, and both are digestible without a YAML
    parser. This module is standard-library only and runs in environments with
    no Torch and no PyYAML; a digest that required materialising the config
    would not be checkable there, which is exactly where the check matters.
    """

    return exp1b_document_sha256(
        {name: block[name] for name in _EFFECTIVE_CONFIG_IDENTITY_FIELDS}
    )


def _effective_config(value: object, *, path: str) -> dict[str, Any]:
    block = _exact_fields(
        value,
        {*_EFFECTIVE_CONFIG_IDENTITY_FIELDS, "derivation", "effective_config_sha256"},
        path=path,
    )
    if _string(block["method_id"], path=f"{path}.method_id") != METHOD_ID:
        raise Exp1bSchemaError(f"{path} does not configure the registered method.")
    if (
        _string(block["schema_name"], path=f"{path}.schema_name")
        != EFFECTIVE_CONFIG_SCHEMA_NAME
        or _integer(block["schema_version"], path=f"{path}.schema_version")
        != EFFECTIVE_CONFIG_SCHEMA_VERSION
        or _string(block["derivation"], path=f"{path}.derivation", identifier=True)
        != EFFECTIVE_CONFIG_DERIVATION
    ):
        raise Exp1bSchemaError(f"{path} identity differs.")
    base_path = _string(block["base_config_path"], path=f"{path}.base_config_path")
    if (
        base_path != f"configs/{PARENT_CONFIG_DIRECTORY}/{METHOD_ID}.yaml"
        or ".." in base_path
    ):
        raise Exp1bSchemaError(f"{path}.base_config_path is not the v2 method template.")
    _sha256(block["base_config_sha256"], path=f"{path}.base_config_sha256")
    _sha256(
        block["base_canonical_config_sha256"],
        path=f"{path}.base_canonical_config_sha256",
    )
    overrides = _mapping(block["overrides"], path=f"{path}.overrides")
    if set(overrides) != set(EFFECTIVE_CONFIG_OVERRIDES):
        raise Exp1bSchemaError(f"{path}.overrides inventory differs.")
    for name, expected in EFFECTIVE_CONFIG_OVERRIDES.items():
        observed = overrides[name]
        # Type-aware: `0` and `False` compare equal, and `0` and `0.0` compare
        # equal, so an int/bool/float substitution would slip past `==` alone.
        if type(observed) is not type(expected) or observed != expected:
            raise Exp1bSchemaError(f"{path}.overrides.{name} differs.")
    recomputed = effective_config_sha256(block)
    if _sha256(
        block["effective_config_sha256"], path=f"{path}.effective_config_sha256"
    ) != recomputed:
        raise Exp1bSchemaError(
            f"{path}.effective_config_sha256 does not match its own inputs."
        )
    return block


def _training_population(value: object, *, path: str) -> dict[str, Any]:
    """The registered train-only dataset and population identity.

    Every field is copied from the frozen v2 protocol's
    ``dataset``/``dataset.splits.train`` blocks, cited by ``source``. The point
    is that the runtime can prove -- before it constructs a model -- that the
    thing it loaded is the registered 1,024-record train split of the registered
    dataset, and not some other directory with the right record count.
    """

    training = _exact_fields(
        value,
        {
            "count",
            "dataset_manifest_sha256",
            "dataset_name",
            "dataset_root",
            "ordered_record_sha256",
            "population_id",
            "resolves_evaluation_data",
            "source",
            "split",
            "split_manifest_sha256",
        },
        path=path,
    )
    if (
        _string(training["split"], path=f"{path}.split") != TRAINING_SPLIT
        or _integer(training["count"], path=f"{path}.count") != TRAINING_RECORD_COUNT
        or _string(training["population_id"], path=f"{path}.population_id", identifier=True)
        != TRAINING_POPULATION_ID
        or _string(training["dataset_name"], path=f"{path}.dataset_name")
        != DATASET_NAME
    ):
        raise Exp1bSchemaError(f"{path} differs from the registered train split.")
    root = _string(training["dataset_root"], path=f"{path}.dataset_root")
    if root != DATASET_ROOT or root.startswith("/") or ".." in root:
        raise Exp1bSchemaError(f"{path}.dataset_root is not the registered root.")
    if (
        _string(training["source"], path=f"{path}.source")
        != f"configs/{PARENT_CONFIG_DIRECTORY}/protocol.json#/dataset/splits/train"
    ):
        raise Exp1bSchemaError(f"{path}.source must cite the frozen v2 train split.")
    for name in (
        "dataset_manifest_sha256",
        "ordered_record_sha256",
        "split_manifest_sha256",
    ):
        _sha256(training[name], path=f"{path}.{name}")
    _false(
        training["resolves_evaluation_data"],
        path=f"{path}.resolves_evaluation_data",
    )
    return training


def _role_attestation(
    value: object, *, path: str, expected_role: str
) -> dict[str, Any]:
    """One PAR's runtime identity, tagged with the role it belongs to.

    Tagged because the two Experiment 1B stages run distinct binaries under
    distinct authorizations. An untagged identity is ambiguous, and comparing a
    producer's identity against an evaluator's live attestation -- which is what
    the untagged form invited -- can never succeed.
    """

    attestation = _exact_fields(
        value,
        {
            "role",
            "source_git_commit",
            "runtime_sha256",
            "launcher_sha256",
            "runtime_authorization_sha256",
        },
        path=path,
    )
    if (
        _string(attestation["role"], path=f"{path}.role", identifier=True)
        != expected_role
    ):
        raise Exp1bSchemaError(f"{path} is not the {expected_role!r} identity.")
    _git_commit(attestation["source_git_commit"], path=f"{path}.source_git_commit")
    for field in ("runtime_sha256", "launcher_sha256", "runtime_authorization_sha256"):
        _sha256(attestation[field], path=f"{path}.{field}")
    return attestation


def _parent_binding(value: object, *, path: str) -> dict[str, Any]:
    binding = _exact_fields(
        value,
        {
            "protocol_id",
            "protocol_schema_name",
            "protocol_schema_version",
            "protocol_sha256",
            "registry_sha256",
            "population_registry_sha256",
            "theory_amendment_sha256",
        },
        path=path,
    )
    if (
        _string(binding["protocol_id"], path=f"{path}.protocol_id")
        != PARENT_PROTOCOL_ID
        or _string(
            binding["protocol_schema_name"], path=f"{path}.protocol_schema_name"
        )
        != PARENT_PROTOCOL_SCHEMA_NAME
        or _integer(
            binding["protocol_schema_version"],
            path=f"{path}.protocol_schema_version",
        )
        != PARENT_PROTOCOL_SCHEMA_VERSION
    ):
        raise Exp1bSchemaError(f"{path} does not cite the frozen v2 namespace.")
    for field in (
        "protocol_sha256",
        "registry_sha256",
        "population_registry_sha256",
        "theory_amendment_sha256",
    ):
        _sha256(binding[field], path=f"{path}.{field}")
    return binding


def _population_binding(value: object, *, path: str) -> dict[str, Any]:
    population = _exact_fields(
        value,
        {
            "population_id",
            "split",
            "count",
            "ordered_record_sha256",
            "binding_sha256",
            "selection_use",
        },
        path=path,
    )
    if (
        _string(population["population_id"], path=f"{path}.population_id")
        != EVALUATION_POPULATION_ID
        or _string(population["split"], path=f"{path}.split") != EVALUATION_SPLIT
        or _integer(population["count"], path=f"{path}.count")
        != EVALUATION_RECORD_COUNT
        or _string(population["selection_use"], path=f"{path}.selection_use") != "none"
    ):
        raise Exp1bSchemaError(
            f"{path} must be the 128-record validation_bridge census with no "
            "selection use."
        )
    _sha256(population["ordered_record_sha256"], path=f"{path}.ordered_record_sha256")
    _sha256(population["binding_sha256"], path=f"{path}.binding_sha256")
    return population


def _seed_block(value: object, *, path: str) -> dict[str, Any]:
    seeds = _exact_fields(
        value, {"registered", "derivation", "namespace", "source"}, path=path
    )
    registered = _sequence(seeds["registered"], path=f"{path}.registered")
    if (
        tuple(
            _integer(item, path=f"{path}.registered[{offset}]", minimum=0)
            for offset, item in enumerate(registered)
        )
        != REGISTERED_SEEDS
    ):
        raise Exp1bSchemaError(
            f"{path}.registered must be the eight pre-outcome v2 confirmatory "
            "seed values, in their registered order."
        )
    if (
        _string(seeds["derivation"], path=f"{path}.derivation") != SEED_DERIVATION
        or _string(seeds["namespace"], path=f"{path}.namespace") != SEED_NAMESPACE
    ):
        raise Exp1bSchemaError(f"{path} must cite the frozen v2 seed derivation.")
    _string(seeds["source"], path=f"{path}.source")
    return seeds


def _analysis_block(value: object, *, path: str) -> dict[str, Any]:
    analysis = _exact_fields(
        value,
        {
            "endpoint",
            "bootstrap",
            "interval",
            "rng_namespace",
            "interval_emission_allowed",
        },
        path=path,
    )
    if _string(analysis["endpoint"], path=f"{path}.endpoint") != (
        "equal_seed_mean_signed_gap_g"
    ):
        raise Exp1bSchemaError(f"{path}.endpoint is not the registered endpoint.")
    bootstrap = _exact_fields(
        analysis["bootstrap"],
        {"scheme", "replicates", "units", "seed", "draw_order"},
        path=f"{path}.bootstrap",
    )
    _string(bootstrap["scheme"], path=f"{path}.bootstrap.scheme")
    _string(bootstrap["draw_order"], path=f"{path}.bootstrap.draw_order")
    if (
        bootstrap["scheme"] != BOOTSTRAP_SCHEME
        or _integer(bootstrap["replicates"], path=f"{path}.bootstrap.replicates")
        != REPLICATES
        or _integer(bootstrap["units"], path=f"{path}.bootstrap.units") != UNITS
        or _integer(bootstrap["seed"], path=f"{path}.bootstrap.seed") != BOOTSTRAP_SEED
        or bootstrap["draw_order"] != "replicate_outer_draw_inner"
    ):
        raise Exp1bSchemaError(f"{path}.bootstrap differs from the frozen D2 contract.")
    interval = _exact_fields(
        analysis["interval"],
        {"convention", "confidence_level", "lower_probability", "upper_probability"},
        path=f"{path}.interval",
    )
    _string(interval["convention"], path=f"{path}.interval.convention")
    if (
        interval["convention"] != INTERVAL_CONVENTION
        or _number(interval["confidence_level"], path=f"{path}.interval")
        != CONFIDENCE_LEVEL
        or _number(interval["lower_probability"], path=f"{path}.interval")
        != LOWER_PROBABILITY
        or _number(interval["upper_probability"], path=f"{path}.interval")
        != UPPER_PROBABILITY
    ):
        raise Exp1bSchemaError(f"{path}.interval differs from the frozen D3 contract.")
    # D2's namespace literal is now owner-approved, so the analysis side is
    # fully frozen and interval emission is open. Execution remains gated
    # separately by base_policy_artifact, runtime_authorization, and
    # execution_gate, none of which this unblocks.
    _namespace_slot(analysis["rng_namespace"], path=f"{path}.rng_namespace")
    _true(
        analysis["interval_emission_allowed"],
        path=f"{path}.interval_emission_allowed",
    )
    return analysis


def validate_exp1b_protocol(value: object) -> dict[str, Any]:
    """Validate the Experiment 1B protocol document."""

    protocol = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "created_date",
            "status",
            "parent",
            "method_id",
            "deployed_depth_n",
            "reference_depth_m",
            "bellman_horizon_K",
            "gamma",
            "mixture_alpha",
            "terminal_environment_interactions",
            "effective_config",
            "training_population",
            "evaluation_population",
            "seeds",
            "analysis",
            "bridge_route",
            "base_policy_artifact",
            "runtime_authorization",
            "amendments",
            "execution_gate",
        },
        path="exp1b_protocol",
    )
    if (
        _string(protocol["schema_name"], path="exp1b_protocol.schema_name")
        != PROTOCOL_SCHEMA_NAME
        or _integer(protocol["schema_version"], path="exp1b_protocol.schema_version")
        != PROTOCOL_SCHEMA_VERSION
        or _string(protocol["protocol_id"], path="exp1b_protocol.protocol_id")
        != PROTOCOL_ID
    ):
        raise Exp1bSchemaError("exp1b_protocol identity differs.")
    _timestamp(protocol["created_date"], path="exp1b_protocol.created_date")
    _string(protocol["status"], path="exp1b_protocol.status", identifier=True)
    _parent_binding(protocol["parent"], path="exp1b_protocol.parent")
    if (
        _string(protocol["method_id"], path="exp1b_protocol.method_id") != METHOD_ID
        or _depth(protocol["deployed_depth_n"], path="exp1b_protocol.deployed_depth_n")
        != DEPLOYED_DEPTH_N
        or _depth(
            protocol["reference_depth_m"], path="exp1b_protocol.reference_depth_m"
        )
        != REFERENCE_DEPTH_M
        or _integer(
            protocol["bellman_horizon_K"], path="exp1b_protocol.bellman_horizon_K"
        )
        != BELLMAN_HORIZON_K
        or _number(protocol["gamma"], path="exp1b_protocol.gamma") != GAMMA
        or _number(protocol["mixture_alpha"], path="exp1b_protocol.mixture_alpha")
        != MIXTURE_ALPHA
        or _integer(
            protocol["terminal_environment_interactions"],
            path="exp1b_protocol.terminal_environment_interactions",
        )
        != TERMINAL_ENVIRONMENT_INTERACTIONS
    ):
        raise Exp1bSchemaError("exp1b_protocol study settings differ.")
    _effective_config(
        protocol["effective_config"], path="exp1b_protocol.effective_config"
    )
    _training_population(
        protocol["training_population"], path="exp1b_protocol.training_population"
    )
    _population_binding(
        protocol["evaluation_population"], path="exp1b_protocol.evaluation_population"
    )
    _seed_block(protocol["seeds"], path="exp1b_protocol.seeds")
    _analysis_block(protocol["analysis"], path="exp1b_protocol.analysis")

    route = _exact_fields(
        protocol["bridge_route"],
        {
            "route_id",
            "request_count",
            "requires_sealed_octet",
            "depths_per_request",
            "scientific_selection",
            "validation_select_access",
            "test_access",
            "v2_route_unchanged",
        },
        path="exp1b_protocol.bridge_route",
    )
    if (
        _string(route["route_id"], path="exp1b_protocol.bridge_route.route_id")
        != BRIDGE_ROUTE_ID
        or _integer(
            route["request_count"], path="exp1b_protocol.bridge_route.request_count"
        )
        != UNITS
    ):
        raise Exp1bSchemaError("exp1b_protocol bridge route identity differs.")
    _true(
        route["requires_sealed_octet"],
        path="exp1b_protocol.bridge_route.requires_sealed_octet",
    )
    depths = _sequence(
        route["depths_per_request"], path="exp1b_protocol.bridge_route.depths_per_request"
    )
    if (
        tuple(
            _depth(
                item,
                path=f"exp1b_protocol.bridge_route.depths_per_request[{offset}]",
            )
            for offset, item in enumerate(depths)
        )
        != (DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M)
    ):
        raise Exp1bSchemaError("exp1b_protocol must compute q=2 and q=8 per request.")
    for field in (
        "scientific_selection",
        "validation_select_access",
        "test_access",
    ):
        _false(route[field], path=f"exp1b_protocol.bridge_route.{field}")
    _true(
        route["v2_route_unchanged"],
        path="exp1b_protocol.bridge_route.v2_route_unchanged",
    )

    _unavailable_slot(
        protocol["base_policy_artifact"], path="exp1b_protocol.base_policy_artifact"
    )
    _unavailable_slot(
        protocol["runtime_authorization"], path="exp1b_protocol.runtime_authorization"
    )
    if _sequence(protocol["amendments"], path="exp1b_protocol.amendments") != []:
        raise Exp1bSchemaError("exp1b_protocol.amendments must start empty.")
    gate = _exact_fields(
        protocol["execution_gate"],
        {"environment_variable", "required_value", "execution_allowed"},
        path="exp1b_protocol.execution_gate",
    )
    _string(gate["environment_variable"], path="exp1b_protocol.execution_gate")
    _string(gate["required_value"], path="exp1b_protocol.execution_gate")
    _false(
        gate["execution_allowed"],
        path="exp1b_protocol.execution_gate.execution_allowed",
    )
    return protocol


def validate_exp1b_registry(
    value: object, *, protocol_sha256: str | None = None
) -> dict[str, Any]:
    """Validate the Experiment 1B registry: exactly eight rows, one per seed."""

    registry = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "row_count",
            "rows",
        },
        path="exp1b_registry",
    )
    if (
        _string(registry["schema_name"], path="exp1b_registry.schema_name")
        != REGISTRY_SCHEMA_NAME
        or _integer(
            registry["schema_version"], path="exp1b_registry.schema_version"
        )
        != REGISTRY_SCHEMA_VERSION
        or _string(registry["protocol_id"], path="exp1b_registry.protocol_id")
        != PROTOCOL_ID
    ):
        raise Exp1bSchemaError("exp1b_registry identity differs.")
    digest = _sha256(registry["protocol_sha256"], path="exp1b_registry.protocol_sha256")
    if protocol_sha256 is not None and digest != protocol_sha256:
        raise Exp1bSchemaError("exp1b_registry cites a different protocol digest.")
    rows = _sequence(registry["rows"], path="exp1b_registry.rows")
    if (
        _integer(registry["row_count"], path="exp1b_registry.row_count") != UNITS
        or len(rows) != UNITS
    ):
        raise Exp1bSchemaError(
            f"exp1b_registry must hold exactly {UNITS} rows, one per seed."
        )
    for offset, (row, seed) in enumerate(zip(rows, REGISTERED_SEEDS)):
        path = f"exp1b_registry.rows[{offset}]"
        checked = _exact_fields(
            row,
            {
                "run_id",
                "seed",
                "seed_position",
                "method_id",
                "n",
                "K",
                "alpha",
                "terminal_environment_interactions",
                "training_split",
                "evaluation_population",
                "evaluation_split",
                "scientific_selection",
                "paper_evidence_eligible",
            },
            path=path,
        )
        _string(checked["run_id"], path=f"{path}.run_id", identifier=True)
        if (
            _integer(checked["seed"], path=f"{path}.seed", minimum=0) != seed
            or _integer(
                checked["seed_position"], path=f"{path}.seed_position", minimum=0
            )
            != offset
        ):
            raise Exp1bSchemaError(
                f"{path} is out of registered seed order."
            )
        if (
            _string(checked["method_id"], path=f"{path}.method_id") != METHOD_ID
            or _depth(checked["n"], path=f"{path}.n") != DEPLOYED_DEPTH_N
            or _integer(checked["K"], path=f"{path}.K") != BELLMAN_HORIZON_K
            or _number(checked["alpha"], path=f"{path}.alpha") != MIXTURE_ALPHA
            or _integer(
                checked["terminal_environment_interactions"],
                path=f"{path}.terminal_environment_interactions",
            )
            != TERMINAL_ENVIRONMENT_INTERACTIONS
            or _string(checked["training_split"], path=f"{path}.training_split")
            != "train"
            or _string(
                checked["evaluation_population"], path=f"{path}.evaluation_population"
            )
            != EVALUATION_POPULATION_ID
            or _string(checked["evaluation_split"], path=f"{path}.evaluation_split")
            != EVALUATION_SPLIT
        ):
            raise Exp1bSchemaError(f"{path} is not the registered reduced-study row.")
        _false(checked["scientific_selection"], path=f"{path}.scientific_selection")
        _true(
            checked["paper_evidence_eligible"],
            path=f"{path}.paper_evidence_eligible",
        )
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != UNITS:
        raise Exp1bSchemaError("exp1b_registry repeats a run ID.")
    return registry


def validate_exp1b_amendment(
    value: object,
    *,
    protocol_sha256: str | None = None,
    registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the Experiment 1B amendment that opens the reduced study."""

    amendment = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "amendment_id",
            "created_at_utc",
            "protocol_id",
            "protocol_sha256",
            "registry_sha256",
            "parent",
            "prior_amendment_history_sha256",
            "bridge_route_id",
            "validation_data_inspected",
            "test_data_opened",
            "outcome_evidence_inspected",
        },
        path="exp1b_amendment",
    )
    if (
        _string(amendment["schema_name"], path="exp1b_amendment.schema_name")
        != AMENDMENT_SCHEMA_NAME
        or _integer(
            amendment["schema_version"], path="exp1b_amendment.schema_version"
        )
        != AMENDMENT_SCHEMA_VERSION
        or _string(amendment["protocol_id"], path="exp1b_amendment.protocol_id")
        != PROTOCOL_ID
        or _string(
            amendment["bridge_route_id"], path="exp1b_amendment.bridge_route_id"
        )
        != BRIDGE_ROUTE_ID
    ):
        raise Exp1bSchemaError("exp1b_amendment identity differs.")
    _string(amendment["amendment_id"], path="exp1b_amendment.amendment_id", identifier=True)
    _timestamp(amendment["created_at_utc"], path="exp1b_amendment.created_at_utc")
    protocol_digest = _sha256(
        amendment["protocol_sha256"], path="exp1b_amendment.protocol_sha256"
    )
    registry_digest = _sha256(
        amendment["registry_sha256"], path="exp1b_amendment.registry_sha256"
    )
    if protocol_sha256 is not None and protocol_digest != protocol_sha256:
        raise Exp1bSchemaError("exp1b_amendment cites a different protocol digest.")
    if registry_sha256 is not None and registry_digest != registry_sha256:
        raise Exp1bSchemaError("exp1b_amendment cites a different registry digest.")
    _parent_binding(amendment["parent"], path="exp1b_amendment.parent")
    _sha256(
        amendment["prior_amendment_history_sha256"],
        path="exp1b_amendment.prior_amendment_history_sha256",
    )
    for field in (
        "validation_data_inspected",
        "test_data_opened",
        "outcome_evidence_inspected",
    ):
        _false(amendment[field], path=f"exp1b_amendment.{field}")
    return amendment


# --- the admission transition ------------------------------------------------
#
# The registered protocol carries two ``unavailable`` slots -- the Experiment 0
# base-policy artifact and the runtime authorization for this commit. Neither
# can be defaulted, inferred, or written by this repository. The transition that
# resolves them is a second, owner-signed amendment, defined here so the
# production boundary is executable and testable now, and so the shape of what
# the owner has to sign is not left implicit.
#
# The transition is strictly monotone: it may only move slots from unavailable
# to available, it must cite the exact protocol, registry, and prior-amendment
# digests it is amending, and it must name the purpose and execution-environment
# gate it is opening. Nothing here opens anything by itself: without the signed
# document, both runtime stages refuse.

ADMISSION_SCHEMA_NAME = "policy_improvement_exp1b_admission_v1"
ADMISSION_SCHEMA_VERSION = 1
#: The one purpose an Experiment 1B admission may open. Matching the launcher
#: purpose token means an authorization minted for a different program cannot be
#: replayed into this one.
EXECUTION_PURPOSE = "policy_improvement_exp1b_reduced_study"
EXECUTION_ENVIRONMENT_VARIABLE = "RUN_UPITRM_FULL_EXPERIMENTS"
EXECUTION_REQUIRED_VALUE = "1"
#: Exactly the slots an admission may flip, in the order it must list them.
ADMISSIBLE_SLOTS = ("base_policy_artifact", "runtime_authorization")
#: The two runtime roles Experiment 1B executes under. They are separate PARs
#: with separate source profiles, separate launcher invocations, and therefore
#: separate runtime digests: one admission that named a single ``runtime_sha256``
#: could authorize at most one of them, so the admission carries one
#: authorization per role and each stage presents its own.
FULL_ROLE = "policy-improvement-full"
THEORY_BRIDGE_ROLE = "policy-improvement-theory-bridge"
ADMISSION_ROLES = (FULL_ROLE, THEORY_BRIDGE_ROLE)


@dataclass(frozen=True)
class Exp1bRoleAuthorization:
    """One role's authorized runtime identity."""

    role: str
    runtime_authorization_sha256: str
    launcher_sha256: str
    runtime_sha256: str
    source_git_commit: str


@dataclass(frozen=True)
class Exp1bExecutionAdmission:
    """The resolved owner inputs, once the signed admission amendment exists."""

    amendment_id: str
    admission_sha256: str
    protocol_sha256: str
    registry_sha256: str
    prior_amendment_sha256: str
    purpose: str
    environment_variable: str
    required_value: str
    base_policy_checkpoint_sha256: str
    base_policy_model_state_sha256: str
    base_policy_amendment_sha256: str
    runtime_authorizations: Mapping[str, Exp1bRoleAuthorization]

    def authorization_for(self, role: str) -> Exp1bRoleAuthorization:
        authorization = self.runtime_authorizations.get(role)
        if authorization is None:
            raise Exp1bSchemaError(
                f"Experiment 1B admission authorizes no {role!r} runtime."
            )
        return authorization

    def require_execution_environment(
        self, environment: Mapping[str, str] | None = None
    ) -> None:
        """Refuse unless the admitted execution-environment gate is set.

        Reads the process environment by default. The v2 program uses the same
        variable for its own full-run gate; Experiment 1B does not widen it,
        redefine it, or supply a fallback value.
        """

        source = os.environ if environment is None else environment
        observed = source.get(self.environment_variable)
        if observed != self.required_value:
            raise Exp1bSchemaError(
                f"Experiment 1B execution requires {self.environment_variable}="
                f"{self.required_value}; it is {observed!r}."
            )

    def require_attestation(
        self,
        *,
        role: str,
        source_git_commit: str,
        runtime_sha256: str,
        launcher_sha256: str,
        runtime_authorization_sha256: str,
    ) -> None:
        """Refuse unless the live runtime is the one admitted **for this role**.

        The role is required, not inferred. Stage A runs the full PAR and Stage B
        runs the theory-bridge PAR; letting either present the other's
        authorization would make the two-PAR split cosmetic.
        """

        admitted = self.authorization_for(role)
        for name, expected, observed in (
            ("source_git_commit", admitted.source_git_commit, source_git_commit),
            ("runtime_sha256", admitted.runtime_sha256, runtime_sha256),
            ("launcher_sha256", admitted.launcher_sha256, launcher_sha256),
            (
                "runtime_authorization_sha256",
                admitted.runtime_authorization_sha256,
                runtime_authorization_sha256,
            ),
        ):
            if expected != observed:
                raise Exp1bSchemaError(
                    f"Experiment 1B admission authorized a different {name} "
                    f"for role {role!r}."
                )


def _available_slot(
    value: object, *, path: str, fields: tuple[str, ...]
) -> dict[str, Any]:
    slot = _exact_fields(value, {"status", "value"}, path=path)
    if slot["status"] != "available":
        raise Exp1bSchemaError(f"{path} must be an available owner input.")
    inner = _exact_fields(slot["value"], set(fields), path=f"{path}.value")
    for name in fields:
        if name == "source_git_commit":
            _git_commit(inner[name], path=f"{path}.value.{name}")
        else:
            _sha256(inner[name], path=f"{path}.value.{name}")
    return inner


def validate_exp1b_admission(
    value: object,
    *,
    protocol_sha256: str,
    registry_sha256: str,
    prior_amendment_sha256: str,
) -> Exp1bExecutionAdmission:
    """Validate the owner-signed amendment that admits Experiment 1B execution.

    Every digest argument is required, not optional: an admission is only
    meaningful relative to the exact documents it admits, and accepting one
    without checking what it was signed against is the whole failure mode this
    guards.
    """

    admission = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "amendment_id",
            "created_at_utc",
            "protocol_id",
            "protocol_sha256",
            "registry_sha256",
            "prior_amendment_sha256",
            "purpose",
            "execution_environment",
            "resolved_slots",
            "base_policy_artifact",
            "runtime_authorization",
            "execution_allowed",
        },
        path="exp1b_admission",
    )
    if (
        _string(admission["schema_name"], path="exp1b_admission.schema_name")
        != ADMISSION_SCHEMA_NAME
        or _integer(
            admission["schema_version"], path="exp1b_admission.schema_version"
        )
        != ADMISSION_SCHEMA_VERSION
        or _string(admission["protocol_id"], path="exp1b_admission.protocol_id")
        != PROTOCOL_ID
        or _string(admission["purpose"], path="exp1b_admission.purpose", identifier=True)
        != EXECUTION_PURPOSE
    ):
        raise Exp1bSchemaError("exp1b_admission identity differs.")
    _string(
        admission["amendment_id"], path="exp1b_admission.amendment_id", identifier=True
    )
    _timestamp(admission["created_at_utc"], path="exp1b_admission.created_at_utc")
    for field, expected in (
        ("protocol_sha256", protocol_sha256),
        ("registry_sha256", registry_sha256),
        ("prior_amendment_sha256", prior_amendment_sha256),
    ):
        if _sha256(admission[field], path=f"exp1b_admission.{field}") != expected:
            raise Exp1bSchemaError(
                f"exp1b_admission.{field} does not admit the document in hand."
            )
    gate = _exact_fields(
        admission["execution_environment"],
        {"environment_variable", "required_value"},
        path="exp1b_admission.execution_environment",
    )
    if (
        _string(
            gate["environment_variable"],
            path="exp1b_admission.execution_environment.environment_variable",
        )
        != EXECUTION_ENVIRONMENT_VARIABLE
        or _string(
            gate["required_value"],
            path="exp1b_admission.execution_environment.required_value",
        )
        != EXECUTION_REQUIRED_VALUE
    ):
        raise Exp1bSchemaError("exp1b_admission opens a different execution gate.")
    slots = _sequence(admission["resolved_slots"], path="exp1b_admission.resolved_slots")
    if tuple(slots) != ADMISSIBLE_SLOTS:
        raise Exp1bSchemaError(
            "exp1b_admission may resolve exactly the two registered slots, in order."
        )
    base = _available_slot(
        admission["base_policy_artifact"],
        path="exp1b_admission.base_policy_artifact",
        fields=("checkpoint_sha256", "model_state_sha256", "amendment_sha256"),
    )
    authorization_slot = _exact_fields(
        admission["runtime_authorization"],
        {"status", "value"},
        path="exp1b_admission.runtime_authorization",
    )
    if authorization_slot["status"] != "available":
        raise Exp1bSchemaError(
            "exp1b_admission.runtime_authorization must be an available owner input."
        )
    per_role = _exact_fields(
        authorization_slot["value"],
        set(ADMISSION_ROLES),
        path="exp1b_admission.runtime_authorization.value",
    )
    authorizations: dict[str, Exp1bRoleAuthorization] = {}
    seen_runtimes: set[str] = set()
    for role in ADMISSION_ROLES:
        path = f"exp1b_admission.runtime_authorization.value.{role}"
        entry = _exact_fields(
            per_role[role],
            {
                "authorization_sha256",
                "launcher_sha256",
                "runtime_sha256",
                "source_git_commit",
            },
            path=path,
        )
        for name in ("authorization_sha256", "launcher_sha256", "runtime_sha256"):
            _sha256(entry[name], path=f"{path}.{name}")
        _git_commit(entry["source_git_commit"], path=f"{path}.source_git_commit")
        runtime_digest = str(entry["runtime_sha256"])
        if runtime_digest in seen_runtimes:
            raise Exp1bSchemaError(
                "exp1b_admission authorizes one runtime artifact for both roles; "
                "the full and theory-bridge PARs are distinct binaries."
            )
        seen_runtimes.add(runtime_digest)
        authorizations[role] = Exp1bRoleAuthorization(
            role=role,
            runtime_authorization_sha256=str(entry["authorization_sha256"]),
            launcher_sha256=str(entry["launcher_sha256"]),
            runtime_sha256=runtime_digest,
            source_git_commit=str(entry["source_git_commit"]),
        )
    _true(admission["execution_allowed"], path="exp1b_admission.execution_allowed")
    return Exp1bExecutionAdmission(
        amendment_id=str(admission["amendment_id"]),
        admission_sha256=exp1b_document_sha256(admission),
        protocol_sha256=protocol_sha256,
        registry_sha256=registry_sha256,
        prior_amendment_sha256=prior_amendment_sha256,
        purpose=EXECUTION_PURPOSE,
        environment_variable=EXECUTION_ENVIRONMENT_VARIABLE,
        required_value=EXECUTION_REQUIRED_VALUE,
        base_policy_checkpoint_sha256=str(base["checkpoint_sha256"]),
        base_policy_model_state_sha256=str(base["model_state_sha256"]),
        base_policy_amendment_sha256=str(base["amendment_sha256"]),
        runtime_authorizations=MappingProxyType(authorizations),
    )


def apply_exp1b_admission(
    protocol: Mapping[str, Any], admission: Exp1bExecutionAdmission
) -> dict[str, Any]:
    """Return the effective protocol with both blocked slots resolved.

    The registered document on disk is never rewritten. This is the transition
    applied in memory at runtime, so the on-disk protocol keeps recording that
    the study was registered execution-blocked, and the admission amendment
    remains the single signed record of what opened it.
    """

    if not isinstance(admission, Exp1bExecutionAdmission):
        raise Exp1bSchemaError("Experiment 1B admission must be validated first.")
    checked = validate_exp1b_protocol(protocol)
    if exp1b_document_sha256(protocol) != admission.protocol_sha256:
        raise Exp1bSchemaError(
            "Experiment 1B admission does not admit this protocol revision."
        )
    effective = dict(checked)
    effective["base_policy_artifact"] = {
        "status": "available",
        "value": {
            "amendment_sha256": admission.base_policy_amendment_sha256,
            "checkpoint_sha256": admission.base_policy_checkpoint_sha256,
            "model_state_sha256": admission.base_policy_model_state_sha256,
        },
    }
    effective["runtime_authorization"] = {
        "status": "available",
        "value": {
            role: {
                "authorization_sha256": item.runtime_authorization_sha256,
                "launcher_sha256": item.launcher_sha256,
                "runtime_sha256": item.runtime_sha256,
                "source_git_commit": item.source_git_commit,
            }
            for role, item in sorted(admission.runtime_authorizations.items())
        },
    }
    effective["execution_gate"] = {
        "environment_variable": admission.environment_variable,
        "execution_allowed": True,
        "required_value": admission.required_value,
    }
    effective["amendments"] = [admission.admission_sha256]
    return effective




def _checkpoint_identity(
    value: object,
    *,
    path: str,
    registry_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One sealed budget-final checkpoint identity, strictly typed.

    ``registry_rows`` binds the run ID to an exact committed registry row. The
    review's reproducer — provenance with repeated run IDs accepted because only
    the file digests were unique — is closed by that binding plus the caller's
    uniqueness check on both fields.
    """

    identity = _exact_fields(
        value,
        {
            "run_id",
            "seed",
            "seed_position",
            "environment_interactions",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "model_state_sha256",
            "sealed",
        },
        path=path,
    )
    run_id = _string(identity["run_id"], path=f"{path}.run_id", identifier=True)
    seed = _integer(identity["seed"], path=f"{path}.seed", minimum=0)
    position = _integer(
        identity["seed_position"], path=f"{path}.seed_position", minimum=0
    )
    if (
        _integer(
            identity["environment_interactions"],
            path=f"{path}.environment_interactions",
        )
        != TERMINAL_ENVIRONMENT_INTERACTIONS
    ):
        raise Exp1bSchemaError(f"{path} is not the budget-final checkpoint.")
    _sha256(identity["checkpoint_sha256"], path=f"{path}.checkpoint_sha256")
    _integer(
        identity["checkpoint_size_bytes"],
        path=f"{path}.checkpoint_size_bytes",
        minimum=1,
    )
    _sha256(identity["model_state_sha256"], path=f"{path}.model_state_sha256")
    _true(identity["sealed"], path=f"{path}.sealed")

    if registry_rows is not None:
        row = registry_rows.get(run_id)
        if row is None:
            raise Exp1bSchemaError(
                f"{path}.run_id is not a committed Experiment 1B registry row."
            )
        if row["seed"] != seed or row["seed_position"] != position:
            raise Exp1bSchemaError(
                f"{path} does not match its registry row's seed and position."
            )
    return identity


def registry_rows_by_run_id(
    registry: Mapping[str, object],
) -> dict[str, Mapping[str, Any]]:
    """Index a validated Experiment 1B registry by run ID."""

    rows = registry.get("rows")
    if not isinstance(rows, Sequence):
        raise Exp1bSchemaError("Experiment 1B registry has no rows.")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        assert isinstance(row, Mapping)
        run_id = str(row["run_id"])
        if run_id in indexed:
            raise Exp1bSchemaError("Experiment 1B registry repeats a run ID.")
        indexed[run_id] = row
    return indexed


# --- the five required secondary diagnostics ---------------------------------
#
# ``EXPERIMENT_1A_READINESS.md:250`` requires target lag, centering/parity,
# mixture identity, deployment mismatch, and persistent-state checks to travel as
# **separate fields**, not folded into the signed-gap arithmetic. None of these is
# invented here: each mirrors a quantity the policy-improvement-v2 theory bridge
# already computes, cited below, so Experiment 1B reports the same thing under the
# same name rather than a second definition of it.
#
#   centering_parity       -> `constructed_centering_roundoff` /
#                             `training_estimator_parity_max_abs_error`
#                             (scripts/policy_improvement_theory_bridge_v2.py:1139-1143)
#   mixture_identity       -> `exact_mixture_deployment_identity_tv`
#                             (scripts/policy_improvement_theory_bridge_v2.py:1105-1121)
#   deployment_mismatch    -> `candidate_current_tv` / `deployment_discrepancy_delta_dep`
#                             (scripts/policy_improvement_theory_bridge_v2.py:1104)
#   persistent_state       -> `persistent_endpoint_witness`
#                             (policy_improvement_full_backend.py:3318-3370)
#
# ``target_lag`` is the exception and is labelled as such. ``propagated_target_lag``
# exists in the v1 result inventory (scripts/policy_improvement_schema.py:2770) as a
# *name only*: every producer emits it as
# ``_unavailable("not_collected_by_registered_protocol")``. No file in this
# repository computes it. The semantics used here -- the online value head against
# the EMA target head at the same state, latent, and depth -- follow the trainer's
# own target construction (rl/upi_trm_trainer.py:503-539), but the functional form
# is a construction, not a citation. It therefore carries an explicit ``kind``
# literal and **no tolerance**: it is recorded and audited for shape, never used to
# refuse a seed, because this repository has no registered threshold for it.
SECONDARY_DIAGNOSTIC_SCHEMA_NAME = "policy_improvement_exp1b_secondary_diagnostics_v1"
SECONDARY_DIAGNOSTIC_SCHEMA_VERSION = 1

TARGET_LAG_KIND = "ema_target_head_endpoint_lag_v1"
CENTERING_PARITY_KIND = "exact_statewise_centering_parity_v1"
MIXTURE_IDENTITY_KIND = "exact_mixture_identity"
DEPLOYMENT_MISMATCH_KIND = "deployment_discrepancy_nonselection_diagnostic"
PERSISTENT_STATE_KIND = "persistent_endpoint_carry_v1"

#: Not invented. Both are the frozen v2 theory amendment's own values, so
#: Experiment 1B refuses at exactly the threshold the parent program refuses at:
#:   configs/policy_improvement_v2/amendments/theory_bridge_v2.json
#:     deployment_contract.exact_mixture_identity_tv_absolute_tolerance = 1e-06
#:     centering_contract.constructed_centering_roundoff_absolute_tolerance = 1e-06
MIXTURE_IDENTITY_TOLERANCE = 1e-06
CENTERING_PARITY_TOLERANCE = 1e-06
#: Also the frozen v2 amendment's own literal:
#:   centering_contract.trainer_reconstruction
TRAINER_RECONSTRUCTION_CONTRACT = (
    "exact_frozen_checkpoint_advantage_tensor_with_registered_action_mask_"
    "clipping_and_recentering"
)

SECONDARY_DIAGNOSTIC_NAMES = (
    "target_lag",
    "centering_parity",
    "mixture_identity",
    "deployment_mismatch",
    "persistent_state",
)

_SECONDARY_FIELDS: dict[str, tuple[str, ...]] = {
    "target_lag": (
        "kind",
        "maximum_absolute_target_lag",
        "target_lag_witness_state_id",
        "deployed_depth",
        "reference_depth",
        "target_ema_tau",
        "state_count",
        "folded_into_signed_gap",
    ),
    "centering_parity": (
        "kind",
        "constructed_centering_roundoff",
        "constructed_centering_tolerance",
        "centering_witness_state_id",
        "centering_scheme",
        "state_count",
        "folded_into_signed_gap",
        # Finding 1: the v2 oracle compares an independently constructed
        # advantage tensor with the trainer's own reconstruction. Without these
        # four, a regression in the trainer's masking, clipping, or centering
        # could not affect the check.
        "training_estimator_centering_defect",
        "training_estimator_parity_max_abs_error",
        "training_estimator_parity_tolerance",
        "parity_witness_state_id",
        "centering_defect_witness_state_id",
        "clipping_kind",
        "clip_value",
        "trainer_reconstruction",
    ),
    "mixture_identity": (
        "kind",
        "maximum_identity_total_variation",
        "identity_tolerance",
        "identity_witness_state_id",
        "mixture_alpha",
        "policy_epsilon",
        "identity_holds",
        "state_count",
        "deployed_reconstructed_from_mixture",
        "operator_used_base_probabilities",
    ),
    "deployment_mismatch": (
        "kind",
        "maximum_candidate_base_total_variation",
        "mismatch_witness_state_id",
        "maximum_normalization_mass_error",
        "state_count",
        "folded_into_signed_gap",
    ),
    "persistent_state": (
        "kind",
        "latent_mode",
        "deployed_transition_depth",
        "endpoint_depths",
        "carried_successor_latent_sha256",
        "action_probabilities_sha256",
        "states_with_carried_latent",
        "state_count",
    ),
}


def _secondary_real(value: object, *, path: str) -> float:
    """A secondary-diagnostic real must be a binary64 float, not an int.

    ``_number`` accepts ``int`` because several registered protocol scalars are
    written as integers. These are not: they are measured quantities that
    canonical JSON renders differently for ``0`` and ``0.0``, so a document
    carrying the integer form would hash differently than the one that was
    audited.
    """

    if isinstance(value, bool) or not isinstance(value, float):
        raise Exp1bSchemaError(f"{path} must be a binary64 float.")
    if not math.isfinite(value):
        raise Exp1bSchemaError(f"{path} must be finite.")
    return value


def _secondary_common(block: Mapping[str, Any], *, path: str, kind: str) -> None:
    """The shape every secondary record shares, plus the contract tripwires."""

    if _string(block["kind"], path=f"{path}.kind", identifier=False) != kind:
        raise Exp1bSchemaError(f"{path} is not the registered {kind!r} diagnostic.")
    if (
        _integer(block["state_count"], path=f"{path}.state_count", minimum=1)
        != EVALUATION_RECORD_COUNT
    ):
        raise Exp1bSchemaError(
            f"{path} did not cover the registered census."
        )


def _secondary_target_lag(block: Mapping[str, Any], *, path: str) -> None:
    _secondary_common(block, path=path, kind=TARGET_LAG_KIND)
    lag = _secondary_real(
        block["maximum_absolute_target_lag"], path=f"{path}.maximum_absolute_target_lag"
    )
    if lag < 0.0:
        raise Exp1bSchemaError(f"{path} target lag must be nonnegative.")
    _string(block["target_lag_witness_state_id"], path=f"{path}.witness")
    if (
        _depth(block["deployed_depth"], path=f"{path}.deployed_depth")
        != DEPLOYED_DEPTH_N
        or _depth(block["reference_depth"], path=f"{path}.reference_depth")
        != REFERENCE_DEPTH_M
    ):
        raise Exp1bSchemaError(f"{path} was not measured at the registered depths.")
    tau = _secondary_real(block["target_ema_tau"], path=f"{path}.target_ema_tau")
    if not 0.0 <= tau <= 1.0:
        raise Exp1bSchemaError(f"{path} target EMA tau is outside [0, 1].")
    # No tolerance, deliberately: no registered threshold for target lag exists.
    _false(block["folded_into_signed_gap"], path=f"{path}.folded_into_signed_gap")


def _secondary_centering(block: Mapping[str, Any], *, path: str) -> None:
    _secondary_common(block, path=path, kind=CENTERING_PARITY_KIND)
    _false(block["folded_into_signed_gap"], path=f"{path}.folded_into_signed_gap")
    roundoff = _secondary_real(
        block["constructed_centering_roundoff"], path=f"{path}.roundoff"
    )
    tolerance = _secondary_real(
        block["constructed_centering_tolerance"], path=f"{path}.tolerance"
    )
    if roundoff < 0.0:
        raise Exp1bSchemaError(f"{path} centering roundoff must be nonnegative.")
    if tolerance != CENTERING_PARITY_TOLERANCE:
        raise Exp1bSchemaError(f"{path} uses an unregistered centering tolerance.")
    if roundoff > tolerance:
        raise Exp1bSchemaError(
            f"{path} exact-centering parity exceeds its registered tolerance."
        )
    _string(block["centering_witness_state_id"], path=f"{path}.witness")
    if (
        _string(block["centering_scheme"], path=f"{path}.centering_scheme")
        != "exact_statewise"
    ):
        raise Exp1bSchemaError(f"{path} is not the registered centering scheme.")

    # --- the independent-versus-trainer comparison, ported from v2 -----------
    defect = _secondary_real(
        block["training_estimator_centering_defect"], path=f"{path}.trainer_defect"
    )
    parity = _secondary_real(
        block["training_estimator_parity_max_abs_error"], path=f"{path}.parity"
    )
    parity_tolerance = _secondary_real(
        block["training_estimator_parity_tolerance"], path=f"{path}.parity_tolerance"
    )
    if defect < 0.0 or parity < 0.0:
        raise Exp1bSchemaError(f"{path} trainer-estimator errors must be nonnegative.")
    if parity_tolerance != CENTERING_PARITY_TOLERANCE:
        raise Exp1bSchemaError(f"{path} uses an unregistered parity tolerance.")
    if parity > parity_tolerance:
        raise Exp1bSchemaError(
            f"{path} trainer and independently constructed advantage tensors "
            "differ beyond tolerance."
        )
    if defect > parity_tolerance:
        raise Exp1bSchemaError(
            f"{path} trainer exact-advantage centering defect exceeds tolerance."
        )
    _string(block["parity_witness_state_id"], path=f"{path}.parity_witness")
    _string(block["centering_defect_witness_state_id"], path=f"{path}.defect_witness")
    # The clipping identity, so a trainer that silently stopped clipping, or
    # started, is a schema failure rather than an invisible change.
    kind = _string(block["clipping_kind"], path=f"{path}.clipping_kind")
    if kind not in {"none", "clip_then_exact_recenter"}:
        raise Exp1bSchemaError(f"{path} names an unregistered clipping kind.")
    clip_value = block["clip_value"]
    if kind == "none":
        if clip_value is not None:
            raise Exp1bSchemaError(f"{path} unclipped estimator named a clip value.")
    else:
        if _secondary_real(clip_value, path=f"{path}.clip_value") <= 0.0:
            raise Exp1bSchemaError(f"{path} clip value must be positive.")
    if (
        _string(block["trainer_reconstruction"], path=f"{path}.trainer_reconstruction")
        != TRAINER_RECONSTRUCTION_CONTRACT
    ):
        raise Exp1bSchemaError(f"{path} is not the registered reconstruction contract.")


def _secondary_mixture(block: Mapping[str, Any], *, path: str) -> None:
    _secondary_common(block, path=path, kind=MIXTURE_IDENTITY_KIND)
    if _secondary_real(block["policy_epsilon"], path=f"{path}.policy_epsilon") != 0.0:
        raise Exp1bSchemaError(
            f"{path} was measured under epsilon-greedy exploration; the registered "
            "method deploys the exact mixture with policy_epsilon = 0."
        )
    # Two tripwires on the finding-2 contract itself. The deployed law must have
    # been read off the deployed policy, not rebuilt from base and candidate --
    # otherwise the identity check tests this code's arithmetic. And the Bellman
    # operator must have used the frozen base law, which is the defect the third
    # review found.
    _false(
        block["deployed_reconstructed_from_mixture"],
        path=f"{path}.deployed_reconstructed_from_mixture",
    )
    _true(
        block["operator_used_base_probabilities"],
        path=f"{path}.operator_used_base_probabilities",
    )
    observed = _secondary_real(
        block["maximum_identity_total_variation"], path=f"{path}.identity_tv"
    )
    tolerance = _secondary_real(block["identity_tolerance"], path=f"{path}.tolerance")
    if observed < 0.0:
        raise Exp1bSchemaError(f"{path} total variation must be nonnegative.")
    if tolerance != MIXTURE_IDENTITY_TOLERANCE:
        raise Exp1bSchemaError(f"{path} uses an unregistered identity tolerance.")
    if _secondary_real(block["mixture_alpha"], path=f"{path}.mixture_alpha") != MIXTURE_ALPHA:
        raise Exp1bSchemaError(f"{path} was measured at a different mixture alpha.")
    _true(block["identity_holds"], path=f"{path}.identity_holds")
    if observed > tolerance:
        raise Exp1bSchemaError(
            f"{path} exact pointwise mixture identity failed."
        )
    _string(block["identity_witness_state_id"], path=f"{path}.witness")


def _secondary_deployment(block: Mapping[str, Any], *, path: str) -> None:
    _secondary_common(block, path=path, kind=DEPLOYMENT_MISMATCH_KIND)
    _false(block["folded_into_signed_gap"], path=f"{path}.folded_into_signed_gap")
    value = _secondary_real(
        block["maximum_candidate_base_total_variation"], path=f"{path}.tv"
    )
    if not 0.0 <= value <= 1.0:
        raise Exp1bSchemaError(f"{path} total variation is outside [0, 1].")
    mass = _secondary_real(
        block["maximum_normalization_mass_error"], path=f"{path}.mass_error"
    )
    if mass < 0.0:
        raise Exp1bSchemaError(f"{path} normalization mass error must be nonnegative.")
    _string(block["mismatch_witness_state_id"], path=f"{path}.witness")


def _secondary_persistent(block: Mapping[str, Any], *, path: str) -> None:
    _secondary_common(block, path=path, kind=PERSISTENT_STATE_KIND)
    if (
        _string(block["latent_mode"], path=f"{path}.latent_mode", identifier=True)
        != "persistent"
    ):
        raise Exp1bSchemaError(
            f"{path} must record the registered persistent latent mode."
        )
    if (
        _depth(block["deployed_transition_depth"], path=f"{path}.transition_depth")
        != DEPLOYED_DEPTH_N
    ):
        raise Exp1bSchemaError(f"{path} transition depth is not the deployed depth.")
    depths = _sequence(block["endpoint_depths"], path=f"{path}.endpoint_depths")
    if tuple(
        _depth(item, path=f"{path}.endpoint_depths[{offset}]")
        for offset, item in enumerate(depths)
    ) != (DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M):
        raise Exp1bSchemaError(f"{path} endpoint depths are not q=2 and q=8.")
    for name in ("carried_successor_latent_sha256", "action_probabilities_sha256"):
        _sha256(block[name], path=f"{path}.{name}")
    if (
        _integer(block["states_with_carried_latent"], path=f"{path}.carried", minimum=0)
        != EVALUATION_RECORD_COUNT
    ):
        raise Exp1bSchemaError(
            f"{path} did not carry the deployed successor latent at every state."
        )


_SECONDARY_VALIDATORS = {
    "target_lag": _secondary_target_lag,
    "centering_parity": _secondary_centering,
    "mixture_identity": _secondary_mixture,
    "deployment_mismatch": _secondary_deployment,
    "persistent_state": _secondary_persistent,
}


def validate_secondary_diagnostics(value: object, *, path: str) -> dict[str, Any]:
    """Validate one seed's five secondary diagnostic records.

    Separate records with separate names and separate tolerances. They are never
    reduced into the signed gap: the primary arithmetic must stay exactly the
    maxima-and-bounds contract in
    ``scripts/policy_improvement_exp1_diagnostics.py``, and a secondary check
    that failed must refuse the seed rather than perturb a published number.
    """

    block = _exact_fields(
        value,
        {"schema_name", "schema_version", *SECONDARY_DIAGNOSTIC_NAMES},
        path=path,
    )
    if (
        _string(block["schema_name"], path=f"{path}.schema_name")
        != SECONDARY_DIAGNOSTIC_SCHEMA_NAME
        or _integer(block["schema_version"], path=f"{path}.schema_version")
        != SECONDARY_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise Exp1bSchemaError(f"{path} schema differs.")
    for name in SECONDARY_DIAGNOSTIC_NAMES:
        record = _exact_fields(
            block[name], set(_SECONDARY_FIELDS[name]), path=f"{path}.{name}"
        )
        _SECONDARY_VALIDATORS[name](record, path=f"{path}.{name}")
    return block


def role_attestation_document(
    *, role: str, attestation: Mapping[str, str]
) -> dict[str, Any]:
    """Build one role-tagged attestation record from a runtime attestation."""

    if role not in ADMISSION_ROLES:
        raise Exp1bSchemaError(f"{role!r} is not a registered Experiment 1B role.")
    required = {
        "source_git_commit",
        "runtime_sha256",
        "launcher_sha256",
        "runtime_authorization_sha256",
    }
    if not isinstance(attestation, Mapping) or set(attestation) != required:
        raise Exp1bSchemaError("Role attestation inventory differs.")
    return {"role": role, **{name: str(attestation[name]) for name in sorted(required)}}


def attestation_matches_authorization(
    attestation: Mapping[str, Any], authorization: Exp1bRoleAuthorization
) -> bool:
    """Whether a role-tagged attestation is exactly the one admitted for its role."""

    return (
        str(attestation.get("role")) == authorization.role
        and str(attestation.get("source_git_commit")) == authorization.source_git_commit
        and str(attestation.get("runtime_sha256")) == authorization.runtime_sha256
        and str(attestation.get("launcher_sha256")) == authorization.launcher_sha256
        and str(attestation.get("runtime_authorization_sha256"))
        == authorization.runtime_authorization_sha256
    )


def validate_substitute_provenance(
    value: object,
    *,
    registry_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the D1 substitute provenance that authorises the 1B route.

    This is the document that stands in for the Stage 1 ``V_select`` binding the
    v2 route requires. It must be complete: every field below is mandatory, and
    an incomplete or internally inconsistent record refuses the route rather
    than degrading it.

    Schema validity is necessary, not sufficient. The values here are *claims*
    until :mod:`scripts.policy_improvement_exp1b_bridge` compares them against
    stable-file digests and runtime attestation. Pass ``registry_rows`` to bind
    every checkpoint run ID to a committed registry row.
    """

    record = _exact_fields(
        value,
        {
            "schema_name",
            "schema_version",
            "route_id",
            "protocol_id",
            "exp1b_protocol_sha256",
            "exp1b_registry_sha256",
            "exp1b_amendment_sha256",
            "admission_sha256",
            "generation_sha256",
            "run_manifest_sha256s",
            "parent",
            "ordered_population_sha256",
            "population_binding_sha256",
            "seed_ids",
            "base_policy_artifact",
            "sealed_checkpoints",
            "producer_attestation",
            "request_schedule_sha256",
            "access_state",
            "scientific_selection",
            "validation_select_access",
            "test_access",
        },
        path="exp1b_provenance",
    )
    if (
        _string(record["schema_name"], path="exp1b_provenance.schema_name")
        != PROVENANCE_SCHEMA_NAME
        or _integer(
            record["schema_version"], path="exp1b_provenance.schema_version"
        )
        != PROVENANCE_SCHEMA_VERSION
        or _string(record["route_id"], path="exp1b_provenance.route_id")
        != BRIDGE_ROUTE_ID
        or _string(record["protocol_id"], path="exp1b_provenance.protocol_id")
        != PROTOCOL_ID
    ):
        raise Exp1bSchemaError("exp1b_provenance identity differs.")
    for field in (
        "exp1b_protocol_sha256",
        "exp1b_registry_sha256",
        "exp1b_amendment_sha256",
        # The signed admission that opened execution, and the one evidence
        # generation this provenance belongs to. Binding the generation here is
        # what stops the same provenance and checkpoints from being copied into
        # a second caller-named directory and running the state machine twice.
        "admission_sha256",
        "generation_sha256",
        "ordered_population_sha256",
        "population_binding_sha256",
        "request_schedule_sha256",
    ):
        _sha256(record[field], path=f"exp1b_provenance.{field}")
    _parent_binding(record["parent"], path="exp1b_provenance.parent")
    # The producer identity, role-scoped. Stage A writes it; Stage B validates it
    # against the admission's ``policy-improvement-full`` authorization rather
    # than against its own live identity, which is a *different* PAR and is
    # required by the admission schema to have a different runtime digest.
    _role_attestation(
        record["producer_attestation"],
        path="exp1b_provenance.producer_attestation",
        expected_role=FULL_ROLE,
    )

    manifests = _sequence(
        record["run_manifest_sha256s"], path="exp1b_provenance.run_manifest_sha256s"
    )
    if len(manifests) != UNITS:
        raise Exp1bSchemaError(
            f"exp1b_provenance must bind exactly {UNITS} run-manifest digests."
        )
    for offset, item in enumerate(manifests):
        _sha256(item, path=f"exp1b_provenance.run_manifest_sha256s[{offset}]")
    if len(set(manifests)) != UNITS:
        raise Exp1bSchemaError("exp1b_provenance repeats a run-manifest digest.")

    seed_ids = _sequence(record["seed_ids"], path="exp1b_provenance.seed_ids")
    if (
        tuple(
            _integer(item, path=f"exp1b_provenance.seed_ids[{offset}]", minimum=0)
            for offset, item in enumerate(seed_ids)
        )
        != REGISTERED_SEEDS
    ):
        raise Exp1bSchemaError(
            "exp1b_provenance must bind all eight registered seed IDs in order."
        )

    base = _exact_fields(
        record["base_policy_artifact"],
        {
            "initialization_kind",
            "amendment_sha256",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "model_state_sha256",
            "architecture_sha256",
            "producer_git_commit",
            "producer_source_manifest_sha256",
            "training_data_sha256",
            "training_procedure_sha256",
            "shared_across_seeds",
            "not_selected_by_validation_or_test",
        },
        path="exp1b_provenance.base_policy_artifact",
    )
    _string(
        base["initialization_kind"],
        path="exp1b_provenance.base_policy_artifact.initialization_kind",
        identifier=True,
    )
    for field in (
        "amendment_sha256",
        "checkpoint_sha256",
        "model_state_sha256",
        "architecture_sha256",
        "producer_source_manifest_sha256",
        "training_data_sha256",
        "training_procedure_sha256",
    ):
        _sha256(base[field], path=f"exp1b_provenance.base_policy_artifact.{field}")
    _integer(
        base["checkpoint_size_bytes"],
        path="exp1b_provenance.base_policy_artifact.checkpoint_size_bytes",
        minimum=1,
    )
    _git_commit(
        base["producer_git_commit"],
        path="exp1b_provenance.base_policy_artifact.producer_git_commit",
    )
    _true(
        base["shared_across_seeds"],
        path="exp1b_provenance.base_policy_artifact.shared_across_seeds",
    )
    _true(
        base["not_selected_by_validation_or_test"],
        path=(
            "exp1b_provenance.base_policy_artifact."
            "not_selected_by_validation_or_test"
        ),
    )

    checkpoints = _sequence(
        record["sealed_checkpoints"], path="exp1b_provenance.sealed_checkpoints"
    )
    if len(checkpoints) != UNITS:
        raise Exp1bSchemaError(
            f"exp1b_provenance must bind exactly {UNITS} sealed checkpoints; "
            "a partial run set is refused."
        )
    seen_digests: set[str] = set()
    seen_run_ids: set[str] = set()
    for offset, (item, seed) in enumerate(zip(checkpoints, REGISTERED_SEEDS)):
        identity = _checkpoint_identity(
            item,
            path=f"exp1b_provenance.sealed_checkpoints[{offset}]",
            registry_rows=registry_rows,
        )
        if identity["seed"] != seed or identity["seed_position"] != offset:
            raise Exp1bSchemaError(
                f"exp1b_provenance.sealed_checkpoints[{offset}] is out of "
                "registered seed order."
            )
        if identity["checkpoint_sha256"] in seen_digests:
            raise Exp1bSchemaError(
                "exp1b_provenance repeats a sealed checkpoint digest across seeds."
            )
        if identity["run_id"] in seen_run_ids:
            raise Exp1bSchemaError(
                "exp1b_provenance repeats a run ID across seeds."
            )
        seen_digests.add(identity["checkpoint_sha256"])
        seen_run_ids.add(identity["run_id"])

    _string(
        record["access_state"],
        path="exp1b_provenance.access_state",
        choices=frozenset(ACCESS_STATES),
    )
    for field in ("scientific_selection", "validation_select_access", "test_access"):
        _false(record[field], path=f"exp1b_provenance.{field}")
    return record


# --------------------------------------------------------------------------
# Durable, transactional eight-request schedule
# --------------------------------------------------------------------------

SCHEDULE_STATE_SCHEMA_NAME = "policy_improvement_exp1b_schedule_state_v1"
SCHEDULE_STATE_SCHEMA_VERSION = 1

#: Per-slot lifecycle. ``claimed`` is the crash-visible intermediate state.
SLOT_STATES = ("pending", "claimed", "served")


class Exp1bScheduleStateError(Exp1bSchemaError):
    """Raised when durable schedule state is missing, stale, or inconsistent."""


@dataclass(frozen=True)
class _ScheduleSlot:
    seed_position: int
    seed: int
    run_id: str
    checkpoint_sha256: str
    state: str = "pending"
    attempts: int = 0
    #: Owner of a live claim, and the epoch that claim was taken in.
    #:
    #: The owner string alone cannot distinguish a live process from a corpse,
    #: which is why a dead claim used to wedge its slot forever. Liveness is now
    #: decided by whether the slot's ``flock`` lease can be acquired; the owner
    #: and epoch exist so that a *stale* holder cannot finalize a claim that has
    #: since been reclaimed by someone else. The epoch increases on every claim
    #: and never resets, and ``attempts`` is preserved across reclaims.
    claim_owner: str | None = None
    claim_epoch: int = 0


class Exp1bRequestSchedule:
    """The eight-request state machine: durable, unique, and serialized.

    Enforces, per owner decision D1:

    * the route opens only once **all eight** budget-final checkpoints are
      sealed — a partial run set never opens it;
    * exactly eight requests, one per registered seed, in registered seed order;
    * a duplicate, missing, extra, or out-of-order request is refused;
    * a request is served at most once — a post-emission retry is refused
      **permanently**, across processes;
    * both ``q=2`` and ``q=8`` are computed in the same request.

    Uniqueness, serialization, durability
    -------------------------------------
    The state artifact is **derived** from an owner-controlled evidence
    generation, not named by the caller, so the same provenance cannot run two
    independent state machines under two paths. It is created ``O_EXCL`` and
    thereafter updated in place under a lock.

    Every transition runs inside ``generation.exclusive()``, an interprocess
    ``flock``, and **reloads durable state after taking the lock**. Two writers
    that opened before either served can no longer both serve the same seed:
    the loser sees the winner's committed state and is refused. An atomic
    replace alone did not give that, because it does not serialize a
    read-modify-write.

    ``claim`` marks a slot ``claimed`` before any evaluator callback runs.
    The payload is then persisted ``O_EXCL`` under the generation, and only
    after it is durably on disk does ``finalize`` mark the slot ``served`` and
    record the payload digest. A crash between claim and finalize therefore
    leaves a recoverable, retryable attempt; a crash after finalize leaves a
    payload that any later process can reload.

    Pre-emission crash policy (explicit)
    ------------------------------------
    A ``claimed`` slot may be re-claimed: nothing was emitted, so no result
    depended on the partial attempt, and refusing would let one transient fault
    destroy the octet. Every attempt is counted durably and surfaced in
    :meth:`attempt_counts`. A ``served`` slot may never be re-claimed.
    """

    def __init__(
        self,
        provenance: Mapping[str, object],
        *,
        generation: Any,
        registry_rows: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        record = validate_substitute_provenance(
            provenance, registry_rows=registry_rows
        )
        checkpoints = record["sealed_checkpoints"]
        assert isinstance(checkpoints, list)
        for attribute in (
            "schedule_path",
            "exclusive",
            "write_payload",
            "read_payload",
            "has_payload",
            "claim_lease_path",
            "claim_lease_is_free",
            "generation_sha256",
        ):
            if not hasattr(generation, attribute):
                raise Exp1bScheduleStateError(
                    "Experiment 1B schedule requires an authenticated evidence "
                    "generation, not a caller-named state path."
                )
        # The provenance names the one generation it belongs to. Copying it and
        # the checkpoints into a second caller-named directory therefore cannot
        # open a second state machine: the digests will not agree.
        if record["generation_sha256"] != str(generation.generation_sha256):
            raise Exp1bScheduleStateError(
                "Substitute provenance was finalized for a different evidence "
                "generation; a copied generation does not open a second schedule."
            )
        self._generation = generation
        self._state_path = Path(generation.schedule_path)
        self._provenance = record
        self._binding = exp1b_document_sha256(record)
        # One live claim per process. `flock` is released across evaluation, so
        # the owner is what stops a second process from re-entering a slot that
        # is already being traversed.
        self._owner = f"pid-{os.getpid()}-{id(self):x}"
        self._slots: list[_ScheduleSlot] = [
            _ScheduleSlot(
                seed_position=int(item["seed_position"]),
                seed=int(item["seed"]),
                run_id=str(item["run_id"]),
                checkpoint_sha256=str(item["checkpoint_sha256"]),
            )
            for item in checkpoints
        ]
        self._payload_digests: dict[int, str] = {}
        with generation.exclusive():
            # Clear this module's own leftover links before classifying, so a
            # crash in the link/unlink window is a transient state rather than a
            # permanent one. The generation lock is already held here, so this
            # cannot delete a concurrent writer's staging temporary. Imported at
            # call time to avoid the module cycle.
            from scripts.policy_improvement_exp1b_evidence import reap_partial_links

            reap_partial_links(self._state_path.parent)
            reap_partial_links(generation.payload_directory)
            if self._state_path.exists():
                self._load()
            else:
                self._create()

    # -- durable state ----------------------------------------------------

    def _document(self) -> dict[str, Any]:
        return {
            "schema_name": SCHEDULE_STATE_SCHEMA_NAME,
            "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
            "route_id": BRIDGE_ROUTE_ID,
            "provenance_sha256": self._binding,
            "generation_sha256": str(self._generation.generation_sha256),
            "slots": [
                {
                    "seed_position": slot.seed_position,
                    "seed": slot.seed,
                    "run_id": slot.run_id,
                    "checkpoint_sha256": slot.checkpoint_sha256,
                    "state": slot.state,
                    "attempts": slot.attempts,
                    "claim_owner": slot.claim_owner,
                    "claim_epoch": slot.claim_epoch,
                    "payload_sha256": self._payload_digests.get(slot.seed_position),
                }
                for slot in self._slots
            ],
        }

    def _create(self) -> None:
        """First writer only, through the same staged install every artifact uses.

        The previous version opened the *final* pathname ``O_EXCL`` and did one
        unchecked ``os.write``. A crash in that window left a zero-byte or
        truncated ``schedule_state.json`` at the real name, and every later
        restart saw the pathname, called :meth:`_load`, and failed with "not
        JSON" forever: there was no state from which to recover.

        :func:`install_durable_artifact` writes to a staged temporary with a
        checked write loop, fsyncs it, installs it with ``os.link`` (no-replace,
        and the name only ever appears with all its bytes), removes the
        temporary, and fsyncs the parent. ``O_EXCL`` on the final name is still
        implied -- ``link`` fails if the destination exists -- so a racing
        creator still loses and is adopted.
        """

        # Imported at call time: the evidence module imports this one at module
        # scope, so a module-scope import here would be a cycle.
        from scripts.policy_improvement_exp1b_evidence import (
            Exp1bArtifactExistsError,
            Exp1bEvidenceError,
            install_durable_artifact,
        )

        payload = canonical_json_bytes(self._document()) + b"\n"
        try:
            install_durable_artifact(
                self._state_path, payload, label="Experiment 1B schedule state"
            )
        except Exp1bArtifactExistsError:
            # Another writer installed it between our check and here; adopt it.
            self._load()
            return
        except Exp1bEvidenceError as exc:
            raise Exp1bScheduleStateError("Schedule state cannot be created.") from exc

    def _fsync_parent(self) -> None:
        descriptor = os.open(self._state_path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _persist(self) -> None:
        """Update in place. The path never changes, so there is one artifact."""

        payload = canonical_json_bytes(self._document()) + b"\n"
        # Prefixed so `_reap_partials` actually removes a crash leftover; a
        # `.partial` *suffix* never matched its `.partial-` prefix test.
        temporary = self._state_path.with_name(
            f".partial-{self._state_path.name}.persist"
        )
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise Exp1bScheduleStateError("Schedule state write stalled.")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, self._state_path)
        self._fsync_parent()

    def _load(self) -> None:
        """Reload durable state strictly, then reconcile any crash orphan.

        Strict on purpose: the schedule is a durable authorization artifact, so
        it goes through the shared strict JSON loader and exact field
        inventories rather than ``json.loads`` plus ``.get``. Boolean and
        integral-float aliases compare equal to integers in Python, so a
        ``schema_version: true`` or ``seed: 2081976412.0`` document would
        otherwise load as valid.
        """

        try:
            raw = self._state_path.read_bytes()
        except OSError as exc:
            raise Exp1bScheduleStateError("Schedule state cannot be read.") from exc
        try:
            document = load_strict_json_bytes(raw)
        except PolicyImprovementSchemaError as exc:
            raise Exp1bScheduleStateError("Schedule state is not JSON.") from exc
        try:
            document = _exact_fields(
                document,
                {
                    "schema_name",
                    "schema_version",
                    "route_id",
                    "provenance_sha256",
                    "generation_sha256",
                    "slots",
                },
                path="exp1b_schedule",
            )
        except Exp1bSchemaError as exc:
            raise Exp1bScheduleStateError(str(exc)) from exc
        try:
            if (
                _string(document["schema_name"], path="exp1b_schedule.schema_name")
                != SCHEDULE_STATE_SCHEMA_NAME
                or _integer(
                    document["schema_version"], path="exp1b_schedule.schema_version"
                )
                != SCHEDULE_STATE_SCHEMA_VERSION
                or _string(document["route_id"], path="exp1b_schedule.route_id")
                != BRIDGE_ROUTE_ID
            ):
                raise Exp1bScheduleStateError("Schedule state schema differs.")
            if (
                _sha256(
                    document["provenance_sha256"],
                    path="exp1b_schedule.provenance_sha256",
                )
                != self._binding
            ):
                raise Exp1bScheduleStateError(
                    "Schedule state belongs to a different substitute provenance."
                )
            if (
                _sha256(
                    document["generation_sha256"],
                    path="exp1b_schedule.generation_sha256",
                )
                != str(self._generation.generation_sha256)
            ):
                raise Exp1bScheduleStateError(
                    "Schedule state belongs to a different evidence generation."
                )
            slots = _sequence(document["slots"], path="exp1b_schedule.slots")
        except Exp1bSchemaError as exc:
            raise Exp1bScheduleStateError(str(exc)) from exc
        if len(slots) != UNITS:
            raise Exp1bScheduleStateError("Schedule state has the wrong arity.")
        restored: list[_ScheduleSlot] = []
        digests: dict[int, str] = {}
        reconciled = False
        for offset, (item, slot) in enumerate(zip(slots, self._slots)):
            path = f"exp1b_schedule.slots[{offset}]"
            try:
                entry = _exact_fields(
                    item,
                    {
                        "seed_position",
                        "seed",
                        "run_id",
                        "checkpoint_sha256",
                        "state",
                        "attempts",
                        "claim_owner",
                        "claim_epoch",
                        "payload_sha256",
                    },
                    path=path,
                )
                position = _integer(entry["seed_position"], path=f"{path}.seed_position")
                seed = _integer(entry["seed"], path=f"{path}.seed")
                run_id = _string(entry["run_id"], path=f"{path}.run_id")
                checkpoint = _sha256(
                    entry["checkpoint_sha256"], path=f"{path}.checkpoint_sha256"
                )
                state = _string(
                    entry["state"], path=f"{path}.state", choices=frozenset(SLOT_STATES)
                )
                attempts = _integer(entry["attempts"], path=f"{path}.attempts", minimum=0)
                epoch = _integer(
                    entry["claim_epoch"], path=f"{path}.claim_epoch", minimum=0
                )
            except Exp1bSchemaError as exc:
                raise Exp1bScheduleStateError(str(exc)) from exc
            if (
                position != slot.seed_position
                or seed != slot.seed
                or run_id != slot.run_id
                or checkpoint != slot.checkpoint_sha256
            ):
                raise Exp1bScheduleStateError(
                    f"Schedule slot {offset} identity differs from its provenance."
                )
            owner = entry["claim_owner"]
            if owner is not None and (not isinstance(owner, str) or not owner):
                raise Exp1bScheduleStateError(
                    f"Schedule slot {offset} claim owner is invalid."
                )
            if state != "claimed" and owner is not None:
                raise Exp1bScheduleStateError(
                    f"Schedule slot {offset} is not claimed but names an owner."
                )
            digest = entry["payload_sha256"]
            if state == "served":
                if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                    raise Exp1bScheduleStateError(
                        f"Served schedule slot {offset} has no payload digest."
                    )
                if not self._generation.has_payload(slot.seed_position):
                    raise Exp1bScheduleStateError(
                        f"Served schedule slot {offset} has no persisted payload."
                    )
                digests[slot.seed_position] = digest
            elif digest is not None:
                raise Exp1bScheduleStateError(
                    f"Unserved schedule slot {offset} carries a payload digest."
                )
            elif self._generation.has_payload(slot.seed_position):
                # Crash window: the payload was durably written but the process
                # died before the served marker landed. The artifact is the
                # authority, so adopt it rather than deadlocking on a slot whose
                # O_EXCL payload can never be rewritten.
                recovered, recovered_digest = self._generation.read_payload(
                    slot.seed_position
                )
                if (
                    recovered.get("seed_position") != slot.seed_position
                    or recovered.get("seed") != slot.seed
                    or recovered.get("run_id") != slot.run_id
                    or recovered.get("checkpoint_sha256") != slot.checkpoint_sha256
                    or recovered.get("route_id") != BRIDGE_ROUTE_ID
                ):
                    raise Exp1bScheduleStateError(
                        f"Orphan payload {offset} does not match its schedule slot."
                    )
                state = "served"
                owner = None
                attempts = max(attempts, 1)
                digests[slot.seed_position] = recovered_digest
                reconciled = True
            restored.append(
                _ScheduleSlot(
                    seed_position=slot.seed_position,
                    seed=slot.seed,
                    run_id=slot.run_id,
                    checkpoint_sha256=slot.checkpoint_sha256,
                    state=state,
                    attempts=attempts,
                    claim_owner=owner,
                    claim_epoch=epoch,
                )
            )
        served = [offset for offset, slot in enumerate(restored) if slot.state == "served"]
        if served != list(range(len(served))):
            raise Exp1bScheduleStateError("Served schedule slots are not a prefix.")
        claimed = [offset for offset, slot in enumerate(restored) if slot.state == "claimed"]
        if len(claimed) > 1 or (claimed and claimed[0] != len(served)):
            raise Exp1bScheduleStateError(
                "Schedule state has a misplaced or duplicated claim."
            )
        self._slots = restored
        self._payload_digests = digests
        if reconciled:
            # Durable again before anything reads the adopted state.
            self._persist()

    def reload(self) -> None:
        """Refresh from durable state, taking the generation lock."""

        with self._generation.exclusive():
            self._load()

    def reload_locked(self) -> None:
        """Refresh from durable state when the caller already holds the lock.

        ``flock`` on a second descriptor of the same file blocks even within one
        process, because each ``os.open`` is a distinct open file description.
        A nested :meth:`reload` therefore deadlocks rather than recursing, so a
        caller already inside ``generation.exclusive()`` must use this.
        """

        self._load()

    # -- inspection -------------------------------------------------------

    @property
    def access_state(self) -> str:
        served = sum(1 for slot in self._slots if slot.state == "served")
        if served == UNITS:
            return "payload_emitted"
        if served or any(slot.state == "claimed" for slot in self._slots):
            return "bridge_open"
        return "sealed_octet_complete"

    @property
    def served_positions(self) -> tuple[int, ...]:
        return tuple(
            slot.seed_position for slot in self._slots if slot.state == "served"
        )

    def attempt_counts(self) -> tuple[int, ...]:
        """Attempts per seed position, including pre-emission retries."""

        return tuple(slot.attempts for slot in self._slots)

    def payload_digests(self) -> tuple[str, ...]:
        return tuple(
            self._payload_digests[slot.seed_position]
            for slot in self._slots
            if slot.state == "served"
        )

    def schedule_sha256(self) -> str:
        """Digest of the ordered schedule, bound into substitute provenance."""

        return exp1b_document_sha256(
            [
                {
                    "seed_position": slot.seed_position,
                    "seed": slot.seed,
                    "run_id": slot.run_id,
                    "checkpoint_sha256": slot.checkpoint_sha256,
                }
                for slot in self._slots
            ]
        )

    # -- transitions ------------------------------------------------------

    def _validate_request(
        self,
        *,
        seed_position: int,
        seed: int,
        run_id: str,
        checkpoint_sha256: str,
        evaluation_population: str,
        depths: Sequence[int],
        lease_acquired: bool = False,
    ) -> _ScheduleSlot:
        if evaluation_population != EVALUATION_POPULATION_ID:
            raise Exp1bSchemaError(
                "Experiment 1B bridge route serves only validation_bridge."
            )
        if tuple(depths) != (DEPLOYED_DEPTH_N, REFERENCE_DEPTH_M):
            raise Exp1bSchemaError(
                "Experiment 1B computes q=2 and q=8 in the same request."
            )
        if isinstance(seed_position, bool) or not isinstance(seed_position, int):
            raise Exp1bSchemaError("Request seed position must be an integer.")
        if not 0 <= seed_position < UNITS:
            raise Exp1bSchemaError("Request seed position is outside the octet.")
        slot = self._slots[seed_position]
        if slot.state == "served":
            raise Exp1bSchemaError(
                "Experiment 1B refuses a post-payload retry for a served seed."
            )
        if (
            slot.state == "claimed"
            and slot.claim_owner not in {None, self._owner}
            and not lease_acquired
        ):
            # Reclaiming another process's claim requires proof that it is dead,
            # and the only proof this module accepts is holding its lease. The
            # lease is an ``flock``, released by the kernel on process death, so
            # acquiring it means no live holder exists. A caller that reached
            # here without it is refused rather than allowed to guess.
            raise Exp1bSchemaError(
                f"Experiment 1B seed {seed_position} is claimed by "
                f"{slot.claim_owner!r}; reclaiming requires its evaluation lease."
            )
        expected = len(self.served_positions)
        if seed_position != expected:
            raise Exp1bSchemaError(
                f"Experiment 1B requests are served in registered seed order; "
                f"expected position {expected}, received {seed_position}."
            )
        if (
            slot.seed != seed
            or slot.run_id != run_id
            or slot.checkpoint_sha256 != checkpoint_sha256
        ):
            raise Exp1bSchemaError(
                "Request identity differs from its registered schedule slot."
            )
        return slot

    def claim(
        self,
        *,
        seed_position: int,
        seed: int,
        run_id: str,
        checkpoint_sha256: str,
        evaluation_population: str,
        depths: Sequence[int],
        lease_acquired: bool = False,
    ) -> _ScheduleSlot:
        """Phase one, under the lock: reload, validate, durably claim.

        The reload is what serializes concurrent writers. Without it, a writer
        that read state before a peer's commit would decide on stale slots.
        """

        with self._generation.exclusive():
            self._load()
            slot = self._validate_request(
                seed_position=seed_position,
                seed=seed,
                run_id=run_id,
                checkpoint_sha256=checkpoint_sha256,
                evaluation_population=evaluation_population,
                depths=depths,
                lease_acquired=lease_acquired,
            )
            reclaimed = (
                slot.state == "claimed" and slot.claim_owner not in {None, self._owner}
            )
            self._slots[seed_position] = _ScheduleSlot(
                seed_position=slot.seed_position,
                seed=slot.seed,
                run_id=slot.run_id,
                checkpoint_sha256=slot.checkpoint_sha256,
                state="claimed",
                # Preserved across a reclaim: a dead attempt still happened.
                attempts=slot.attempts + 1,
                claim_owner=self._owner,
                claim_epoch=slot.claim_epoch + 1,
            )
            _ = reclaimed  # recorded by the epoch bump; kept for readability
            self._persist()
            return self._slots[seed_position]

    @property
    def owner(self) -> str:
        return self._owner

    def finalize(
        self,
        *,
        seed_position: int,
        payload_document: Mapping[str, Any],
        claim_epoch: int | None = None,
    ) -> str:
        """Phase two, under the lock: persist the payload, then mark served.

        The payload is written ``O_EXCL`` **before** the slot changes state, so
        the served state is never reachable without a recoverable artifact.
        """

        if isinstance(seed_position, bool) or not isinstance(seed_position, int):
            raise Exp1bSchemaError("Finalized seed position must be an integer.")
        if not 0 <= seed_position < UNITS:
            raise Exp1bSchemaError("Finalized seed position is outside the octet.")
        with self._generation.exclusive():
            self._load()
            slot = self._slots[seed_position]
            if slot.state != "claimed":
                raise Exp1bSchemaError(
                    f"Experiment 1B cannot finalize a slot in state {slot.state!r}."
                )
            if slot.claim_owner != self._owner or (
                claim_epoch is not None and slot.claim_epoch != claim_epoch
            ):
                raise Exp1bSchemaError(
                    "Experiment 1B cannot finalize a claim that has since been "
                    "reclaimed by another process."
                )
            digest = self._generation.write_payload(seed_position, payload_document)
            _sha256(digest, path="payload SHA-256")
            self._slots[seed_position] = _ScheduleSlot(
                seed_position=slot.seed_position,
                seed=slot.seed,
                run_id=slot.run_id,
                checkpoint_sha256=slot.checkpoint_sha256,
                state="served",
                attempts=slot.attempts,
                claim_owner=None,
                claim_epoch=slot.claim_epoch,
            )
            self._payload_digests[seed_position] = digest
            self._persist()
            return digest

    def abandon_claim(
        self, *, seed_position: int, claim_epoch: int | None = None
    ) -> None:
        """Release a claim after a pre-emission failure, keeping the attempt."""

        with self._generation.exclusive():
            self._load()
            slot = self._slots[seed_position]
            if slot.state == "served":
                # `_load` reconciled a durable orphan payload into served state
                # while this process was failing. The artifact is the authority;
                # there is nothing left to abandon.
                return
            if slot.state != "claimed":
                raise Exp1bSchemaError("Only a claimed slot can be abandoned.")
            if slot.claim_owner != self._owner or (
                claim_epoch is not None and slot.claim_epoch != claim_epoch
            ):
                raise Exp1bSchemaError(
                    "Experiment 1B cannot abandon a claim that has since been "
                    "reclaimed by another process."
                )
            self._slots[seed_position] = _ScheduleSlot(
                seed_position=slot.seed_position,
                seed=slot.seed,
                run_id=slot.run_id,
                checkpoint_sha256=slot.checkpoint_sha256,
                state="pending",
                attempts=slot.attempts,
                claim_owner=None,
                claim_epoch=slot.claim_epoch,
            )
            self._persist()

    def require_complete(self) -> None:
        """Refuse an aggregate built from anything but the whole octet."""

        served = self.served_positions
        if len(served) != UNITS:
            raise Exp1bSchemaError(
                f"Experiment 1B aggregation requires all {UNITS} requests; "
                f"{len(served)} were served."
            )
        if served != tuple(range(UNITS)):
            raise Exp1bSchemaError("Experiment 1B served positions are not 0..7.")
        if self.access_state != "payload_emitted":
            raise Exp1bSchemaError("Experiment 1B schedule did not reach its payload.")
