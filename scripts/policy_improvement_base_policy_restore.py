#!/usr/bin/env fbpython
"""Authenticate and restore the registered train-only base policy.

Stage 1 compares one fixed-base proposal against one frozen base policy. Every
row in the screen must therefore start from byte-identical weights, and those
weights must be the artifact the validated base-policy amendment names and
nothing else.

The authentication order matters. The whole-file digest is checked against the
amendment before anything is deserialized, so a substituted or truncated
artifact is rejected while it is still opaque bytes. Only then is the payload
loaded, and its embedded architecture and model-state identities are checked
against both the amendment and the live model configuration.

Stage 0 does not use this path. Its rows keep random initialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.policy_improvement_v2_schema import (
    canonical_json_bytes,
    PolicyImprovementV2SchemaError,
    validate_base_policy_amendment,
)


BASE_POLICY_PRODUCER_SCHEMA_NAME = "policy_improvement_base_policy_producer_v2"
BASE_POLICY_PRODUCER_SCHEMA_VERSION = 1
BASE_POLICY_INITIALIZATION_KIND = "train_only_pretrained"
_READ_SIZE = 1024 * 1024


class BasePolicyRestoreError(RuntimeError):
    """Raised when the registered base policy cannot be authenticated."""


@dataclass(frozen=True)
class AuthenticatedBasePolicy:
    """One base artifact proven to match its amendment before deserialization."""

    path: Path
    size_bytes: int
    checkpoint_sha256: str
    architecture_sha256: str
    model_state_sha256: str
    producer_git_commit: str
    producer_source_manifest_sha256: str
    training_procedure_sha256: str
    training_split_ordered_record_sha256: str
    training_dataset_manifest_sha256: str
    device: int
    inode: int


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def load_base_policy_amendment(path: str | Path) -> dict[str, Any]:
    """Load and validate the amendment through the checked-in v2 schema."""

    location = Path(path)
    if not location.is_absolute() or os.path.realpath(location) != str(location):
        raise BasePolicyRestoreError(
            "Base-policy amendment path must be absolute and canonical."
        )
    try:
        document = json.loads(location.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BasePolicyRestoreError(
            "Base-policy amendment is not strict ASCII JSON."
        ) from exc
    try:
        return validate_base_policy_amendment(document)
    except PolicyImprovementV2SchemaError as exc:
        raise BasePolicyRestoreError(
            "Base-policy amendment failed its registered schema."
        ) from exc


def authenticate_base_policy_artifact(
    artifact_path: str | Path,
    *,
    amendment: Mapping[str, object],
) -> AuthenticatedBasePolicy:
    """Prove the artifact matches its amendment before any deserialization.

    Path shape, size, and the whole-file SHA-256 are all checked here, while
    the file is still opaque bytes. A caller-selected checkpoint that is not
    the registered artifact cannot get past this function.
    """

    try:
        validated = validate_base_policy_amendment(dict(amendment))
    except PolicyImprovementV2SchemaError as exc:
        raise BasePolicyRestoreError(
            "Base-policy amendment failed its registered schema."
        ) from exc
    registered = validated["base_policy_artifact"]
    if registered["status"] != "available":
        raise BasePolicyRestoreError("Base-policy amendment registers no artifact.")
    if registered["initialization_kind"] != BASE_POLICY_INITIALIZATION_KIND:
        raise BasePolicyRestoreError(
            "Stage 1 requires the train-only pretrained base artifact."
        )

    path = Path(artifact_path)
    if not path.is_absolute() or os.path.realpath(path) != str(path):
        raise BasePolicyRestoreError(
            "Base-policy artifact path must be absolute, canonical, and "
            "non-symlinked."
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise BasePolicyRestoreError("Base-policy artifact does not exist.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise BasePolicyRestoreError(
            "Base-policy artifact must be a regular non-symlink file."
        )
    if before.st_size != int(registered["checkpoint_size_bytes"]):
        raise BasePolicyRestoreError(
            "Base-policy artifact size differs from its registered amendment."
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BasePolicyRestoreError(
            "Base-policy artifact cannot be opened safely."
        ) from exc
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(_READ_SIZE), b""):
                digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len({_file_identity(before), _file_identity(opened), _file_identity(after)}) != 1:
        raise BasePolicyRestoreError(
            "Base-policy artifact changed while it was authenticated."
        )
    if digest.hexdigest() != registered["checkpoint_sha256"]:
        raise BasePolicyRestoreError(
            "Base-policy artifact digest differs from its registered amendment."
        )

    return AuthenticatedBasePolicy(
        path=path,
        size_bytes=before.st_size,
        checkpoint_sha256=registered["checkpoint_sha256"],
        architecture_sha256=registered["architecture_sha256"],
        model_state_sha256=registered["model_state_sha256"],
        producer_git_commit=registered["producer_git_commit"],
        producer_source_manifest_sha256=registered[
            "producer_source_manifest_sha256"
        ],
        training_procedure_sha256=registered["training_procedure_sha256"],
        training_split_ordered_record_sha256=registered[
            "training_split_ordered_record_sha256"
        ],
        training_dataset_manifest_sha256=registered[
            "training_dataset_manifest_sha256"
        ],
        device=before.st_dev,
        inode=before.st_ino,
    )


def restore_base_policy_state(
    authenticated: AuthenticatedBasePolicy,
    *,
    model_config: Mapping[str, object],
) -> dict[str, Any]:
    """Deserialize the authenticated artifact and bind its model identity.

    Only reached once the whole-file digest matched. The embedded architecture
    and model-state digests are then checked against the amendment and against
    the architecture this run actually constructed, so a base policy trained
    for a different model shape cannot be loaded.
    """

    import torch

    from rl.persistent_diagnostic_checkpoint import state_dict_sha256

    live_architecture = canonical_sha256(dict(model_config))
    if live_architecture != authenticated.architecture_sha256:
        raise BasePolicyRestoreError(
            "Base-policy architecture differs from the architecture this run "
            "constructed."
        )
    try:
        payload = torch.load(
            authenticated.path, map_location="cpu", weights_only=True
        )
    except Exception as exc:  # noqa: BLE001 - torch raises many unpickling types
        raise BasePolicyRestoreError(
            "Base-policy artifact cannot be deserialized safely."
        ) from exc
    if not isinstance(payload, dict):
        raise BasePolicyRestoreError("Base-policy artifact is not one payload object.")
    if (
        payload.get("schema_name") != BASE_POLICY_PRODUCER_SCHEMA_NAME
        or payload.get("schema_version") != BASE_POLICY_PRODUCER_SCHEMA_VERSION
        or payload.get("initialization_kind") != BASE_POLICY_INITIALIZATION_KIND
    ):
        raise BasePolicyRestoreError("Base-policy artifact schema differs.")
    for field, expected in (
        ("architecture_sha256", authenticated.architecture_sha256),
        ("model_state_sha256", authenticated.model_state_sha256),
        ("training_procedure_sha256", authenticated.training_procedure_sha256),
        ("producer_git_commit", authenticated.producer_git_commit),
    ):
        if payload.get(field) != expected:
            raise BasePolicyRestoreError(
                f"Base-policy artifact {field} differs from its amendment."
            )
    embedded_config = payload.get("model_config")
    if not isinstance(embedded_config, Mapping) or canonical_sha256(
        dict(embedded_config)
    ) != authenticated.architecture_sha256:
        raise BasePolicyRestoreError(
            "Base-policy embedded model configuration differs from its identity."
        )
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise BasePolicyRestoreError("Base-policy artifact carries no model state.")
    restored = {str(name): tensor for name, tensor in state.items()}
    if state_dict_sha256(restored) != authenticated.model_state_sha256:
        raise BasePolicyRestoreError(
            "Base-policy model state differs from its registered identity."
        )
    return restored


def apply_base_policy_state(
    model: Any,
    authenticated: AuthenticatedBasePolicy,
    *,
    model_config: Mapping[str, object],
) -> str:
    """Load the registered base weights into a freshly constructed model.

    Must run before any current-policy snapshot, candidate copy, target
    network, optimizer, or trainer is built, so every derived module inherits
    the same frozen base. Returns the loaded model-state digest.
    """

    from rl.persistent_diagnostic_checkpoint import state_dict_sha256

    state = restore_base_policy_state(authenticated, model_config=model_config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise BasePolicyRestoreError(
            "Base-policy model state does not fit the constructed model."
        ) from exc
    loaded = state_dict_sha256(model.state_dict())
    if loaded != authenticated.model_state_sha256:
        raise BasePolicyRestoreError(
            "Model state after restore differs from the registered base policy."
        )
    return loaded
