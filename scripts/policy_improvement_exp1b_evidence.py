#!/usr/bin/env fbpython
"""Experiment 1B owner-controlled evidence generation: locking, artifacts, state.

Everything the bridge route needs that must outlive a process, be unique, and be
serialized between concurrent writers lives here.

Why this module exists
----------------------
The previous revision let the caller name the schedule file and asserted that
eight checkpoints were sealed. Both were holes:

* two writers could open the same provenance and each serve seed 0, because an
  atomic ``os.replace`` prevents a torn write but does not serialize a
  read-modify-write;
* the same provenance under ``state-a.json`` and ``state-b.json`` produced two
  independent state machines;
* a provenance JSON claiming ``sealed: true`` eight times opened the route with
  no checkpoint file anywhere on disk;
* a payload lived only in the serving process, so the documented
  one-process-per-seed flow could never assemble the octet.

An *evidence generation* fixes all four. It is a single owner-controlled
directory whose layout is derived, not chosen::

    <evidence_root>/<generation_id>/
        provenance.json          the authenticated substitute provenance
        schedule_state.json      created O_EXCL; never replaced by a new path
        .lock                    interprocess mutex, held across every transition
        checkpoints/seed-<p>/checkpoint.pt
        checkpoints/seed-<p>/run_manifest.json
        payloads/seed-<p>.json   written O_EXCL before the slot is marked served

The schedule path is *computed* from the generation, so there is no second path
to open. The lock is held across reload, precondition, claim, payload write,
finalize, and fsync, so the transition is serialized rather than merely atomic.

Ownership and permissions
-------------------------
The root and generation directories must be owned by the effective user and must
not be group- or world-writable. That is what makes "owner-controlled" checkable
rather than aspirational.

Opaque-bytes discipline
-----------------------
Checkpoint artifacts are authenticated by size and SHA-256 over their raw bytes.
Nothing here deserializes a checkpoint. The model-state digest comes from a
sibling run manifest, itself digest-bound, exactly as the v2 base-policy
restoration authenticates before it deserializes.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.policy_improvement_exp1b_schema import (
    attestation_matches_authorization,
    BRIDGE_ROUTE_ID,
    _role_attestation,
    Exp1bRoleAuthorization,
    FULL_ROLE,
    role_attestation_document,
    validate_secondary_diagnostics,
    validate_substitute_provenance,
    Exp1bSchemaError,
    PROTOCOL_ID,
    PROVENANCE_SCHEMA_NAME,
    PROVENANCE_SCHEMA_VERSION,
    exp1b_document_sha256,
    REGISTERED_SEEDS,
    TERMINAL_ENVIRONMENT_INTERACTIONS,
    TRAINING_RECORD_COUNT,
    UNITS,
)
from scripts.policy_improvement_schema import (
    canonical_json_bytes,
    load_strict_json_bytes,
    PolicyImprovementSchemaError,
)

__all__ = [
    "AuthenticatedCheckpoint",
    "Exp1bArtifactExistsError",
    "Exp1bClaimLeaseHeld",
    "Exp1bEvidenceError",
    "Exp1bEvidenceGeneration",
    "Exp1bFinalizedEvidence",
    "authenticate_base_artifact",
    "authenticated_checkpoint_bytes",
    "install_durable_artifact",
    "reap_partial_links",
    "finalize_exp1b_evidence",
    "authenticate_sealed_octet_checkpoints",
    "open_evidence_generation",
    "stable_bytes",
]

#: The one result schema this generation may publish. Duplicated from
#: ``scripts.policy_improvement_exp1b_aggregate`` because that module imports
#: this one; a consistency test pins the two together.
RESULT_SCHEMA_NAME = "policy_improvement_exp1b_result_v1"

_SHA256_HEX = 64
_RUN_MANIFEST_SCHEMA = "policy_improvement_exp1b_run_manifest_v1"
_PAYLOAD_SCHEMA = "policy_improvement_exp1b_seed_payload_v1"
_RETIREMENT_SCHEMA = "policy_improvement_exp1b_retired_checkpoint_v1"

#: The two durable names that make up a published result. Named once so the
#: accessors and the raw-writer refusal below cannot drift apart.
_RESULT_DOCUMENT_NAME = "exp1b_result.json"
_RESULT_DIGEST_NAME = "exp1b_result.sha256"
_RESULT_ARTIFACT_NAMES = frozenset({_RESULT_DOCUMENT_NAME, _RESULT_DIGEST_NAME})


class Exp1bEvidenceError(RuntimeError):
    """Raised when evidence is missing, unowned, non-unique, or unauthentic."""


class Exp1bClaimLeaseHeld(Exp1bEvidenceError):
    """Raised when a live process already holds a slot's evaluation lease."""


class Exp1bArtifactExistsError(Exp1bEvidenceError):
    """Raised when a durable artifact already occupies its final pathname.

    Typed rather than string-matched: the retry path has to distinguish "this
    already exists, check whether it is ours" from every other failure, and
    substring-matching an error message is how a partial finalization previously
    read as a successful race.
    """


def _strict_int(value: object, *, label: str, minimum: int | None = None) -> int:
    """An exact integer. ``True`` and ``10000.0`` are not integers here.

    Python compares ``True == 1`` and ``10000.0 == 10000``, so a durable
    document carrying either would pass an equality check while round-tripping
    to a different canonical digest than the one it claims.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise Exp1bEvidenceError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise Exp1bEvidenceError(f"{label} is below its minimum.")
    return value


def _strict_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise Exp1bEvidenceError(f"{label} must be nonempty ASCII.")
    return value


def _strict_false(value: object, *, label: str) -> None:
    if value is not False:
        raise Exp1bEvidenceError(f"{label} must be exactly false.")


def _digest_text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Exp1bEvidenceError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def stable_bytes(path: Path, *, label: str) -> tuple[bytes, str, int]:
    """Read a file's opaque bytes and prove they did not change during the read.

    Returns ``(payload, sha256, size)``. Rejects symlinks, multiply-linked
    inodes, non-regular files, and any inode-metadata change across the read.
    Never interprets the bytes.
    """

    if not path.is_absolute() or ".." in path.parts:
        raise Exp1bEvidenceError(f"{label} path must be absolute and canonical.")
    try:
        if path.resolve(strict=True) != path:
            raise Exp1bEvidenceError(f"{label} path must not traverse an alias.")
    except OSError as exc:
        raise Exp1bEvidenceError(f"{label} is unavailable.") from exc

    descriptor = -1
    try:
        before_path = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before_path.st_nlink != 1
            or (before_path.st_dev, before_path.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise Exp1bEvidenceError(f"{label} is a link alias or the wrong type.")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise Exp1bEvidenceError(f"{label} cannot be authenticated.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        identity(before_path) != identity(before)
        or identity(before) != identity(after)
        or identity(after) != identity(after_path)
    ):
        raise Exp1bEvidenceError(f"{label} changed while it was read.")
    payload = b"".join(chunks)
    return payload, hashlib.sha256(payload).hexdigest(), before.st_size


def _stable_json(path: Path, *, label: str) -> tuple[Any, str]:
    payload, _raw, _size = stable_bytes(path, label=label)
    try:
        document = load_strict_json_bytes(payload)
    except PolicyImprovementSchemaError as exc:
        raise Exp1bEvidenceError(f"{label} is not strict JSON.") from exc
    return document, hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _require_owned_directory(path: Path, *, label: str) -> None:
    """The directory must exist, be ours, and not be group/world writable."""

    if not path.is_absolute() or ".." in path.parts:
        raise Exp1bEvidenceError(f"{label} must be an absolute canonical directory.")
    try:
        info = path.lstat()
    except OSError as exc:
        raise Exp1bEvidenceError(f"{label} does not exist.") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise Exp1bEvidenceError(f"{label} must be a real directory, not a link.")
    if info.st_uid != os.geteuid():
        raise Exp1bEvidenceError(f"{label} is not owned by the running user.")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise Exp1bEvidenceError(f"{label} is group- or world-writable.")


_TEMP_PREFIX = ".partial-"


def reap_partial_links(directory: Path) -> None:
    """Public entry point for the reaper, for recovery readers.

    ``_reap_partials`` used to run only before the next install. Every recovery
    path *reads* first -- payload-orphan reconciliation, result-document
    completion, publication-state classification -- so each one hit
    :func:`stable_bytes`'s ``st_nlink != 1`` refusal before any cleanup could
    run, and the link/unlink crash window was therefore permanent rather than
    transient. Callers hold the generation lock.
    """

    _reap_partials(directory)


def _reap_partials(directory: Path) -> None:
    """Remove this module's leftover temporaries before installing.

    Two crash windows leave one behind. A crash after the temp is written but
    before it is linked leaves an orphan. A crash *between* the link and the
    temp's removal leaves the final artifact with ``st_nlink == 2``, which
    :func:`stable_bytes` refuses outright -- so the reaper is not housekeeping,
    it is what makes that window recoverable.
    """

    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(_TEMP_PREFIX):
            continue
        try:
            entry.unlink()
        except OSError:
            continue


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> None:
    """Install a complete file at ``path``, never replacing an existing one.

    Write to a temporary in the same directory, fsync it, then ``os.link`` it
    into place and unlink the temporary. Three properties matter and the
    previous version only had the third:

    * **Atomic content.** ``O_EXCL`` on the final name published an empty
      directory entry that was only filled afterwards, so a crash mid-write left
      a truncated artifact at the real pathname. The link publishes a name that
      already has all its bytes.
    * **No-replace.** ``os.link`` fails with ``FileExistsError`` if the
      destination exists. ``os.rename``/``os.replace`` cannot express this: they
      silently unlink the destination, and CPython exposes no ``RENAME_NOREPLACE``.
    * **Durability.** The data is fsynced before the name exists, and the
      directory is fsynced after, so the name and its bytes survive together.
    """

    directory = path.parent
    _reap_partials(directory)
    temporary = directory / f"{_TEMP_PREFIX}{path.name}.{os.getpid()}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise Exp1bEvidenceError(f"{label} staging file cannot be created.") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Exp1bEvidenceError(f"{label} write stalled.")
            offset += written
        os.fsync(descriptor)
        if os.fstat(descriptor).st_size != len(payload):
            raise Exp1bEvidenceError(f"{label} staging file is the wrong size.")
    except BaseException:
        os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)

    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise Exp1bArtifactExistsError(
            f"{label} already exists and is never replaced."
        ) from exc
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise Exp1bEvidenceError(f"{label} cannot be installed.") from exc
    try:
        temporary.unlink()
    except OSError:
        pass
    parent = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def install_durable_artifact(path: Path, payload: bytes, *, label: str) -> None:
    """Public entry point for the staged install, for the schedule module.

    The schedule lives in a different module but has the same durability
    requirement, and its initial write used a raw ``os.write`` onto the final
    pathname. A crash there left a zero-byte or truncated ``schedule_state.json``
    that no restart could parse or repair.

    It refuses the two result names outright. A generic exported writer plus a
    result pathname is a raw-writer bypass of
    :meth:`Exp1bEvidenceGeneration.publish_result`, which is the only thing
    allowed to publish a result and the only thing that runs the validator and
    the independent audit to earn that. This does not pretend to stop arbitrary
    code from calling :func:`open`; it stops *this module's own exported
    surface* from being the tool that does it.
    """

    if path.name in _RESULT_ARTIFACT_NAMES:
        raise Exp1bEvidenceError(
            "The Experiment 1B result is published only through "
            "Exp1bEvidenceGeneration.publish_result, which runs the full "
            "validator and the independent audit itself."
        )
    _write_exclusive(path, payload, label=label)


def _install_or_adopt(path: Path, payload: bytes, *, label: str) -> str:
    """Install ``payload`` at ``path``, or adopt a byte-identical existing file.

    Retry after a crash is the normal case, not an exceptional one: eight Stage A
    runs and a finalization step, any of which can be interrupted. Refusing every
    existing artifact made a retry unrecoverable; silently overwriting one would
    make the evidence meaningless. Adopting only a byte-identical artifact is the
    only version of "retry" that preserves both.
    """

    try:
        _write_exclusive(path, payload, label=label)
        return "installed"
    except Exp1bArtifactExistsError:
        pass
    existing, digest, size = stable_bytes(path, label=label)
    if size != len(payload) or digest != hashlib.sha256(payload).hexdigest():
        raise Exp1bEvidenceError(
            f"{label} already exists and differs from what this run would "
            "write; it cannot be adopted."
        )
    if existing != payload:
        raise Exp1bEvidenceError(
            f"{label} already exists with different bytes; it cannot be adopted."
        )
    return "adopted"


def authenticated_checkpoint_bytes(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> bytes:
    """Return the exact bytes a caller is allowed to deserialize, or refuse.

    The bytes come back to the caller so that deserialization happens **from
    memory**. Hashing a path and then reopening it is two different reads of two
    possibly different files: between the two, the artifact can be replaced, and
    a loader that reopens deserializes a byte sequence nothing authenticated.
    Evidence files are owner-writable, so this is not a theoretical window.

    Everything below is one open: :func:`stable_bytes` opens once with
    ``O_NOFOLLOW``, rejects symlinks, multiply-linked inodes, and non-regular
    files, and proves the inode metadata did not change across the read. This
    function adds the registered size and digest comparison and one final
    re-authentication, so a replacement that happened after the read is also
    caught before the caller acts on the payload.

    Lives here, not in the Torch backend, for two reasons: it is standard-library
    only, and the replacement regression can therefore run in a checkout without
    Torch -- which is the only kind of checkout this repository is tested in.
    """

    expected_digest = _digest_text(expected_sha256, label=f"{label} SHA-256")
    expected_size = _strict_int(
        expected_size_bytes, label=f"{label} size", minimum=1
    )
    payload, observed_digest, observed_size = stable_bytes(path, label=label)
    if observed_size != expected_size:
        raise Exp1bEvidenceError(
            f"{label} size differs from its registered descriptor."
        )
    if observed_digest != expected_digest:
        raise Exp1bEvidenceError(
            f"{label} bytes differ from their registered descriptor."
        )
    if len(payload) != expected_size:
        raise Exp1bEvidenceError(f"{label} read returned a short buffer.")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise Exp1bEvidenceError(
            f"{label} in-memory buffer does not hash to the authenticated digest."
        )
    # Re-authenticate the path. The returned buffer is already fixed, so this
    # cannot change what the caller deserializes; what it establishes is that the
    # artifact at that path is still the one that was authenticated, so a
    # replacement is reported rather than silently tolerated.
    _recheck, recheck_digest, recheck_size = stable_bytes(
        path, label=f"{label} re-authentication"
    )
    if recheck_digest != expected_digest or recheck_size != expected_size:
        raise Exp1bEvidenceError(
            f"{label} was replaced between authentication and use."
        )
    return payload


@dataclass(frozen=True)
class AuthenticatedCheckpoint:
    """One budget-final checkpoint proved to exist, with authenticated bytes."""

    seed_position: int
    seed: int
    run_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    model_state_sha256: str
    effective_config_sha256: str
    environment_interactions: int
    run_manifest_sha256: str


@dataclass(frozen=True)
class Exp1bEvidenceGeneration:
    """One owner-controlled evidence generation with a derived layout."""

    root: Path
    generation_id: str
    directory: Path
    provenance_path: Path
    schedule_path: Path
    payload_directory: Path
    lock_path: Path
    checkpoint_directory: Path
    generation_sha256: str

    def reap_under_lock(self) -> None:
        """Clear this module's leftover staging links across the generation.

        Takes the generation lock, so it can never delete a live writer's
        temporary: a writer holds that same lock for the whole install. Every
        recovery path calls this *before* it reads or classifies, because
        :func:`stable_bytes` refuses an artifact left at ``st_nlink == 2`` by a
        crash in the link/unlink window, and reading first made that state
        permanent.

        An earlier version reaped inside the readers themselves. That closed the
        crash window and opened a worse one: an unlocked reader in one process
        deleted a lock-holding writer's staging temporary in another, losing the
        write.
        """

        with self.exclusive():
            self._reap_all()

    def _reap_all(self) -> None:
        """Reap every directory this generation writes into. Lock must be held."""

        _reap_partials(self.directory)
        _reap_partials(self.payload_directory)
        base = self.directory / "base_policy"
        if base.is_dir():
            _reap_partials(base)
        for position in range(UNITS):
            directory = self.checkpoint_directory_for(position)
            if directory.is_dir():
                _reap_partials(directory)

    def claim_lease_path(self, seed_position: int) -> Path:
        """The per-slot liveness lease.

        An ``flock`` on this file is held for exactly as long as a process is
        evaluating that seed. The kernel releases it when the process dies, so
        "can I acquire this lease?" is a proof of the previous owner's death
        that needs no clock and no timeout. That matters: the schedule document
        is canonical JSON with a digest, and a wall-clock deadline in it would
        make the record nondeterministic.
        """

        if (
            isinstance(seed_position, bool)
            or not isinstance(seed_position, int)
            or not 0 <= seed_position < UNITS
        ):
            raise Exp1bEvidenceError("Seed position is outside the registered octet.")
        return self.directory / f".claim-{seed_position}.lock"

    @contextmanager
    def claim_lease(self, seed_position: int) -> Iterator[int]:
        """Hold one slot's lease for the duration of an evaluation.

        Separate from :meth:`exclusive`, which is the generation-wide mutex held
        only across a state transition. This lease spans the *whole* evaluation,
        which is precisely the window the generation lock is not held for.
        """

        descriptor = os.open(
            self.claim_lease_path(seed_position), os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise Exp1bClaimLeaseHeld(
                    f"Experiment 1B seed {seed_position} is being evaluated by a "
                    "live process."
                ) from exc
            try:
                yield descriptor
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def claim_lease_is_free(self, seed_position: int) -> bool:
        """Whether no live process holds this slot's lease.

        A free lease on a slot the schedule records as ``claimed`` means the
        claiming process died before it emitted. That is the only evidence this
        module will accept for reclaiming it.
        """

        descriptor = os.open(
            self.claim_lease_path(seed_position), os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return True
        finally:
            os.close(descriptor)

    def payload_path(self, seed_position: int) -> Path:
        if (
            isinstance(seed_position, bool)
            or not isinstance(seed_position, int)
            or not 0 <= seed_position < UNITS
        ):
            raise Exp1bEvidenceError("Seed position is outside the registered octet.")
        return self.payload_directory / f"seed-{seed_position}.json"

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the interprocess mutex across a whole state transition.

        ``flock`` is advisory but process-wide and released on process death,
        which is what a crash-window needs: a writer that dies mid-transition
        does not wedge the generation.
        """

        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def write_payload(self, seed_position: int, document: Mapping[str, Any]) -> str:
        """Persist one canonical seed payload, no-replace, and return its digest.

        Called *before* the slot is marked served, so a crash between the two
        leaves a recoverable payload and an unserved slot rather than a served
        slot with nothing behind it.
        """

        payload = canonical_json_bytes(document) + b"\n"
        _write_exclusive(
            self.payload_path(seed_position),
            payload,
            label=f"Experiment 1B payload {seed_position}",
        )
        return hashlib.sha256(canonical_json_bytes(document)).hexdigest()

    def read_payload(self, seed_position: int) -> tuple[dict[str, Any], str]:
        """Reload one persisted payload and recompute its canonical digest.

        Callers hold the generation lock and have already reaped; see
        :meth:`reap_under_lock`.
        """

        document, digest = _stable_json(
            self.payload_path(seed_position),
            label=f"Experiment 1B payload {seed_position}",
        )
        if not isinstance(document, Mapping):
            raise Exp1bEvidenceError(
                f"Experiment 1B payload {seed_position} is not an object."
            )
        expected = {
            "schema_name",
            "schema_version",
            "route_id",
            "seed_position",
            "seed",
            "run_id",
            "checkpoint_sha256",
            "census_ordering_sha256",
            "state_count",
            "maximum_endpoint_discrepancy",
            "maximum_absolute_residual_n",
            "maximum_absolute_residual_m",
            "direct_residual_bound_n",
            "finite_reference_bound",
            "signed_gap",
            "discrepancy_witness_state_id",
            "residual_n_witness_state_id",
            "residual_m_witness_state_id",
            "secondary_diagnostics",
        }
        label = f"Experiment 1B payload {seed_position}"
        if set(document) != expected:
            raise Exp1bEvidenceError(f"{label} field inventory differs.")
        if (
            _strict_text(document["schema_name"], label=f"{label} schema name")
            != _PAYLOAD_SCHEMA
            or _strict_int(document["schema_version"], label=f"{label} version") != 2
        ):
            raise Exp1bEvidenceError(f"{label} has the wrong schema.")
        if (
            _strict_int(document["seed_position"], label=f"{label} position", minimum=0)
            != seed_position
        ):
            raise Exp1bEvidenceError(f"{label} is not the payload for its own slot.")
        _strict_int(document["seed"], label=f"{label} seed", minimum=0)
        _strict_int(document["state_count"], label=f"{label} state count", minimum=1)
        _strict_text(document["run_id"], label=f"{label} run ID")
        # The route, not merely a route. A crash orphan carrying another ASCII
        # route identifier was previously adopted, served alongside seven real
        # seeds, and aggregated under the Experiment 1B route ID.
        if _strict_text(document["route_id"], label=f"{label} route ID") != (
            BRIDGE_ROUTE_ID
        ):
            raise Exp1bEvidenceError(
                f"{label} was produced under route "
                f"{document['route_id']!r}, not {BRIDGE_ROUTE_ID!r}."
            )
        for name in (
            "checkpoint_sha256",
            "census_ordering_sha256",
        ):
            _digest_text(document[name], label=f"{label} {name}")
        for name in (
            "maximum_endpoint_discrepancy",
            "maximum_absolute_residual_n",
            "maximum_absolute_residual_m",
            "direct_residual_bound_n",
            "finite_reference_bound",
            "signed_gap",
        ):
            value = document[name]
            if isinstance(value, bool) or not isinstance(value, float):
                raise Exp1bEvidenceError(f"{label} {name} is not a binary64 float.")
            if value != value or value in (float("inf"), float("-inf")):
                raise Exp1bEvidenceError(f"{label} {name} is not finite.")
        for name in (
            "discrepancy_witness_state_id",
            "residual_n_witness_state_id",
            "residual_m_witness_state_id",
        ):
            _strict_text(document[name], label=f"{label} {name}")
        try:
            validate_secondary_diagnostics(
                document["secondary_diagnostics"],
                path=f"{label}.secondary_diagnostics",
            )
        except Exp1bSchemaError as exc:
            raise Exp1bEvidenceError(f"{label} secondary diagnostics: {exc}") from exc
        return dict(document), digest

    def has_payload(self, seed_position: int) -> bool:
        return self.payload_path(seed_position).is_file()

    # --- Stage A publication -------------------------------------------------

    def checkpoint_directory_for(self, seed_position: int) -> Path:
        if (
            isinstance(seed_position, bool)
            or not isinstance(seed_position, int)
            or not 0 <= seed_position < UNITS
        ):
            raise Exp1bEvidenceError("Seed position is outside the registered octet.")
        return self.checkpoint_directory / f"seed-{seed_position}"

    def checkpoint_publication_state(self, seed_position: int) -> str:
        """Where one seed's Stage A publication stands.

        ``absent`` (nothing), ``checkpoint_only`` (crash between the two files),
        or ``complete``. Named so a retry can tell "not yet run" from "ran and
        was interrupted", which is the distinction that decides whether the
        operator has to retrain.
        """

        directory = self.checkpoint_directory_for(seed_position)
        has_checkpoint = (directory / "checkpoint.pt").is_file()
        has_manifest = (directory / "run_manifest.json").is_file()
        if has_checkpoint and has_manifest:
            return "complete"
        if has_checkpoint:
            return "checkpoint_only"
        if has_manifest:
            raise Exp1bEvidenceError(
                f"Experiment 1B seed {seed_position} has a run manifest with no "
                "checkpoint; the publication order cannot produce that state."
            )
        return "absent"

    def retire_unreferenced_checkpoint(self, seed_position: int) -> str:
        """Clear a checkpoint left behind by a crash between the pair's two files.

        Returns the transition taken: ``absent``, ``complete``, or ``retired``.

        Why this exists. Publication writes the checkpoint first and the manifest
        second, so a crash between them leaves ``checkpoint_only``. That is not a
        published seed -- the manifest carries counters and identities only the
        run itself can produce -- so the seed has to run again. But the rerun is a
        fresh 10,000-interaction training run, and training is not bit
        reproducible, so its serialization differs and ``_install_or_adopt``
        refuses the checkpoint that is already sitting there. The documented crash
        state was therefore recoverable only by luck.

        The fix is to make the leftover *go away* before the rerun, and to do it
        only when it is provably unreferenced. A checkpoint enters the evidence
        graph solely through its manifest: manifest to sealed octet to payload to
        result to provenance. With no manifest, nothing can name it -- provided no
        later artifact exists that would imply the manifest once did. So this
        refuses if a schedule, a payload, a result, or a provenance exists.

        A complete pair is never touched, here or anywhere: once both files are
        installed the seed is immutable and a retry adopts it instead.

        The whole transition runs under the generation mutex, so a concurrent
        writer cannot install a manifest between the classification and the move.
        """

        directory = self.checkpoint_directory_for(seed_position)
        with self.exclusive():
            _reap_partials(directory)
            state = self.checkpoint_publication_state(seed_position)
            if state != "checkpoint_only":
                # `complete` is immutable and `absent` has nothing to retire.
                return state
            for path, what in (
                (self.provenance_path, "finalized provenance"),
                (self.result_path, "published result"),
                (self.schedule_path, "Stage B request schedule"),
                (self.payload_path(seed_position), "sealed seed payload"),
            ):
                if path.exists():
                    raise Exp1bEvidenceError(
                        f"Experiment 1B seed {seed_position} has no run manifest "
                        f"but the generation carries a {what}; that is not a "
                        "reachable crash state, and the checkpoint it names is "
                        "not this module's to retire."
                    )

            checkpoint = directory / "checkpoint.pt"
            # Hash before moving, while the inode is still singly linked:
            # `stable_bytes` refuses `st_nlink != 1`, which is exactly why the
            # retirement is a rename and not a link-then-unlink. A rename has no
            # intermediate state in which both names exist, so a crash inside it
            # cannot wedge the next attempt.
            _payload, digest, size = stable_bytes(
                checkpoint, label=f"Experiment 1B checkpoint {seed_position}"
            )
            del _payload
            retired = directory / "retired"
            try:
                retired.mkdir(mode=0o700, parents=False, exist_ok=True)
            except OSError as exc:
                raise Exp1bEvidenceError(
                    f"Experiment 1B seed {seed_position} retirement directory "
                    "cannot be created."
                ) from exc
            _require_owned_directory(
                retired, label=f"Experiment 1B retirement directory {seed_position}"
            )
            # Content-addressed, with an index only if the same bytes were
            # retired before. Nothing durable is ever replaced, including here.
            target = retired / f"checkpoint-{digest}.pt"
            attempt = 0
            while target.exists():
                attempt += 1
                target = retired / f"checkpoint-{digest}.{attempt}.pt"
            try:
                os.rename(checkpoint, target)
            except OSError as exc:
                raise Exp1bEvidenceError(
                    f"Experiment 1B seed {seed_position} checkpoint cannot be "
                    "retired."
                ) from exc
            for path in (retired, directory):
                handle = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(handle)
                finally:
                    os.close(handle)
            # An audit trail, written no-replace after the move. Advisory: a
            # crash between the rename and this write loses the record, not the
            # bytes, and the retired file is content-named either way.
            _write_exclusive(
                retired / f"{target.stem}.json",
                canonical_json_bytes(
                    {
                        "schema_name": _RETIREMENT_SCHEMA,
                        "generation_sha256": str(self.generation_sha256),
                        "seed_position": seed_position,
                        "checkpoint_sha256": digest,
                        "checkpoint_size_bytes": size,
                        "reason": "crash_between_checkpoint_and_run_manifest",
                        "retired_file": target.name,
                    }
                )
                + b"\n",
                label=f"Experiment 1B retirement record {seed_position}",
            )
        return "retired"

    def publish_checkpoint(
        self,
        *,
        seed_position: int,
        checkpoint_bytes: bytes,
        run_manifest: Mapping[str, Any],
    ) -> tuple[str, str]:
        """Durably publish one seed's budget-final checkpoint and run manifest.

        Both artifacts are created ``O_EXCL`` and fsynced, checkpoint first: a
        crash between the two leaves an unreferenced checkpoint, which
        authentication rejects for having no manifest, rather than a manifest
        pointing at bytes that were never written. Returns
        ``(checkpoint_sha256, run_manifest_sha256)``.

        The whole call runs under the generation mutex, so two Stage A processes
        racing on the same seed cannot interleave; the loser fails on ``O_EXCL``.
        """

        if not isinstance(checkpoint_bytes, (bytes, bytearray)):
            raise Exp1bEvidenceError("Checkpoint publication requires bytes.")
        payload = bytes(checkpoint_bytes)
        if not payload:
            raise Exp1bEvidenceError("Checkpoint publication requires nonempty bytes.")
        if not isinstance(run_manifest, Mapping):
            raise Exp1bEvidenceError("Checkpoint publication requires a run manifest.")
        directory = self.checkpoint_directory_for(seed_position)
        checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        recorded = run_manifest.get("checkpoint_sha256")
        if recorded != checkpoint_sha256:
            raise Exp1bEvidenceError(
                "Run manifest checkpoint digest differs from the bytes being published."
            )
        if run_manifest.get("checkpoint_size_bytes") != len(payload):
            raise Exp1bEvidenceError(
                "Run manifest checkpoint size differs from the bytes being published."
            )
        manifest_bytes = canonical_json_bytes(run_manifest) + b"\n"
        with self.exclusive():
            # `exist_ok=True`: a retry after a crash finds its own directory,
            # possibly with only some of its files. Refusing it outright made
            # every interrupted Stage A seed permanently unretryable.
            try:
                directory.mkdir(mode=0o700, parents=False, exist_ok=True)
            except OSError as exc:
                raise Exp1bEvidenceError(
                    f"Experiment 1B checkpoint directory for seed {seed_position} "
                    "cannot be created."
                ) from exc
            _require_owned_directory(
                directory,
                label=f"Experiment 1B checkpoint directory {seed_position}",
            )
            parent = os.open(self.checkpoint_directory, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            # Per file, not per directory: a crash between the checkpoint and
            # its manifest leaves the first installed and the second absent.
            _install_or_adopt(
                directory / "checkpoint.pt",
                payload,
                label=f"Experiment 1B checkpoint {seed_position}",
            )
            _install_or_adopt(
                directory / "run_manifest.json",
                manifest_bytes,
                label=f"Experiment 1B run manifest {seed_position}",
            )
        return checkpoint_sha256, hashlib.sha256(
            canonical_json_bytes(run_manifest)
        ).hexdigest()

    # --- Stage B publication -------------------------------------------------

    @property
    def result_path(self) -> Path:
        return self.directory / _RESULT_DOCUMENT_NAME

    @property
    def result_digest_path(self) -> Path:
        return self.directory / _RESULT_DIGEST_NAME

    def _durable_provenance_locked(self, route: Any) -> dict[str, Any]:
        """The provenance on disk *right now*, proved to be the authenticated one.

        The generation lock must already be held.

        ``route.provenance_document`` is an in-memory copy taken when the route
        opened, and a cached anchor is not an anchor. After authentication the
        file can be replaced with a strict-JSON provenance naming another study,
        or deleted outright, and a stale route would still finalize a result that
        claims it -- while a newly opened route refuses the same generation. So
        the anchor is re-read from disk, strictly revalidated, and required to be
        byte-for-byte the document this route authenticated.

        Both the digest and the content are compared. Digest equality already
        implies content equality for canonical JSON; the content comparison is
        what makes that reasoning unnecessary to trust.
        """

        # A crash in the link/unlink window leaves the provenance at
        # `st_nlink == 2`, which `stable_bytes` refuses. Reap first: this is a
        # recovery read like every other one, and we hold the lock.
        _reap_partials(self.directory)
        try:
            document, digest = _stable_json(
                self.provenance_path, label="Experiment 1B durable provenance"
            )
        except Exp1bEvidenceError as exc:
            raise Exp1bEvidenceError(
                "Experiment 1B publication cannot read the durable provenance "
                f"its result would claim: {exc}"
            ) from exc
        try:
            validate_substitute_provenance(document)
        except Exp1bSchemaError as exc:
            raise Exp1bEvidenceError(
                f"Durable Experiment 1B provenance is not a valid substitute "
                f"provenance: {exc}"
            ) from exc
        expected = _digest_text(
            getattr(route, "provenance_sha256", None),
            label="Experiment 1B authenticated provenance digest",
        )
        if digest != expected:
            raise Exp1bEvidenceError(
                "Durable Experiment 1B provenance is not the one this route "
                "authenticated; it changed after the route opened."
            )
        if document != dict(getattr(route, "provenance_document", {})):
            raise Exp1bEvidenceError(
                "Durable Experiment 1B provenance differs in content from the "
                "one this route authenticated."
            )
        return document

    def _validated_result_bytes(
        self,
        *,
        result: Any,
        route: Any,
        protocol: Mapping[str, Any],
        registry: Mapping[str, Any],
        amendment: Mapping[str, Any],
        admission: Mapping[str, Any],
    ) -> tuple[bytes, str]:
        """Run the full validator and the independent audit here, and return bytes.

        The document is no longer an input. Two earlier shapes of this boundary
        both failed the same way, because both let the caller choose the bytes:

        * ``audit=Callable[[bytes], None]`` -- ``audit=lambda _: None`` published
          anything, since the writer cannot tell a real auditor from a stub;
        * ``proof=Exp1bPublicationProof`` -- the mint accepted arbitrary bytes, so
          minting a proof and handing it back published a four-key document that
          satisfied a shallow structural floor and omitted the entire registered
          result. An issuance registry records *that* something was minted, never
          *what was checked*, and a leading underscore is a naming convention, not
          an authorization boundary.

        So publication takes the authenticated route and the four study documents
        and does the work itself: ``validated_exp1b_document`` re-derives every
        durable claim from this generation under its lock, serializes the
        canonical bytes, and runs the standalone auditor -- the same independent
        consumer an outside reviewer would run -- over exactly those bytes. What
        gets written is what was validated, because it is the same object. There
        is nothing for a caller to substitute.

        Imported at call time: ``aggregate`` imports ``bridge``, which imports
        this module, so a module-scope import here would be a cycle.
        """

        from scripts.policy_improvement_exp1b_aggregate import (
            Exp1bAggregateError,
            validated_exp1b_document,
        )

        # First, because a route into another generation validates perfectly well
        # against its own durable state and there is no reason to run that audit
        # before refusing it. This does not establish that the route is genuine --
        # anything can hold a reference to this generation -- but
        # `validated_exp1b_document` refuses a non-`Exp1bBridgeRoute` outright, so
        # between the two, only the route that served this generation gets through.
        if getattr(route, "generation", None) != self:
            raise Exp1bEvidenceError(
                "Experiment 1B publication requires the route that served this "
                "evidence generation, not a route into another one."
            )
        # Type-checked here rather than relying on `validated_exp1b_document` to
        # do it first: the durable-provenance read below reads identity off the
        # route, and it should not be reading it off an arbitrary object even to
        # refuse it a moment later. Call-time import for the same cycle reason.
        from scripts.policy_improvement_exp1b_bridge import Exp1bBridgeRoute

        if not isinstance(route, Exp1bBridgeRoute):
            raise Exp1bEvidenceError(
                "Experiment 1B publication requires the authenticated route that "
                "served the octet, not an object that merely exposes its methods."
            )

        # The anchor, re-read from disk under the generation lock and proved to be
        # the one the route authenticated. Taken here and released before
        # `validated_exp1b_document`, which takes the same lock itself; `flock`
        # does not recurse within one process.
        with self.exclusive():
            durable_provenance = self._durable_provenance_locked(route)

        try:
            payload = validated_exp1b_document(
                result,
                route=route,
                protocol=protocol,
                registry=registry,
                amendment=amendment,
                # The freshly read durable document, not the route's cache and
                # not a caller argument. This is what the auditor's study-digest
                # binding is anchored to, so the anchor has to be the live one.
                provenance=durable_provenance,
                admission=admission,
            )
        except Exp1bAggregateError as exc:
            raise Exp1bEvidenceError(
                f"Experiment 1B result did not pass publication validation: {exc}"
            ) from exc
        if not isinstance(payload, bytes) or not payload:
            raise Exp1bEvidenceError(
                "Experiment 1B validation produced no document bytes."
            )

        parsed = load_strict_json_bytes(
            payload[:-1] if payload.endswith(b"\n") else payload
        )
        if not isinstance(parsed, Mapping):
            raise Exp1bEvidenceError("Experiment 1B result is not an object.")
        interval = parsed.get("interval")
        if (
            parsed.get("schema_name") != RESULT_SCHEMA_NAME
            or parsed.get("protocol_id") != PROTOCOL_ID
            or parsed.get("route_id") != BRIDGE_ROUTE_ID
            or not isinstance(interval, Mapping)
            or interval.get("status") != "available"
        ):
            raise Exp1bEvidenceError(
                "Experiment 1B publication refuses a document that is not a "
                "registered result with an available interval."
            )
        # Derived from the validated bytes. A caller-supplied digest was one more
        # thing the writer had to check instead of know.
        return payload, exp1b_document_sha256(parsed)

    def publish_result(
        self,
        *,
        result: Any,
        route: Any,
        protocol: Mapping[str, Any],
        registry: Mapping[str, Any],
        amendment: Mapping[str, Any],
        admission: Mapping[str, Any],
    ) -> Path:
        """Validate, audit, and durably publish this generation's one result.

        No-replace and fsynced, like every other artifact here, and the digest
        sidecar is written second so a crash cannot leave a sidecar pointing at
        a result that was never durable. Publishing twice is an error: a
        generation has exactly one result, and a retry after emission is a
        permanent refusal, not an overwrite.
        """

        payload, digest = self._validated_result_bytes(
            result=result,
            route=route,
            protocol=protocol,
            registry=registry,
            amendment=amendment,
            admission=admission,
        )
        with self.exclusive():
            # Finding 3: validation released the lock, so the provenance could be
            # replaced or deleted between the audit and this install. Re-prove it
            # here, under the same lock the installation happens in, so the two
            # artifacts appear only while the anchor they claim is still on disk.
            self._durable_provenance_locked(route)
            _install_or_adopt(
                self.result_path, payload, label="Experiment 1B result document"
            )
            _install_or_adopt(
                self.result_digest_path,
                (digest + "\n").encode("ascii"),
                label="Experiment 1B result digest",
            )
        return self.result_path

    def result_publication_state(self) -> str:
        """Where the two-file publication journal stands: absent, partial, complete.

        The document is written first and the digest sidecar second, so the only
        reachable partial state is ``document_only``. Naming it makes the crash
        window recoverable instead of a dead end.
        """

        has_document = self.result_path.is_file()
        has_sidecar = self.result_digest_path.is_file()
        if not has_document and not has_sidecar:
            return "absent"
        if has_document and not has_sidecar:
            return "document_only"
        if has_document and has_sidecar:
            return "complete"
        raise Exp1bEvidenceError(
            "Experiment 1B result sidecar exists without its document; the "
            "publication journal is not in a reachable state."
        )

    def complete_result_publication(
        self,
        *,
        result: Any,
        route: Any,
        protocol: Mapping[str, Any],
        registry: Mapping[str, Any],
        amendment: Mapping[str, Any],
        admission: Mapping[str, Any],
    ) -> None:
        """Finish a publication that crashed between the document and the sidecar.

        Recovery is a publication too, so it runs the identical gate: the same
        validator, the same independent audit, the same authenticated route and
        study documents. Completing a journal is not a cheaper operation than
        starting one, and a document-only crash state must not become the way to
        get a sidecar onto bytes nothing validated.

        Only ever completes a journal whose durable document is byte-identical
        to what this process just validated. A different document means two
        different results claim the same generation, which is a refusal.
        """

        payload, digest = self._validated_result_bytes(
            result=result,
            route=route,
            protocol=protocol,
            registry=registry,
            amendment=amendment,
            admission=admission,
        )
        with self.exclusive():
            self._durable_provenance_locked(route)
            _reap_partials(self.directory)
            if self.result_digest_path.is_file():
                raise Exp1bEvidenceError(
                    "Experiment 1B result publication is already complete."
                )
            durable, _sha, _size = stable_bytes(
                self.result_path, label="Experiment 1B result document"
            )
            if durable != payload:
                raise Exp1bEvidenceError(
                    "Durable Experiment 1B result differs from the one being "
                    "published; the generation already carries another result."
                )
            _write_exclusive(
                self.result_digest_path,
                (digest + "\n").encode("ascii"),
                label="Experiment 1B result digest",
            )

    def read_published_result(self) -> tuple[bytes, str]:
        """Reload the published result and check it against its own sidecar."""

        payload, _, _ = stable_bytes(
            self.result_path, label="Experiment 1B result document"
        )
        sidecar, _, _ = stable_bytes(
            self.result_digest_path, label="Experiment 1B result digest"
        )
        recorded = _digest_text(
            sidecar.decode("ascii", "strict").strip(),
            label="Experiment 1B result digest",
        )
        body = payload[:-1] if payload.endswith(b"\n") else payload
        try:
            parsed = load_strict_json_bytes(body)
        except PolicyImprovementSchemaError as exc:
            raise Exp1bEvidenceError(
                "Published Experiment 1B result is not strict JSON."
            ) from exc
        if exp1b_document_sha256(parsed) != recorded:
            raise Exp1bEvidenceError(
                "Published Experiment 1B result does not match its recorded digest."
            )
        return payload, recorded


def open_evidence_generation(
    *,
    evidence_root: str | os.PathLike[str],
    generation_id: str,
) -> Exp1bEvidenceGeneration:
    """Resolve and authenticate one owner-controlled evidence generation.

    The schedule, payload, lock, and checkpoint locations are *derived* from the
    generation, never supplied. There is therefore exactly one schedule artifact
    per generation, and the previous alternate-path reset is unreachable.
    """

    if (
        not isinstance(generation_id, str)
        or not generation_id
        or not generation_id.isascii()
        or "/" in generation_id
        or generation_id in {".", ".."}
        or not all(
            character.isalnum() or character in "-_." for character in generation_id
        )
    ):
        raise Exp1bEvidenceError("Evidence generation ID is not a safe identifier.")
    root = Path(evidence_root)
    _require_owned_directory(root, label="Experiment 1B evidence root")
    directory = root / generation_id
    _require_owned_directory(directory, label="Experiment 1B evidence generation")
    payload_directory = directory / "payloads"
    checkpoint_directory = directory / "checkpoints"
    _require_owned_directory(payload_directory, label="Experiment 1B payload directory")
    _require_owned_directory(
        checkpoint_directory, label="Experiment 1B checkpoint directory"
    )
    generation = Exp1bEvidenceGeneration(
        root=root,
        generation_id=generation_id,
        directory=directory,
        provenance_path=directory / "provenance.json",
        schedule_path=directory / "schedule_state.json",
        payload_directory=payload_directory,
        lock_path=directory / ".lock",
        checkpoint_directory=checkpoint_directory,
        generation_sha256=exp1b_document_sha256(
            {
                "root": str(root),
                "generation_id": generation_id,
                "schedule": "schedule_state.json",
                "payloads": "payloads",
                "checkpoints": "checkpoints",
            }
        ),
    )
    return generation


@dataclass(frozen=True)
class Exp1bFinalizedEvidence:
    """What one evidence-finalization pass published."""

    provenance_sha256: str
    generation_sha256: str
    base_artifact_sha256: str
    checkpoint_count: int


def finalize_exp1b_evidence(
    *,
    generation: Exp1bEvidenceGeneration,
    admitted_base_artifact: Path,
    admission_sha256: str,
    base_descriptor: Mapping[str, Any],
    registry_rows: Mapping[str, Mapping[str, Any]],
    exp1b_protocol_sha256: str,
    exp1b_registry_sha256: str,
    exp1b_amendment_sha256: str,
    parent: Mapping[str, Any],
    ordered_population_sha256: str,
    population_binding_sha256: str,
    attestation: Mapping[str, str],
    producer_authorization: Exp1bRoleAuthorization,
) -> Exp1bFinalizedEvidence:
    """Publish the provenance and base artifact Stage B requires, exactly once.

    This is the missing production step: before it existed the only code that
    wrote ``provenance.json`` and ``base_policy/base_policy.pt`` was a test
    fixture, so eight successful Stage A runs still left a generation Stage B
    could not open.

    Nothing here is authored from a caller's claims. Every sealed-checkpoint
    descriptor is *derived* from the eight run manifests Stage A already
    published and re-verified against the checkpoint bytes on disk; the base
    artifact is copied from the already-authenticated admitted path under a
    no-replace rule; and the resulting provenance binds the signed admission and
    this generation's identity, so it cannot be replayed into a second
    generation.
    """

    required = {
        "source_git_commit",
        "runtime_sha256",
        "launcher_sha256",
        "runtime_authorization_sha256",
    }
    if not isinstance(attestation, Mapping) or set(attestation) != required:
        raise Exp1bEvidenceError("Evidence finalization needs the runtime attestation.")
    if not attestation_matches_authorization(
        role_attestation_document(role=FULL_ROLE, attestation=attestation),
        producer_authorization,
    ):
        raise Exp1bEvidenceError(
            "Evidence finalization ran under a runtime the admission did not "
            "authorize for the producer role."
        )
    _digest_text(admission_sha256, label="admission digest")
    for name, value in (
        ("exp1b protocol", exp1b_protocol_sha256),
        ("exp1b registry", exp1b_registry_sha256),
        ("exp1b amendment", exp1b_amendment_sha256),
        ("ordered population", ordered_population_sha256),
        ("population binding", population_binding_sha256),
    ):
        _digest_text(value, label=f"{name} digest")
    # The descriptor is supplied by the caller's already-authenticated base
    # policy, not authored here; the two derived fields are recomputed below
    # from the bytes actually copied into the generation.
    descriptor_fields = {
        "initialization_kind",
        "amendment_sha256",
        "model_state_sha256",
        "architecture_sha256",
        "producer_git_commit",
        "producer_source_manifest_sha256",
        "training_data_sha256",
        "training_procedure_sha256",
        "shared_across_seeds",
        "not_selected_by_validation_or_test",
    }
    if not isinstance(base_descriptor, Mapping) or set(base_descriptor) != (
        descriptor_fields
    ):
        raise Exp1bEvidenceError(
            "Evidence finalization needs the authenticated base-artifact descriptor."
        )
    base_policy_model_state_sha256 = str(base_descriptor["model_state_sha256"])

    with generation.exclusive():
        # The admitted base bytes, copied in whole under the generation so Stage
        # B authenticates the same object Stage A initialized from.
        base_bytes, base_digest, base_size = stable_bytes(
            Path(admitted_base_artifact), label="admitted base-policy artifact"
        )
        base_directory = generation.directory / "base_policy"
        try:
            base_directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        except OSError as exc:
            raise Exp1bEvidenceError(
                "Experiment 1B base-policy directory cannot be created."
            ) from exc
        _install_or_adopt(
            base_directory / "base_policy.pt",
            base_bytes,
            label="Experiment 1B generation base artifact",
        )

        declared_producer = role_attestation_document(
            role=FULL_ROLE, attestation=attestation
        )
        attested_producers: list[Mapping[str, Any]] = []
        descriptors: list[dict[str, Any]] = []
        for position in range(UNITS):
            seed = REGISTERED_SEEDS[position]
            directory = generation.checkpoint_directory_for(position)
            _payload, checkpoint_sha256, size = stable_bytes(
                directory / "checkpoint.pt",
                label=f"Experiment 1B checkpoint {position}",
            )
            manifest, manifest_digest = _stable_json(
                directory / "run_manifest.json",
                label=f"Experiment 1B run manifest {position}",
            )
            if not isinstance(manifest, Mapping):
                raise Exp1bEvidenceError(
                    f"Experiment 1B run manifest {position} is not an object."
                )
            run_id = str(manifest.get("run_id"))
            row = registry_rows.get(run_id)
            if (
                row is None
                or row["seed"] != seed
                or row["seed_position"] != position
            ):
                raise Exp1bEvidenceError(
                    f"Run manifest {position} is not a committed registry row."
                )
            if (
                manifest.get("checkpoint_sha256") != checkpoint_sha256
                or manifest.get("checkpoint_size_bytes") != size
            ):
                raise Exp1bEvidenceError(
                    f"Run manifest {position} does not describe its checkpoint."
                )
            if (
                manifest.get("initialization_artifact_sha256") != base_digest
                or manifest.get("restored_base_model_state_sha256")
                != base_policy_model_state_sha256
            ):
                raise Exp1bEvidenceError(
                    f"Run manifest {position} did not initialize from the admitted "
                    "base artifact."
                )
            # Finding 4: the provenance's producer identity and admission digest
            # are *derived* from what the eight runs actually recorded, and all
            # eight must agree. A caller-supplied attestation that no run
            # attested to can no longer be sealed alongside them.
            if manifest.get("admission_sha256") != admission_sha256:
                raise Exp1bEvidenceError(
                    f"Run manifest {position} names a different signed admission."
                )
            try:
                run_producer = _role_attestation(
                    manifest.get("producer_attestation"),
                    path=f"run manifest {position}.producer_attestation",
                    expected_role=FULL_ROLE,
                )
            except Exp1bSchemaError as exc:
                raise Exp1bEvidenceError(
                    f"Run manifest {position} producer attestation: {exc}"
                ) from exc
            if run_producer != declared_producer:
                raise Exp1bEvidenceError(
                    f"Run manifest {position} names a different producer runtime "
                    "than the finalizing process."
                )
            attested_producers.append(run_producer)
            descriptors.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "seed_position": position,
                    "environment_interactions": TERMINAL_ENVIRONMENT_INTERACTIONS,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_size_bytes": size,
                    "model_state_sha256": str(manifest["model_state_sha256"]),
                    "sealed": True,
                    "run_manifest_sha256": manifest_digest,
                }
            )

        if len(attested_producers) != UNITS or any(
            item != declared_producer for item in attested_producers
        ):
            raise Exp1bEvidenceError(
                "The eight run manifests do not unanimously attest one producer "
                "runtime; provenance cannot be derived from them."
            )
        schedule_sha256 = exp1b_document_sha256(
            [
                {
                    "seed_position": item["seed_position"],
                    "seed": item["seed"],
                    "run_id": item["run_id"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                }
                for item in descriptors
            ]
        )
        provenance = {
            "schema_name": PROVENANCE_SCHEMA_NAME,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "route_id": BRIDGE_ROUTE_ID,
            "protocol_id": PROTOCOL_ID,
            "exp1b_protocol_sha256": exp1b_protocol_sha256,
            "exp1b_registry_sha256": exp1b_registry_sha256,
            "exp1b_amendment_sha256": exp1b_amendment_sha256,
            "admission_sha256": admission_sha256,
            "generation_sha256": generation.generation_sha256,
            "parent": dict(parent),
            "ordered_population_sha256": ordered_population_sha256,
            "population_binding_sha256": population_binding_sha256,
            "seed_ids": list(REGISTERED_SEEDS),
            "base_policy_artifact": {
                **{key: base_descriptor[key] for key in descriptor_fields},
                "checkpoint_sha256": base_digest,
                "checkpoint_size_bytes": base_size,
            },
            "sealed_checkpoints": [
                {key: item[key] for key in sorted(item) if key != "run_manifest_sha256"}
                for item in descriptors
            ],
            "run_manifest_sha256s": [
                item["run_manifest_sha256"] for item in descriptors
            ],
            # Derived from the eight authenticated manifests, all of which
            # attested this identity, rather than asserted independently.
            "producer_attestation": dict(attested_producers[0]),
            "request_schedule_sha256": schedule_sha256,
            "access_state": "sealed_octet_complete",
            "scientific_selection": False,
            "validation_select_access": False,
            "test_access": False,
        }
        # Adopt-or-refuse, not refuse-outright. A finalization interrupted after
        # the base copy but before the provenance previously left the generation
        # permanently unfinishable; now the second attempt installs the missing
        # provenance, and a *different* provenance is a refusal rather than a
        # silently tolerated race.
        _install_or_adopt(
            generation.provenance_path,
            canonical_json_bytes(provenance) + b"\n",
            label="Experiment 1B substitute provenance",
        )
    return Exp1bFinalizedEvidence(
        provenance_sha256=exp1b_document_sha256(provenance),
        generation_sha256=generation.generation_sha256,
        base_artifact_sha256=base_digest,
        checkpoint_count=len(descriptors),
    )


def _authenticate_run_manifest(
    path: Path,
    *,
    label: str,
    expected_run_id: str,
    expected_seed: int,
    expected_position: int,
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
    expected_effective_config_sha256: str,
    expected_train_ordered_record_sha256: str,
) -> dict[str, Any]:
    document, digest = _stable_json(path, label=label)
    if not isinstance(document, Mapping):
        raise Exp1bEvidenceError(f"{label} is not an object.")
    required = {
        "schema_name",
        "schema_version",
        "run_id",
        "seed",
        "seed_position",
        "environment_interactions",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "model_state_sha256",
        "effective_config_sha256",
        "train_split_ordered_record_sha256",
        "train_record_count",
        "resolved_evaluation_data",
        "initialization_kind",
        "initialization_artifact_sha256",
        "restored_base_model_state_sha256",
        "applied_seed",
        "counters",
        # Finding 4: each run says, in its own durable record, which signed
        # admission opened it and which producer runtime produced it. Provenance
        # is then derived from these eight rather than asserted alongside them.
        "admission_sha256",
        "producer_attestation",
    }
    if set(document) != required:
        raise Exp1bEvidenceError(f"{label} field inventory differs.")
    if (
        _strict_text(document["schema_name"], label=f"{label} schema name")
        != _RUN_MANIFEST_SCHEMA
        or _strict_int(document["schema_version"], label=f"{label} schema version") != 1
    ):
        raise Exp1bEvidenceError(f"{label} schema differs.")
    if (
        _strict_text(document["run_id"], label=f"{label} run ID") != expected_run_id
        or _strict_int(document["seed"], label=f"{label} seed", minimum=0)
        != expected_seed
        or _strict_int(document["seed_position"], label=f"{label} position", minimum=0)
        != expected_position
    ):
        raise Exp1bEvidenceError(f"{label} identity differs from its registry row.")
    if (
        _strict_int(
            document["environment_interactions"], label=f"{label} interactions"
        )
        != TERMINAL_ENVIRONMENT_INTERACTIONS
    ):
        raise Exp1bEvidenceError(f"{label} is not the budget-final run.")
    if (
        _digest_text(document["checkpoint_sha256"], label=f"{label} checkpoint digest")
        != checkpoint_sha256
        or _strict_int(
            document["checkpoint_size_bytes"], label=f"{label} checkpoint size",
            minimum=1,
        )
        != checkpoint_size_bytes
    ):
        raise Exp1bEvidenceError(
            f"{label} does not describe the checkpoint bytes on disk."
        )
    _digest_text(document["model_state_sha256"], label=f"{label} model state")
    _digest_text(
        document["initialization_artifact_sha256"],
        label=f"{label} initialization artifact",
    )
    _digest_text(
        document["restored_base_model_state_sha256"],
        label=f"{label} restored base state",
    )
    _strict_text(document["initialization_kind"], label=f"{label} initialization kind")
    _digest_text(document["admission_sha256"], label=f"{label} admission digest")
    try:
        _role_attestation(
            document["producer_attestation"],
            path=f"{label}.producer_attestation",
            expected_role=FULL_ROLE,
        )
    except Exp1bSchemaError as exc:
        raise Exp1bEvidenceError(f"{label} producer attestation: {exc}") from exc
    if (
        _strict_int(document["applied_seed"], label=f"{label} applied seed", minimum=0)
        != expected_seed
    ):
        raise Exp1bEvidenceError(
            f"{label} did not apply its registered seed to the run."
        )
    counters = document["counters"]
    if not isinstance(counters, Mapping) or not counters:
        raise Exp1bEvidenceError(f"{label} carries no trainer counters.")
    for name, value in counters.items():
        _strict_int(value, label=f"{label} counter {name!r}", minimum=0)
    if (
        counters.get("environment_interactions")
        != TERMINAL_ENVIRONMENT_INTERACTIONS
    ):
        raise Exp1bEvidenceError(
            f"{label} counters disagree with the registered budget."
        )
    if (
        _digest_text(
            document["effective_config_sha256"], label=f"{label} effective config"
        )
        != expected_effective_config_sha256
    ):
        raise Exp1bEvidenceError(
            f"{label} was not produced under the registered effective configuration."
        )
    if (
        _digest_text(
            document["train_split_ordered_record_sha256"], label=f"{label} train split"
        )
        != expected_train_ordered_record_sha256
    ):
        raise Exp1bEvidenceError(
            f"{label} did not train on the registered ordered train split."
        )
    if (
        _strict_int(document["train_record_count"], label=f"{label} record count")
        != TRAINING_RECORD_COUNT
    ):
        raise Exp1bEvidenceError(
            f"{label} did not train on all {TRAINING_RECORD_COUNT} records."
        )
    _strict_false(
        document["resolved_evaluation_data"],
        label=f"{label} resolved_evaluation_data",
    )
    return {**document, "run_manifest_sha256": digest}


def authenticate_sealed_octet_checkpoints(
    *,
    generation: Exp1bEvidenceGeneration,
    provenance: Mapping[str, Any],
    registry_rows: Mapping[str, Mapping[str, Any]],
    effective_config_sha256: str,
    train_ordered_record_sha256: str,
) -> tuple[AuthenticatedCheckpoint, ...]:
    """Prove all eight budget-final checkpoints exist before the route opens.

    Each descriptor in the provenance must be matched by real bytes under the
    evidence generation: the checkpoint file's size and SHA-256, and a sibling
    run manifest carrying the model-state and effective-config digests, the
    registry-bound identity, the terminal interaction count, the 1,024-record
    train identity, and an explicit no-evaluation-access claim.

    Nothing is deserialized. This authenticates opaque bytes only.
    """

    descriptors = provenance.get("sealed_checkpoints")
    if not isinstance(descriptors, Sequence) or len(descriptors) != UNITS:
        raise Exp1bEvidenceError(
            f"Experiment 1B requires exactly {UNITS} sealed checkpoint descriptors."
        )
    recorded_manifests = provenance.get("run_manifest_sha256s")
    if not isinstance(recorded_manifests, Sequence) or len(
        recorded_manifests
    ) != UNITS:
        raise Exp1bEvidenceError(
            f"Experiment 1B requires exactly {UNITS} run-manifest digests."
        )
    recorded_admission = provenance.get("admission_sha256")
    provenance_producer = provenance.get("producer_attestation")
    if not isinstance(provenance_producer, Mapping):
        raise Exp1bEvidenceError("Substitute provenance has no producer attestation.")
    authenticated: list[AuthenticatedCheckpoint] = []
    seen_paths: set[Path] = set()
    for offset, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise Exp1bEvidenceError(f"Checkpoint descriptor {offset} is not an object.")
        run_id = str(descriptor["run_id"])
        seed = int(descriptor["seed"])
        position = int(descriptor["seed_position"])
        if position != offset or seed != REGISTERED_SEEDS[offset]:
            raise Exp1bEvidenceError(
                f"Checkpoint descriptor {offset} is out of registered seed order."
            )
        row = registry_rows.get(run_id)
        if row is None or row["seed"] != seed or row["seed_position"] != position:
            raise Exp1bEvidenceError(
                f"Checkpoint descriptor {offset} is not a committed registry row."
            )
        seed_directory = generation.checkpoint_directory / f"seed-{position}"
        checkpoint_path = seed_directory / "checkpoint.pt"
        manifest_path = seed_directory / "run_manifest.json"
        if checkpoint_path in seen_paths:
            raise Exp1bEvidenceError("Two seeds resolve to one checkpoint artifact.")
        seen_paths.add(checkpoint_path)

        _payload, checkpoint_sha, size = stable_bytes(
            checkpoint_path, label=f"Experiment 1B checkpoint {position}"
        )
        if checkpoint_sha != descriptor["checkpoint_sha256"]:
            raise Exp1bEvidenceError(
                f"Checkpoint {position} bytes differ from their sealed descriptor."
            )
        if size != descriptor["checkpoint_size_bytes"]:
            raise Exp1bEvidenceError(
                f"Checkpoint {position} size differs from its sealed descriptor."
            )
        manifest = _authenticate_run_manifest(
            manifest_path,
            label=f"Experiment 1B run manifest {position}",
            expected_run_id=run_id,
            expected_seed=seed,
            expected_position=position,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_size_bytes=size,
            expected_effective_config_sha256=_digest_text(
                effective_config_sha256, label="registered effective config"
            ),
            expected_train_ordered_record_sha256=_digest_text(
                train_ordered_record_sha256, label="registered train split"
            ),
        )
        if manifest["model_state_sha256"] != descriptor["model_state_sha256"]:
            raise Exp1bEvidenceError(
                f"Checkpoint {position} model-state digest differs from its descriptor."
            )
        # The provenance records a digest per manifest; recomputing one and not
        # comparing it made the record decorative. An edited manifest previously
        # still opened the route under the original provenance.
        if manifest["run_manifest_sha256"] != recorded_manifests[position]:
            raise Exp1bEvidenceError(
                f"Run manifest {position} does not match the digest its "
                "provenance recorded for that position."
            )
        # Unanimity: all eight runs must name the same signed admission and the
        # same producer runtime, and both must be the ones the provenance was
        # derived from.
        if manifest["admission_sha256"] != recorded_admission:
            raise Exp1bEvidenceError(
                f"Run manifest {position} names a different signed admission "
                "than the provenance it was finalized into."
            )
        if dict(manifest["producer_attestation"]) != dict(provenance_producer):
            raise Exp1bEvidenceError(
                f"Run manifest {position} names a different producer runtime "
                "than the provenance it was finalized into."
            )
        authenticated.append(
            AuthenticatedCheckpoint(
                seed_position=position,
                seed=seed,
                run_id=run_id,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_size_bytes=size,
                model_state_sha256=str(manifest["model_state_sha256"]),
                effective_config_sha256=str(manifest["effective_config_sha256"]),
                environment_interactions=int(manifest["environment_interactions"]),
                run_manifest_sha256=str(manifest["run_manifest_sha256"]),
            )
        )
    configs = {item.effective_config_sha256 for item in authenticated}
    if len(configs) != 1:
        raise Exp1bEvidenceError(
            "The eight Experiment 1B runs do not share one effective configuration."
        )
    if len({item.checkpoint_sha256 for item in authenticated}) != UNITS:
        raise Exp1bEvidenceError("Two Experiment 1B runs produced identical bytes.")
    return tuple(authenticated)


def authenticate_base_artifact(
    *,
    generation: Exp1bEvidenceGeneration,
    provenance: Mapping[str, Any],
) -> str:
    """Prove the admitted base artifact's bytes exist and match its descriptor.

    Opaque bytes only; the model-state digest is carried by the descriptor and
    re-checked against the run manifests by
    :func:`authenticate_sealed_octet_checkpoints`.
    """

    descriptor = provenance.get("base_policy_artifact")
    if not isinstance(descriptor, Mapping):
        raise Exp1bEvidenceError("Substitute provenance has no base-artifact block.")
    path = generation.directory / "base_policy" / "base_policy.pt"
    _payload, digest, size = stable_bytes(path, label="Experiment 1B base artifact")
    if digest != descriptor["checkpoint_sha256"]:
        raise Exp1bEvidenceError(
            "Base-artifact bytes differ from the admitted descriptor."
        )
    if size != descriptor["checkpoint_size_bytes"]:
        raise Exp1bEvidenceError(
            "Base-artifact size differs from the admitted descriptor."
        )
    return digest
