#!/usr/bin/env python3
"""Generation-atomic publication for Phase 4 figure artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from utils.run_identity import canonical_json_bytes, canonical_json_sha256


PHASE4_FIGURE_PUBLICATION_SCHEMA_VERSION = 1
PHASE4_FIGURE_OUTPUTS = frozenset(
    {
        "fig_phase4_2x2_norm_ablation.pdf",
        "fig_phase4_2x2_norm_ablation.png",
        "fig_phase4_bar_comparison.pdf",
        "table_phase4_2x2_norm_ablation.tex",
    }
)

_MANIFEST_NAME = "MANIFEST.json"
_CURRENT_NAME = "CURRENT.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_READ_SIZE = 1024 * 1024

_CHECKPOINT_DESIGN_FIELDS = (
    "condition",
    "seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "model_state_sha256",
    "checkpoint_step",
    "training_run_id",
    "config_sha256",
    "rl_config_sha256",
    "model_config_sha256",
    "dataset_provenance_sha256",
    "producer_git_commit",
    "producer_source_manifest_sha256",
    "training_runtime_artifact_sha256",
    "initialization_kind",
    "checkpoint_schema_version",
    "training_invocation_schema_version",
    "enable_contraction",
    "disable_value_head_norm",
    "latent_projection_mode",
    "latent_ball_radius",
)


class Phase4FigurePublicationError(RuntimeError):
    """Raised when a Phase 4 figure generation cannot be published safely."""


@dataclass(frozen=True)
class Phase4FigurePublicationIdentity:
    """Immutable inputs that identify one Phase 4 figure generation."""

    summary_schema_version: int
    summary_experiment: str
    summary_sha256: str
    summary_identity_sha256: str
    evaluator_git_commit: str
    evaluator_source_manifest_sha256: str
    evaluator_runtime_artifact_sha256: str
    producer_git_commit: str
    producer_source_manifest_sha256: str
    training_runtime_artifact_sha256: str
    figure_git_commit: str
    figure_source_manifest_sha256: str
    figure_runtime_artifact_sha256: str
    checkpoint_design_identity_sha256: str
    checkpoint_run_count: int
    diagnostic_dataset_name: str
    diagnostic_dataset_sha256: str
    diagnostic_ordered_states_sha256: str
    diagnostic_state_count: int


@dataclass(frozen=True)
class PublishedPhase4Generation:
    """Paths and hashes for one immutable published generation."""

    publication_root: Path
    generation_id: str
    generation_path: Path
    manifest_sha256: str
    current_path: Path


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Phase4FigurePublicationError(
            f"{label} must be 64 lowercase hexadecimal characters."
        )
    return value


def _require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise Phase4FigurePublicationError(
            f"{label} must be 40 lowercase hexadecimal characters."
        )
    return value


def _strict_json_object(
    pairs: Sequence[tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase4FigurePublicationError(
                f"JSON object contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def strict_json_loads(payload: bytes, label: str) -> Dict[str, Any]:
    """Parse one strict UTF-8 JSON object and reject duplicate keys."""

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except Phase4FigurePublicationError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Phase4FigurePublicationError(
            f"{label} is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Phase4FigurePublicationError(f"{label} must be a JSON object.")
    return value


def _stable_read_regular_file(
    path: Path,
    label: str,
    *,
    flush: bool = False,
    require_single_link: bool = False,
) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            f"{label} cannot be opened without following symlinks."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Phase4FigurePublicationError(f"{label} is not a regular file.")
        if require_single_link and before.st_nlink != 1:
            raise Phase4FigurePublicationError(
                f"{label} must have exactly one hard link."
            )
        chunks = []
        while True:
            block = os.read(descriptor, _READ_SIZE)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        if flush:
            _fsync_fd(descriptor, f"file:{label}")
        after = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            f"{label} changed or could not be read."
        ) from exc
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_nlink",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(path_stat, field)
        for field in stable_fields
    ) or len(payload) != before.st_size:
        raise Phase4FigurePublicationError(f"{label} changed while being read.")
    return payload, hashlib.sha256(payload).hexdigest()


def stable_regular_file_bytes(
    path: str | Path,
    label: str,
) -> tuple[bytes, str]:
    """Return stable bytes and SHA-256 for one non-symlink regular file."""

    return _stable_read_regular_file(Path(path), label)


def verify_runtime_archive_sha256(
    path_value: str | Path,
    expected_sha256: str,
    label: str,
) -> str:
    """Rehash one explicitly absolute runtime archive without following links."""

    _require_sha256(expected_sha256, f"Expected {label} SHA-256")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise Phase4FigurePublicationError(
            f"{label} path must be absolute."
        )
    _, actual_sha256 = _stable_read_regular_file(path, label)
    if actual_sha256 != expected_sha256:
        raise Phase4FigurePublicationError(
            f"{label} bytes differ from the authorized SHA-256."
        )
    return actual_sha256


def build_phase4_figure_publication_identity(
    summary: Mapping[str, Any],
    *,
    summary_sha256: str,
    evaluator_runtime_artifact_sha256: str,
    training_runtime_artifact_sha256: str,
    figure_git_commit: str,
    figure_source_manifest_sha256: str,
    figure_runtime_artifact_sha256: str,
    expected_producer_source: Mapping[str, Any],
) -> Phase4FigurePublicationIdentity:
    """Build the immutable identity recorded by a figure generation."""

    _require_sha256(summary_sha256, "Summary SHA-256")
    runs = summary.get("all_results")
    if not isinstance(runs, list) or not runs:
        raise Phase4FigurePublicationError(
            "Summary has no checkpoint design rows."
        )
    ordered_runs = []
    try:
        for run in sorted(runs, key=lambda value: (value["condition"], value["seed"])):
            ordered_runs.append(
                {field: run[field] for field in _CHECKPOINT_DESIGN_FIELDS}
            )
        producer_commits = {run["producer_git_commit"] for run in runs}
        producer_manifests = {
            run["producer_source_manifest_sha256"] for run in runs
        }
        training_runtimes = {
            run["training_runtime_artifact_sha256"] for run in runs
        }
        diagnostic = summary["diagnostic_dataset"]
    except (KeyError, TypeError) as exc:
        raise Phase4FigurePublicationError(
            "Summary is missing a publication identity field."
        ) from exc
    if producer_commits != {expected_producer_source.get("git_commit")} or (
        producer_manifests
        != {expected_producer_source.get("source_manifest_sha256")}
    ):
        raise Phase4FigurePublicationError(
            "Summary producer identity differs from the authorized producer."
        )
    if training_runtimes != {training_runtime_artifact_sha256}:
        raise Phase4FigurePublicationError(
            "Summary training runtime differs from the authorized archive."
        )
    if summary.get("evaluator_runtime_artifact_sha256") != (
        evaluator_runtime_artifact_sha256
    ):
        raise Phase4FigurePublicationError(
            "Summary evaluator runtime differs from the authorized archive."
        )
    checkpoint_design = {
        "schema_version": 1,
        "runs": ordered_runs,
    }
    return Phase4FigurePublicationIdentity(
        summary_schema_version=int(summary["schema_version"]),
        summary_experiment=str(summary["experiment"]),
        summary_sha256=summary_sha256,
        summary_identity_sha256=canonical_json_sha256(summary),
        evaluator_git_commit=_require_commit(
            summary["evaluator_git_commit"], "Evaluator Git commit"
        ),
        evaluator_source_manifest_sha256=_require_sha256(
            summary["evaluator_source_manifest_sha256"],
            "Evaluator source manifest SHA-256",
        ),
        evaluator_runtime_artifact_sha256=_require_sha256(
            evaluator_runtime_artifact_sha256,
            "Evaluator runtime SHA-256",
        ),
        producer_git_commit=_require_commit(
            next(iter(producer_commits)), "Producer Git commit"
        ),
        producer_source_manifest_sha256=_require_sha256(
            next(iter(producer_manifests)),
            "Producer source manifest SHA-256",
        ),
        training_runtime_artifact_sha256=_require_sha256(
            training_runtime_artifact_sha256,
            "Training runtime SHA-256",
        ),
        figure_git_commit=_require_commit(
            figure_git_commit, "Figure-generator Git commit"
        ),
        figure_source_manifest_sha256=_require_sha256(
            figure_source_manifest_sha256,
            "Figure-generator source manifest SHA-256",
        ),
        figure_runtime_artifact_sha256=_require_sha256(
            figure_runtime_artifact_sha256,
            "Figure-generator runtime SHA-256",
        ),
        checkpoint_design_identity_sha256=canonical_json_sha256(
            checkpoint_design
        ),
        checkpoint_run_count=len(ordered_runs),
        diagnostic_dataset_name=str(diagnostic["dataset_name"]),
        diagnostic_dataset_sha256=_require_sha256(
            summary["diagnostic_dataset_sha256"],
            "Diagnostic dataset SHA-256",
        ),
        diagnostic_ordered_states_sha256=_require_sha256(
            diagnostic["ordered_states_sha256"],
            "Diagnostic ordered-states SHA-256",
        ),
        diagnostic_state_count=int(diagnostic["total_selected_records"]),
    )


def _semantic_generation_identity(
    identity: Phase4FigurePublicationIdentity,
) -> Dict[str, Any]:
    return {
        "output_schema_version": PHASE4_FIGURE_PUBLICATION_SCHEMA_VERSION,
        "summary_sha256": identity.summary_sha256,
        "evaluator_runtime_artifact_sha256": (
            identity.evaluator_runtime_artifact_sha256
        ),
        "training_runtime_artifact_sha256": (
            identity.training_runtime_artifact_sha256
        ),
        "figure_source_manifest_sha256": (
            identity.figure_source_manifest_sha256
        ),
        "figure_runtime_artifact_sha256": (
            identity.figure_runtime_artifact_sha256
        ),
    }


def phase4_figure_generation_id(
    identity: Phase4FigurePublicationIdentity,
) -> str:
    """Return the deterministic generation ID for immutable inputs."""

    return canonical_json_sha256(_semantic_generation_identity(identity))


def resolve_owned_publication_root(
    owner_root_value: str | Path,
    publication_root_value: str | Path,
) -> tuple[Path, Path]:
    """Resolve a dedicated publication root strictly within its owner root."""

    requested_owner = Path(owner_root_value).expanduser()
    if requested_owner.is_symlink():
        raise Phase4FigurePublicationError(
            "Publication owner root must not be a symlink."
        )
    try:
        owner_root = requested_owner.resolve(strict=True)
        owner_stat = os.lstat(owner_root)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Publication owner root does not exist."
        ) from exc
    if not stat.S_ISDIR(owner_stat.st_mode):
        raise Phase4FigurePublicationError(
            "Publication owner root must be a directory."
        )
    requested_root = Path(publication_root_value).expanduser()
    if not requested_root.is_absolute():
        requested_root = owner_root / requested_root
    if requested_root.is_symlink():
        raise Phase4FigurePublicationError(
            "Publication root must not be a symlink."
        )
    publication_root = requested_root.resolve(strict=False)
    try:
        relative = publication_root.relative_to(owner_root)
    except ValueError as exc:
        raise Phase4FigurePublicationError(
            "Publication root escapes its declared owner root."
        ) from exc
    if not relative.parts:
        raise Phase4FigurePublicationError(
            "Publication root must be a dedicated child of its owner root."
        )
    return owner_root, publication_root


def _fsync_fd(descriptor: int, label: str) -> None:
    del label
    os.fsync(descriptor)


def _fsync_directory(path: Path, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            f"Directory {label} cannot be opened safely."
        ) from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise Phase4FigurePublicationError(
                f"Directory {label} is not a directory."
            )
        _fsync_fd(descriptor, f"directory:{label}")
    except OSError as exc:
        raise Phase4FigurePublicationError(
            f"Directory {label} could not be flushed."
        ) from exc
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes, label: str) -> None:
    del label
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _write_new_fsynced(path: Path, payload: bytes, label: str) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            f"{label} cannot be created exclusively."
        ) from exc
    try:
        _write_all(descriptor, payload, label)
        _fsync_fd(descriptor, f"file:{label}")
    except OSError as exc:
        raise Phase4FigurePublicationError(
            f"{label} could not be written and flushed."
        ) from exc
    finally:
        os.close(descriptor)


def _directory_entry_modes(path: Path) -> Dict[str, int]:
    try:
        with os.scandir(path) as entries:
            result = {
                entry.name: entry.stat(follow_symlinks=False).st_mode
                for entry in entries
            }
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Publication directory cannot be enumerated safely."
        ) from exc
    return result


def _create_publication_path_durably(
    owner_root: Path,
    publication_root: Path,
) -> None:
    relative_parts = publication_root.relative_to(owner_root).parts
    current = owner_root
    for index, part in enumerate(relative_parts):
        child = current / part
        try:
            child_stat = os.lstat(child)
        except FileNotFoundError:
            try:
                os.mkdir(child, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise Phase4FigurePublicationError(
                    "Publication path component cannot be created safely."
                ) from exc
            child_stat = os.lstat(child)
        except OSError as exc:
            raise Phase4FigurePublicationError(
                "Publication path component cannot be inspected safely."
            ) from exc
        if not stat.S_ISDIR(child_stat.st_mode):
            raise Phase4FigurePublicationError(
                "Publication path component must be a non-symlink directory."
            )
        label = (
            "publication owner after child creation"
            if index == 0
            else "publication root parent after child creation"
        )
        _fsync_directory(current, label)
        current = child


def _validate_root_layout(
    owner_root: Path,
    publication_root: Path,
    *,
    create: bool,
) -> Path:
    try:
        if create:
            _create_publication_path_durably(owner_root, publication_root)
        root_stat = os.lstat(publication_root)
    except OSError as exc:
        action = "created" if create else "read"
        raise Phase4FigurePublicationError(
            f"Publication root cannot be {action} safely."
        ) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise Phase4FigurePublicationError(
            "Publication root must be a non-symlink directory."
        )
    allowed = {"generations", _CURRENT_NAME}
    unexpected = sorted(
        name
        for name in _directory_entry_modes(publication_root)
        if name not in allowed and not name.startswith(".CURRENT.")
    )
    if unexpected:
        raise Phase4FigurePublicationError(
            "Publication root contains historical or unowned content: "
            f"{unexpected}."
        )
    generations = publication_root / "generations"
    try:
        if create:
            try:
                os.mkdir(generations, 0o700)
            except FileExistsError:
                pass
        generations_stat = os.lstat(generations)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Generation root cannot be created safely."
        ) from exc
    if not stat.S_ISDIR(generations_stat.st_mode):
        raise Phase4FigurePublicationError(
            "Generation root must be a non-symlink directory."
        )
    # Fsync even when another publisher, or an earlier failed attempt, left a
    # visible generations directory.  Observing the name does not prove that
    # its parent-directory entry is crash durable.
    _fsync_directory(
        publication_root,
        "publication root after generations validation",
    )
    current = publication_root / _CURRENT_NAME
    try:
        current_stat = os.lstat(current)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Current-generation pointer cannot be inspected."
        ) from exc
    else:
        if not stat.S_ISREG(current_stat.st_mode):
            raise Phase4FigurePublicationError(
                "Current-generation pointer must be a regular file."
            )
    return generations


def _collect_output_records(
    generation_path: Path,
    *,
    flush: bool,
    allow_manifest: bool = False,
) -> list[Dict[str, Any]]:
    modes = _directory_entry_modes(generation_path)
    actual_names = frozenset(modes)
    expected_names = set(PHASE4_FIGURE_OUTPUTS)
    if allow_manifest:
        expected_names.add(_MANIFEST_NAME)
    if actual_names != expected_names:
        raise Phase4FigurePublicationError(
            "Phase 4 figure generation produced an incomplete artifact set: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}."
        )
    records = []
    inodes: set[tuple[int, int]] = set()
    for name in sorted(PHASE4_FIGURE_OUTPUTS):
        if not stat.S_ISREG(modes[name]):
            raise Phase4FigurePublicationError(
                f"Staged output {name!r} is not a regular file."
            )
        file_stat = os.stat(generation_path / name, follow_symlinks=False)
        inode = (file_stat.st_dev, file_stat.st_ino)
        if file_stat.st_nlink != 1 or inode in inodes:
            raise Phase4FigurePublicationError(
                "Staged outputs contain a hard link or inode alias."
            )
        inodes.add(inode)
        payload, digest = _stable_read_regular_file(
            generation_path / name,
            f"staged output {name}",
            flush=flush,
            require_single_link=True,
        )
        records.append({"name": name, "bytes": len(payload), "sha256": digest})
    return records


def _manifest_value(
    identity: Phase4FigurePublicationIdentity,
    generation_id: str,
    outputs: list[Dict[str, Any]],
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": PHASE4_FIGURE_PUBLICATION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "outputs": outputs,
        "summary_identity": {
            "schema_version": identity.summary_schema_version,
            "experiment": identity.summary_experiment,
            "canonical_sha256": identity.summary_identity_sha256,
        },
        "summary_sha256": identity.summary_sha256,
        "evaluator": {
            "git_commit": identity.evaluator_git_commit,
            "source_manifest_sha256": (
                identity.evaluator_source_manifest_sha256
            ),
            "runtime_artifact_sha256": (
                identity.evaluator_runtime_artifact_sha256
            ),
        },
        "producer": {
            "git_commit": identity.producer_git_commit,
            "source_manifest_sha256": (
                identity.producer_source_manifest_sha256
            ),
        },
        "authorized_training_runtime_sha256": (
            identity.training_runtime_artifact_sha256
        ),
        "figure_generator": {
            "git_commit": identity.figure_git_commit,
            "source_manifest_sha256": (
                identity.figure_source_manifest_sha256
            ),
            "runtime_artifact_sha256": (
                identity.figure_runtime_artifact_sha256
            ),
        },
        "checkpoint_design_identity": {
            "schema_version": 1,
            "run_count": identity.checkpoint_run_count,
            "sha256": identity.checkpoint_design_identity_sha256,
        },
        "diagnostic_dataset_identity": {
            "dataset_name": identity.diagnostic_dataset_name,
            "manifest_sha256": identity.diagnostic_dataset_sha256,
            "ordered_states_sha256": (
                identity.diagnostic_ordered_states_sha256
            ),
            "state_count": identity.diagnostic_state_count,
        },
        "generated_at": generated_at,
    }


def _generation_exists(generation_path: Path) -> bool:
    try:
        mode = os.lstat(generation_path).st_mode
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Final generation path cannot be inspected."
        ) from exc
    if stat.S_ISLNK(mode):
        raise Phase4FigurePublicationError(
            "Final generation path must not be a symlink."
        )
    raise Phase4FigurePublicationError(
        "Final generation path already exists and will not be overwritten."
    )


def _rename_directory_no_replace(
    parent_path: Path,
    source_name: str,
    destination_name: str,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(parent_path, flags)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise Phase4FigurePublicationError(
                "Host has no atomic no-replace publication primitive."
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            descriptor,
            os.fsencode(source_name),
            descriptor,
            os.fsencode(destination_name),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise Phase4FigurePublicationError(
                "Final generation path already exists and will not be overwritten."
            )
        raise Phase4FigurePublicationError(
            "Atomic generation publication failed: "
            f"{os.strerror(error_number)}."
        )
    finally:
        os.close(descriptor)


def _replace_pointer(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _inode_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))


def _unlink_if_same(path: Path, identity: tuple[int, int, int]) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError:
        return
    if _inode_identity(current) == identity:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _remove_tree_if_same(path: Path, identity: tuple[int, int, int]) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError:
        return
    if _inode_identity(current) == identity and stat.S_ISDIR(current.st_mode):
        shutil.rmtree(path, ignore_errors=True)


def _verify_pointer_temp(
    path: Path,
    identity: tuple[int, int, int],
    expected_bytes: bytes,
) -> None:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Current-generation pointer temporary file is unavailable."
        ) from exc
    if _inode_identity(before) != identity:
        raise Phase4FigurePublicationError(
            "Current-generation pointer temporary path was replaced."
        )
    payload, _ = _stable_read_regular_file(
        path,
        "current-generation pointer temporary file",
        require_single_link=True,
    )
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Current-generation pointer temporary file changed."
        ) from exc
    if _inode_identity(after) != identity or payload != expected_bytes:
        raise Phase4FigurePublicationError(
            "Current-generation pointer temporary file changed before replace."
        )


def _current_pointer_bytes_if_present(
    owner_root: Path,
    publication_root: Path,
) -> bytes | None:
    current_path = publication_root / _CURRENT_NAME
    try:
        os.lstat(current_path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Current-generation pointer cannot be inspected."
        ) from exc
    resolve_current_phase4_generation(owner_root, publication_root)
    payload, _ = _stable_read_regular_file(
        current_path,
        "prior current-generation pointer",
        require_single_link=True,
    )
    return payload


def _restore_prior_pointer(
    publication_root: Path,
    prior_pointer_bytes: bytes | None,
    replaced_identity: tuple[int, int, int],
) -> None:
    current_path = publication_root / _CURRENT_NAME
    try:
        current_stat = os.lstat(current_path)
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Invalid current-generation pointer cannot be recovered."
        ) from exc
    if _inode_identity(current_stat) != replaced_identity:
        raise Phase4FigurePublicationError(
            "Current-generation pointer changed before recovery."
        )
    if prior_pointer_bytes is None:
        current_path.unlink()
        _fsync_directory(publication_root, "publication root after pointer recovery")
        return

    descriptor, restore_name = tempfile.mkstemp(
        prefix=".CURRENT.restore.",
        suffix=".tmp",
        dir=publication_root,
    )
    restore_path = Path(restore_name)
    restore_identity = _inode_identity(os.fstat(descriptor))
    try:
        _write_all(
            descriptor,
            prior_pointer_bytes,
            "current-generation pointer recovery",
        )
        _fsync_fd(descriptor, "file:current-generation pointer recovery")
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Prior current-generation pointer cannot be recovered."
        ) from exc
    finally:
        os.close(descriptor)
    try:
        _verify_pointer_temp(
            restore_path,
            restore_identity,
            prior_pointer_bytes,
        )
        os.replace(restore_path, current_path)
        _fsync_directory(publication_root, "publication root after pointer recovery")
    finally:
        _unlink_if_same(restore_path, restore_identity)


def _identity_from_manifest(
    manifest: Mapping[str, Any],
) -> Phase4FigurePublicationIdentity:
    expected_fields = {
        "schema_version",
        "generation_id",
        "outputs",
        "summary_identity",
        "summary_sha256",
        "evaluator",
        "producer",
        "authorized_training_runtime_sha256",
        "figure_generator",
        "checkpoint_design_identity",
        "diagnostic_dataset_identity",
        "generated_at",
    }
    if set(manifest) != expected_fields:
        raise Phase4FigurePublicationError(
            "Generation manifest has missing or unexpected fields."
        )
    if manifest["schema_version"] != PHASE4_FIGURE_PUBLICATION_SCHEMA_VERSION:
        raise Phase4FigurePublicationError(
            "Generation manifest schema version is unsupported."
        )
    summary = manifest["summary_identity"]
    evaluator = manifest["evaluator"]
    producer = manifest["producer"]
    figure = manifest["figure_generator"]
    checkpoint = manifest["checkpoint_design_identity"]
    diagnostic = manifest["diagnostic_dataset_identity"]
    nested_fields = (
        (summary, {"schema_version", "experiment", "canonical_sha256"}),
        (
            evaluator,
            {"git_commit", "source_manifest_sha256", "runtime_artifact_sha256"},
        ),
        (producer, {"git_commit", "source_manifest_sha256"}),
        (
            figure,
            {"git_commit", "source_manifest_sha256", "runtime_artifact_sha256"},
        ),
        (checkpoint, {"schema_version", "run_count", "sha256"}),
        (
            diagnostic,
            {
                "dataset_name",
                "manifest_sha256",
                "ordered_states_sha256",
                "state_count",
            },
        ),
    )
    if any(not isinstance(value, dict) or set(value) != fields for value, fields in nested_fields):
        raise Phase4FigurePublicationError(
            "Generation manifest contains malformed identity metadata."
        )
    if not isinstance(manifest["generated_at"], str) or not manifest["generated_at"]:
        raise Phase4FigurePublicationError(
            "Generation manifest timestamp is missing."
        )
    for value, label in (
        (summary["schema_version"], "Summary schema version"),
        (checkpoint["schema_version"], "Checkpoint identity schema version"),
        (checkpoint["run_count"], "Checkpoint run count"),
        (diagnostic["state_count"], "Diagnostic state count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Phase4FigurePublicationError(f"{label} must be positive.")
    for value, label in (
        (summary["experiment"], "Summary experiment"),
        (diagnostic["dataset_name"], "Diagnostic dataset name"),
    ):
        if not isinstance(value, str) or not value:
            raise Phase4FigurePublicationError(f"{label} must be non-empty.")
    return Phase4FigurePublicationIdentity(
        summary_schema_version=summary["schema_version"],
        summary_experiment=summary["experiment"],
        summary_sha256=_require_sha256(
            manifest["summary_sha256"], "Summary SHA-256"
        ),
        summary_identity_sha256=_require_sha256(
            summary["canonical_sha256"], "Summary identity SHA-256"
        ),
        evaluator_git_commit=_require_commit(
            evaluator["git_commit"], "Evaluator Git commit"
        ),
        evaluator_source_manifest_sha256=_require_sha256(
            evaluator["source_manifest_sha256"],
            "Evaluator source manifest SHA-256",
        ),
        evaluator_runtime_artifact_sha256=_require_sha256(
            evaluator["runtime_artifact_sha256"], "Evaluator runtime SHA-256"
        ),
        producer_git_commit=_require_commit(
            producer["git_commit"], "Producer Git commit"
        ),
        producer_source_manifest_sha256=_require_sha256(
            producer["source_manifest_sha256"],
            "Producer source manifest SHA-256",
        ),
        training_runtime_artifact_sha256=_require_sha256(
            manifest["authorized_training_runtime_sha256"],
            "Training runtime SHA-256",
        ),
        figure_git_commit=_require_commit(
            figure["git_commit"], "Figure-generator Git commit"
        ),
        figure_source_manifest_sha256=_require_sha256(
            figure["source_manifest_sha256"],
            "Figure-generator source manifest SHA-256",
        ),
        figure_runtime_artifact_sha256=_require_sha256(
            figure["runtime_artifact_sha256"],
            "Figure-generator runtime SHA-256",
        ),
        checkpoint_design_identity_sha256=_require_sha256(
            checkpoint["sha256"], "Checkpoint design identity SHA-256"
        ),
        checkpoint_run_count=checkpoint["run_count"],
        diagnostic_dataset_name=diagnostic["dataset_name"],
        diagnostic_dataset_sha256=_require_sha256(
            diagnostic["manifest_sha256"], "Diagnostic dataset SHA-256"
        ),
        diagnostic_ordered_states_sha256=_require_sha256(
            diagnostic["ordered_states_sha256"],
            "Diagnostic ordered-states SHA-256",
        ),
        diagnostic_state_count=diagnostic["state_count"],
    )


def _validate_generation_directory(
    generation_path: Path,
    expected_generation_id: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[Dict[str, Any], str]:
    try:
        generation_mode = os.lstat(generation_path).st_mode
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Published generation is unavailable."
        ) from exc
    if not stat.S_ISDIR(generation_mode):
        raise Phase4FigurePublicationError(
            "Published generation must be a non-symlink directory."
        )
    modes = _directory_entry_modes(generation_path)
    expected_names = set(PHASE4_FIGURE_OUTPUTS) | {_MANIFEST_NAME}
    if set(modes) != expected_names:
        raise Phase4FigurePublicationError(
            "Published generation has missing or unexpected entries."
        )
    if any(not stat.S_ISREG(mode) for mode in modes.values()):
        raise Phase4FigurePublicationError(
            "Published generation entries must be regular files."
        )
    entry_stats = {
        name: os.stat(generation_path / name, follow_symlinks=False)
        for name in expected_names
    }
    entry_inodes = {(value.st_dev, value.st_ino) for value in entry_stats.values()}
    if len(entry_inodes) != len(entry_stats) or any(
        value.st_nlink != 1 for value in entry_stats.values()
    ):
        raise Phase4FigurePublicationError(
            "Published generation contains a hard link or inode alias."
        )
    manifest_bytes, manifest_sha256 = _stable_read_regular_file(
        generation_path / _MANIFEST_NAME,
        "generation manifest",
        require_single_link=True,
    )
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise Phase4FigurePublicationError(
            "Generation manifest differs from CURRENT.json."
        )
    manifest = strict_json_loads(manifest_bytes, "Generation manifest")
    if manifest_bytes != canonical_json_bytes(manifest) + b"\n":
        raise Phase4FigurePublicationError(
            "Generation manifest is not canonically encoded."
        )
    if manifest.get("generation_id") != expected_generation_id:
        raise Phase4FigurePublicationError(
            "Generation manifest ID differs from its directory."
        )
    identity = _identity_from_manifest(manifest)
    if phase4_figure_generation_id(identity) != expected_generation_id:
        raise Phase4FigurePublicationError(
            "Generation ID does not match immutable manifest inputs."
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(PHASE4_FIGURE_OUTPUTS):
        raise Phase4FigurePublicationError(
            "Generation manifest output inventory is incomplete."
        )
    output_records: Dict[str, Dict[str, Any]] = {}
    for record in outputs:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "bytes",
            "sha256",
        }:
            raise Phase4FigurePublicationError(
                "Generation manifest output record is malformed."
            )
        name = record["name"]
        if not isinstance(name, str) or name in output_records:
            raise Phase4FigurePublicationError(
                "Generation manifest output names are invalid or duplicated."
            )
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
        ):
            raise Phase4FigurePublicationError(
                "Generation manifest output byte count is invalid."
            )
        _require_sha256(record["sha256"], f"Output {name} SHA-256")
        output_records[name] = record
    if set(output_records) != PHASE4_FIGURE_OUTPUTS:
        raise Phase4FigurePublicationError(
            "Generation manifest output names differ from the registered set."
        )
    for name in sorted(PHASE4_FIGURE_OUTPUTS):
        payload, digest = _stable_read_regular_file(
            generation_path / name,
            f"published output {name}",
            require_single_link=True,
        )
        record = output_records[name]
        if len(payload) != record["bytes"] or digest != record["sha256"]:
            raise Phase4FigurePublicationError(
                f"Published output {name!r} differs from the manifest."
            )
    return manifest, manifest_sha256


def resolve_current_phase4_generation(
    owner_root_value: str | Path,
    publication_root_value: str | Path,
) -> PublishedPhase4Generation:
    """Resolve CURRENT.json and validate the complete immutable generation."""

    owner_root, publication_root = resolve_owned_publication_root(
        owner_root_value,
        publication_root_value,
    )
    generations = _validate_root_layout(
        owner_root,
        publication_root,
        create=False,
    )
    current_path = publication_root / _CURRENT_NAME
    current_bytes, _ = _stable_read_regular_file(
        current_path,
        "current-generation pointer",
        require_single_link=True,
    )
    current = strict_json_loads(current_bytes, "Current-generation pointer")
    if set(current) != {"schema_version", "generation_id", "manifest_sha256"}:
        raise Phase4FigurePublicationError(
            "Current-generation pointer has missing or unexpected fields."
        )
    if current["schema_version"] != PHASE4_FIGURE_PUBLICATION_SCHEMA_VERSION:
        raise Phase4FigurePublicationError(
            "Current-generation pointer schema version is unsupported."
        )
    generation_id = _require_sha256(current["generation_id"], "Generation ID")
    manifest_sha256 = _require_sha256(
        current["manifest_sha256"], "Manifest SHA-256"
    )
    if current_bytes != canonical_json_bytes(current) + b"\n":
        raise Phase4FigurePublicationError(
            "Current-generation pointer is not canonically encoded."
        )
    generation_path = generations / generation_id
    _validate_generation_directory(
        generation_path,
        generation_id,
        manifest_sha256,
    )
    return PublishedPhase4Generation(
        publication_root=publication_root,
        generation_id=generation_id,
        generation_path=generation_path,
        manifest_sha256=manifest_sha256,
        current_path=current_path,
    )


def publish_phase4_generation(
    owner_root_value: str | Path,
    publication_root_value: str | Path,
    identity: Phase4FigurePublicationIdentity,
    generate: Callable[[Path], None],
    revalidate: Callable[[], Phase4FigurePublicationIdentity],
    *,
    generated_at: str | None = None,
) -> PublishedPhase4Generation:
    """Generate and atomically publish one complete Phase 4 artifact set."""

    owner_root, publication_root = resolve_owned_publication_root(
        owner_root_value,
        publication_root_value,
    )
    generations = _validate_root_layout(
        owner_root,
        publication_root,
        create=True,
    )
    current_path = publication_root / _CURRENT_NAME
    if current_path.exists():
        resolve_current_phase4_generation(owner_root_value, publication_root)

    generation_id = phase4_figure_generation_id(identity)
    generation_path = generations / generation_id
    _generation_exists(generation_path)
    stage = Path(
        tempfile.mkdtemp(prefix=".phase4-generation.stage.", dir=generations)
    )
    stage_identity = _inode_identity(os.lstat(stage))
    pointer_temp: Path | None = None
    pointer_temp_identity: tuple[int, int, int] | None = None
    generation_published = False
    try:
        generate(stage)
        outputs = _collect_output_records(stage, flush=True)
        timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        if not isinstance(timestamp, str) or not timestamp:
            raise Phase4FigurePublicationError(
                "Generation timestamp must be non-empty."
            )
        manifest = _manifest_value(
            identity,
            generation_id,
            outputs,
            timestamp,
        )
        manifest_bytes = canonical_json_bytes(manifest) + b"\n"
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _write_new_fsynced(
            stage / _MANIFEST_NAME,
            manifest_bytes,
            "generation manifest",
        )
        _fsync_directory(stage, "staging generation")

        final_identity = revalidate()
        if final_identity != identity:
            raise Phase4FigurePublicationError(
                "Publication identity changed after artifact generation."
            )
        final_outputs = _collect_output_records(
            stage,
            flush=True,
            allow_manifest=True,
        )
        if final_outputs != outputs:
            raise Phase4FigurePublicationError(
                "Staged outputs changed during final identity validation."
            )
        final_manifest_bytes, final_manifest_sha256 = _stable_read_regular_file(
            stage / _MANIFEST_NAME,
            "generation manifest",
            flush=True,
            require_single_link=True,
        )
        if (
            final_manifest_bytes != manifest_bytes
            or final_manifest_sha256 != manifest_sha256
        ):
            raise Phase4FigurePublicationError(
                "Generation manifest changed before publication."
            )
        if set(_directory_entry_modes(stage)) != (
            set(PHASE4_FIGURE_OUTPUTS) | {_MANIFEST_NAME}
        ):
            raise Phase4FigurePublicationError(
                "Staging generation changed before publication."
            )
        _fsync_directory(stage, "staging generation final")
        _generation_exists(generation_path)
        _rename_directory_no_replace(
            generations,
            stage.name,
            generation_id,
        )
        generation_published = True
        try:
            published_stat = os.lstat(generation_path)
        except OSError as exc:
            raise Phase4FigurePublicationError(
                "Published generation cannot be reidentified after rename."
            ) from exc
        if (
            not stat.S_ISDIR(published_stat.st_mode)
            or _inode_identity(published_stat) != stage_identity
        ):
            raise Phase4FigurePublicationError(
                "Published generation is not the validated staging directory."
            )
        _validate_generation_directory(
            generation_path,
            generation_id,
            manifest_sha256,
        )
        _fsync_directory(generations, "generations")
        _fsync_directory(publication_root, "publication root after generation")

        prior_pointer_bytes = _current_pointer_bytes_if_present(
            owner_root,
            publication_root,
        )
        current = {
            "schema_version": PHASE4_FIGURE_PUBLICATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "manifest_sha256": manifest_sha256,
        }
        current_bytes = canonical_json_bytes(current) + b"\n"
        descriptor, pointer_name = tempfile.mkstemp(
            prefix=".CURRENT.",
            suffix=".tmp",
            dir=publication_root,
        )
        pointer_temp = Path(pointer_name)
        pointer_temp_identity = _inode_identity(os.fstat(descriptor))
        try:
            _write_all(descriptor, current_bytes, "current-generation pointer")
            _fsync_fd(descriptor, "file:current-generation pointer")
        except OSError as exc:
            raise Phase4FigurePublicationError(
                "Current-generation pointer could not be written and flushed."
            ) from exc
        finally:
            os.close(descriptor)
        try:
            current_mode = os.lstat(current_path).st_mode
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise Phase4FigurePublicationError(
                "Current-generation pointer cannot be inspected before update."
            ) from exc
        else:
            if not stat.S_ISREG(current_mode):
                raise Phase4FigurePublicationError(
                    "Current-generation pointer must remain a regular file."
                )
        assert pointer_temp_identity is not None
        _verify_pointer_temp(
            pointer_temp,
            pointer_temp_identity,
            current_bytes,
        )
        _replace_pointer(pointer_temp, current_path)
        try:
            replaced_pointer_identity = _inode_identity(os.lstat(current_path))
        except OSError as exc:
            raise Phase4FigurePublicationError(
                "Current-generation pointer is unavailable after replace."
            ) from exc
        pointer_temp = None
        expected_pointer_identity = pointer_temp_identity
        pointer_temp_identity = None
        try:
            resolved_current = resolve_current_phase4_generation(
                owner_root,
                publication_root,
            )
        except Phase4FigurePublicationError as exc:
            _restore_prior_pointer(
                publication_root,
                prior_pointer_bytes,
                replaced_pointer_identity,
            )
            if prior_pointer_bytes is not None:
                resolve_current_phase4_generation(owner_root, publication_root)
            raise Phase4FigurePublicationError(
                "Current-generation pointer failed post-replace validation and "
                "the prior pointer was restored."
            ) from exc
        if (
            replaced_pointer_identity != expected_pointer_identity
            and resolved_current.generation_id == generation_id
        ):
            _restore_prior_pointer(
                publication_root,
                prior_pointer_bytes,
                replaced_pointer_identity,
            )
            if prior_pointer_bytes is not None:
                resolve_current_phase4_generation(owner_root, publication_root)
            raise Phase4FigurePublicationError(
                "Current-generation pointer did not publish the verified "
                "temporary inode; the prior pointer was restored."
            )
        _fsync_directory(publication_root, "publication root after pointer")
        return PublishedPhase4Generation(
            publication_root=publication_root,
            generation_id=generation_id,
            generation_path=generation_path,
            manifest_sha256=manifest_sha256,
            current_path=current_path,
        )
    except Phase4FigurePublicationError:
        raise
    except OSError as exc:
        raise Phase4FigurePublicationError(
            "Phase 4 generation publication failed."
        ) from exc
    finally:
        if pointer_temp is not None and pointer_temp_identity is not None:
            _unlink_if_same(pointer_temp, pointer_temp_identity)
        if not generation_published:
            _remove_tree_if_same(stage, stage_identity)
