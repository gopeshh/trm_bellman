#!/usr/bin/env fbpython
"""Generate one fail-closed policy-improvement runtime authorization v3."""

# pyre-strict

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

from confirmatory_runtime_launcher import (
    ConfirmatoryRuntimeError,
    SOURCE_MANIFEST_RELATIVE_PATH,
    validate_archive_layout,
    validate_confirmatory_archive_sources,
)
from phase4_runtime_profile import (
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
    AuthorizedPhase4Profile,
    AuthorizedTrainingSource,
    Phase4RuntimeProfileError,
    assert_phase4_archive_matches_profile,
    authorize_phase4_source_profile,
    authorize_phase4_training_source,
)
from scripts.policy_improvement_populations import validate_v2_populations
from scripts.policy_improvement_schema import (
    RUNTIME_ROLES_V2,
    canonical_json_bytes,
    validate_runtime_authorization,
)
from scripts.policy_improvement_v2_registry import (
    load_v2_base_configs,
    validate_v2_registry_document,
)
from scripts.policy_improvement_v2_schema import (
    PolicyImprovementV2SchemaError,
    load_strict_json,
    sha256_json,
    validate_v2_protocol,
)


_PROTOCOL_RELATIVE_PATH = "configs/policy_improvement_v2/protocol.json"
_POPULATIONS_RELATIVE_PATH = "configs/policy_improvement_v2/populations.json"
_REGISTRY_RELATIVE_PATH = "configs/policy_improvement_v2/registry.json"
_THEORY_AMENDMENT_RELATIVE_PATH = (
    "configs/policy_improvement_v2/amendments/theory_bridge_v2.json"
)
_PROFILE_NAMES = (
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE,
    POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE,
)
_RUNTIME_NAMES = ("training", "full", "theory", "audit", "analysis")
_LOWER_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_READ_SIZE = 1024 * 1024
# These identities are from the optimized platform010 phase4 launcher. A Buck
# bootstrap or toolchain change must fail closed until this reviewed boundary
# is updated from a freshly built launcher.
#
# Rebound on 2026-08-24 against the launcher built by
#   buck2 build --local-only @fbcode//mode/opt \
#     fbcode//buiksat_trm:phase4_runtime_launcher
# from two independent cold Buck2 isolation directories that produced
# byte-identical artifacts. Three inputs moved since the previous binding:
# fbsource updated llvm-fb/21/platform010 on 2026-08-20, which relinked both
# native members; seven pinned support sources changed upstream on 2026-08-18
# and 2026-08-19; and the launcher target now sets imports_monitor = False so
# fbcode//python/imports_monitor stays out of the pre-authentication closure.
# The pinned closure therefore stays at exactly the members below plus the two
# dynamic members and the authorized profile sources.
#
# Rebound again on 2026-09-11 against the standalone-packaged launcher built
# from the clean checkout at 73c8780d70d9063fd88b5e8545515998c8d58a9f with
#   buck2 build @fbcode//mode/opt \
#     fbcode//buiksat_trm:phase4_runtime_launcher
# Two of the five pinned values could not hold across that rebuild, and both
# are build-environment bytes that no amount of source review can reproduce:
# _LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256 moved because the fbcode-provided
# support sources inside the pinned closure changed upstream again after the
# 2026-08-24 binding, and _LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256 moved
# because both native members were relinked. Refusing to re-measure them would
# mean the launcher could never be rebuilt on a moved toolchain, which is not a
# security property this boundary is meant to have; it guards against a swapped
# or tampered launcher, and the member inventories are what make that guard
# work. Those inventories did not move: the same 44 pinned support members and
# the same 2 native members are present, byte-for-byte the same set as before.
# _LAUNCHER_ARCHIVE_PREFIX_SIZE, _LAUNCHER_ARCHIVE_PREFIX_SHA256, and
# _LAUNCHER_STARTUP_LOADER_NORMALIZED_SHA256 re-measured identical to their
# 2026-08-24 values on the 2026-09-11 launcher and are deliberately untouched.
# :test_policy_improvement_launcher_identity validates every constant here
# against a real Buck-built launcher PAR; keep that target green when
# rebinding. Run it under @fbcode//mode/opt: _launcher_executable_manifest
# requires fbmake.build_mode == "opt", so a dev-mode launcher is refused before
# these pins are ever consulted, and the target's own reviewed-configuration
# guard errors out instead of skipping when it sees one.
_LAUNCHER_DYNAMIC_SUPPORT_MEMBERS = frozenset(
    {
        "__manifest__.py",
        "__par__/__startup_function_loader__.py",
    }
)
_LAUNCHER_PINNED_SUPPORT_MEMBERS = (
    "__main__.py",
    "__main__.pyc",
    "__par__/__init__.py",
    "__par__/bootstrap.py",
    "__par__/meta_only/__init__.py",
    "__par__/meta_only/bootstrap.py",
    "__par__/meta_only/devfd_zipimport.py",
    "__par__/meta_only/multiprocessing_fork_default.py",
    "__par__/meta_only/pexutil.py",
    "__par__/meta_only/process_title.py",
    "clifoundation/__init__.py",
    "clifoundation/lib/__init__.py",
    "clifoundation/lib/py/__init__.py",
    "clifoundation/lib/py/error/__init__.py",
    "clifoundation/lib/py/error/state.py",
    "clifoundation/lib/py/error/typing.py",
    "clifoundation/lib/py/error/utils.py",
    "clifoundation/lib/py/scrut.py",
    "clifoundation/lib/py/usage/__init__.py",
    "clifoundation/lib/py/usage/additional.py",
    "clifoundation/lib/py/usage/bootstrap.py",
    "clifoundation/lib/py/usage/cinder.py",
    "clifoundation/lib/py/usage/consts.py",
    "clifoundation/lib/py/usage/logger_cat.py",
    "clifoundation/lib/py/usage/sample.py",
    "clifoundation/lib/py/usage/scribe_cat.py",
    "clifoundation/lib/py/usage/state.py",
    "clifoundation/lib/py/usage/typing.py",
    "fbvscode/__init__.py",
    "fbvscode/__main__.py",
    "fbvscode/bootstrapping.py",
    "fbvscode/common.py",
    "fbvscode/pid_inject.py",
    "fbvscode/scribe_logging.py",
    "fbvscode/socket.py",
    "python/__init__.py",
    "python/debuggers/__init__.py",
    "python/debuggers/debugpy.py",
    "python/debuggers/determine_par_type.py",
    "python/debuggers/guess_main_breakpoint.py",
    "python/debuggers/pdb.py",
    "python/debuggers/sys_path_trampoline.py",
    "sitecustomize.py",
    "static_extension_finder.py",
)
_LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256 = (
    "e51996140e698c24a7d66801b415a0f81a526634f6c168f0976f380000b5265e"
)
_LAUNCHER_ARCHIVE_PREFIX_SIZE = 8215
_LAUNCHER_ARCHIVE_PREFIX_SHA256 = (
    "87e71b36ae3f0dff3321a09ad2d09f1255ababea6f2e3577419c7f01c8e50f05"
)
_LAUNCHER_NATIVE_SUPPORT_MEMBERS = (
    "runtime/bin/phase4_runtime_launcher#native-main#platform-runtime#python#py_version_3_12",
    "runtime/lib/__python_generated_allocator_preload",
)
_LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256 = (
    "abb8f9887e32087cf11c4d9c188a96bd41ccab678f3e8262708c9dd807d3b04a"
)
_LAUNCHER_STARTUP_LOADER_NORMALIZED_SHA256 = (
    "a0e47cb96c58fe681f56c0b690c3de15dc6a756dbff74fb7a79ee9e5622dd280"
)
# Exactly the two startup hooks fbcode installs for this target: the static
# extension loader, and the fork-default hook that fbcode/PACKAGE enables
# repository-wide. Both modules are already inside the pinned closure. The
# imports-monitor hook is excluded at the Buck target, not accepted here.
_LAUNCHER_STARTUP_FUNCTIONS = {
    "00_MULTIPROCESSING_FORK_DEFAULT": (
        "__par__.meta_only.multiprocessing_fork_default:set_fork_default"
    ),
    "00_STATIC_EXTENSION_FINDER": "static_extension_finder:_initialize",
}
# The optional group matches only the literal fbcode opt-by-default python
# modifier suffix. Everything else stays anchored: opt build, linux-x86_64,
# platform010, no sanitizer, and a 16-hex configuration hash.
_LAUNCHER_LABEL = re.compile(
    r"^(?:fbcode|fbsource)//[A-Za-z0-9_./-]+:phase4_runtime_launcher "
    r"\(cfg:opt-linux-x86_64-fbcode-platform010-clang[0-9]+-no-san"
    r"(?:-opt-by-default)?#[0-9a-f]{16}\)$"
)


class RuntimeAuthorizationGenerationError(RuntimeError):
    """Raised when authorization inputs cannot be bound without ambiguity."""


@dataclass(frozen=True)
class SourceAuthorization:
    training: AuthorizedTrainingSource
    profiles: Mapping[str, AuthorizedPhase4Profile]


@dataclass(frozen=True)
class PolicyRegistration:
    protocol: dict[str, Any]
    protocol_sha256: str
    registry: dict[str, Any]
    registry_sha256: str
    amendment: dict[str, Any]
    amendment_sha256: str


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class ArtifactAuthorization:
    launcher: ArtifactIdentity
    runtimes: Mapping[str, ArtifactIdentity]


@dataclass(frozen=True)
class PublishedAuthorization:
    path: Path
    sha256: str
    document: dict[str, Any]


def _run_git(root: Path, *arguments: str) -> str:
    environment = {
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise RuntimeAuthorizationGenerationError(
            "Source Git identity cannot be authenticated."
        ) from exc
    return completed.stdout.strip()


def _canonical_existing_directory(path_value: str, *, label: str) -> Path:
    if not os.path.isabs(path_value) or os.path.realpath(path_value) != path_value:
        raise RuntimeAuthorizationGenerationError(
            f"{label} must be an absolute canonical directory path."
        )
    path = Path(path_value)
    try:
        status = path.lstat()
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(f"{label} does not exist.") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise RuntimeAuthorizationGenerationError(f"{label} must be a directory.")
    return path


def require_clean_exact_git_commit(
    project_root: str,
    expected_git_commit: str,
) -> Path:
    """Require one canonical Git root at the exact clean commit."""

    if _LOWER_COMMIT.fullmatch(expected_git_commit) is None:
        raise RuntimeAuthorizationGenerationError(
            "Expected Git commit must be 40 lowercase hexadecimal characters."
        )
    root = _canonical_existing_directory(project_root, label="Source project root")
    top_level = _run_git(root, "rev-parse", "--show-toplevel")
    if os.path.realpath(top_level) != str(root):
        raise RuntimeAuthorizationGenerationError(
            "Source project root must be the Git top level."
        )
    if _run_git(root, "rev-parse", "--verify", "HEAD^{commit}") != expected_git_commit:
        raise RuntimeAuthorizationGenerationError(
            "Source checkout differs from the expected commit."
        )
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeAuthorizationGenerationError("Source checkout is not clean.")
    return root


def _load_canonical_document(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    try:
        value = load_strict_json(path)
    except PolicyImprovementV2SchemaError as exc:
        raise RuntimeAuthorizationGenerationError(
            f"Registered document {relative_path!r} is not strict JSON."
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeAuthorizationGenerationError(
            f"Registered document {relative_path!r} must be one JSON object."
        )
    # Source profiles bind the checked-in spelling. Authorization identities use
    # the canonical parsed bytes, matching the pre-import launcher contract.
    canonical_json_bytes(value)
    return dict(value)


def _validate_theory_amendment(
    amendment: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    populations_sha256: str,
    registry: Mapping[str, Any],
    registry_sha256: str,
) -> None:
    expected_fields = {
        "amendment_id",
        "analysis",
        "bellman_estimators",
        "centering_contract",
        "checkpoint_schedule",
        "created_at_utc",
        "deployment_contract",
        "evaluator_contract",
        "metrics",
        "multi_fidelity_exact_method_screen",
        "outcome_evidence_inspected",
        "population_registry_sha256",
        "predictive_return_estimator",
        "prior_amendment_history_sha256",
        "protocol_id",
        "protocol_schema_name",
        "protocol_schema_version",
        "protocol_sha256",
        "reference_depths",
        "result_schema",
        "schema_name",
        "schema_version",
        "smoke_policy",
        "source_registry_schema_name",
        "source_registry_schema_version",
        "source_registry_sha256",
        "test_data_opened",
    }
    if set(amendment) != expected_fields:
        raise RuntimeAuthorizationGenerationError(
            "Theory amendment field inventory differs."
        )
    required = {
        "schema_name": "policy_improvement_theory_bridge_amendment_v2",
        "schema_version": 1,
        "amendment_id": "pre-stage1-theory-bridge-v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_schema_name": protocol["schema_name"],
        "protocol_schema_version": protocol["schema_version"],
        "protocol_sha256": protocol_sha256,
        "population_registry_sha256": populations_sha256,
        "source_registry_schema_name": registry["schema_name"],
        "source_registry_schema_version": registry["registry_schema_version"],
        "source_registry_sha256": registry_sha256,
        "prior_amendment_history_sha256": hashlib.sha256(b"[]").hexdigest(),
        "test_data_opened": False,
        "outcome_evidence_inspected": False,
    }
    if any(amendment.get(field) != expected for field, expected in required.items()):
        raise RuntimeAuthorizationGenerationError(
            "Theory amendment identity or pre-outcome attestations differ."
        )


def _load_policy_registration(root: Path) -> PolicyRegistration:
    protocol = validate_v2_protocol(
        _load_canonical_document(root, _PROTOCOL_RELATIVE_PATH)
    )
    populations = validate_v2_populations(
        _load_canonical_document(root, _POPULATIONS_RELATIVE_PATH)
    )
    population_sha256 = sha256_json(populations)
    if protocol["population_registry"]["sha256"] != population_sha256:
        raise RuntimeAuthorizationGenerationError(
            "Protocol population-registry digest differs from canonical bytes."
        )
    configs = load_v2_base_configs(protocol, root)
    registry = validate_v2_registry_document(
        _load_canonical_document(root, _REGISTRY_RELATIVE_PATH),
        protocol,
        populations,
        base_configs=configs,
    )
    protocol_sha256 = sha256_json(protocol)
    registry_sha256 = sha256_json(registry)
    amendment = _load_canonical_document(root, _THEORY_AMENDMENT_RELATIVE_PATH)
    _validate_theory_amendment(
        amendment,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        populations_sha256=population_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
    )
    return PolicyRegistration(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        registry=registry,
        registry_sha256=registry_sha256,
        amendment=amendment,
        amendment_sha256=sha256_json(amendment),
    )


def _authenticate_sources(root: Path, expected_git_commit: str) -> SourceAuthorization:
    training = authorize_phase4_training_source(root, expected_git_commit)
    profiles = {
        profile: authorize_phase4_source_profile(root, expected_git_commit, profile)
        for profile in _PROFILE_NAMES
    }
    return SourceAuthorization(training=training, profiles=profiles)


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return (status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode))


def _hash_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(_READ_SIZE), b""):
        digest.update(block)
    return digest.hexdigest()


def _authenticate_artifact(
    path_value: str,
    *,
    label: str,
    archive_validator: Callable[[ZipFile], None] | None = None,
) -> ArtifactIdentity:
    if not os.path.isabs(path_value) or os.path.realpath(path_value) != path_value:
        raise RuntimeAuthorizationGenerationError(
            f"{label} path must be absolute, canonical, and non-symlinked."
        )
    path = Path(path_value)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(f"{label} does not exist.") from exc
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise RuntimeAuthorizationGenerationError(f"{label} must be a regular file.")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(f"{label} cannot be opened.") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            descriptor_before = os.fstat(descriptor)
            if (
                _file_identity(descriptor_before) != _file_identity(path_before)
                or descriptor_before.st_size <= 0
            ):
                raise RuntimeAuthorizationGenerationError(
                    f"{label} changed before authentication."
                )
            digest = _hash_open_file(handle)
            if archive_validator is not None:
                handle.seek(0)
                try:
                    with ZipFile(handle, "r") as archive:
                        archive_validator(archive)
                except BadZipFile as exc:
                    raise RuntimeAuthorizationGenerationError(
                        f"{label} is not a valid ZIP-based PAR."
                    ) from exc
            descriptor_after = os.fstat(descriptor)
        path_after = path.lstat()
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            f"{label} could not be authenticated."
        ) from exc
    finally:
        os.close(descriptor)
    identities = {
        _file_identity(path_before),
        _file_identity(descriptor_before),
        _file_identity(descriptor_after),
        _file_identity(path_after),
    }
    if len(identities) != 1:
        raise RuntimeAuthorizationGenerationError(
            f"{label} changed during authentication."
        )
    return ArtifactIdentity(
        path=path,
        sha256=digest,
        device=descriptor_before.st_dev,
        inode=descriptor_before.st_ino,
    )


def _training_archive_validator(
    expected_manifest_bytes: bytes,
) -> Callable[[ZipFile], None]:
    def validate(archive: ZipFile) -> None:
        validate_archive_layout(archive)
        validate_confirmatory_archive_sources(archive)
        try:
            embedded_manifest = archive.read(SOURCE_MANIFEST_RELATIVE_PATH)
        except (BadZipFile, KeyError, OSError, RuntimeError) as exc:
            raise RuntimeAuthorizationGenerationError(
                "Training PAR lacks its exact producer manifest."
            ) from exc
        if embedded_manifest != expected_manifest_bytes:
            raise RuntimeAuthorizationGenerationError(
                "Training PAR producer manifest differs from the clean checkout."
            )

    return validate


def _profile_archive_validator(
    authorized: AuthorizedPhase4Profile,
) -> Callable[[ZipFile], None]:
    def validate(archive: ZipFile) -> None:
        validate_archive_layout(archive)
        assert_phase4_archive_matches_profile(archive, authorized)

    return validate


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR manifest contains a duplicate JSON key."
            )
        value[key] = item
    return value


def _launcher_member_bytes(archive: ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except (BadZipFile, KeyError, OSError, RuntimeError, zlib.error) as exc:
        raise RuntimeAuthorizationGenerationError(
            f"Launcher PAR member {name!r} cannot be authenticated."
        ) from exc


def _validate_launcher_archive_prefix(archive: ZipFile) -> bytes:
    handle = archive.fp
    if handle is None:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR archive is not backed by an open file."
        )
    infos = archive.infolist()
    if not infos:
        raise RuntimeAuthorizationGenerationError("Launcher PAR is empty.")
    prefix_size = min(info.header_offset for info in infos)
    try:
        position = handle.tell()
        handle.seek(0)
        prefix = handle.read(prefix_size)
        handle.seek(position)
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR executable prefix cannot be authenticated."
        ) from exc
    if (
        prefix_size != _LAUNCHER_ARCHIVE_PREFIX_SIZE
        or hashlib.sha256(prefix).hexdigest() != _LAUNCHER_ARCHIVE_PREFIX_SHA256
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR executable prefix differs from the pinned Buck bootstrap."
        )
    return prefix


def _validate_launcher_buildstamp(archive: ZipFile, prefix: bytes) -> None:
    raw_stamp = _launcher_member_bytes(archive, "BUILDSTAMP")
    try:
        stamp = raw_stamp.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR BUILDSTAMP is not ASCII."
        ) from exc
    if re.fullmatch(r"[0-9a-f]{32}", stamp) is None:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR BUILDSTAMP has an invalid identity."
        )

    # Buck defines BUILDSTAMP as the MD5 of the complete executable PAR before
    # appending the BUILDSTAMP member. Reconstruct that exact pre-stamp archive
    # so a caller cannot redirect extraction by editing the cache key alone.
    unstamped = io.BytesIO()
    unstamped.write(prefix)
    try:
        with ZipFile(unstamped, "a", allowZip64=True) as rebuilt:
            for info in archive.infolist():
                if info.filename == "BUILDSTAMP":
                    continue
                rebuilt.writestr(
                    copy.copy(info),
                    _launcher_member_bytes(archive, info.filename),
                )
    except (BadZipFile, OSError, RuntimeError, zlib.error) as exc:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR pre-BUILDSTAMP bytes cannot be reconstructed."
        ) from exc
    derived = hashlib.md5(unstamped.getvalue()).hexdigest()
    if stamp != derived:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR BUILDSTAMP differs from its archive content."
        )


def _validate_launcher_native_support(archive: ZipFile) -> None:
    runtime_members = {
        info.filename
        for info in archive.infolist()
        if info.filename.startswith("runtime/")
    }
    if runtime_members != set(_LAUNCHER_NATIVE_SUPPORT_MEMBERS):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR native runtime member inventory differs."
        )
    identity = {
        name: hashlib.sha256(_launcher_member_bytes(archive, name)).hexdigest()
        for name in _LAUNCHER_NATIVE_SUPPORT_MEMBERS
    }
    if hashlib.sha256(canonical_json_bytes(identity)).hexdigest() != (
        _LAUNCHER_NATIVE_SUPPORT_MANIFEST_SHA256
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR native runtime support differs."
        )


def _launcher_executable_manifest(archive: ZipFile) -> dict[str, object]:
    try:
        manifest = json.loads(
            _launcher_member_bytes(archive, "__manifest__.json").decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR lacks a valid executable manifest."
        ) from exc
    expected_fields = {
        "buck_labels",
        "env",
        "fbmake",
        "library_versions",
        "python_features",
        "startup_functions",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR executable manifest inventory differs."
        )
    fbmake = manifest["fbmake"]
    if not isinstance(fbmake, dict):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR executable manifest has no fbmake identity."
        )
    build_rule = fbmake.get("build_rule")
    if (
        fbmake.get("main_module") != "phase4_runtime_launcher"
        or "main_function" not in fbmake
        or fbmake["main_function"] is not None
        or fbmake.get("build_rule_type") != "python_binary"
        or type(fbmake.get("rule_type_is_unit_test")) is not int
        or fbmake["rule_type_is_unit_test"] != 0
        or fbmake.get("build_tool") != "buck2"
        or fbmake.get("build_mode") != "opt"
        or fbmake.get("par_style") != "fastzip"
        or fbmake.get("platform") != "platform010"
        or fbmake.get("link_strategy") != "native"
        or not isinstance(build_rule, str)
        or not build_rule.endswith(":phase4_runtime_launcher")
        or manifest["startup_functions"] != _LAUNCHER_STARTUP_FUNCTIONS
        or manifest["env"] != {}
        or manifest["library_versions"] != []
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR does not execute the authenticated launcher module."
        )
    return manifest


def _launcher_expected_modules(
    authorized: AuthorizedPhase4Profile,
) -> frozenset[str]:
    excluded = {
        "__main__.py",
        "sitecustomize.py",
        "static_extension_finder.py",
    }
    paths = {
        *(
            path
            for path in _LAUNCHER_PINNED_SUPPORT_MEMBERS
            if path.endswith(".py")
        ),
        *(path for path in authorized.sources if path.endswith(".py")),
    }
    return frozenset(
        path.removesuffix(".py").replace("/", ".")
        for path in paths - excluded
    )


def _validate_launcher_python_manifest(
    archive: ZipFile,
    authorized: AuthorizedPhase4Profile,
    executable_manifest: Mapping[str, object],
) -> None:
    try:
        tree = ast.parse(
            _launcher_member_bytes(archive, "__manifest__.py").decode("utf-8"),
            filename="__manifest__.py",
        )
        values: dict[str, object] = {}
        for statement in tree.body:
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                raise ValueError("non-literal launcher manifest statement")
            name = statement.targets[0].id
            if name in values:
                raise ValueError("duplicate launcher manifest assignment")
            values[name] = ast.literal_eval(statement.value)
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR Python manifest is not literal-only."
        ) from exc
    expected_fields = {
        "buck_labels",
        "env",
        "fbmake",
        "library_versions",
        "modules",
        "origins",
        "python_features",
        "startup_functions",
    }
    if set(values) != expected_fields or any(
        values[field] != executable_manifest[field]
        for field in expected_fields - {"modules", "origins"}
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR Python and JSON manifests differ."
        )
    modules = values["modules"]
    origins = values["origins"]
    expected_modules = _launcher_expected_modules(authorized)
    if (
        not isinstance(modules, list)
        or not all(isinstance(module, str) for module in modules)
        or len(modules) != len(set(modules))
        or frozenset(modules) != expected_modules
        or not isinstance(origins, tuple)
        or len(origins) != len(modules)
        or not all(
            isinstance(origin, str) and "\n" not in origin and "\r" not in origin
            for origin in origins
        )
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR module provenance differs from its executable closure."
        )


def _validate_launcher_startup_loader(archive: ZipFile) -> None:
    payload = _launcher_member_bytes(
        archive,
        "__par__/__startup_function_loader__.py",
    )
    try:
        lines = payload.decode("utf-8").splitlines(keepends=True)
        variable_lines = [line for line in lines if line.startswith("VARS = ")]
        if len(variable_lines) != 1:
            raise ValueError("launcher startup variables are ambiguous")
        variables = ast.literal_eval(variable_lines[0].removeprefix("VARS = ").strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR startup loader has an invalid generated identity."
        ) from exc
    if (
        not isinstance(variables, dict)
        or set(variables) != {"label", "name"}
        or variables["name"] != "phase4_runtime_launcher"
        or not isinstance(variables["label"], str)
        or _LAUNCHER_LABEL.fullmatch(variables["label"]) is None
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR startup loader identifies another target."
        )
    normalized = "".join(
        line for line in lines if not line.startswith("VARS = ")
    ).encode("utf-8")
    if hashlib.sha256(normalized).hexdigest() != (
        _LAUNCHER_STARTUP_LOADER_NORMALIZED_SHA256
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR startup loader implementation differs."
        )


def _validate_launcher_executable_closure(
    archive: ZipFile,
    authorized: AuthorizedPhase4Profile,
) -> None:
    executable_members = {
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and info.filename.endswith((".py", ".pyc", ".pyo", ".so", ".pyd", ".dylib"))
    }
    profiled_python = {
        path for path in authorized.sources if path.endswith(".py")
    }
    expected = {
        *_LAUNCHER_PINNED_SUPPORT_MEMBERS,
        *_LAUNCHER_DYNAMIC_SUPPORT_MEMBERS,
        *profiled_python,
    }
    if executable_members != expected:
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR executable member inventory differs."
        )
    pinned_identity = {
        name: hashlib.sha256(_launcher_member_bytes(archive, name)).hexdigest()
        for name in _LAUNCHER_PINNED_SUPPORT_MEMBERS
    }
    if hashlib.sha256(canonical_json_bytes(pinned_identity)).hexdigest() != (
        _LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256
    ):
        raise RuntimeAuthorizationGenerationError(
            "Launcher PAR pinned support implementation differs."
        )
    executable_manifest = _launcher_executable_manifest(archive)
    _validate_launcher_python_manifest(archive, authorized, executable_manifest)
    _validate_launcher_startup_loader(archive)


def _launcher_archive_validator(
    authorized: AuthorizedPhase4Profile,
) -> Callable[[ZipFile], None]:
    validate_profile = _profile_archive_validator(authorized)

    def validate(archive: ZipFile) -> None:
        try:
            validate_profile(archive)
        except (ConfirmatoryRuntimeError, Phase4RuntimeProfileError) as exc:
            raise RuntimeAuthorizationGenerationError(
                "Launcher PAR source profile differs from the clean checkout."
            ) from exc
        prefix = _validate_launcher_archive_prefix(archive)
        _validate_launcher_buildstamp(archive, prefix)
        _validate_launcher_native_support(archive)
        _validate_launcher_executable_closure(archive, authorized)

    return validate


def _authenticate_artifacts(
    *,
    launcher_path: str,
    runtime_paths: Mapping[str, str],
    sources: SourceAuthorization,
) -> ArtifactAuthorization:
    if set(runtime_paths) != set(_RUNTIME_NAMES):
        raise RuntimeAuthorizationGenerationError(
            "Exactly five named runtime PAR paths are required."
        )
    launcher = _authenticate_artifact(
        launcher_path,
        label="Launcher",
        archive_validator=_launcher_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE]
        ),
    )
    validators = {
        "training": _training_archive_validator(sources.training.manifest_bytes),
        "full": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE]
        ),
        "theory": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE]
        ),
        "audit": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE]
        ),
        "analysis": _profile_archive_validator(
            sources.profiles[POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE]
        ),
    }
    runtimes = {
        name: _authenticate_artifact(
            runtime_paths[name],
            label=f"{name} runtime PAR",
            archive_validator=validators[name],
        )
        for name in _RUNTIME_NAMES
    }
    identities = {(launcher.device, launcher.inode)}
    identities.update(
        (artifact.device, artifact.inode) for artifact in runtimes.values()
    )
    if len(identities) != len(_RUNTIME_NAMES) + 1:
        raise RuntimeAuthorizationGenerationError(
            "Launcher and the five runtime PAR paths must identify distinct files."
        )
    return ArtifactAuthorization(launcher=launcher, runtimes=runtimes)


def build_runtime_authorization(
    *,
    authorization_id: str,
    created_at_utc: str,
    expected_git_commit: str,
    registration: PolicyRegistration,
    sources: SourceAuthorization,
    artifacts: ArtifactAuthorization,
) -> dict[str, Any]:
    if (
        registration.protocol_sha256 != sha256_json(registration.protocol)
        or registration.registry_sha256 != sha256_json(registration.registry)
        or registration.amendment_sha256 != sha256_json(registration.amendment)
    ):
        raise RuntimeAuthorizationGenerationError(
            "Canonical registration digests differ from their documents."
        )
    _validate_theory_amendment(
        registration.amendment,
        protocol=registration.protocol,
        protocol_sha256=registration.protocol_sha256,
        populations_sha256=registration.protocol["population_registry"]["sha256"],
        registry=registration.registry,
        registry_sha256=registration.registry_sha256,
    )
    if (
        sources.training.git_commit != expected_git_commit
        or set(sources.profiles) != set(_PROFILE_NAMES)
        or any(
            profile.git_commit != expected_git_commit
            for profile in sources.profiles.values()
        )
    ):
        raise RuntimeAuthorizationGenerationError(
            "Source authorizations do not bind the expected exact commit."
        )
    if set(artifacts.runtimes) != set(_RUNTIME_NAMES):
        raise RuntimeAuthorizationGenerationError(
            "Artifact authorization lacks one of the five runtime PARs."
        )
    artifact_identities = {(artifacts.launcher.device, artifacts.launcher.inode)}
    artifact_identities.update(
        (artifact.device, artifact.inode) for artifact in artifacts.runtimes.values()
    )
    if len(artifact_identities) != len(_RUNTIME_NAMES) + 1:
        raise RuntimeAuthorizationGenerationError(
            "Artifact authorization contains aliased files."
        )
    profile_digests = {
        "training": sources.training.source_manifest_sha256,
        "evaluation": sources.training.source_manifest_sha256,
        "audit": sources.profiles[
            POLICY_IMPROVEMENT_AUDIT_SOURCE_PROFILE
        ].source_manifest_sha256,
        "analysis": sources.profiles[
            POLICY_IMPROVEMENT_ANALYSIS_SOURCE_PROFILE
        ].source_manifest_sha256,
        "full": sources.profiles[
            POLICY_IMPROVEMENT_FULL_SOURCE_PROFILE
        ].source_manifest_sha256,
        "theory": sources.profiles[
            POLICY_IMPROVEMENT_THEORY_BRIDGE_SOURCE_PROFILE
        ].source_manifest_sha256,
    }
    runtime_digests = {
        "training": artifacts.runtimes["training"].sha256,
        "evaluation": artifacts.runtimes["training"].sha256,
        "audit": artifacts.runtimes["audit"].sha256,
        "analysis": artifacts.runtimes["analysis"].sha256,
        "full": artifacts.runtimes["full"].sha256,
        "theory": artifacts.runtimes["theory"].sha256,
    }
    role_keys = ("training", "evaluation", "audit", "analysis", "full", "theory")
    document = {
        "schema_name": "policy_improvement_runtime_authorization_v3",
        "schema_version": 3,
        "authorization_id": authorization_id,
        "created_at_utc": created_at_utc,
        "protocol_sha256": registration.protocol_sha256,
        "protocol": {
            "schema_name": registration.protocol["schema_name"],
            "schema_version": registration.protocol["schema_version"],
            "protocol_id": registration.protocol["protocol_id"],
            "sha256": registration.protocol_sha256,
        },
        "registry": {
            "schema_name": registration.registry["schema_name"],
            "schema_version": registration.registry["registry_schema_version"],
            "sha256": registration.registry_sha256,
        },
        "amendments": [
            {
                "schema_name": registration.amendment["schema_name"],
                "schema_version": registration.amendment["schema_version"],
                "amendment_id": registration.amendment["amendment_id"],
                "sha256": registration.amendment_sha256,
            }
        ],
        "producer_git_commit": expected_git_commit,
        "producer_source_manifest_sha256": sources.training.source_manifest_sha256,
        "launcher_sha256": artifacts.launcher.sha256,
        "roles": [
            {
                "role": RUNTIME_ROLES_V2[index],
                "source_git_commit": expected_git_commit,
                "runtime_sha256": runtime_digests[key],
                "runtime_profile_sha256": profile_digests[key],
                "selected_source_manifest_sha256": profile_digests[key],
            }
            for index, key in enumerate(role_keys)
        ],
    }
    validated = validate_runtime_authorization(document)
    if canonical_json_bytes(validated) != canonical_json_bytes(document):
        raise RuntimeAuthorizationGenerationError(
            "Runtime authorization changed during self-validation."
        )
    return validated


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeAuthorizationGenerationError(
                "Runtime authorization publication made no progress."
            )
        offset += written


def _publish_new_private_file(
    *,
    owner_directory: str,
    output_name: str,
    project_root: Path,
    payload: bytes,
) -> Path:
    owner = _canonical_existing_directory(
        owner_directory,
        label="Authorization owner directory",
    )
    try:
        inside_project = os.path.commonpath((str(project_root), str(owner))) == str(
            project_root
        )
    except ValueError:
        inside_project = False
    if inside_project:
        raise RuntimeAuthorizationGenerationError(
            "Runtime authorization cannot be published inside the source repository."
        )
    if (
        not output_name
        or output_name in {".", ".."}
        or Path(output_name).name != output_name
        or not output_name.isascii()
    ):
        raise RuntimeAuthorizationGenerationError(
            "Authorization output name must be one ASCII filename."
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(owner, flags)
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Authorization owner directory cannot be opened safely."
        ) from exc
    temporary_name = f".{output_name}.{secrets.token_hex(16)}.tmp"
    temporary_descriptor = -1
    published = False
    complete = False
    temporary_identity: tuple[int, int] | None = None
    try:
        directory_status = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_status.st_mode)
            or directory_status.st_uid != os.geteuid()
            or stat.S_IMODE(directory_status.st_mode) != 0o700
        ):
            raise RuntimeAuthorizationGenerationError(
                "Authorization owner directory must be owner-only mode 0700."
            )
        try:
            os.stat(output_name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeAuthorizationGenerationError(
                "Authorization output already exists."
            )
        temporary_descriptor = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        os.fchmod(temporary_descriptor, 0o600)
        _write_all(temporary_descriptor, payload)
        os.fsync(temporary_descriptor)
        status = os.fstat(temporary_descriptor)
        temporary_identity = (status.st_dev, status.st_ino)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_size != len(payload)
        ):
            raise RuntimeAuthorizationGenerationError(
                "Authorization temporary file has an invalid identity."
            )
        os.lseek(temporary_descriptor, 0, os.SEEK_SET)
        if (
            b"".join(iter(lambda: os.read(temporary_descriptor, _READ_SIZE), b""))
            != payload
        ):
            raise RuntimeAuthorizationGenerationError(
                "Authorization temporary file differs from canonical bytes."
            )
        os.link(
            temporary_name,
            output_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        published = True
        output_status = os.stat(
            output_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            (output_status.st_dev, output_status.st_ino) != temporary_identity
            or not stat.S_ISREG(output_status.st_mode)
            or output_status.st_uid != os.geteuid()
            or stat.S_IMODE(output_status.st_mode) != 0o600
            or output_status.st_nlink != 2
        ):
            raise RuntimeAuthorizationGenerationError(
                "Published authorization has an invalid identity."
            )
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        temporary_name = ""
        os.fsync(directory_descriptor)
        final_status = os.stat(
            output_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        current_directory_status = owner.lstat()
        if (
            (final_status.st_dev, final_status.st_ino) != temporary_identity
            or final_status.st_nlink != 1
            or stat.S_IMODE(final_status.st_mode) != 0o600
            or _directory_identity(current_directory_status)
            != _directory_identity(directory_status)
            or current_directory_status.st_uid != os.geteuid()
            or stat.S_IMODE(current_directory_status.st_mode) != 0o700
        ):
            raise RuntimeAuthorizationGenerationError(
                "Final authorization or owner-directory identity changed."
            )
        complete = True
    except FileExistsError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Authorization output already exists."
        ) from exc
    except OSError as exc:
        raise RuntimeAuthorizationGenerationError(
            "Runtime authorization could not be published atomically."
        ) from exc
    finally:
        if published and not complete and temporary_identity is not None:
            try:
                status = os.stat(
                    output_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (status.st_dev, status.st_ino) == temporary_identity:
                    os.unlink(output_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        os.close(directory_descriptor)
    return owner / output_name


def generate_runtime_authorization(
    *,
    project_root: str,
    expected_git_commit: str,
    authorization_id: str,
    created_at_utc: str,
    launcher_path: str,
    runtime_paths: Mapping[str, str],
    owner_directory: str,
    output_name: str,
) -> PublishedAuthorization:
    root = require_clean_exact_git_commit(project_root, expected_git_commit)
    registration_before = _load_policy_registration(root)
    sources_before = _authenticate_sources(root, expected_git_commit)
    artifacts = _authenticate_artifacts(
        launcher_path=launcher_path,
        runtime_paths=runtime_paths,
        sources=sources_before,
    )
    registration_after = _load_policy_registration(root)
    sources_after = _authenticate_sources(root, expected_git_commit)
    require_clean_exact_git_commit(project_root, expected_git_commit)
    if registration_after != registration_before or sources_after != sources_before:
        raise RuntimeAuthorizationGenerationError(
            "Source or registered protocol changed during authorization."
        )
    document = build_runtime_authorization(
        authorization_id=authorization_id,
        created_at_utc=created_at_utc,
        expected_git_commit=expected_git_commit,
        registration=registration_before,
        sources=sources_before,
        artifacts=artifacts,
    )
    payload = canonical_json_bytes(document)
    output_path = _publish_new_private_file(
        owner_directory=owner_directory,
        output_name=output_name,
        project_root=root,
        payload=payload,
    )
    return PublishedAuthorization(
        path=output_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        document=document,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--training-runtime", required=True)
    parser.add_argument("--full-runtime", required=True)
    parser.add_argument("--theory-runtime", required=True)
    parser.add_argument("--audit-runtime", required=True)
    parser.add_argument("--analysis-runtime", required=True)
    parser.add_argument("--owner-directory", required=True)
    parser.add_argument("--output-name", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    published = generate_runtime_authorization(
        project_root=arguments.project_root,
        expected_git_commit=arguments.expected_git_commit,
        authorization_id=arguments.authorization_id,
        created_at_utc=arguments.created_at_utc,
        launcher_path=arguments.launcher,
        runtime_paths={
            "training": arguments.training_runtime,
            "full": arguments.full_runtime,
            "theory": arguments.theory_runtime,
            "audit": arguments.audit_runtime,
            "analysis": arguments.analysis_runtime,
        },
        owner_directory=arguments.owner_directory,
        output_name=arguments.output_name,
    )
    summary = canonical_json_bytes(
        {
            "path": str(published.path),
            "runtime_authorization_sha256": published.sha256,
        }
    )
    sys.stdout.buffer.write(summary + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
