#!/usr/bin/env fbpython
"""Validate the checked-in launcher identity against a real Buck-built PAR.

Every other launcher test replaces the pinned identity constants with
``mock.patch.object`` and writes a synthetic archive, so none of them can tell
whether the checked-in pins still describe the artifact Buck actually produces.
That gap let the pins drift until runtime authorization failed.

This test closes it. Buck builds ``:phase4_runtime_launcher`` and materializes
the real PAR as a resource. The positive assertions run the unmodified
authorization validators against those unmodified bytes with no patching. The
negative assertions each apply one targeted mutation to a copy of that same
real artifact, so they exercise tamper detection without inventing an archive.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from phase4_runtime_profile import (
    AuthorizedPhase4Profile,
    POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS,
    POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
)
from scripts import policy_improvement_runtime_authorization as authorization


# Resources are materialized into the Buck link tree next to this module.
# Do not resolve: the test sources inside the link tree are symlinks back to
# the checkout, and resolving them would point at the repository instead, where
# no Buck-built artifact exists.
ROOT = Path(__file__).absolute().parents[1]
LAUNCHER_PAR = ROOT / "phase4_runtime_launcher.par"
BUCK_FILE = ROOT / "BUCK"
EXECUTABLE_SUFFIXES = (".py", ".pyc", ".pyo", ".so", ".pyd", ".dylib")


def _profile_inventory(archive: ZipFile) -> AuthorizedPhase4Profile:
    """Authorized profile carrying the checked-in launcher path inventory.

    ``_validate_launcher_executable_closure`` and ``_launcher_expected_modules``
    read only the keys of ``sources``, never the digests, so taking the digests
    from the archive cannot mask a drifted pin here. Binding the profile
    digests to a clean checkout is the authorization generator's job, through
    ``assert_phase4_archive_matches_profile``.
    """

    names = {info.filename for info in archive.infolist()}
    missing = sorted(set(POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS) - names)
    if missing:
        raise AssertionError(f"Launcher PAR lacks profile sources {missing!r}.")
    return AuthorizedPhase4Profile(
        git_commit="0" * 40,
        source_manifest_sha256="0" * 64,
        profile=POLICY_IMPROVEMENT_LAUNCHER_SOURCE_PROFILE,
        sources={
            path: hashlib.sha256(archive.read(path)).hexdigest()
            for path in POLICY_IMPROVEMENT_LAUNCHER_PROFILE_PATHS
        },
    )


def _rewrite(source: ZipFile, destination: Path, *, replace=None, add=None) -> Path:
    """Copy every member of a real PAR, applying one targeted mutation."""

    replace = replace or {}
    add = add or {}
    prefix_size = min(info.header_offset for info in source.infolist())
    handle = source.fp
    assert handle is not None
    position = handle.tell()
    handle.seek(0)
    prefix = handle.read(prefix_size)
    handle.seek(position)
    with destination.open("wb") as out:
        out.write(prefix)
    with ZipFile(destination, "a") as rebuilt:
        for info in source.infolist():
            payload = replace.get(info.filename, source.read(info.filename))
            rebuilt.writestr(copy.copy(info), payload)
        for name, payload in add.items():
            rebuilt.writestr(name, payload)
    return destination


class LauncherIdentityTest(unittest.TestCase):
    """The checked-in pins must describe the launcher Buck builds today."""

    @classmethod
    def setUpClass(cls) -> None:
        if not LAUNCHER_PAR.is_file():
            raise AssertionError(
                f"Launcher PAR resource is missing at {LAUNCHER_PAR}. The Buck "
                "target must materialize the real artifact."
            )
        cls.payload = LAUNCHER_PAR.read_bytes()

    def _archive(self) -> ZipFile:
        archive = ZipFile(LAUNCHER_PAR, "r")
        self.addCleanup(archive.close)
        return archive

    def _require_reviewed_configuration(self, archive: ZipFile) -> dict:
        manifest = authorization._launcher_executable_manifest(archive)
        fbmake = manifest["fbmake"]
        assert isinstance(fbmake, dict)
        if (
            fbmake.get("build_mode") != "opt"
            or fbmake.get("par_style") != "fastzip"
            or fbmake.get("link_strategy") != "native"
            or fbmake.get("platform") != "platform010"
        ):
            self.skipTest(
                "Launcher identity is pinned for the optimized platform010 "
                "fastzip native launcher. This build is "
                f"{fbmake.get('build_mode')}/{fbmake.get('par_style')}/"
                f"{fbmake.get('link_strategy')}/{fbmake.get('platform')}; run "
                "under @fbcode//mode/opt to enforce the pins."
            )
        return manifest

    def test_buck_target_excludes_imports_monitor(self) -> None:
        """The narrow closure depends on this Buck attribute staying set."""

        tree = ast.parse(BUCK_FILE.read_text(), filename="BUCK")
        settings = [
            {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "python_binary"
            and any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "phase4_runtime_launcher"
                for keyword in node.keywords
            )
        ]
        self.assertEqual(len(settings), 1)
        imports_monitor = settings[0].get("imports_monitor")
        self.assertIsInstance(imports_monitor, ast.Constant)
        self.assertIs(
            imports_monitor.value,
            False,
            "phase4_runtime_launcher must keep imports_monitor = False so "
            "fbcode//python/imports_monitor stays out of the launcher's "
            "pre-authentication executable closure.",
        )
        self.assertIs(settings[0]["compile"].value, False)

    def test_executable_prefix_matches_pin(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        prefix = authorization._validate_launcher_archive_prefix(archive)
        self.assertEqual(len(prefix), authorization._LAUNCHER_ARCHIVE_PREFIX_SIZE)
        self.assertEqual(
            hashlib.sha256(prefix).hexdigest(),
            authorization._LAUNCHER_ARCHIVE_PREFIX_SHA256,
        )

    def test_buildstamp_binds_archive_content(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        prefix = authorization._validate_launcher_archive_prefix(archive)
        authorization._validate_launcher_buildstamp(archive, prefix)

    def test_native_support_matches_pin(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        runtime_members = sorted(
            info.filename
            for info in archive.infolist()
            if info.filename.startswith("runtime/")
        )
        self.assertEqual(
            runtime_members,
            sorted(authorization._LAUNCHER_NATIVE_SUPPORT_MEMBERS),
        )
        authorization._validate_launcher_native_support(archive)

    def test_pinned_support_matches_pin(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        names = {info.filename for info in archive.infolist()}
        self.assertEqual(
            sorted(set(authorization._LAUNCHER_PINNED_SUPPORT_MEMBERS) - names),
            [],
        )
        identity = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in authorization._LAUNCHER_PINNED_SUPPORT_MEMBERS
        }
        self.assertEqual(
            hashlib.sha256(
                authorization.canonical_json_bytes(identity)
            ).hexdigest(),
            authorization._LAUNCHER_PINNED_SUPPORT_MANIFEST_SHA256,
        )

    def test_startup_functions_match_pin(self) -> None:
        archive = self._archive()
        manifest = self._require_reviewed_configuration(archive)
        self.assertEqual(
            manifest["startup_functions"],
            authorization._LAUNCHER_STARTUP_FUNCTIONS,
        )
        self.assertNotIn(
            "01_PYTHON_IMPORTS_MONITOR",
            manifest["startup_functions"],
            "Import-time telemetry must not run before authentication.",
        )

    def test_startup_loader_matches_pin(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        authorization._validate_launcher_startup_loader(archive)

    def test_executable_closure_matches_pin(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        authorized = _profile_inventory(archive)
        expected = {
            *authorization._LAUNCHER_PINNED_SUPPORT_MEMBERS,
            *authorization._LAUNCHER_DYNAMIC_SUPPORT_MEMBERS,
            *(path for path in authorized.sources if path.endswith(".py")),
        }
        actual = {
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.endswith(EXECUTABLE_SUFFIXES)
        }
        self.assertEqual(sorted(actual - expected), [])
        self.assertEqual(sorted(expected - actual), [])
        authorization._validate_launcher_executable_closure(archive, authorized)

    def test_full_launcher_validation_accepts_the_real_artifact(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        authorized = _profile_inventory(archive)
        prefix = authorization._validate_launcher_archive_prefix(archive)
        authorization._validate_launcher_buildstamp(archive, prefix)
        authorization._validate_launcher_native_support(archive)
        authorization._validate_launcher_executable_closure(archive, authorized)

    def test_altered_native_member_is_rejected(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        native_main = authorization._LAUNCHER_NATIVE_SUPPORT_MEMBERS[0]
        mutated = bytearray(archive.read(native_main))
        mutated[-1] ^= 0xFF
        target = Path(self.enterContext(tempfile.TemporaryDirectory())) / "native.par"
        _rewrite(archive, target, replace={native_main: bytes(mutated)})
        with ZipFile(target, "r") as tampered:
            with self.assertRaises(
                authorization.RuntimeAuthorizationGenerationError
            ) as caught:
                authorization._validate_launcher_native_support(tampered)
        self.assertIn("native runtime support differs", str(caught.exception))

    def test_reintroduced_imports_monitor_is_rejected(self) -> None:
        """Reproduce the closure mismatch that blocked the earlier cycles."""

        archive = self._archive()
        self._require_reviewed_configuration(archive)
        authorized = _profile_inventory(archive)
        target = Path(self.enterContext(tempfile.TemporaryDirectory())) / "monitor.par"
        _rewrite(
            archive,
            target,
            add={
                "python/imports_monitor/__init__.py": b"",
                "python/imports_monitor/imports_monitor.py": (
                    b"def install(*a):\n    pass\n"
                ),
            },
        )
        with ZipFile(target, "r") as widened:
            with self.assertRaises(
                authorization.RuntimeAuthorizationGenerationError
            ) as caught:
                authorization._validate_launcher_executable_closure(
                    widened, authorized
                )
        self.assertIn("executable member inventory differs", str(caught.exception))

    def test_extra_startup_function_is_rejected(self) -> None:
        archive = self._archive()
        self._require_reviewed_configuration(archive)
        loader = "__par__/__startup_function_loader__.py"
        widened = archive.read(loader).replace(
            b"STARTUP_FUNCTIONS=[",
            b"STARTUP_FUNCTIONS=['''python.imports_monitor.imports_monitor:install''',",
        )
        self.assertNotEqual(widened, archive.read(loader))
        target = Path(self.enterContext(tempfile.TemporaryDirectory())) / "startup.par"
        _rewrite(archive, target, replace={loader: widened})
        with ZipFile(target, "r") as tampered:
            with self.assertRaises(
                authorization.RuntimeAuthorizationGenerationError
            ) as caught:
                authorization._validate_launcher_startup_loader(tampered)
        self.assertIn("startup loader implementation differs", str(caught.exception))

    def test_label_pattern_stays_narrow(self) -> None:
        pattern = authorization._LAUNCHER_LABEL
        base = (
            "fbcode//buiksat_trm:phase4_runtime_launcher "
            "(cfg:opt-linux-x86_64-fbcode-platform010-clang21-no-san"
        )
        self.assertIsNotNone(pattern.fullmatch(f"{base}#8b1a31e17a6261c4)"))
        self.assertIsNotNone(
            pattern.fullmatch(f"{base}-opt-by-default#8b1a31e17a6261c4)")
        )
        for rejected in (
            f"{base}-opt-by-defaults#8b1a31e17a6261c4)",
            f"{base}-asan#8b1a31e17a6261c4)",
            f"{base}-opt-by-default#8b1a31e17a6261c)",
            f"{base}-opt-by-default#8b1a31e17a6261c4) trailing",
            "fbcode//buiksat_trm:other (cfg:opt-linux-x86_64-fbcode-platform010-"
            "clang21-no-san-opt-by-default#8b1a31e17a6261c4)",
            "fbcode//buiksat_trm:phase4_runtime_launcher (cfg:dev-linux-x86_64-"
            "fbcode-platform010-clang21-no-san#8b1a31e17a6261c4)",
        ):
            self.assertIsNone(pattern.fullmatch(rejected), rejected)


if __name__ == "__main__":
    unittest.main()
