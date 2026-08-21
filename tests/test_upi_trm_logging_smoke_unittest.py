"""
Tests for UPI-TRM logging smoke test - unittest version.
Converts pytest-style tests to unittest.TestCase for Buck2 compatibility.
"""

import copy
import fcntl
import hashlib
import importlib
import io
import json
import os
import py_compile
import random
import shutil
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.sudoku_checkers import dummy_checker
from rl.training_setup import (
    DummyPuzzleDataset,
    build_dataset_from_paths,
    offset_puzzle_identifiers,
)
from rl.upi_trm_trainer import UPITrmTrainer
import upi_trm_train
from upi_trm_train import (
    _capture_rng_state,
    _checkpoint_training_invocation,
    _claim_confirmatory_attempt_paths,
    _config_dict,
    _fixed_base_effective_config,
    _phase4_condition_from_run_id,
    _phase4_run_requested,
    _policy_smoke_requested,
    _phase4_training_invocation,
    _preflight_confirmatory_runtime,
    _preflight_training_runtime,
    _remember_latest_optimization_metrics,
    _reject_confirmatory_resume,
    _resolve_train_pool_size,
    _resolve_registered_rl_config,
    _restore_rng_state,
    _validate_ppo_exact_budget_schedule,
    _validate_registered_confirmatory_assignment,
    _validate_materialized_split_manifest,
    _validate_evidence_identity,
    _validate_confirmatory_attempt_index,
    _validate_expected_producer_commit,
    _validate_run_identity_bindings,
    _verify_producer_source_matches_runtime,
    resume_from_checkpoint,
    resume_from_checkpoint_for_theory_evaluation,
    save_checkpoint,
    validate_checkpoint_state_for_audit,
)
from utils.dataset_provenance import (
    build_dataset_provenance,
    dataset_sample_sha256s,
)
from utils.run_identity import (
    build_checkpoint_lineage,
    canonical_json_sha256,
    file_sha256,
)
from utils.source_identity import (
    SOURCE_MANIFEST_RELATIVE_PATH,
    build_producer_source_manifest,
)


def _num_actions(seq_len: int, vocab_size: int) -> int:
    return seq_len * vocab_size + 1


def _tiny_trm_cfg(seq_len: int, vocab_size: int, num_identifiers: int, batch_size: int):
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=max(num_identifiers, batch_size),
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=False,
        rl_enable_policy_head=True,
        rl_num_actions=_num_actions(seq_len, vocab_size),
    )


class TestUPITrmLoggingSmoke(unittest.TestCase):
    def test_theory_checkpoint_restore_is_quiet_without_changing_default(self):
        observed_emit_progress = []

        def resume_impl(*args, emit_progress=True, **kwargs):
            observed_emit_progress.append(emit_progress)
            if emit_progress:
                print("checkpoint progress")
            return 7

        with patch.object(
            upi_trm_train,
            "_PREVERIFIED_RUNTIME_SHA256",
            "b" * 64,
        ), patch(
            "upi_trm_train._resume_from_checkpoint_impl",
            side_effect=resume_impl,
        ):
            theory_stdout = io.StringIO()
            with redirect_stdout(theory_stdout):
                theory_result = resume_from_checkpoint_for_theory_evaluation(
                    11,
                    MagicMock(),
                    MagicMock(),
                    "cpu",
                    expected_dataset_provenance={},
                    expected_run_identity=None,
                    expected_checkpoint_sha256="a" * 64,
                )
            normal_stdout = io.StringIO()
            with redirect_stdout(normal_stdout):
                normal_result = resume_from_checkpoint(
                    11,
                    MagicMock(),
                    MagicMock(),
                    "cpu",
                    expected_dataset_provenance={},
                    expected_run_identity=None,
                    expected_checkpoint_sha256="a" * 64,
                )

        self.assertEqual(theory_result, 7)
        self.assertEqual(theory_stdout.getvalue(), "")
        self.assertEqual(normal_result, 7)
        self.assertEqual(normal_stdout.getvalue(), "checkpoint progress\n")
        self.assertEqual(observed_emit_progress, [False, True])

    def test_audit_checkpoint_restore_suppresses_progress_output(self):
        with patch(
            "upi_trm_train._resume_from_checkpoint_impl",
            return_value=7,
        ) as resume_impl:
            result = validate_checkpoint_state_for_audit(
                "/tmp/checkpoint.pt",
                MagicMock(),
                MagicMock(),
                "cuda",
                expected_dataset_provenance={},
                expected_run_identity=None,
                expected_checkpoint_sha256="a" * 64,
                authorized_originating_runtime_sha256="b" * 64,
            )

        self.assertEqual(result, 7)
        self.assertFalse(resume_impl.call_args.kwargs["emit_progress"])

    @staticmethod
    def _open_private_unpack(path: Path) -> tuple[int, str]:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        os.set_inheritable(descriptor, True)
        return descriptor, f"/proc/self/fd/{descriptor}"

    def test_checkpoint_invocation_binds_seed_run_and_config_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "condition.yaml"
            config_path.write_text("enable_contraction: false\n")
            rl_config = {"enable_contraction": False}
            model_config = {"rl_enable_contraction": False}

            invocation = _checkpoint_training_invocation(
                training_seed=41,
                training_run_id="phase4_2x2_norm_ablation.nc_nv.seed41",
                config_source_paths=[str(config_path)],
                rl_config=rl_config,
                model_config=model_config,
            )

        self.assertEqual(invocation["schema_version"], 1)
        self.assertEqual(invocation["training_seed"], 41)
        self.assertEqual(
            invocation["run_id"],
            "phase4_2x2_norm_ablation.nc_nv.seed41",
        )
        self.assertEqual(
            invocation["config_sources"],
            [
                {
                    "name": "condition.yaml",
                    "sha256": hashlib.sha256(
                        b"enable_contraction: false\n"
                    ).hexdigest(),
                }
            ],
        )
        self.assertEqual(
            invocation["rl_config_sha256"],
            canonical_json_sha256(rl_config),
        )
        self.assertEqual(
            invocation["model_config_sha256"],
            canonical_json_sha256(model_config),
        )

    def test_phase4_invocation_binds_preverified_runtime_artifact(self):
        runtime_sha256 = "d" * 64
        invocation = _phase4_training_invocation(
            source_context={
                "runtime_artifact_sha256": runtime_sha256,
                "config_sources": [
                    {"name": "nc_nv.yaml", "sha256": "c" * 64}
                ],
                "producer_source": {
                    "git_commit": "a" * 40,
                    "git_clean": True,
                    "source_manifest_sha256": "b" * 64,
                },
            },
            training_seed=41,
            run_id="phase4_2x2_norm_ablation.nc_nv.seed41",
            rl_config={"training_protocol": "legacy"},
            model_config={"hidden_size": 64},
            dataset_provenance={"schema_version": 1},
            initialization_kind="random",
            initialization_artifact_sha256=None,
        )

        self.assertEqual(invocation["schema_version"], 3)
        self.assertEqual(invocation["runtime_artifact_sha256"], runtime_sha256)
        with self.assertRaisesRegex(RuntimeError, "64 lowercase"):
            _phase4_training_invocation(
                source_context={
                    "runtime_artifact_sha256": "not-a-digest",
                    "config_sources": [],
                    "producer_source": {},
                },
                training_seed=41,
                run_id="phase4_2x2_norm_ablation.nc_nv.seed41",
                rl_config={},
                model_config={},
                dataset_provenance={},
                initialization_kind="random",
                initialization_artifact_sha256=None,
            )

    def test_phase4_run_id_requires_registered_seed_and_cell(self):
        self.assertEqual(
            _phase4_condition_from_run_id(
                "phase4_2x2_norm_ablation.yc_yv.seed42",
                42,
            ),
            "yc_yv",
        )
        with self.assertRaisesRegex(RuntimeError, "registered seed"):
            _phase4_condition_from_run_id(
                "phase4_2x2_norm_ablation.yc_yv.seed44",
                44,
            )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _phase4_condition_from_run_id(
                "phase4_2x2_norm_ablation.nc_nv.seed42",
                41,
            )

    def test_phase4_preimport_detection_covers_launcher_and_run_id_forms(self):
        self.assertTrue(_phase4_run_requested(["--phase4-publication"]))
        self.assertTrue(
            _phase4_run_requested(
                [
                    "--run-id",
                    "phase4_2x2_norm_ablation.nc_nv.seed41",
                ]
            )
        )
        self.assertTrue(
            _phase4_run_requested(
                [
                    "--run-id=phase4_2x2_norm_ablation.yc_yv.seed43",
                ]
            )
        )
        self.assertFalse(_phase4_run_requested(["--run-id", "ordinary.seed41"]))

    def test_policy_smoke_preflight_requires_its_separate_attestation(self):
        self.assertTrue(
            _policy_smoke_requested(
                ["--policy-improvement-smoke-entrypoint"]
            )
        )
        self.assertFalse(_policy_smoke_requested(["--phase4-publication"]))
        with self.assertRaisesRegex(RuntimeError, "packaged-runtime launcher"):
            _preflight_training_runtime(
                argv=["--policy-improvement-smoke-entrypoint"],
                module_file=__file__,
                environ={},
            )

        sentinel = upi_trm_train.RuntimePreflight(
            runtime_sha256="d" * 64,
            private_unpack_descriptor=11,
            runtime_descriptor=12,
            phase4_role="policy-improvement-smoke",
            source_git_commit="a" * 40,
            source_manifest_sha256="b" * 64,
        )
        with patch.object(
            upi_trm_train,
            "preflight_runtime",
            return_value=sentinel,
        ) as preflight:
            result = _preflight_training_runtime(
                argv=["--policy-improvement-smoke-entrypoint"],
                module_file="/proc/self/fd/12/upi_trm_train.py",
                environ={},
            )
        self.assertIs(result, sentinel)
        self.assertEqual(
            preflight.call_args.kwargs["allowed_phase4_roles"],
            frozenset({"policy-improvement-smoke"}),
        )

    def test_policy_smoke_is_mutually_exclusive_with_existing_evidence_modes(self):
        for conflict in ("--confirmatory", "--phase4-publication"):
            with self.subTest(conflict=conflict), self.assertRaisesRegex(
                RuntimeError, "cannot be combined"
            ):
                _preflight_training_runtime(
                    argv=[
                        "--policy-improvement-smoke-entrypoint",
                        conflict,
                    ],
                    module_file=__file__,
                    environ={},
                )

    def test_phase4_preflight_requires_training_attestation(self):
        with self.assertRaisesRegex(RuntimeError, "packaged-runtime launcher"):
            _preflight_training_runtime(
                argv=[
                    "--run-id",
                    "phase4_2x2_norm_ablation.nc_nv.seed41",
                ],
                module_file=__file__,
                environ={},
            )

        sentinel = upi_trm_train.RuntimePreflight(
            runtime_sha256="d" * 64,
            private_unpack_descriptor=11,
            runtime_descriptor=12,
            phase4_role="training",
            source_git_commit="a" * 40,
            source_manifest_sha256="b" * 64,
        )
        with patch.object(
            upi_trm_train,
            "preflight_runtime",
            return_value=sentinel,
        ) as preflight:
            result = _preflight_training_runtime(
                argv=[
                    "--phase4-publication",
                    "--run-id=phase4_2x2_norm_ablation.nc_nv.seed41",
                ],
                module_file="/proc/self/fd/12/upi_trm_train.py",
                environ={},
            )
        self.assertIs(result, sentinel)
        self.assertEqual(
            preflight.call_args.kwargs["allowed_phase4_roles"],
            frozenset({"training"}),
        )
        self.assertTrue(preflight.call_args.kwargs["attestation_required"])

    def test_phase4_and_confirmatory_modes_are_mutually_exclusive_preimport(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            _preflight_training_runtime(
                argv=[
                    "--confirmatory",
                    "--run-id=phase4_2x2_norm_ablation.nc_nv.seed41",
                ],
                module_file=__file__,
                environ={},
            )

    def test_confirmatory_preflight_requires_packaged_launcher(self):
        with self.assertRaisesRegex(RuntimeError, "packaged-runtime launcher"):
            _preflight_confirmatory_runtime(
                argv=["--confirmatory"],
                module_file=__file__,
                environ={},
            )

    def test_confirmatory_preflight_rejects_unsealed_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.par"
            with zipfile.ZipFile(runtime, "w") as archive:
                archive.writestr("upi_trm_train.py", b"# packaged\n")
            descriptor = os.memfd_create(
                "upi_trm_unsealed_preflight_test",
                os.MFD_ALLOW_SEALING,
            )
            os.write(descriptor, runtime.read_bytes())
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.set_inheritable(descriptor, True)
            descriptor_path = f"/proc/self/fd/{descriptor}"
            private_unpack = Path(directory) / "private-unpack"
            private_unpack.mkdir(mode=0o700)
            private_descriptor, private_descriptor_path = self._open_private_unpack(
                private_unpack
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "immutable attested"):
                    _preflight_confirmatory_runtime(
                        argv=["--confirmatory"],
                        module_file=f"{descriptor_path}/upi_trm_train.py",
                        environ={
                            "UPI_TRM_VERIFIED_RUNTIME_PATH": descriptor_path,
                            "UPI_TRM_VERIFIED_RUNTIME_SHA256": hashlib.sha256(
                                runtime.read_bytes()
                            ).hexdigest(),
                            "UPI_TRM_VERIFIED_RUNTIME_FD": str(descriptor),
                            "UPI_TRM_PRIVATE_UNPACK_BASE": private_descriptor_path,
                            "UPI_TRM_PRIVATE_UNPACK_FD": str(private_descriptor),
                            "FB_PAR_FILENAME": descriptor_path,
                            "FB_PAR_UNPACK_BASEDIR": private_descriptor_path,
                        },
                    )
            finally:
                os.close(descriptor)
                os.close(private_descriptor)

    def test_confirmatory_preflight_binds_and_consumes_runtime_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.par"
            with zipfile.ZipFile(runtime, "w") as archive:
                archive.writestr("upi_trm_train.py", b"# packaged\n")
            digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
            descriptor = os.memfd_create(
                "upi_trm_preflight_test",
                os.MFD_ALLOW_SEALING,
            )
            os.write(descriptor, runtime.read_bytes())
            os.fchmod(descriptor, 0o500)
            fcntl.fcntl(
                descriptor,
                fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_SEAL,
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.set_inheritable(descriptor, True)
            descriptor_path = f"/proc/self/fd/{descriptor}"
            private_unpack = Path(directory) / "private-unpack"
            private_unpack.mkdir(mode=0o700)
            private_descriptor, private_descriptor_path = self._open_private_unpack(
                private_unpack
            )
            try:
                environment = {
                    "UPI_TRM_VERIFIED_RUNTIME_PATH": descriptor_path,
                    "UPI_TRM_VERIFIED_RUNTIME_SHA256": digest,
                    "UPI_TRM_VERIFIED_RUNTIME_FD": str(descriptor),
                    "FB_PAR_FILENAME": descriptor_path,
                    "FB_PAR_UNPACK_BASEDIR": private_descriptor_path,
                    "UPI_TRM_PRIVATE_UNPACK_BASE": private_descriptor_path,
                    "UPI_TRM_PRIVATE_UNPACK_FD": str(private_descriptor),
                }
                preflight = _preflight_confirmatory_runtime(
                    argv=[
                        "--confirmatory",
                        "--confirmatory-tier",
                        "debug",
                        "--prepare-confirmatory-lock",
                    ],
                    module_file=f"{descriptor_path}/upi_trm_train.py",
                    environ=environment,
                )
                self.assertEqual(preflight[0], digest)
                self.assertEqual(preflight[1], private_descriptor)
                self.assertEqual(preflight[2], descriptor)
                self.assertNotIn("UPI_TRM_VERIFIED_RUNTIME_PATH", environment)
                self.assertNotIn("UPI_TRM_VERIFIED_RUNTIME_SHA256", environment)
                self.assertNotIn("UPI_TRM_VERIFIED_RUNTIME_FD", environment)
                self.assertNotIn("UPI_TRM_PRIVATE_UNPACK_BASE", environment)
                self.assertNotIn("UPI_TRM_PRIVATE_UNPACK_FD", environment)
                self.assertNotIn("FB_PAR_UNPACK_BASEDIR", environment)
                self.assertFalse(os.get_inheritable(descriptor))
                self.assertFalse(os.get_inheritable(private_descriptor))
                os.set_inheritable(descriptor, True)
                os.set_inheritable(private_descriptor, True)

                uppercase_environment = {
                    "UPI_TRM_VERIFIED_RUNTIME_PATH": descriptor_path,
                    "UPI_TRM_VERIFIED_RUNTIME_SHA256": digest.upper(),
                    "UPI_TRM_VERIFIED_RUNTIME_FD": str(descriptor),
                    "FB_PAR_FILENAME": descriptor_path,
                    "FB_PAR_UNPACK_BASEDIR": private_descriptor_path,
                    "UPI_TRM_PRIVATE_UNPACK_BASE": private_descriptor_path,
                    "UPI_TRM_PRIVATE_UNPACK_FD": str(private_descriptor),
                }
                with self.assertRaisesRegex(RuntimeError, "lowercase"):
                    _preflight_confirmatory_runtime(
                        argv=["--confirmatory"],
                        module_file=f"{descriptor_path}/upi_trm_train.py",
                        environ=uppercase_environment,
                    )

                wrong_digest_environment = {
                    "UPI_TRM_VERIFIED_RUNTIME_PATH": descriptor_path,
                    "UPI_TRM_VERIFIED_RUNTIME_SHA256": "0" * 64,
                    "UPI_TRM_VERIFIED_RUNTIME_FD": str(descriptor),
                    "FB_PAR_FILENAME": descriptor_path,
                    "FB_PAR_UNPACK_BASEDIR": private_descriptor_path,
                    "UPI_TRM_PRIVATE_UNPACK_BASE": private_descriptor_path,
                    "UPI_TRM_PRIVATE_UNPACK_FD": str(private_descriptor),
                }
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    _preflight_confirmatory_runtime(
                        argv=["--confirmatory"],
                        module_file=f"{descriptor_path}/upi_trm_train.py",
                        environ=wrong_digest_environment,
                    )
            finally:
                os.close(descriptor)
                os.close(private_descriptor)

    def test_preflight_rejects_attestation_without_confirmatory_mode(self):
        with self.assertRaisesRegex(RuntimeError, "authenticated evidence mode"):
            _preflight_confirmatory_runtime(
                argv=[],
                module_file=__file__,
                environ={"UPI_TRM_VERIFIED_RUNTIME_PATH": "/tmp/runtime.par"},
            )

    def test_runtime_bytecode_cache_ignores_checkout_pyc(self):
        self.assertEqual(
            Path(sys.pycache_prefix or "").resolve(),
            upi_trm_train._RUNTIME_BYTECODE_CACHE_ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_name = "runtime_bytecode_identity_probe"
            source_path = root / f"{module_name}.py"
            source_path.write_text('VALUE = "evill"\n', encoding="ascii")
            malicious_stat = source_path.stat()
            local_cache = (
                root
                / "__pycache__"
                / f"{module_name}.{sys.implementation.cache_tag}.pyc"
            )
            local_cache.parent.mkdir()
            py_compile.compile(
                str(source_path),
                cfile=str(local_cache),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
            )
            source_path.write_text('VALUE = "clean"\n', encoding="ascii")
            os.utime(
                source_path,
                ns=(malicious_stat.st_atime_ns, malicious_stat.st_mtime_ns),
            )

            sys.path.insert(0, str(root))
            importlib.invalidate_caches()
            try:
                module = importlib.import_module(module_name)
                self.assertEqual(module.VALUE, "clean")
                cached_path = Path(module.__cached__).resolve()
                self.assertTrue(
                    cached_path.is_relative_to(
                        upi_trm_train._RUNTIME_BYTECODE_CACHE_ROOT
                    )
                )
            finally:
                sys.modules.pop(module_name, None)
                sys.path.remove(str(root))

    """Smoke tests for UPI-TRM logging and evaluation hooks."""

    def test_producer_root_must_match_executing_sources(self):
        runtime_root = Path(upi_trm_train.__file__).resolve().parent
        with patch("upi_trm_train.assert_git_files_match_head"):
            _verify_producer_source_matches_runtime(runtime_root)
        with tempfile.TemporaryDirectory() as directory:
            unrelated_root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "source manifest"):
                _verify_producer_source_matches_runtime(unrelated_root)

    def test_source_tree_runtime_hashes_executing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            producer_root = Path(directory) / "producer"
            for relative_path in (
                "confirmatory_runtime_launcher.py",
                "phase4_runtime_profile.py",
                "policy_improvement_checkpoint_allowlist.py",
                "policy_improvement_smoke_checkpoint.py",
                "policy_improvement_smoke_runtime.py",
                "puzzle_dataset.py",
                "runtime_archive_preflight.py",
                "scripts/policy_improvement_populations.py",
                "scripts/policy_improvement_registry.py",
                "scripts/policy_improvement_schema.py",
                "scripts/policy_improvement_v2_registry.py",
                "scripts/policy_improvement_v2_schema.py",
                "upi_trm_train.py",
            ):
                destination = producer_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"# {relative_path}\n", encoding="ascii")
            for source_directory in (
                "dataset",
                "evaluators",
                "models",
                "rl",
                "utils",
            ):
                destination = producer_root / source_directory / "module.py"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    f"# {source_directory}\n",
                    encoding="ascii",
                )
            config_root = producer_root / "configs" / "iclr_confirmatory"
            config_root.mkdir(parents=True)
            (config_root / "cell.yaml").write_text(
                "gamma: 0.9\n",
                encoding="ascii",
            )
            policy_root = producer_root / "configs" / "policy_improvement_v1"
            policy_root.mkdir(parents=True)
            (policy_root / "protocol.json").write_text(
                "{}\n", encoding="ascii"
            )
            manifest = build_producer_source_manifest(producer_root)
            (producer_root / SOURCE_MANIFEST_RELATIVE_PATH).write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="ascii",
            )

            runtime_root = Path(directory) / "runtime"
            shutil.copytree(producer_root, runtime_root)
            runtime_module_path = runtime_root / "upi_trm_train.py"
            with (
                patch.object(upi_trm_train, "__file__", str(runtime_module_path)),
                patch("upi_trm_train.assert_git_files_match_head"),
            ):
                _verify_producer_source_matches_runtime(producer_root)

            (runtime_root / "rl" / "module.py").write_text(
                "# tampered runtime\n",
                encoding="ascii",
            )
            with (
                patch.object(upi_trm_train, "__file__", str(runtime_module_path)),
                patch("upi_trm_train.assert_git_files_match_head"),
                self.assertRaisesRegex(RuntimeError, "source manifest"),
            ):
                _verify_producer_source_matches_runtime(producer_root)

    def test_standalone_runtime_rejects_tampered_archive_source(self):
        with tempfile.TemporaryDirectory() as directory:
            producer_root = Path(directory) / "producer"
            for relative_path in (
                "confirmatory_runtime_launcher.py",
                "phase4_runtime_profile.py",
                "policy_improvement_checkpoint_allowlist.py",
                "policy_improvement_smoke_checkpoint.py",
                "policy_improvement_smoke_runtime.py",
                "puzzle_dataset.py",
                "runtime_archive_preflight.py",
                "scripts/policy_improvement_populations.py",
                "scripts/policy_improvement_registry.py",
                "scripts/policy_improvement_schema.py",
                "scripts/policy_improvement_v2_registry.py",
                "scripts/policy_improvement_v2_schema.py",
                "upi_trm_train.py",
            ):
                destination = producer_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(f"# {relative_path}\n", encoding="ascii")
            for source_directory in (
                "dataset",
                "evaluators",
                "models",
                "rl",
                "utils",
            ):
                destination = producer_root / source_directory / "module.py"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    f"# {source_directory}\n",
                    encoding="ascii",
                )
            config_root = producer_root / "configs" / "iclr_confirmatory"
            config_root.mkdir(parents=True)
            (config_root / "cell.yaml").write_text(
                "gamma: 0.9\n",
                encoding="ascii",
            )
            policy_root = producer_root / "configs" / "policy_improvement_v1"
            policy_root.mkdir(parents=True)
            (policy_root / "protocol.json").write_text(
                "{}\n", encoding="ascii"
            )

            manifest = build_producer_source_manifest(producer_root)
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode("ascii")
            archive_path = Path(directory) / "runtime.par"

            def write_archive(*, tamper: bool) -> None:
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for relative_path in manifest["sources"]:
                        if not relative_path.endswith(".py"):
                            continue
                        source_bytes = (producer_root / relative_path).read_bytes()
                        if tamper and relative_path == "rl/module.py":
                            source_bytes += b"# tampered\n"
                        archive.writestr(relative_path, source_bytes)
                    archive.writestr(
                        SOURCE_MANIFEST_RELATIVE_PATH,
                        manifest_bytes,
                    )

            runtime_module_path = archive_path / "upi_trm_train.py"
            write_archive(tamper=False)
            with (
                patch.object(upi_trm_train, "__file__", str(runtime_module_path)),
                patch("upi_trm_train.assert_git_files_match_head"),
            ):
                _verify_producer_source_matches_runtime(producer_root)

            write_archive(tamper=True)
            with (
                patch.object(upi_trm_train, "__file__", str(runtime_module_path)),
                patch("upi_trm_train.assert_git_files_match_head"),
                self.assertRaisesRegex(RuntimeError, "source manifest"),
            ):
                _verify_producer_source_matches_runtime(producer_root)

    def test_confirmatory_launch_binds_registered_producer_commit(self):
        identity = {"git_commit": "a" * 40, "git_clean": True}
        self.assertEqual(
            _validate_expected_producer_commit("a" * 40, identity),
            "a" * 40,
        )
        with self.assertRaisesRegex(RuntimeError, "differs"):
            _validate_expected_producer_commit("b" * 40, identity)
        with self.assertRaisesRegex(RuntimeError, "40-character"):
            _validate_expected_producer_commit("A" * 40, identity)

    def test_confirmatory_evidence_identity_binds_full_config(self):
        config = {"algorithm": "trm_ppo", "num_steps": 80}
        identity = {
            "schema_version": 1,
            "run_id": "c2-ppo-seed101",
            "algorithm": "trm_ppo",
            "training_seed": 101,
            "producer_git_commit": "a" * 40,
            "effective_config_sha256": canonical_json_sha256(config),
            "effective_config": config,
            "dataset_provenance_sha256": "b" * 64,
        }
        self.assertEqual(_validate_evidence_identity(identity), identity)
        mutated = copy.deepcopy(identity)
        mutated["effective_config"]["num_steps"] = 81
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            _validate_evidence_identity(mutated)

        runtime_hash = "c" * 64
        current_config = {
            **config,
            "runtime_artifact_sha256": runtime_hash,
        }
        current_identity = {
            **identity,
            "schema_version": 2,
            "runtime_artifact_sha256": runtime_hash,
            "effective_config": current_config,
            "effective_config_sha256": canonical_json_sha256(current_config),
        }
        self.assertEqual(
            _validate_evidence_identity(current_identity),
            current_identity,
        )
        wrong_runtime = copy.deepcopy(current_identity)
        wrong_runtime["runtime_artifact_sha256"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "runtime artifact"):
            _validate_evidence_identity(wrong_runtime)

    def test_fixed_base_lock_binds_registration_data_and_initialization(self):
        args = SimpleNamespace(
            config=[],
            confirmatory_cell="C2_UPI_TRM",
            confirmatory_tier="confirmatory",
            attempt_index=0,
            run_id="c2-upi-seed101",
            seed=101,
            backbone="trm",
            train_split="train",
            eval_split="test",
            env_step_budget=80_000,
            save_interval=0,
            log_env_interval=10_000,
            eval_env_interval=10_000,
            save_env_interval=10_000,
            puzzle_emb_lr=0.01,
            puzzle_emb_weight_decay=0.0,
            imitation_pretrain=False,
            imitation_epochs=0,
            debug_checks=False,
        )
        provenance = {"ordered_records": {"train": ["a"], "eval": ["b"]}}

        def build(
            *,
            args_overrides=None,
            dataset_provenance=None,
            runtime_artifact_sha256="e" * 64,
            **initialization,
        ):
            local_args = SimpleNamespace(**vars(args))
            for name, value in (args_overrides or {}).items():
                setattr(local_args, name, value)
            return _fixed_base_effective_config(
                args=local_args,
                rl_config={
                    "num_train_steps": 20_000,
                    "log_interval": 50,
                    "eval_interval": 100,
                    "eval_num_episodes": 512,
                    "eval_seed": 26080311,
                },
                model_config={"hidden_size": 64},
                execution_device="cpu",
                train_record_count=1024,
                eval_record_count=512,
                dataset_provenance=(dataset_provenance or provenance),
                initialization_kind=initialization.get("kind", "random"),
                initialization_artifact_sha256=initialization.get("sha256"),
                registered_assignment={
                    "attempt_index": local_args.attempt_index,
                    "registry_sha256": "f" * 64,
                },
                runtime_artifact_sha256=runtime_artifact_sha256,
            )

        base = build()
        self.assertEqual(base["effective_config_schema_version"], 4)
        self.assertEqual(base["runtime_artifact_sha256"], "e" * 64)
        variants = (
            build(args_overrides={"seed": 102, "run_id": "c2-upi-seed102"}),
            build(args_overrides={"confirmatory_cell": "C2_TRM_PPO"}),
            build(args_overrides={"attempt_index": 1}),
            build(
                dataset_provenance={
                    "ordered_records": {"train": ["x"], "eval": ["b"]}
                }
            ),
            build(kind="weights_checkpoint", sha256="a" * 64),
            build(runtime_artifact_sha256="d" * 64),
        )
        for variant in variants:
            self.assertNotEqual(
                canonical_json_sha256(base),
                canonical_json_sha256(variant),
            )

    def test_registered_assignment_rejects_mislabeled_cell_and_seed(self):
        root = Path(upi_trm_train.__file__).resolve().parent
        config_dir = root / "configs" / "iclr_confirmatory"
        args = SimpleNamespace(
            confirmatory_cell="B0_I00",
            confirmatory_tier="confirmatory",
            attempt_index=0,
            prepare_confirmatory_lock=True,
            seed=101,
            run_id="b0_i00-seed101",
            config=[
                str(config_dir / "bridge_base.yaml"),
                str(config_dir / "bridge_b0.yaml"),
            ],
            dataset_paths=[str(root / "data" / "iclr-confirmatory-sudoku4x4-v1")],
            train_split="train",
            eval_split="test",
            train_pool_size=1024,
            eval_pool_size=512,
            train_manifest_sha256=(
                "8def4f59387c1ab9466d043c40a7fdd3c7c670e2778b8d949295811ae7b6088a"
            ),
            eval_manifest_sha256=(
                "163a083a9f5744b7cc485663b269b89acc3103d9e1ec64c7e93f78e36fa79d40"
            ),
            env_step_budget=80_000,
            log_env_interval=10_000,
            eval_env_interval=10_000,
            save_env_interval=10_000,
            save_interval=0,
            backbone="trm",
            hidden_size=64,
            h_cycles=2,
            l_cycles=2,
            l_layers=1,
            puzzle_emb_ndim=0,
            producer_repo_root=str(root),
        )
        rl_cfg = _resolve_registered_rl_config(
            [config_dir / "bridge_base.yaml", config_dir / "bridge_b0.yaml"]
        )
        assignment = _validate_registered_confirmatory_assignment(
            args=args,
            selected_baseline=None,
            rl_cfg=rl_cfg,
        )
        self.assertEqual(assignment["cell"], "B0_I00")

        args.confirmatory_cell = "Bd_I01"
        args.run_id = "bd_i01-seed101"
        with self.assertRaisesRegex(RuntimeError, "YAML config layers"):
            _validate_registered_confirmatory_assignment(
                args=args,
                selected_baseline=None,
                rl_cfg=rl_cfg,
            )
        args.confirmatory_cell = "B0_I00"
        args.run_id = "b0_i00-seed111"
        args.seed = 111
        with self.assertRaisesRegex(RuntimeError, "seed is not registered"):
            _validate_registered_confirmatory_assignment(
                args=args,
                selected_baseline=None,
                rl_cfg=rl_cfg,
            )

        args.seed = 101
        args.run_id = "b0_i00-seed101"
        mutated_rl_cfg = RLConfig(
            **{**rl_cfg.model_dump(), "batch_size": rl_cfg.batch_size * 2}
        )
        with self.assertRaisesRegex(RuntimeError, "Resolved RLConfig"):
            _validate_registered_confirmatory_assignment(
                args=args,
                selected_baseline=None,
                rl_cfg=mutated_rl_cfg,
            )

    def test_cuda_rng_capture_and_restore_use_all_devices(self):
        cuda_states = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
        with patch("upi_trm_train.torch.cuda.is_available", return_value=True), patch(
            "upi_trm_train.torch.cuda.get_rng_state_all",
            return_value=cuda_states,
        ) as get_all:
            state = _capture_rng_state()
        get_all.assert_called_once_with()
        self.assertEqual(state["torch_cuda"], cuda_states)

        with patch("upi_trm_train.torch.cuda.set_rng_state_all") as set_all:
            _restore_rng_state(state)
        set_all.assert_called_once_with(cuda_states)

    @staticmethod
    def _write_dataset_root(
        root: Path,
        num_puzzles: int,
        split: str = "test",
    ) -> None:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        inputs = np.arange(num_puzzles * 2, dtype=np.int32).reshape(num_puzzles, 2)
        np.save(split_dir / "all__inputs.npy", inputs)
        np.save(split_dir / "all__labels.npy", inputs + 1)
        np.save(
            split_dir / "all__puzzle_identifiers.npy",
            np.arange(num_puzzles, dtype=np.int32),
        )
        np.save(
            split_dir / "all__puzzle_indices.npy",
            np.arange(num_puzzles + 1, dtype=np.int32),
        )
        np.save(
            split_dir / "all__group_indices.npy",
            np.arange(num_puzzles + 1, dtype=np.int32),
        )
        (split_dir / "dataset.json").write_text(
            json.dumps(
                {
                    "seq_len": 2,
                    "vocab_size": 16,
                    "pad_id": 0,
                    "ignore_label_id": None,
                    "blank_identifier_id": 0,
                    "num_puzzle_identifiers": num_puzzles,
                    "total_groups": num_puzzles,
                    "mean_puzzle_examples": 1,
                    "total_puzzles": num_puzzles,
                    "sets": ["all"],
                }
            )
        )

    @staticmethod
    def _make_persistent_budget_trainer(
        training_protocol="legacy",
        batch_size=8,
        episodic_latent=False,
        capture_preinterpolation_policy_pair=False,
        evaluation_policy_mode="configured",
    ):
        class FixedDataset:
            seq_len = 2
            vocab_size = 4
            num_identifiers = 1

            def __init__(self):
                self.samples = [
                    {
                        "inputs": torch.ones(2, dtype=torch.long),
                        "puzzle_identifiers": torch.tensor(0, dtype=torch.long),
                        "initial_plan": torch.ones(2, dtype=torch.long),
                    }
                ]

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = FixedDataset()
        fixed_base = training_protocol == "fixed_base_exact"
        stop_action_mode = "terminal" if fixed_base else "disabled"
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.9,
            reward_shaping=False,
            task_type="dummy",
            vocab_size=dataset.vocab_size,
            stop_action_mode=stop_action_mode,
            fail_terminal_reward=-1.0,
        )
        env = PlanEditEnv(
            dataset=dataset,
            checker=lambda _x, _y: 0.0,
            config=env_cfg,
        )
        env.set_stop_action_id(
            stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1
        )
        cfg = RLConfig(
            batch_size=batch_size,
            replay_capacity=32,
            rollout_episodes_per_step=1,
            max_edits=3,
            gamma=env_cfg.gamma,
            K=1,
            inner_unroll_n=1,
            episodic_latent=episodic_latent,
            task_name="dummy",
            solved_threshold=None,
            stop_action_mode=stop_action_mode,
            reward_shaping=False,
            fail_terminal_reward=-1.0,
            value_target_clip=None if fixed_base else 20.0,
            enable_contraction=False,
            opnorm_clamp_interval=0,
            latent_projection_mode="disabled",
            latent_ball_radius=None,
            lr_schedule="constant",
            use_tqdm=False,
            training_protocol=training_protocol,
            theory_exact_mixture=fixed_base,
            exact_k_step_targets=fixed_base,
            exact_baseline_summation=fixed_base,
            capture_preinterpolation_policy_pair=(
                capture_preinterpolation_policy_pair
            ),
            evaluation_policy_mode=evaluation_policy_mode,
        )
        model = TinyRecursiveReasoningModel_ACTV1(
            _tiny_trm_cfg(
                dataset.seq_len,
                dataset.vocab_size,
                dataset.num_identifiers,
                cfg.batch_size,
            )
        )
        if fixed_base:
            # Keep terminal STOP in the declared action space while making tests
            # that exercise a full three-edit budget deterministic.
            with torch.no_grad():
                model.edit_policy.mlp[-1].bias[-1] = -1e9
        trainer = UPITrmTrainer(model, env, cfg, torch.device("cpu"))
        return model, trainer, cfg

    @staticmethod
    def _stub_updates(trainer):
        trainer.value_update = MagicMock(
            return_value={"loss_value": 0.0, "value_optimizer_step": 1.0}
        )
        trainer.policy_update = MagicMock(
            return_value={"loss_policy": 0.0, "policy_optimizer_step": 1.0}
        )

    @staticmethod
    def _checkpoint_provenance(trainer, *, generation_seed=123):
        dataset = trainer.env.dataset
        stop_action_id = trainer.env.stop_action_id
        return build_dataset_provenance(
            builder_name="tests.FixedDataset",
            builder_version=1,
            generation_seed=generation_seed,
            train_record_sha256s=dataset_sample_sha256s(dataset),
            eval_record_sha256s=["e" * 64 for _ in range(len(dataset))],
            train_split="unit-train",
            eval_split="unit-eval",
            environment_config=dict(vars(trainer.env.config)),
            action_mask_config={
                "task_config_class": (
                    type(trainer.env.task_config).__name__
                    if trainer.env.task_config is not None
                    else None
                ),
                "disable_constraint_masking": (
                    trainer.env.config.disable_constraint_masking
                ),
                "stop_action_mode": trainer.env._stop_mode,
                "stop_action_id": stop_action_id,
                "enable_undo": trainer.env._enable_undo,
                "undo_action_id": trainer.env.undo_action_id,
                "vocab_size": trainer.env.vocab_size,
                "num_actions": (
                    trainer.env.undo_action_id + 1
                    if trainer.env.undo_action_id is not None
                    else stop_action_id + 1
                ),
                "masked_token_ids": [0, 1],
            },
        )

    def test_phase4_checkpoint_rejects_runtime_identity_mismatch(self):
        model, trainer, cfg = self._make_persistent_budget_trainer("legacy")
        provenance = self._checkpoint_provenance(trainer)
        invocation = _phase4_training_invocation(
            source_context={
                "runtime_artifact_sha256": "d" * 64,
                "config_sources": [],
                "producer_source": {
                    "git_commit": "a" * 40,
                    "git_clean": True,
                    "source_manifest_sha256": "b" * 64,
                },
            },
            training_seed=41,
            run_id="phase4_2x2_norm_ablation.nc_nv.seed41",
            rl_config=_config_dict(cfg),
            model_config=_config_dict(model.config),
            dataset_provenance=provenance,
            initialization_kind="random",
            initialization_artifact_sha256=None,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            upi_trm_train,
            "_PREVERIFIED_RUNTIME_SHA256",
            "e" * 64,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "preverified publication identity",
            ):
                save_checkpoint(
                    model,
                    trainer,
                    step=0,
                    checkpoint_dir=directory,
                    rl_cfg=cfg,
                    dataset_provenance=provenance,
                    training_seed=41,
                    training_run_id=(
                        "phase4_2x2_norm_ablation.nc_nv.seed41"
                    ),
                )
            with self.assertRaisesRegex(RuntimeError, "wrong runtime artifact"):
                save_checkpoint(
                    model,
                    trainer,
                    step=0,
                    checkpoint_dir=directory,
                    rl_cfg=cfg,
                    dataset_provenance=provenance,
                    training_seed=41,
                    training_run_id=(
                        "phase4_2x2_norm_ablation.nc_nv.seed41"
                    ),
                    checkpoint_training_invocation=invocation,
                )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def _run_identity(
        self,
        model,
        trainer,
        provenance,
        *,
        run_id="unit.seed17",
        seed=17,
        environment_interactions=None,
        effective_config_schema_version=4,
        runtime_artifact_sha256="e" * 64,
    ):
        rl_config = (
            trainer.rl_cfg.model_dump()
            if hasattr(trainer.rl_cfg, "model_dump")
            else trainer.rl_cfg.dict()
        )
        model_config = (
            model.config.model_dump()
            if hasattr(model.config, "model_dump")
            else model.config.dict()
        )
        effective_config = {
            "effective_config_schema_version": effective_config_schema_version,
            "algorithm": "upi_trm",
            "training_protocol": "fixed_base_exact",
            "backbone": "trm",
            "rl_config": rl_config,
            "model_config": model_config,
            "execution_device": "cpu",
            "runtime_fingerprint_sha256": canonical_json_sha256(
                upi_trm_train._runtime_fingerprint()
            ),
            "dataset": {
                "train_split": provenance["splits"]["train"],
                "eval_split": provenance["splits"]["eval"],
                "train_record_count": provenance["ordered_records"]["train"][
                    "count"
                ],
                "eval_record_count": provenance["ordered_records"]["eval"][
                    "count"
                ],
            },
            "budget": {
                "outer_train_steps": rl_config["num_train_steps"],
                "environment_interactions": environment_interactions,
            },
            "schedule": {
                "log_outer_interval": rl_config["log_interval"],
                "eval_outer_interval": rl_config["eval_interval"],
                "save_outer_interval": 1,
                "log_environment_interval": None,
                "eval_environment_interval": None,
                "save_environment_interval": None,
            },
            "evaluation": {
                "episode_count": rl_config["eval_num_episodes"],
                "seed": rl_config["eval_seed"],
                "pool_size": provenance["ordered_records"]["eval"]["count"],
            },
            "puzzle_embedding_optimizer": {
                "learning_rate": 0.01,
                "weight_decay": 0.1,
            },
            "imitation": {"enabled": False, "epochs": 0},
            "external_logging": "disabled",
            "debug_checks": False,
            "config_source_sha256s": [],
        }
        if effective_config_schema_version in {2, 3, 4}:
            registration = {
                "cell": "C2_UPI_TRM",
                "tier": "debug",
                "run_id": run_id,
                "training_seed": seed,
                "registry_sha256": "f" * 64,
            }
            if effective_config_schema_version in {3, 4}:
                registration["attempt_index"] = 0
            effective_config.update(
                {
                    "registration": registration,
                    "dataset_provenance_sha256": canonical_json_sha256(
                        provenance
                    ),
                    "initialization": {
                        "kind": "random",
                        "artifact_sha256": None,
                    },
                }
            )
        if effective_config_schema_version == 4:
            if runtime_artifact_sha256 is None:
                raise AssertionError("Schema-4 test identity requires a runtime hash.")
            effective_config["runtime_artifact_sha256"] = runtime_artifact_sha256
            runtime_patcher = patch.object(
                upi_trm_train,
                "_PREVERIFIED_RUNTIME_SHA256",
                runtime_artifact_sha256,
            )
            runtime_patcher.start()
            self.addCleanup(runtime_patcher.stop)
        return {
            "run_identity_schema_version": 1,
            "run_id": run_id,
            "training_seed": seed,
            "producer": {"git_commit": "a" * 40, "git_clean": True},
            "effective_config": effective_config,
            "effective_config_sha256": canonical_json_sha256(effective_config),
            "dataset_provenance_sha256": canonical_json_sha256(provenance),
            "initialization": {"kind": "random", "artifact_sha256": None},
        }

    def test_schema_v5_save_rejects_historical_effective_config_schemas(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            for schema_version in (1, 2, 3):
                with self.subTest(schema_version=schema_version):
                    identity = self._run_identity(
                        model,
                        trainer,
                        provenance,
                        effective_config_schema_version=schema_version,
                        runtime_artifact_sha256=None,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "effective configuration schema 4",
                    ):
                        save_checkpoint(
                            model,
                            trainer,
                            step=0,
                            checkpoint_dir=tmp,
                            rl_cfg=cfg,
                            dataset_provenance=provenance,
                            run_identity=identity,
                            checkpoint_lineage=self._root_lineage(),
                        )
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_schema_four_run_identity_binds_active_runtime_artifact(self):
        model, trainer, _ = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        runtime_sha256 = "e" * 64
        identity = self._run_identity(
            model,
            trainer,
            provenance,
            runtime_artifact_sha256=runtime_sha256,
        )
        rl_config = _config_dict(trainer.rl_cfg)
        model_config = _config_dict(model.config)
        runtime_fingerprint = upi_trm_train._runtime_fingerprint()
        self.assertEqual(
            _validate_run_identity_bindings(
                identity,
                rl_config=rl_config,
                model_config=model_config,
                dataset_provenance=provenance,
                execution_device="cpu",
                runtime_fingerprint=runtime_fingerprint,
                runtime_artifact_sha256=runtime_sha256,
            ),
            identity,
        )
        with self.assertRaisesRegex(RuntimeError, "runtime artifact differs"):
            _validate_run_identity_bindings(
                identity,
                rl_config=rl_config,
                model_config=model_config,
                dataset_provenance=provenance,
                execution_device="cpu",
                runtime_fingerprint=runtime_fingerprint,
                runtime_artifact_sha256="d" * 64,
            )

    @staticmethod
    def _root_lineage():
        return build_checkpoint_lineage(
            parent_checkpoint_sha256=None,
            parent_checkpoint_step=None,
            parent_environment_steps=None,
        )

    def _assert_replay_equal(self, expected, actual):
        self.assertEqual(len(expected.replay), len(actual.replay))
        for left, right in zip(expected.replay.storage, actual.replay.storage):
            self.assertEqual(left.episode_id, right.episode_id)
            self.assertEqual(left.timestep, right.timestep)
            self.assertTrue(torch.equal(left.action, right.action))
            self.assertTrue(torch.equal(left.reward, right.reward))
            self.assertTrue(torch.equal(left.done, right.done))
            self.assertTrue(torch.equal(left.y, right.y))
            self.assertTrue(torch.equal(left.y_next, right.y_next))
            for key in left.x:
                self.assertTrue(torch.equal(left.x[key], right.x[key]))
                self.assertTrue(torch.equal(left.x_next[key], right.x_next[key]))
            self.assertIsNotNone(left.latent)
            self.assertIsNotNone(right.latent)
            torch.testing.assert_close(left.latent.z_H, right.latent.z_H)
            torch.testing.assert_close(left.latent.z_L, right.latent.z_L)
            if bool(left.done.item()):
                self.assertIsNone(left.next_latent)
                self.assertIsNone(right.next_latent)
            else:
                self.assertIsNotNone(left.next_latent)
                self.assertIsNotNone(right.next_latent)
                torch.testing.assert_close(
                    left.next_latent.z_H,
                    right.next_latent.z_H,
                )
                torch.testing.assert_close(
                    left.next_latent.z_L,
                    right.next_latent.z_L,
                )

    def _assert_nested_equal(self, expected, actual):
        if torch.is_tensor(expected) or torch.is_tensor(actual):
            self.assertTrue(torch.is_tensor(expected) and torch.is_tensor(actual))
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            return
        if isinstance(expected, dict) or isinstance(actual, dict):
            self.assertTrue(isinstance(expected, dict) and isinstance(actual, dict))
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self._assert_nested_equal(expected[key], actual[key])
            return
        if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
            self.assertIs(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self._assert_nested_equal(expected_item, actual_item)
            return
        self.assertEqual(expected, actual)

    def test_logging_and_eval_hooks_run(self):
        """Test that logging and evaluation hooks run without errors."""
        dataset = DummyPuzzleDataset(num_instances=5, seq_len=8, vocab_size=16)
        env_cfg = PlanEditEnvConfig(max_edits=3, gamma=0.9, reward_shaping=True, vocab_size=dataset.vocab_size)
        env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
        env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

        rl_cfg = RLConfig(
            batch_size=2,
            num_train_steps=3,
            rollout_episodes_per_step=1,
            max_edits=3,
            log_interval=1,
            eval_interval=2,
            eval_num_episodes=5,
            use_tqdm=False,
            gamma=env_cfg.gamma,
        )

        model_cfg = _tiny_trm_cfg(
            seq_len=dataset.seq_len,
            vocab_size=dataset.vocab_size,
            num_identifiers=dataset.num_identifiers,
            batch_size=rl_cfg.batch_size,
        )
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
        trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))

        for _ in range(rl_cfg.num_train_steps):
            metrics = trainer.train_step()
            self.assertIn("loss_value", metrics)
            self.assertIn("loss_policy", metrics)

        before_eval = trainer.compute_accounting_snapshot()
        self.assertGreater(
            before_eval["model_work"]["training"]["policy_api_calls"], 0
        )
        self.assertEqual(
            before_eval["model_work"]["evaluation"]["policy_api_calls"], 0
        )
        success_rate = trainer.evaluate_policy_success_rate(env_cfg=env_cfg, dataset=dataset, checker=dummy_checker)
        self.assertIsInstance(success_rate, float)
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)
        after_eval = trainer.compute_accounting_snapshot()
        self.assertEqual(
            after_eval["model_work"]["training"],
            before_eval["model_work"]["training"],
        )
        self.assertGreater(
            after_eval["model_work"]["evaluation"]["policy_api_calls"], 0
        )
        self.assertIsNone(after_eval["peak_memory_bytes"]["cuda_allocated"])
        self.assertIsNone(after_eval["peak_memory_bytes"]["cuda_reserved"])

    def test_dataset_bootstrap_fallback_is_explicit(self):
        with patch("rl.training_setup.PuzzleDataset", side_effect=RuntimeError("boom")):
            with patch("builtins.print") as mock_print:
                dataset, *_ = build_dataset_from_paths(
                    ["missing-dataset"], pool_size=4, allow_dummy_fallback=True
                )

        self.assertIsInstance(dataset, DummyPuzzleDataset)
        emitted = "\n".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_print.call_args_list
        )
        self.assertIn("falling back to dummy dataset", emitted)

    def test_confirmatory_training_pool_size_is_explicit(self):
        self.assertEqual(
            _resolve_train_pool_size(
                requested_size=None,
                batch_size=32,
                require_explicit=False,
            ),
            32,
        )
        with self.assertRaisesRegex(RuntimeError, "explicit training pool size"):
            _resolve_train_pool_size(
                requested_size=None,
                batch_size=32,
                require_explicit=True,
            )
        self.assertEqual(
            _resolve_train_pool_size(
                requested_size=1024,
                batch_size=32,
                require_explicit=True,
            ),
            1024,
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _resolve_train_pool_size(
                requested_size=0,
                batch_size=32,
                require_explicit=False,
            )

    def test_ppo_exact_budget_schedule_requires_complete_rollouts(self):
        valid = {
            "env_step_budget": 80_000,
            "restored_env_steps": 0,
            "rollout_steps": 80,
            "log_env_interval": 10_000,
            "eval_env_interval": 10_000,
            "save_env_interval": 10_000,
        }
        _validate_ppo_exact_budget_schedule(**valid)
        _validate_ppo_exact_budget_schedule(
            **{**valid, "log_env_interval": None, "save_env_interval": 0}
        )

        for field, value in {
            "env_step_budget": 80_001,
            "restored_env_steps": 1,
            "log_env_interval": 10_001,
            "eval_env_interval": 10_001,
            "save_env_interval": 10_001,
        }.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_ppo_exact_budget_schedule(**{**valid, field: value})

        for field, value in {
            "env_step_budget": 0,
            "restored_env_steps": -1,
            "rollout_steps": 0,
            "log_env_interval": -1,
            "eval_env_interval": True,
            "save_env_interval": -1,
        }.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_ppo_exact_budget_schedule(**{**valid, field: value})

        with self.assertRaisesRegex(ValueError, "final endpoint"):
            _validate_ppo_exact_budget_schedule(
                **{**valid, "env_step_budget": 80_000, "eval_env_interval": 12_000}
            )

    def test_confirmatory_resume_is_rejected_until_eval_publication_is_atomic(self):
        _reject_confirmatory_resume(None)
        with self.assertRaisesRegex(RuntimeError, "Restart.*from scratch"):
            _reject_confirmatory_resume("checkpoint_step_10000.pt")

    def test_confirmatory_attempt_paths_are_fresh_and_attempt_indexed(self):
        self.assertEqual(_validate_confirmatory_attempt_index(0), 0)
        for invalid in (None, True, -1, 1.5, "1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                RuntimeError, "attempt-index"
            ):
                _validate_confirmatory_attempt_index(invalid)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_0, evaluation_0 = _claim_confirmatory_attempt_paths(
                checkpoint_root=root / "checkpoints" / "b0-seed101",
                evaluation_root=root / "evaluations",
                run_id="b0-seed101",
                attempt_index=0,
            )
            self.assertEqual(checkpoint_0.name, "attempt_0000")
            self.assertEqual(
                evaluation_0.relative_to(root / "evaluations").as_posix(),
                "b0-seed101/attempt_0000",
            )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                _claim_confirmatory_attempt_paths(
                    checkpoint_root=root / "checkpoints" / "b0-seed101",
                    evaluation_root=root / "evaluations",
                    run_id="b0-seed101",
                    attempt_index=0,
                )

            checkpoint_1, evaluation_1 = _claim_confirmatory_attempt_paths(
                checkpoint_root=root / "checkpoints" / "b0-seed101",
                evaluation_root=root / "evaluations",
                run_id="b0-seed101",
                attempt_index=1,
            )
            self.assertTrue(checkpoint_0.is_dir())
            self.assertTrue(evaluation_0.is_dir())
            self.assertEqual(checkpoint_1.name, "attempt_0001")
            self.assertEqual(evaluation_1.name, "attempt_0001")

    def test_environment_logging_retains_latest_real_update(self):
        first_update = {
            "optimization_performed": 1.0,
            "loss_value": 2.5,
            "env_steps_total": 9_992.0,
        }
        pending = _remember_latest_optimization_metrics(None, first_update)
        self.assertEqual(pending, first_update)
        self.assertIsNot(pending, first_update)

        cap_only = {
            "optimization_performed": 0.0,
            "loss_value": 0.0,
            "env_steps_total": 10_000.0,
        }
        self.assertEqual(
            _remember_latest_optimization_metrics(pending, cap_only),
            first_update,
        )
        self.assertIsNone(_remember_latest_optimization_metrics(None, cap_only))

    def test_config_dict_supports_ppo_dataclass(self):
        config = upi_trm_train.PPOConfig(num_steps=80, num_minibatches=4)
        payload = _config_dict(config)
        self.assertEqual(payload["num_steps"], 80)
        self.assertEqual(payload["num_minibatches"], 4)

    def test_registered_split_manifest_binds_loaded_record_order(self):
        torch.manual_seed(777)
        dataset = DummyPuzzleDataset(num_instances=2, seq_len=4, vocab_size=4)
        record_hashes = dataset_sample_sha256s(dataset)
        input_hashes = upi_trm_train.dataset_input_sha256s(dataset)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            (root / "train").mkdir()
            payload_path = root / "train" / "payload.bin"
            payload_path.write_bytes(b"registered records")
            manifest = {
                "generated_count": 2,
                "record_sha256s": record_hashes,
                "input_sha256s": input_hashes,
                "ordered_record_sha256": upi_trm_train.ordered_record_sha256(
                    record_hashes
                ),
                "files": {
                    "train/payload.bin": {
                        "bytes": payload_path.stat().st_size,
                        "sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                    }
                },
            }
            manifest_path = root / "manifests" / "train.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            registered_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            validated = _validate_materialized_split_manifest(
                dataset_root=root,
                split="train",
                registered_sha256=registered_hash,
                dataset=dataset,
            )
            self.assertEqual(validated["record_sha256s"], record_hashes)

            dataset.samples.reverse()
            with self.assertRaisesRegex(RuntimeError, "pool differs"):
                _validate_materialized_split_manifest(
                    dataset_root=root,
                    split="train",
                    registered_sha256=registered_hash,
                    dataset=dataset,
                )

    def test_confirmatory_evaluation_failure_is_fatal(self):
        trainer = MagicMock()
        trainer.evaluate_policy_metrics.side_effect = RuntimeError("evaluation failed")

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            upi_trm_train._run_eval_and_log(
                progress_step=10_000,
                outer_step=125,
                trainer=trainer,
                env_cfg=MagicMock(),
                dataset=MagicMock(),
                checker_fn=MagicMock(),
                step_iter=None,
                use_wandb=False,
                strict=True,
            )

    def test_confirmatory_source_is_checked_around_artifact_publication(self):
        trainer = MagicMock()
        trainer.evaluate_policy_metrics.return_value = {
            "mean_score": 0.0,
            "success_rate": 0.0,
            "eval_policy_mode": "greedy",
            "eval_seed": 1729,
            "solved_count": 0,
            "total_episodes": 1,
            "score_min": 0.0,
            "score_max": 0.0,
            "initial_score_mean": 0.0,
            "per_instance": [],
        }
        trainer.compute_accounting_snapshot.return_value = {
            "model_work": {"uninstrumented_roles": []}
        }
        trainer.get_debug_stats.return_value = {}
        source_events = []

        with patch(
            "upi_trm_train.write_evaluation_artifact",
            return_value={
                "summary_sha256": "a" * 64,
                "per_instance_sha256": "b" * 64,
            },
        ):
            upi_trm_train._run_eval_and_log(
                progress_step=10_000,
                outer_step=125,
                trainer=trainer,
                env_cfg=MagicMock(),
                dataset=MagicMock(),
                checker_fn=MagicMock(),
                step_iter=None,
                use_wandb=False,
                strict=True,
                artifact_output_dir="unused",
                artifact_metadata={"policy_mode": "greedy"},
                source_revalidation_fn=source_events.append,
            )

        self.assertEqual(
            source_events,
            [
                "before evaluation",
                "before evaluation artifact publication",
                "after evaluation artifact publication",
            ],
        )

    def test_training_materialization_uses_requested_pool_not_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            self._write_dataset_root(root, num_puzzles=12, split="train")
            dataset, *_ = build_dataset_from_paths(
                [str(root)],
                pool_size=10,
                split="train",
            )

        self.assertEqual(len(dataset), 10)

    def test_requested_dataset_failure_is_fatal_by_default(self):
        with patch("rl.training_setup.PuzzleDataset", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "Failed to load the requested dataset"):
                build_dataset_from_paths(["missing-dataset"], pool_size=4)

    def test_eval_puzzle_identifiers_can_be_made_disjoint(self):
        train = DummyPuzzleDataset(num_instances=3, seq_len=4, vocab_size=4)
        evaluation = DummyPuzzleDataset(num_instances=2, seq_len=4, vocab_size=4)
        offset_puzzle_identifiers(evaluation, train.num_identifiers)

        train_ids = {
            int(sample["puzzle_identifiers"].item()) for sample in train.samples
        }
        eval_ids = {
            int(sample["puzzle_identifiers"].item()) for sample in evaluation.samples
        }
        self.assertFalse(train_ids.intersection(eval_ids))
        self.assertEqual(min(eval_ids), train.num_identifiers)

    def test_multiple_dataset_roots_use_disjoint_identifier_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            self._write_dataset_root(first, num_puzzles=2)
            self._write_dataset_root(second, num_puzzles=3)
            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=0,
                    dataset_paths=[str(first), str(second)],
                    global_batch_size=5,
                    test_set_mode=True,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="test",
            )

            identifiers = []
            for _set_name, batch, valid_count in dataset:
                identifiers.extend(
                    int(value)
                    for value in batch["puzzle_identifiers"][:valid_count]
                )

        self.assertEqual(identifiers, [0, 1, 2, 3, 4])
        self.assertEqual(dataset.metadata.num_puzzle_identifiers, 5)

    def test_exact_cap_pauses_persistent_episode_without_optimizing(self):
        torch.manual_seed(101)
        _, trainer, _ = self._make_persistent_budget_trainer()
        self._stub_updates(trainer)

        first_metrics = trainer.train_step(max_env_steps_to_collect=1)
        self.assertEqual(trainer.get_env_step_count(), 1)
        self.assertEqual(trainer._train_step_count, 0)
        self.assertEqual(trainer._next_episode_id, 0)
        self.assertIsNotNone(trainer._active_episode)
        self.assertEqual(first_metrics["optimization_performed"], 0.0)
        trainer.value_update.assert_not_called()
        trainer.policy_update.assert_not_called()

        first = trainer.replay.storage[0]
        second_metrics = trainer.train_step(max_env_steps_to_collect=1)
        second = trainer.replay.storage[1]
        self.assertEqual(second.episode_id, first.episode_id)
        self.assertEqual(second.timestep, first.timestep + 1)
        self.assertTrue(torch.equal(first.y_next, second.y))
        for key in first.x_next:
            self.assertTrue(torch.equal(first.x_next[key], second.x[key]))
        torch.testing.assert_close(first.next_latent.z_H, second.latent.z_H)
        torch.testing.assert_close(first.next_latent.z_L, second.latent.z_L)
        self.assertEqual(second_metrics["optimization_performed"], 0.0)
        trainer.value_update.assert_not_called()
        trainer.policy_update.assert_not_called()

        final_metrics = trainer.train_step(max_env_steps_to_collect=1)
        self.assertEqual(final_metrics["optimization_performed"], 1.0)
        self.assertEqual(trainer.get_env_step_count(), 3)
        self.assertEqual(trainer._train_step_count, 1)
        self.assertEqual(trainer._next_episode_id, 1)
        self.assertIsNone(trainer._active_episode)
        trainer.value_update.assert_called_once()
        trainer.policy_update.assert_called_once()

    def test_collection_pause_schedule_matches_uninterrupted_episode(self):
        torch.manual_seed(202)
        _, uninterrupted, _ = self._make_persistent_budget_trainer()
        self._stub_updates(uninterrupted)
        torch.manual_seed(303)
        uninterrupted.train_step()
        uninterrupted_next_rng = torch.rand(4)

        torch.manual_seed(202)
        _, paused, _ = self._make_persistent_budget_trainer()
        self._stub_updates(paused)
        torch.manual_seed(303)
        for _ in range(3):
            paused.train_step(max_env_steps_to_collect=1)
        paused_next_rng = torch.rand(4)

        self._assert_replay_equal(uninterrupted, paused)
        torch.testing.assert_close(uninterrupted_next_rng, paused_next_rng)
        self.assertEqual(uninterrupted._train_step_count, paused._train_step_count)
        self.assertEqual(uninterrupted._next_episode_id, paused._next_episode_id)
        self.assertEqual(
            uninterrupted.value_update.call_count, paused.value_update.call_count
        )
        self.assertEqual(
            uninterrupted.policy_update.call_count, paused.policy_update.call_count
        )

    def test_schema_v5_fixed_base_resume_continues_persistent_episode_exactly(self):
        torch.manual_seed(404)
        model, original, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(original)
        run_identity = self._run_identity(
            model,
            original,
            provenance,
            environment_interactions=3,
        )
        self._stub_updates(original)
        random.seed(505)
        np.random.seed(505)
        torch.manual_seed(505)
        original.train_step(max_env_steps_to_collect=1)
        saved_transition = original.replay.storage[0]
        saved_step = original.get_env_step_count()
        model.eval()
        original.policy_model_old.eval()
        original.policy_model_candidate.train()
        original.target_model.eval()
        saved_gradient = torch.full_like(next(model.parameters()), 0.25)
        next(model.parameters()).grad = saved_gradient.clone()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                original,
                step=saved_step,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=run_identity,
                checkpoint_lineage=self._root_lineage(),
            )
            original.train_step(max_env_steps_to_collect=2)
            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(4)

            torch.manual_seed(999)
            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact"
            )
            self._stub_updates(restored)
            start_update = resume_from_checkpoint(
                checkpoint_path,
                restored_model,
                restored,
                "cpu",
                expected_dataset_provenance=provenance,
                expected_run_identity=run_identity,
                expected_checkpoint_sha256=file_sha256(checkpoint_path),
            )

            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["checkpoint_schema_version"], 5)
            self.assertEqual(payload["training_protocol"], "fixed_base_exact")
            self.assertEqual(
                payload["rl_config"]["training_protocol"],
                "fixed_base_exact",
            )
            self.assertEqual(payload["execution_device"], "cpu")
            self.assertFalse(
                (Path(tmp) / f"model_step_{saved_step}.pt").exists()
            )
            self.assertFalse(restored_model.training)
            self.assertFalse(restored.policy_model_old.training)
            self.assertTrue(restored.policy_model_candidate.training)
            self.assertFalse(restored.target_model.training)
            torch.testing.assert_close(
                next(restored_model.parameters()).grad,
                saved_gradient,
            )
            saved_compute = payload["trainer_state"][
                "compute_accounting_state"
            ]
            restored_compute = restored.compute_accounting_checkpoint_state()
            self.assertEqual(
                restored_compute["model_compute_state"],
                saved_compute["model_compute_state"],
            )
            self.assertEqual(
                restored_compute["training_model_work"],
                saved_compute["training_model_work"],
            )
            self.assertEqual(
                restored_compute["evaluation_model_work"],
                saved_compute["evaluation_model_work"],
            )
            self.assertEqual(
                restored_compute["training_wall_time_seconds"],
                saved_compute["training_wall_time_seconds"],
            )
            self.assertEqual(
                restored_compute["evaluation_wall_time_seconds"],
                saved_compute["evaluation_wall_time_seconds"],
            )
            self.assertGreaterEqual(
                restored_compute["peak_process_rss_bytes"],
                saved_compute["peak_process_rss_bytes"],
            )

            self.assertEqual(start_update, 0)
            self.assertEqual(restored.get_env_step_count(), 1)
            self.assertIsNotNone(restored._active_episode)
            self.assertEqual(restored._active_episode["timestep"], 1)
            torch.testing.assert_close(
                restored._active_episode["latent"].z_H,
                saved_transition.next_latent.z_H,
            )
            torch.testing.assert_close(
                restored._active_episode["latent"].z_L,
                saved_transition.next_latent.z_L,
            )

            restored.train_step(max_env_steps_to_collect=2)
            actual_python = random.random()
            actual_numpy = float(np.random.rand())
            actual_torch = torch.rand(4)

        self._assert_replay_equal(original, restored)
        self.assertEqual(expected_python, actual_python)
        self.assertEqual(expected_numpy, actual_numpy)
        torch.testing.assert_close(expected_torch, actual_torch)
        self.assertEqual(original._train_step_count, restored._train_step_count)
        self.assertEqual(original._next_episode_id, restored._next_episode_id)
        self.assertIsNone(restored._active_episode)
        self.assertEqual(
            original.compute_accounting_snapshot()["model_work"],
            restored.compute_accounting_snapshot()["model_work"],
        )

    def test_schema_v5_episodic_bridge_cell_round_trips_without_latents(self):
        model, original, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact",
            episodic_latent=True,
        )
        provenance = self._checkpoint_provenance(original)
        identity = self._run_identity(
            model,
            original,
            provenance,
            run_id="unit.episodic17",
            environment_interactions=3,
        )
        self._stub_updates(original)
        original.train_step(max_env_steps_to_collect=1)
        self.assertIsNone(original.replay.storage[0].latent)
        self.assertIsNone(original.replay.storage[0].next_latent)
        self.assertIsNone(original._active_episode["latent"])

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                original,
                step=1,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact",
                episodic_latent=True,
            )
            self._stub_updates(restored)
            resume_from_checkpoint(
                checkpoint_path,
                restored_model,
                restored,
                "cpu",
                expected_dataset_provenance=provenance,
                expected_run_identity=identity,
                expected_checkpoint_sha256=file_sha256(checkpoint_path),
            )

        self.assertIsNone(restored.replay.storage[0].latent)
        self.assertIsNone(restored.replay.storage[0].next_latent)
        self.assertIsNone(restored._active_episode["latent"])
        self.assertEqual(restored.get_env_step_count(), 1)

    def test_schema_v5_resume_rejects_wrong_persistent_latent_shape(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        identity = self._run_identity(
            model,
            trainer,
            provenance,
            environment_interactions=3,
        )
        self._stub_updates(trainer)
        trainer.train_step(max_env_steps_to_collect=1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=trainer._env_step_count,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            wrong_latent = payload["replay_transitions"][0].latent
            hidden_size = wrong_latent.z_H.shape[-1]
            wrong_latent.z_H = torch.zeros(1, 1, hidden_size)
            wrong_latent.z_L = torch.zeros(1, 1, hidden_size)
            corrupt_path = Path(tmp) / "wrong_latent.pt"
            torch.save(payload, corrupt_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact"
            )
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load:
                with self.assertRaisesRegex(RuntimeError, "wrong model shape"):
                    resume_from_checkpoint(
                        str(corrupt_path),
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                        expected_run_identity=identity,
                        expected_checkpoint_sha256=file_sha256(corrupt_path),
                    )
            model_load.assert_not_called()

            missing_successor = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertFalse(
                bool(missing_successor["replay_transitions"][0].done.item())
            )
            missing_successor["replay_transitions"][0].next_latent = None
            missing_successor_path = Path(tmp) / "missing_successor_latent.pt"
            torch.save(missing_successor, missing_successor_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact"
            )
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "nonterminal replay requires current and successor latents",
                ):
                    resume_from_checkpoint(
                        str(missing_successor_path),
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                        expected_run_identity=identity,
                        expected_checkpoint_sha256=file_sha256(
                            missing_successor_path
                        ),
                    )
            model_load.assert_not_called()

    def test_schema_v5_resume_enforces_terminal_replay_latent_contract(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        identity = self._run_identity(
            model,
            trainer,
            provenance,
            environment_interactions=3,
        )
        self._stub_updates(trainer)
        trainer.train_step()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=trainer._env_step_count,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            base = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            terminal_index = next(
                index
                for index, transition in enumerate(base["replay_transitions"])
                if bool(transition.done.item())
            )
            terminal = base["replay_transitions"][terminal_index]
            self.assertIsNotNone(terminal.latent)
            self.assertIsNone(terminal.next_latent)

            cases = []
            missing_current = copy.deepcopy(base)
            missing_current["replay_transitions"][terminal_index].latent = None
            cases.append(
                (
                    "terminal_missing_current.pt",
                    missing_current,
                    "persistent replay requires a current latent",
                )
            )
            successor_on_terminal = copy.deepcopy(base)
            corrupted_terminal = successor_on_terminal["replay_transitions"][
                terminal_index
            ]
            corrupted_terminal.next_latent = copy.deepcopy(
                corrupted_terminal.latent
            )
            cases.append(
                (
                    "terminal_with_successor.pt",
                    successor_on_terminal,
                    "terminal replay must not carry a successor latent",
                )
            )

            for filename, payload, expected_message in cases:
                with self.subTest(filename=filename):
                    corrupt_path = Path(tmp) / filename
                    torch.save(payload, corrupt_path)
                    restored_model, restored, _ = (
                        self._make_persistent_budget_trainer("fixed_base_exact")
                    )
                    with patch.object(
                        restored_model,
                        "load_state_dict",
                        wraps=restored_model.load_state_dict,
                    ) as model_load:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            expected_message,
                        ):
                            resume_from_checkpoint(
                                str(corrupt_path),
                                restored_model,
                                restored,
                                "cpu",
                                expected_dataset_provenance=provenance,
                                expected_run_identity=identity,
                                expected_checkpoint_sha256=file_sha256(
                                    corrupt_path
                                ),
                            )
                    model_load.assert_not_called()

    def test_schema_v5_terminal_reason_marker_rejects_corrupt_new_checkpoint(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        identity = self._run_identity(
            model,
            trainer,
            provenance,
            environment_interactions=3,
        )
        self._stub_updates(trainer)
        trainer.train_step()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=trainer._env_step_count,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                payload["trainer_state"]["terminal_reason_replay_version"],
                1,
            )
            terminal = next(
                transition
                for transition in payload["replay_transitions"]
                if bool(transition.done.item())
            )
            terminal.terminal_reason = None
            corrupt_path = Path(tmp) / "missing_terminal_reason.pt"
            torch.save(payload, corrupt_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact"
            )
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load:
                with self.assertRaisesRegex(RuntimeError, "replay record"):
                    resume_from_checkpoint(
                        str(corrupt_path),
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                        expected_run_identity=identity,
                        expected_checkpoint_sha256=file_sha256(corrupt_path),
                    )
            model_load.assert_not_called()

            del payload["trainer_state"]["terminal_reason_replay_version"]
            markerless_path = Path(tmp) / "markerless_schema_v5.pt"
            torch.save(payload, markerless_path)
            markerless_model, markerless_trainer, _ = (
                self._make_persistent_budget_trainer("fixed_base_exact")
            )
            with patch.object(
                markerless_model,
                "load_state_dict",
                wraps=markerless_model.load_state_dict,
            ) as model_load, patch.object(
                markerless_trainer.value_opt,
                "load_state_dict",
                wraps=markerless_trainer.value_opt.load_state_dict,
            ) as optimizer_load, patch.object(
                markerless_trainer.replay,
                "clear",
                wraps=markerless_trainer.replay.clear,
            ) as replay_clear, patch(
                "upi_trm_train._restore_rng_state"
            ) as restore_rng:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "missing terminal-reason replay metadata",
                ):
                    resume_from_checkpoint(
                        str(markerless_path),
                        markerless_model,
                        markerless_trainer,
                        "cpu",
                        expected_dataset_provenance=provenance,
                        expected_run_identity=identity,
                        expected_checkpoint_sha256=file_sha256(markerless_path),
                    )
            model_load.assert_not_called()
            optimizer_load.assert_not_called()
            replay_clear.assert_not_called()
            restore_rng.assert_not_called()

    def test_schema_v5_resume_rejects_wrong_timestep_zero_active_latent(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        identity = self._run_identity(model, trainer, provenance)
        trainer._start_episode()
        # This fixture invokes a private collector primitive only to construct
        # an invalid timestep-zero latent. Production starts episodes inside a
        # train_step accounting boundary.
        trainer.policy_model_old.reset_compute_counters()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            active_latent = payload["trainer_state"]["collection_state"][
                "active_episode"
            ]["latent"]
            hidden_size = active_latent.z_H.shape[-1]
            active_latent.z_H = torch.zeros(1, 1, hidden_size)
            active_latent.z_L = torch.zeros(1, 1, hidden_size)
            corrupt_path = Path(tmp) / "wrong_active_latent.pt"
            torch.save(payload, corrupt_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact"
            )
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load:
                with self.assertRaisesRegex(RuntimeError, "active latent component"):
                    resume_from_checkpoint(
                        str(corrupt_path),
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                        expected_run_identity=identity,
                        expected_checkpoint_sha256=file_sha256(corrupt_path),
                    )
            model_load.assert_not_called()

    def test_fixed_base_save_requires_identity_and_never_overwrites(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        identity = self._run_identity(model, trainer, provenance)
        with tempfile.TemporaryDirectory() as tmp:
            trainer._train_step_active = True
            with self.assertRaisesRegex(RuntimeError, "idle trainer"):
                save_checkpoint(
                    model,
                    trainer,
                    step=0,
                    checkpoint_dir=tmp,
                    rl_cfg=cfg,
                    dataset_provenance=provenance,
                    run_identity=identity,
                    checkpoint_lineage=self._root_lineage(),
                )
            trainer._train_step_active = False
            with self.assertRaisesRegex(RuntimeError, "require a run identity"):
                save_checkpoint(
                    model,
                    trainer,
                    step=0,
                    checkpoint_dir=tmp,
                    rl_cfg=cfg,
                    dataset_provenance=provenance,
                )
            self.assertEqual(list(Path(tmp).iterdir()), [])

            with patch.object(
                trainer,
                "exact_centering_checkpoint_state",
                None,
            ), self.assertRaisesRegex(
                RuntimeError,
                "require exact-centering state",
            ):
                save_checkpoint(
                    model,
                    trainer,
                    step=0,
                    checkpoint_dir=tmp,
                    rl_cfg=cfg,
                    dataset_provenance=provenance,
                    run_identity=identity,
                    checkpoint_lineage=self._root_lineage(),
                )
            self.assertEqual(list(Path(tmp).iterdir()), [])

            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            original_bytes = Path(checkpoint_path).read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                save_checkpoint(
                    model,
                    trainer,
                    step=0,
                    checkpoint_dir=tmp,
                    rl_cfg=cfg,
                    dataset_provenance=provenance,
                    run_identity=identity,
                    checkpoint_lineage=self._root_lineage(),
                )
            self.assertEqual(Path(checkpoint_path).read_bytes(), original_bytes)

    def test_schema_v5_resume_matches_real_next_optimizer_update(self):
        random.seed(1201)
        np.random.seed(1201)
        torch.manual_seed(1201)
        model, uninterrupted, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact",
            batch_size=1,
        )
        uninterrupted.set_checker_fn(dummy_checker)
        provenance = self._checkpoint_provenance(uninterrupted)
        identity = self._run_identity(model, uninterrupted, provenance)

        first_metrics = uninterrupted.train_step()
        self.assertEqual(first_metrics["value_optimizer_step"], 1.0)
        self.assertEqual(first_metrics["policy_optimizer_step"], 1.0)
        self.assertEqual(uninterrupted._value_optimizer_step_count, 1)
        self.assertEqual(uninterrupted._policy_optimizer_step_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                uninterrupted,
                step=uninterrupted._train_step_count,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            checkpoint_payload, _ = upi_trm_train._load_checkpoint_payload(
                checkpoint_path
            )
            centering_state = checkpoint_payload["trainer_state"][
                "exact_centering_state"
            ]
            self.assertEqual(centering_state["batch_count"], 1)
            self.assertTrue(centering_state["history_complete"])
            self.assertLessEqual(
                centering_state["maximum_observed"],
                centering_state["tolerance"],
            )

            expected_metrics = uninterrupted.train_step()
            expected_centering_state = (
                uninterrupted.exact_centering_checkpoint_state()
            )
            expected_modules = {
                "model": copy.deepcopy(model.state_dict()),
                "old": copy.deepcopy(uninterrupted.policy_model_old.state_dict()),
                "candidate": copy.deepcopy(
                    uninterrupted.policy_model_candidate.state_dict()
                ),
                "target": copy.deepcopy(uninterrupted.target_model.state_dict()),
            }
            expected_value_optimizer = copy.deepcopy(
                uninterrupted.value_opt.state_dict()
            )
            expected_policy_optimizer = copy.deepcopy(
                uninterrupted.policy_opt.state_dict()
            )
            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(4)

            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored_model, restored, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact",
                batch_size=1,
            )
            restored.set_checker_fn(dummy_checker)
            resume_from_checkpoint(
                checkpoint_path,
                restored_model,
                restored,
                "cpu",
                expected_dataset_provenance=provenance,
                expected_run_identity=identity,
                expected_checkpoint_sha256=file_sha256(checkpoint_path),
            )
            self.assertEqual(
                restored.exact_centering_checkpoint_state(),
                centering_state,
            )
            actual_metrics = restored.train_step()
            actual_centering_state = restored.exact_centering_checkpoint_state()
            actual_python = random.random()
            actual_numpy = float(np.random.rand())
            actual_torch = torch.rand(4)

        self._assert_nested_equal(expected_modules["model"], restored_model.state_dict())
        self._assert_nested_equal(
            expected_modules["old"], restored.policy_model_old.state_dict()
        )
        self._assert_nested_equal(
            expected_modules["candidate"],
            restored.policy_model_candidate.state_dict(),
        )
        self._assert_nested_equal(
            expected_modules["target"], restored.target_model.state_dict()
        )
        self._assert_nested_equal(
            expected_value_optimizer, restored.value_opt.state_dict()
        )
        self._assert_nested_equal(
            expected_policy_optimizer, restored.policy_opt.state_dict()
        )
        self.assertEqual(
            uninterrupted._value_optimizer_step_count,
            restored._value_optimizer_step_count,
        )
        self.assertEqual(
            uninterrupted._policy_optimizer_step_count,
            restored._policy_optimizer_step_count,
        )
        self.assertEqual(
            expected_metrics["value_optimizer_step"],
            actual_metrics["value_optimizer_step"],
        )
        self.assertEqual(
            expected_metrics["policy_optimizer_step"],
            actual_metrics["policy_optimizer_step"],
        )
        for metric in (
            "exact_centering_defect_max",
            "exact_centering_defect_max_observed",
            "exact_centering_tolerance",
            "exact_centering_batches_total",
            "exact_centering_history_complete",
        ):
            self.assertEqual(expected_metrics[metric], actual_metrics[metric])
        self.assertEqual(expected_centering_state, actual_centering_state)
        self.assertEqual(expected_python, actual_python)
        self.assertEqual(expected_numpy, actual_numpy)
        torch.testing.assert_close(expected_torch, actual_torch, rtol=0, atol=0)

    def test_schema_v5_identity_and_runtime_mismatch_fail_before_mutation(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        provenance = self._checkpoint_provenance(trainer)
        identity = self._run_identity(model, trainer, provenance)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
                run_identity=identity,
                checkpoint_lineage=self._root_lineage(),
            )
            base = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            variants = []
            for name, field, value in (
                ("seed", "training_seed", 18),
                ("run_id", "run_id", "unit.seed18"),
            ):
                payload = copy.deepcopy(base)
                payload["run_identity"][field] = value
                variants.append((name, payload, "Run identity mismatch"))

            commit = copy.deepcopy(base)
            commit["run_identity"]["producer"]["git_commit"] = "b" * 40
            variants.append(("commit", commit, "Run identity mismatch"))

            config = copy.deepcopy(base)
            config["run_identity"]["effective_config"]["debug_checks"] = True
            config["run_identity"]["effective_config_sha256"] = (
                canonical_json_sha256(
                    config["run_identity"]["effective_config"]
                )
            )
            variants.append(("config", config, "Run identity mismatch"))

            dataset = copy.deepcopy(base)
            dataset["run_identity"]["dataset_provenance_sha256"] = "f" * 64
            variants.append(("dataset", dataset, "Run identity mismatch"))

            runtime = copy.deepcopy(base)
            runtime["runtime_fingerprint"]["torch_version"] = "different"
            variants.append(("runtime", runtime, "Runtime fingerprint"))

            schema4 = copy.deepcopy(base)
            schema4["checkpoint_schema_version"] = 4
            variants.append(("schema4", schema4, "Schema-v4 fixed-base"))

            for name, invalid_version in (
                ("schema_bool", True),
                ("schema_float", 5.0),
                ("schema_string", "5"),
            ):
                invalid_schema = copy.deepcopy(base)
                invalid_schema["checkpoint_schema_version"] = invalid_version
                variants.append(
                    (name, invalid_schema, "schema version must be an integer")
                )

            markerless = copy.deepcopy(base)
            del markerless["trainer_state"]["terminal_reason_replay_version"]
            variants.append(
                (
                    "missing_terminal_reason_marker",
                    markerless,
                    "missing terminal-reason replay metadata",
                )
            )

            terminal_marker_float = copy.deepcopy(base)
            terminal_marker_float["trainer_state"][
                "terminal_reason_replay_version"
            ] = 1.0
            variants.append(
                (
                    "terminal_reason_marker_float",
                    terminal_marker_float,
                    "terminal-reason replay version is invalid",
                )
            )

            missing_centering = copy.deepcopy(base)
            del missing_centering["trainer_state"]["exact_centering_state"]
            variants.append(
                (
                    "missing_exact_centering_state",
                    missing_centering,
                    "missing exact-centering state",
                )
            )

            centering_schema_float = copy.deepcopy(base)
            centering_schema_float["trainer_state"]["exact_centering_state"][
                "schema_version"
            ] = 1.0
            variants.append(
                (
                    "exact_centering_schema_float",
                    centering_schema_float,
                    "exact-centering evidence is invalid",
                )
            )

            collector_schema_bool = copy.deepcopy(base)
            collector_schema_bool["trainer_state"]["collection_state"][
                "schema_version"
            ] = True
            variants.append(
                (
                    "collector_schema_bool",
                    collector_schema_bool,
                    "collector checkpoint schema",
                )
            )

            environment_schema_float = copy.deepcopy(base)
            environment_schema_float["trainer_state"]["environment_state"][
                "schema_version"
            ] = 1.0
            variants.append(
                (
                    "environment_schema_float",
                    environment_schema_float,
                    "PlanEditEnv checkpoint schema",
                )
            )

            compute_schema_string = copy.deepcopy(base)
            compute_schema_string["trainer_state"]["compute_accounting_state"][
                "schema_version"
            ] = "1"
            variants.append(
                (
                    "compute_schema_string",
                    compute_schema_string,
                    "compute accounting state is invalid",
                )
            )

            module_state = copy.deepcopy(base)
            removed_parameter = next(
                name
                for name in module_state["model_state_dict"]
                if name.startswith("value_head.")
            )
            module_state["model_state_dict"].pop(removed_parameter)
            variants.append(("module_state", module_state, "model state failed"))

            optimizer_state = copy.deepcopy(base)
            optimizer_state["value_optimizer_state_dict"]["param_groups"][0][
                "params"
            ] = []
            variants.append(
                (
                    "optimizer_state",
                    optimizer_state,
                    "value_optimizer_state_dict failed",
                )
            )

            gradients = copy.deepcopy(base)
            first_gradient_name = next(
                iter(gradients["parameter_gradients"]["model"])
            )
            gradients["parameter_gradients"]["model"][
                first_gradient_name
            ] = torch.zeros(1)
            variants.append(("gradients", gradients, "model state failed"))

            recurrent_name = next(
                name
                for name, value in base["model_state_dict"].items()
                if name.startswith("inner.")
                and not name.startswith(("inner.lm_head.", "inner.q_head."))
                and torch.is_floating_point(value)
            )
            candidate_recurrence = copy.deepcopy(base)
            candidate_recurrence["policy_model_candidate_state_dict"][
                recurrent_name
            ] = (
                candidate_recurrence["policy_model_candidate_state_dict"][
                    recurrent_name
                ].clone()
                + 1.0
            )
            variants.append(
                (
                    "candidate_recurrence",
                    candidate_recurrence,
                    "shared frozen recurrent map",
                )
            )

            target_recurrence = copy.deepcopy(base)
            target_recurrence["target_model_state_dict"][recurrent_name] = (
                target_recurrence["target_model_state_dict"][
                    recurrent_name
                ].clone()
                + 1.0
            )
            variants.append(
                (
                    "target_recurrence",
                    target_recurrence,
                    "shared frozen recurrent map",
                )
            )

            trainer_state = copy.deepcopy(base)
            trainer_state["trainer_state"]["term_stats"] = 1
            variants.append(
                ("trainer_state", trainer_state, "term_stats has an invalid")
            )

            for name, payload, expected_message in variants:
                with self.subTest(name=name):
                    variant_path = Path(tmp) / f"{name}.pt"
                    torch.save(payload, variant_path)
                    restored_model, restored, _ = (
                        self._make_persistent_budget_trainer("fixed_base_exact")
                    )
                    live_modules = {
                        "model": restored_model,
                        "old": restored.policy_model_old,
                        "candidate": restored.policy_model_candidate,
                        "target": restored.target_model,
                    }
                    before_modules = {
                        module_name: {
                            key: value.detach().clone()
                            for key, value in module.state_dict().items()
                        }
                        for module_name, module in live_modules.items()
                    }
                    with patch.object(
                        restored_model,
                        "load_state_dict",
                        wraps=restored_model.load_state_dict,
                    ) as model_load, patch.object(
                        restored.policy_model_old,
                        "load_state_dict",
                        wraps=restored.policy_model_old.load_state_dict,
                    ) as old_load, patch.object(
                        restored.policy_model_candidate,
                        "load_state_dict",
                        wraps=restored.policy_model_candidate.load_state_dict,
                    ) as candidate_load, patch.object(
                        restored.target_model,
                        "load_state_dict",
                        wraps=restored.target_model.load_state_dict,
                    ) as target_load, patch.object(
                        restored.value_opt,
                        "load_state_dict",
                        wraps=restored.value_opt.load_state_dict,
                    ) as optimizer_load, patch.object(
                        restored.replay,
                        "clear",
                        wraps=restored.replay.clear,
                    ) as replay_clear, patch(
                        "upi_trm_train._restore_rng_state"
                    ) as restore_rng:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            expected_message,
                        ):
                            resume_from_checkpoint(
                                str(variant_path),
                                restored_model,
                                restored,
                                "cpu",
                                expected_dataset_provenance=provenance,
                                expected_run_identity=identity,
                                expected_checkpoint_sha256=file_sha256(
                                    variant_path
                                ),
                            )
                    model_load.assert_not_called()
                    old_load.assert_not_called()
                    candidate_load.assert_not_called()
                    target_load.assert_not_called()
                    optimizer_load.assert_not_called()
                    replay_clear.assert_not_called()
                    restore_rng.assert_not_called()
                    for module_name, module in live_modules.items():
                        for key, value in module.state_dict().items():
                            torch.testing.assert_close(
                                value,
                                before_modules[module_name][key],
                            )

    def test_legacy_checkpoint_requires_explicit_weights_only_warm_start(self):
        torch.manual_seed(606)
        model, trainer, cfg = self._make_persistent_budget_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=self._checkpoint_provenance(trainer),
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            schema3_payload = copy.deepcopy(payload)
            schema3_payload["checkpoint_schema_version"] = 3
            schema3_payload.pop("training_protocol", None)
            schema3_payload["rl_config"].pop("training_protocol", None)
            schema3_path = str(Path(tmp) / "schema3_legacy.pt")
            torch.save(schema3_payload, schema3_path)

            legacy_model, legacy_trainer, _ = self._make_persistent_budget_trainer()
            start_step = resume_from_checkpoint(
                schema3_path,
                legacy_model,
                legacy_trainer,
                "cpu",
                expected_dataset_provenance=self._checkpoint_provenance(trainer),
            )
            self.assertEqual(start_step, 0)

            fixed_model, fixed_trainer, _ = self._make_persistent_budget_trainer(
                "fixed_base_exact"
            )
            with patch.object(
                fixed_model,
                "load_state_dict",
                wraps=fixed_model.load_state_dict,
            ) as model_load, patch.object(
                fixed_trainer.value_opt,
                "load_state_dict",
                wraps=fixed_trainer.value_opt.load_state_dict,
            ) as optimizer_load, patch.object(
                fixed_trainer.replay,
                "clear",
                wraps=fixed_trainer.replay.clear,
            ) as replay_clear:
                with self.assertRaisesRegex(RuntimeError, "Schema-v3"):
                    resume_from_checkpoint(
                        schema3_path,
                        fixed_model,
                        fixed_trainer,
                        "cpu",
                        expected_dataset_provenance=self._checkpoint_provenance(
                            fixed_trainer
                        ),
                    )
            model_load.assert_not_called()
            optimizer_load.assert_not_called()
            replay_clear.assert_not_called()

            payload["checkpoint_schema_version"] = 2
            legacy_path = str(Path(tmp) / "legacy.pt")
            torch.save(payload, legacy_path)

            _, strict_trainer, _ = self._make_persistent_budget_trainer()
            with self.assertRaisesRegex(RuntimeError, "schema-v3"):
                resume_from_checkpoint(
                    legacy_path,
                    strict_trainer.model,
                    strict_trainer,
                    "cpu",
                )

            warm_model, warm_trainer, _ = self._make_persistent_budget_trainer()
            before = {
                key: value.detach().clone()
                for key, value in warm_model.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "resume is refused"):
                resume_from_checkpoint(
                    legacy_path,
                    warm_model,
                    warm_trainer,
                    "cpu",
                    allow_legacy_warm_start=True,
                )
            for key, value in warm_model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertEqual(warm_trainer.get_env_step_count(), 0)
            self.assertEqual(len(warm_trainer.replay), 0)
            self.assertIsNone(warm_trainer._active_episode)

    def test_fixed_base_checkpoint_rejects_stale_single_model_artifact(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            "fixed_base_exact"
        )
        with tempfile.TemporaryDirectory() as tmp:
            stale_path = Path(tmp) / "model_step_2.pt"
            stale_path.write_bytes(b"stale legacy weights")

            with self.assertRaisesRegex(
                RuntimeError,
                "stale single-model artifact",
            ):
                save_checkpoint(
                    model,
                    trainer,
                    step=3,
                    checkpoint_dir=tmp,
                    rl_cfg=cfg,
                    dataset_provenance=self._checkpoint_provenance(trainer),
                )

            self.assertEqual(stale_path.read_bytes(), b"stale legacy weights")
            self.assertFalse((Path(tmp) / "rl_checkpoint_step_3.pt").exists())

    def test_preinterpolation_mixture_checkpoint_publishes_complete_pair_only(self):
        model, trainer, cfg = self._make_persistent_budget_trainer(
            capture_preinterpolation_policy_pair=True,
            evaluation_policy_mode="preinterpolation_exact_mixture",
        )
        with torch.no_grad():
            next(trainer.policy_model_candidate.edit_policy.parameters()).add_(0.5)
        trainer._capture_current_preinterpolation_pair()
        expected_base = {
            name: value.detach().clone()
            for name, value in trainer.preinterpolation_policy_base.state_dict().items()
        }
        expected_candidate = {
            name: value.detach().clone()
            for name, value in trainer.preinterpolation_policy_candidate.state_dict().items()
        }
        provenance = self._checkpoint_provenance(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            self.assertIn("preinterpolation_policy_base_state_dict", payload)
            self.assertIn("preinterpolation_policy_candidate_state_dict", payload)
            self.assertEqual(payload["preinterpolation_pair_generation"], 1)
            self.assertFalse((Path(tmp) / "model_step_0.pt").exists())

            restored_model, restored, _ = self._make_persistent_budget_trainer(
                capture_preinterpolation_policy_pair=True,
                evaluation_policy_mode="preinterpolation_exact_mixture",
            )
            resume_from_checkpoint(
                checkpoint_path,
                restored_model,
                restored,
                "cpu",
                expected_dataset_provenance=provenance,
            )
            self.assertEqual(restored._preinterpolation_pair_generation, 1)
            for name, value in restored.preinterpolation_policy_base.state_dict().items():
                torch.testing.assert_close(value, expected_base[name])
            for name, value in restored.preinterpolation_policy_candidate.state_dict().items():
                torch.testing.assert_close(value, expected_candidate[name])

    def test_checkpoint_roundtrip_restores_target_replay_and_counters(self):
        dataset = DummyPuzzleDataset(num_instances=4, seq_len=8, vocab_size=16)
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.9,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )

        def make_trainer():
            env = PlanEditEnv(dataset, dummy_checker, env_cfg)
            env.set_stop_action_id(_num_actions(dataset.seq_len, dataset.vocab_size) - 1)
            cfg = RLConfig(batch_size=2, max_edits=3, gamma=env_cfg.gamma)
            model = TinyRecursiveReasoningModel_ACTV1(
                _tiny_trm_cfg(
                    dataset.seq_len,
                    dataset.vocab_size,
                    dataset.num_identifiers,
                    cfg.batch_size,
                )
            )
            return model, UPITrmTrainer(model, env, cfg, torch.device("cpu")), cfg

        model, trainer, cfg = make_trainer()
        trainer.collect_episode()
        trainer._train_step_count = 7
        trainer._env_step_count = 11
        with torch.no_grad():
            next(iter(trainer.target_model.parameters())).fill_(0.123)

        with tempfile.TemporaryDirectory() as tmp:
            provenance = self._checkpoint_provenance(trainer)
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=7,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            expected_random = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(3)
            checkpoint_payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint_payload["checkpoint_schema_version"], 4)
            self.assertEqual(checkpoint_payload["training_protocol"], "legacy")
            self.assertTrue((Path(tmp) / "model_step_7.pt").exists())
            self.assertEqual(checkpoint_payload["dataset_provenance"], provenance)
            self.assertIn("rng_state", checkpoint_payload)

            random.seed(9999)
            np.random.seed(9999)
            torch.manual_seed(9999)
            self.assertEqual(checkpoint_payload["model_config"]["hidden_size"], 32)
            self.assertEqual(checkpoint_payload["model_config"]["H_cycles"], 1)
            restored_model, restored, _ = make_trainer()
            with patch("upi_trm_train.torch.load", wraps=torch.load) as checkpoint_load:
                start_step = resume_from_checkpoint(
                    checkpoint_path,
                    restored_model,
                    restored,
                    "cpu",
                    expected_dataset_provenance=provenance,
                )

            self.assertEqual(checkpoint_load.call_args.kwargs["map_location"], "cpu")
            # Resume is data-only: the restricted unpickler must be in force so
            # a hostile checkpoint cannot execute during a strict restore.
            self.assertTrue(checkpoint_load.call_args.kwargs["weights_only"])
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.rand()), expected_numpy)
            torch.testing.assert_close(torch.rand(3), expected_torch)

        self.assertEqual(start_step, 7)
        self.assertEqual(restored._train_step_count, 7)
        self.assertEqual(restored._env_step_count, 11)
        self.assertEqual(len(restored.replay), len(trainer.replay))
        restored_transition = restored.replay.storage[0]
        for value in (
            *restored_transition.x.values(),
            restored_transition.y,
            *restored_transition.x_next.values(),
            restored_transition.y_next,
            restored_transition.action,
            restored_transition.reward,
            restored_transition.done,
        ):
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
        torch.testing.assert_close(
            next(iter(restored.target_model.parameters())),
            next(iter(trainer.target_model.parameters())),
        )

    def test_resume_rejects_dataset_mismatch_before_mutating_model(self):
        dataset = DummyPuzzleDataset(num_instances=3, seq_len=4, vocab_size=4)
        env_cfg = PlanEditEnvConfig(
            max_edits=2,
            gamma=0.9,
            reward_shaping=True,
            vocab_size=dataset.vocab_size,
        )

        def make_trainer():
            env = PlanEditEnv(dataset, dummy_checker, env_cfg)
            env.set_stop_action_id(
                _num_actions(dataset.seq_len, dataset.vocab_size) - 1
            )
            cfg = RLConfig(batch_size=2, max_edits=2, gamma=env_cfg.gamma)
            model = TinyRecursiveReasoningModel_ACTV1(
                _tiny_trm_cfg(
                    dataset.seq_len,
                    dataset.vocab_size,
                    dataset.num_identifiers,
                    cfg.batch_size,
                )
            )
            return model, UPITrmTrainer(model, env, cfg, torch.device("cpu")), cfg

        model, trainer, cfg = make_trainer()
        saved_provenance = self._checkpoint_provenance(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=1,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=saved_provenance,
            )
            restored_model, restored, _ = make_trainer()
            before = {
                key: value.detach().clone()
                for key, value in restored_model.state_dict().items()
            }
            python_rng_before = random.getstate()
            numpy_rng_before = np.random.get_state()
            torch_rng_before = torch.random.get_rng_state().clone()
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load, patch.object(
                restored.value_opt,
                "load_state_dict",
                wraps=restored.value_opt.load_state_dict,
            ) as value_optimizer_load, patch.object(
                restored.policy_opt,
                "load_state_dict",
                wraps=restored.policy_opt.load_state_dict,
            ) as policy_optimizer_load, patch.object(
                restored.replay,
                "clear",
                wraps=restored.replay.clear,
            ) as replay_clear, patch(
                "upi_trm_train._restore_rng_state"
            ) as restore_rng:
                with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    resume_from_checkpoint(
                        checkpoint_path,
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=self._checkpoint_provenance(
                            restored,
                            generation_seed=999,
                        ),
                    )
            model_load.assert_not_called()
            value_optimizer_load.assert_not_called()
            policy_optimizer_load.assert_not_called()
            replay_clear.assert_not_called()
            restore_rng.assert_not_called()
            for key, value in restored_model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertEqual(random.getstate(), python_rng_before)
            numpy_rng_after = np.random.get_state()
            self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
            np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
            self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
            torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)

    def test_resume_preflights_environment_before_mutating_model(self):
        model, trainer, cfg = self._make_persistent_budget_trainer()
        provenance = self._checkpoint_provenance(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            payload["trainer_state"]["environment_state"]["config"][
                "max_edits"
            ] += 1
            corrupt_path = str(Path(tmp) / "bad_environment.pt")
            torch.save(payload, corrupt_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer()
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load, patch.object(
                restored.value_opt,
                "load_state_dict",
                wraps=restored.value_opt.load_state_dict,
            ) as optimizer_load, patch.object(
                restored.replay,
                "clear",
                wraps=restored.replay.clear,
            ) as replay_clear, patch(
                "upi_trm_train._restore_rng_state"
            ) as restore_rng:
                with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                    resume_from_checkpoint(
                        corrupt_path,
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                    )

            model_load.assert_not_called()
            optimizer_load.assert_not_called()
            replay_clear.assert_not_called()
            restore_rng.assert_not_called()

    def test_resume_rejects_execution_device_mismatch_before_mutation(self):
        model, trainer, cfg = self._make_persistent_budget_trainer()
        provenance = self._checkpoint_provenance(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            payload["execution_device"] = "cuda:0"
            mismatch_path = str(Path(tmp) / "bad_device.pt")
            torch.save(payload, mismatch_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer()
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load, patch.object(
                restored.value_opt,
                "load_state_dict",
                wraps=restored.value_opt.load_state_dict,
            ) as optimizer_load, patch.object(
                restored.replay,
                "clear",
                wraps=restored.replay.clear,
            ) as replay_clear:
                with self.assertRaisesRegex(RuntimeError, "execution device"):
                    resume_from_checkpoint(
                        mismatch_path,
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                    )

            model_load.assert_not_called()
            optimizer_load.assert_not_called()
            replay_clear.assert_not_called()

    def test_resume_rejects_nonfinite_model_state_before_mutation(self):
        model, trainer, cfg = self._make_persistent_budget_trainer()
        provenance = self._checkpoint_provenance(trainer)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = save_checkpoint(
                model,
                trainer,
                step=0,
                checkpoint_dir=tmp,
                rl_cfg=cfg,
                dataset_provenance=provenance,
            )
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            first_name = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][first_name].view(-1)[0] = float("nan")
            corrupt_path = str(Path(tmp) / "nonfinite_model.pt")
            torch.save(payload, corrupt_path)

            restored_model, restored, _ = self._make_persistent_budget_trainer()
            with patch.object(
                restored_model,
                "load_state_dict",
                wraps=restored_model.load_state_dict,
            ) as model_load, patch.object(
                restored.value_opt,
                "load_state_dict",
                wraps=restored.value_opt.load_state_dict,
            ) as optimizer_load, patch.object(
                restored.replay,
                "clear",
                wraps=restored.replay.clear,
            ) as replay_clear:
                with self.assertRaisesRegex(RuntimeError, "failed preflight"):
                    resume_from_checkpoint(
                        corrupt_path,
                        restored_model,
                        restored,
                        "cpu",
                        expected_dataset_provenance=provenance,
                    )

            model_load.assert_not_called()
            optimizer_load.assert_not_called()
            replay_clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
