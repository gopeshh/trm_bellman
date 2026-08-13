#!/usr/bin/env python3

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from scripts import audit_phase4_paper_ready
from scripts import eval_phase4_2x2_norm_ablation
from scripts import make_paper_figures_phase4
from scripts.phase4_result_schema import (
    PHASE4_METRIC_AVAILABILITY,
    PHASE4_SCHEMA_VERSION,
    Phase4SummaryValidationError,
    phase4_metric_availability,
    validate_phase4_summary,
    write_phase4_summary,
)


RETIRED_RUN_FIELDS = {
    "success_trivial",
    "success_hard",
    "final_loss",
    "has_nan",
}
RETIRED_AGGREGATE_FIELDS = {
    "success_trivial_mean",
    "success_trivial_std",
    "success_hard_mean",
    "success_hard_std",
    "nan_count",
}


CONDITION_TOGGLES = {
    "nc_nv": (False, True),
    "nc_yv": (False, False),
    "yc_nv": (True, True),
    "yc_yv": (True, False),
}


def _condition_result(
    condition: str = "nc_nv", seed: int = 41
) -> eval_phase4_2x2_norm_ablation.ConditionResult:
    condition_index = eval_phase4_2x2_norm_ablation.CONDITIONS.index(condition)
    offset = condition_index * 0.05 + (seed - 41) * 0.01
    enable_contraction, disable_value_head_norm = CONDITION_TOGGLES[condition]
    return eval_phase4_2x2_norm_ablation.ConditionResult(
        condition=condition,
        seed=seed,
        checkpoint_path=f"results/{condition}_s{seed}/model_step_5000.pt",
        enable_contraction=enable_contraction,
        disable_value_head_norm=disable_value_head_norm,
        latent_projection_mode="enabled",
        latent_ball_radius=10.0,
        L_preproj=0.4 + offset,
        L_preproj_std=0.02,
        var_V=0.3 + offset,
        projection_active_rate=0.1 + offset,
        argmax_agreement_4x=0.9 - offset,
        argmax_agreement_8x=0.8 - offset,
        delta_V_4x=0.1 + offset,
        delta_V_8x=0.2 + offset,
    )


def _valid_summary():
    results = [
        _condition_result(condition, seed)
        for condition in eval_phase4_2x2_norm_ablation.CONDITIONS
        for seed in eval_phase4_2x2_norm_ablation.SEEDS
    ]
    aggregates = eval_phase4_2x2_norm_ablation.aggregate_results(results)
    return {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "metric_availability": phase4_metric_availability(),
        "experiment": "Phase4_2x2_norm_ablation",
        "description": "test summary",
        "generated_at": "2026-08-12T00:00:00",
        "git_sha": "deadbeef",
        "conditions": list(eval_phase4_2x2_norm_ablation.CONDITIONS),
        "seeds": list(eval_phase4_2x2_norm_ablation.SEEDS),
        "all_results": [asdict(result) for result in results],
        "aggregates": [asdict(aggregate) for aggregate in aggregates],
    }


class Phase4ReportingTest(unittest.TestCase):
    def test_schema_v2_has_exact_availability_and_no_retired_fields(self) -> None:
        summary = _valid_summary()

        validate_phase4_summary(summary)

        self.assertEqual(summary["metric_availability"], PHASE4_METRIC_AVAILABILITY)
        self.assertFalse(RETIRED_RUN_FIELDS.intersection(summary["all_results"][0]))
        self.assertFalse(
            RETIRED_AGGREGATE_FIELDS.intersection(summary["aggregates"][0])
        )
        self.assertTrue(
            audit_phase4_paper_ready.check_publication_schema(summary).passed
        )

    def test_legacy_placeholder_and_null_fields_are_rejected(self) -> None:
        for field, value in (
            ("success_trivial", 0.0),
            ("success_hard", None),
            ("final_loss", 0.0),
            ("has_nan", False),
        ):
            with self.subTest(field=field, value=value):
                summary = _valid_summary()
                summary["all_results"][0][field] = value
                with self.assertRaises(Phase4SummaryValidationError):
                    validate_phase4_summary(summary)

        summary = _valid_summary()
        summary["aggregates"][0]["success_trivial_mean"] = None
        with self.assertRaises(Phase4SummaryValidationError):
            validate_phase4_summary(summary)

    def test_changed_availability_metadata_is_rejected(self) -> None:
        summary = _valid_summary()
        summary["metric_availability"]["final_loss"]["reason"] = "unknown"

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "metric_availability"
        ):
            validate_phase4_summary(summary)

    def test_aggregate_seed_count_must_match_measured_runs(self) -> None:
        summary = _valid_summary()
        summary["aggregates"][0]["n_seeds"] = 2

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "n_seeds must be 3"
        ):
            validate_phase4_summary(summary)

    def test_partial_design_is_not_publishable(self) -> None:
        summary = _valid_summary()
        summary["all_results"].pop()

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "complete four-condition"
        ):
            validate_phase4_summary(summary)

    def test_aggregate_values_are_recomputed_from_runs(self) -> None:
        summary = _valid_summary()
        summary["aggregates"][0]["var_V_mean"] += 0.01

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "recomputed from all_results"
        ):
            validate_phase4_summary(summary)

    def test_condition_toggles_are_bound_by_schema(self) -> None:
        summary = _valid_summary()
        summary["all_results"][0]["enable_contraction"] = True

        with self.assertRaisesRegex(
            Phase4SummaryValidationError, "does not match condition"
        ):
            validate_phase4_summary(summary)

    def test_aggregation_exposes_only_measured_metrics(self) -> None:
        aggregate = eval_phase4_2x2_norm_ablation.aggregate_results(
            [_condition_result()]
        )[0]

        self.assertFalse(RETIRED_RUN_FIELDS.intersection(asdict(_condition_result())))
        self.assertFalse(RETIRED_AGGREGATE_FIELDS.intersection(asdict(aggregate)))

    def test_claims_and_latex_do_not_publish_unmeasured_values(self) -> None:
        summary = _valid_summary()
        aggregates = eval_phase4_2x2_norm_ablation.aggregate_results(
            [
                _condition_result(condition, seed)
                for condition in eval_phase4_2x2_norm_ablation.CONDITIONS
                for seed in eval_phase4_2x2_norm_ablation.SEEDS
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            eval_phase4_2x2_norm_ablation.generate_claims_md(
                aggregates, output_dir
            )
            make_paper_figures_phase4.generate_latex_table(summary, output_dir)

            claims = (output_dir / "CLAIMS.md").read_text()
            latex = (
                output_dir / "table_phase4_2x2_norm_ablation.tex"
            ).read_text()

        self.assertNotIn("| Success |", claims)
        self.assertNotIn("NaN", claims)
        self.assertNotIn("expected to show", claims)
        for metric, metadata in PHASE4_METRIC_AVAILABILITY.items():
            self.assertIn(f"`{metric}`", claims)
            self.assertIn(f"`{metadata['reason']}`", claims)
        self.assertNotIn("Success", latex)
        self.assertNotIn("success_trivial", latex)

    def test_invalid_summary_fails_before_figure_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            summary_path = temp_path / "legacy_summary.json"
            output_path = temp_path / "figures"
            legacy_summary = _valid_summary()
            del legacy_summary["schema_version"]
            summary_path.write_text(json.dumps(legacy_summary))

            with mock.patch.object(
                sys,
                "argv",
                [
                    "make_paper_figures_phase4",
                    "--summary_json",
                    str(summary_path),
                    "--out_dir",
                    str(output_path),
                ],
            ):
                self.assertEqual(make_paper_figures_phase4.main(), 1)

            self.assertFalse(output_path.exists())

    def test_nonfinite_measurement_is_rejected_before_serialization(self) -> None:
        summary = _valid_summary()
        summary["all_results"][0]["var_V"] = float("nan")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"
            with self.assertRaisesRegex(
                Phase4SummaryValidationError, "must be finite"
            ):
                write_phase4_summary(summary, output_path)
            self.assertFalse(output_path.exists())

    def test_schema_less_summary_is_historical_not_publishable(self) -> None:
        summary = copy.deepcopy(_valid_summary())
        del summary["schema_version"]

        result = audit_phase4_paper_ready.check_publication_schema(summary)

        self.assertFalse(result.passed)
        self.assertIn("historical", result.message.lower())
        self.assertIn("non-publishable", result.message.lower())

    def test_historical_audit_does_not_overwrite_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "AUDIT.md"
            audit_path.write_text("historical artifact\n")
            results = [
                audit_phase4_paper_ready.AuditResult(
                    "Publication Schema",
                    False,
                    "legacy summary",
                )
            ]

            audit_phase4_paper_ready.write_audit_md(
                Path(temp_dir), results, all_passed=False
            )

            self.assertEqual(audit_path.read_text(), "historical artifact\n")


if __name__ == "__main__":
    unittest.main()
