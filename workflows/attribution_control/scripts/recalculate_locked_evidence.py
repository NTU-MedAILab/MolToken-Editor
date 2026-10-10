#!/usr/bin/env python3
"""Recalculate the locked A05 matched-control results from frozen evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
TABLES = RESULTS / "manuscript_tables"
FIGURES = RESULTS / "figure_source_data"
PROVENANCE = RESULTS / "provenance" / "a05"

TOLERANCE = 1e-10


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: object, expected: object, label: str) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise AssertionError(
            f"{label}: expected {expected}, observed {actual}"
        )


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_endpoint(
    frame: pd.DataFrame,
    *terms: str,
) -> pd.Series:
    wanted = tuple(normalized(term) for term in terms)
    matches = frame[
        frame["endpoint"].map(
            lambda value: all(term in normalized(value) for term in wanted)
        )
    ]
    check(
        len(matches) == 1,
        f"Expected one endpoint matching {terms}, observed {len(matches)}",
    )
    return matches.iloc[0]


def verify_endpoint(
    *,
    parent_effects: pd.DataFrame,
    bootstrap: pd.DataFrame,
    summary: pd.DataFrame,
    manuscript: pd.DataFrame,
    endpoint_terms: tuple[str, ...],
    treated_column: str,
    control_column: str,
    effect_column: str,
    bootstrap_column: str,
) -> dict[str, float]:
    summary_row = find_endpoint(summary, *endpoint_terms)
    manuscript_row = find_endpoint(manuscript, *endpoint_terms)

    treated = float(parent_effects[treated_column].mean())
    control = float(parent_effects[control_column].mean())
    effect = float(parent_effects[effect_column].mean())

    close(effect, treated - control, f"{endpoint_terms} paired contrast")
    close(
        summary_row["treated_parent_mean"],
        treated,
        f"{endpoint_terms} summary treated mean",
    )
    close(
        summary_row["control_parent_mean"],
        control,
        f"{endpoint_terms} summary control mean",
    )
    close(
        summary_row["paired_effect_parent_mean"],
        effect,
        f"{endpoint_terms} summary paired effect",
    )

    ci_low = float(bootstrap[bootstrap_column].quantile(0.025))
    ci_high = float(bootstrap[bootstrap_column].quantile(0.975))

    close(
        summary_row["ci_lower_2_5_percent"],
        ci_low,
        f"{endpoint_terms} lower confidence limit",
    )
    close(
        summary_row["ci_upper_97_5_percent"],
        ci_high,
        f"{endpoint_terms} upper confidence limit",
    )

    close(
        manuscript_row["treated"],
        treated,
        f"{endpoint_terms} manuscript treated mean",
    )
    close(
        manuscript_row["control"],
        control,
        f"{endpoint_terms} manuscript control mean",
    )
    close(
        manuscript_row["treated_minus_control"],
        effect,
        f"{endpoint_terms} manuscript effect",
    )
    close(
        manuscript_row["ci95_low"],
        ci_low,
        f"{endpoint_terms} manuscript lower confidence limit",
    )
    close(
        manuscript_row["ci95_high"],
        ci_high,
        f"{endpoint_terms} manuscript upper confidence limit",
    )

    return {
        "treated": treated,
        "control": control,
        "effect": effect,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def main() -> None:
    audit = json.loads(
        (
            PROVENANCE
            / "final"
            / "A05_FINAL_AUDIT.json"
        ).read_text(encoding="utf-8")
    )

    check(audit["status"].lower() == "pass", "A05 audit is not PASS")
    check(audit["A05_status"] == "CLOSED", "A05 is not CLOSED")

    universe = audit["matching_universe"]
    check(universe["parents"] == 90, "Matching universe is not 90 parents")
    check(
        universe["primary_treated_regions"] == 178,
        "Matching universe is not 178 treated regions",
    )
    check(
        universe["strictly_matched_regions"] == 92,
        "Strictly matched region count is not 92",
    )
    check(
        universe["matched_parents"] == 65,
        "Matched parent count is not 65",
    )
    check(universe["matching_seeds"] == 5, "Matching seed count is not 5")
    check(
        universe["matched_pair_rows"] == 460,
        "Matched-pair row count is not 460",
    )

    roster = pd.read_csv(
        FIGURES / "attribution_matched_control_pair_roster_460.csv"
    )
    check(len(roster) == 460, "Frozen pair roster does not contain 460 rows")
    check(roster["seed"].nunique() == 5, "Pair roster does not contain 5 seeds")
    check(
        set(roster.groupby("seed").size()) == {92},
        "Each matching seed must contain 92 pairs",
    )

    step10 = pd.read_csv(
        PROVENANCE
        / "step10"
        / "A05_step10_parent_seed_method_B5_metrics_650.csv"
    )
    check(len(step10) == 650, "STEP10 metric table does not contain 650 rows")
    check(
        set(step10["fixed_slot_denominator"]) == {5},
        "STEP10 denominator is not fixed at B=5",
    )
    check(
        step10["seed"].nunique() == 5,
        "STEP10 table does not contain 5 seeds",
    )
    check(
        step10["mol_id"].nunique() == 65,
        "STEP10 table does not contain 65 parents",
    )
    check(
        step10["method_label"].nunique() == 2,
        "STEP10 table does not contain two paired methods",
    )

    manuscript = pd.read_csv(
        TABLES / "attribution_matched_control.csv"
    )
    check(len(manuscript) == 4, "A05 manuscript table must contain four rows")

    primary_parent = pd.read_csv(
        FIGURES
        / "attribution_matched_control_primary_parent_effects_65.csv"
    )
    primary_seed = pd.read_csv(
        FIGURES
        / "attribution_matched_control_primary_parent_seed_325.csv"
    )
    primary_bootstrap = pd.read_csv(
        FIGURES
        / "attribution_matched_control_primary_bootstrap_5000.csv"
    )
    primary_summary = pd.read_csv(
        FIGURES
        / "attribution_matched_control_primary_summary.csv"
    )

    check(len(primary_parent) == 65, "Primary analysis is not 65 parents")
    check(
        set(primary_parent["seed_count"]) == {5},
        "Primary parents do not all retain five seeds",
    )
    check(len(primary_seed) == 325, "Primary paired table is not 325 rows")
    check(
        len(primary_bootstrap) == 5000,
        "Primary bootstrap does not contain 5000 replicates",
    )

    target_window = verify_endpoint(
        parent_effects=primary_parent,
        bootstrap=primary_bootstrap,
        summary=primary_summary,
        manuscript=manuscript,
        endpoint_terms=("target", "window"),
        treated_column=(
            "target_window_hit_rate_B5_treated_five_seed_mean"
        ),
        control_column=(
            "target_window_hit_rate_B5_control_five_seed_mean"
        ),
        effect_column=(
            "target_window_hit_rate_B5_paired_effect_five_seed_mean"
        ),
        bootstrap_column="target_window_hit_rate_B5_paired_effect",
    )

    strict_success = verify_endpoint(
        parent_effects=primary_parent,
        bootstrap=primary_bootstrap,
        summary=primary_summary,
        manuscript=manuscript,
        endpoint_terms=("strict", "success"),
        treated_column=(
            "strict_success_rate_B5_treated_five_seed_mean"
        ),
        control_column=(
            "strict_success_rate_B5_control_five_seed_mean"
        ),
        effect_column=(
            "strict_success_rate_B5_paired_effect_five_seed_mean"
        ),
        bootstrap_column="strict_success_rate_B5_paired_effect",
    )

    sensitivity_parent = pd.read_csv(
        FIGURES
        / "attribution_matched_control_sensitivity_parent_effects_58.csv"
    )
    sensitivity_seed = pd.read_csv(
        FIGURES
        / "attribution_matched_control_sensitivity_parent_seed_290.csv"
    )
    sensitivity_bootstrap = pd.read_csv(
        FIGURES
        / "attribution_matched_control_sensitivity_bootstrap_5000.csv"
    )
    sensitivity_summary = pd.read_csv(
        FIGURES
        / "attribution_matched_control_sensitivity_summary.csv"
    )
    complete_roster = pd.read_csv(
        FIGURES
        / "attribution_matched_control_complete_parent_roster_58.csv"
    )
    incomplete_roster = pd.read_csv(
        FIGURES
        / "attribution_matched_control_incomplete_parent_roster_7.csv"
    )

    check(
        len(sensitivity_parent) == 58,
        "Sensitivity analysis is not 58 parents",
    )
    check(
        set(sensitivity_parent["seed_count"]) == {5},
        "Sensitivity parents do not all retain five seeds",
    )
    check(
        len(sensitivity_seed) == 290,
        "Sensitivity paired table is not 290 rows",
    )
    check(
        len(sensitivity_bootstrap) == 5000,
        "Sensitivity bootstrap does not contain 5000 replicates",
    )
    check(len(complete_roster) == 58, "Complete roster is not 58 parents")
    check(len(incomplete_roster) == 7, "Incomplete roster is not 7 parents")

    target_distance = verify_endpoint(
        parent_effects=sensitivity_parent,
        bootstrap=sensitivity_bootstrap,
        summary=sensitivity_summary,
        manuscript=manuscript,
        endpoint_terms=("target", "distance"),
        treated_column=(
            "mean_target_distance_improvement_observed_treated_five_seed_mean"
        ),
        control_column=(
            "mean_target_distance_improvement_observed_control_five_seed_mean"
        ),
        effect_column=(
            "mean_target_distance_improvement_observed_paired_effect_five_seed_mean"
        ),
        bootstrap_column=(
            "mean_target_distance_improvement_observed_paired_effect"
        ),
    )

    final_score = verify_endpoint(
        parent_effects=sensitivity_parent,
        bootstrap=sensitivity_bootstrap,
        summary=sensitivity_summary,
        manuscript=manuscript,
        endpoint_terms=("final", "edit", "score"),
        treated_column=(
            "mean_final_edit_score_observed_treated_five_seed_mean"
        ),
        control_column=(
            "mean_final_edit_score_observed_control_five_seed_mean"
        ),
        effect_column=(
            "mean_final_edit_score_observed_paired_effect_five_seed_mean"
        ),
        bootstrap_column=(
            "mean_final_edit_score_observed_paired_effect"
        ),
    )

    inference = audit["inference"]
    check(
        inference["statistical_unit"] == "parent",
        "Inferential unit is not parent",
    )
    check(
        inference["bootstrap_replicates"] == 5000,
        "Audit bootstrap count is not 5000",
    )
    check(
        inference["bootstrap_seed"] == 20260703,
        "Audit bootstrap seed differs from the lock",
    )
    check(
        inference["p_values_computed"] is False,
        "Unexpected p-value claim",
    )

    limits = audit["interpretation_limits"]
    check(
        limits["causal_claim_supported"] is False,
        "Unexpected causal claim",
    )
    check(
        limits["wet_lab_validation"] is False,
        "Unexpected wet-lab claim",
    )

    result = {
        "status": "PASS",
        "mode": "locked-evidence recalculation",
        "matching": {
            "parents": 90,
            "treated_regions": 178,
            "strictly_matched_regions": 92,
            "matched_parents": 65,
            "matching_seeds": 5,
            "matched_pair_rows": 460,
        },
        "primary_analysis": {
            "parents": 65,
            "parent_seed_pairs": 325,
            "target_window": target_window,
            "strict_success": strict_success,
        },
        "complete_observation_sensitivity": {
            "parents": 58,
            "parent_seed_pairs": 290,
            "target_distance": target_distance,
            "final_edit_score": final_score,
        },
        "bootstrap": {
            "replicates": 5000,
            "seed_recorded_in_audit": 20260703,
            "confidence_limits_recalculated_from_frozen_distributions": True,
        },
        "scope": (
            "This verifies frozen matched-control evidence. "
            "It does not rerun matching, candidate generation, "
            "model inference, property scoring, or B=5 selection."
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
