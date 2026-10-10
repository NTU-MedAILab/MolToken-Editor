#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE10D_SCRIPT = ROOT / "moltoken/scripts/generate_edits.py"

PHASE19 = ROOT / "localization"
TABLES = PHASE19 / "tables"
REPORTS = PHASE19 / "reports"
PREDICTIONS = PHASE19 / "predictions"

METHOD_ORDER = [
    "moltoken_editor",
    "brics_edit",
    "recap_edit",
    "rgroup_enumeration",
    "random_action_matched",
]


def bool_value(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "pass"}


def import_phase10d():
    if not PHASE10D_SCRIPT.exists():
        raise FileNotFoundError(f"Missing Phase10-D script: {PHASE10D_SCRIPT}")

    sys.path.insert(0, str(PHASE10D_SCRIPT.parent))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path.cwd()))

    spec = importlib.util.spec_from_file_location("phase10d_original_for_phase19c", PHASE10D_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    # Redirect Phase10-D outputs into Phase19 folders.
    mod.TABLES = TABLES
    mod.REPORTS = REPORTS
    mod.PREDICTIONS = PREDICTIONS

    return mod


def top_by_group_parent(scored: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    valid = scored[
        scored["edit_valid"].map(bool_value)
        & scored["edited_main_pred"].notna()
    ].copy()

    if valid.empty:
        return valid

    if "evidence_ready" in valid.columns:
        ready = valid[valid["evidence_ready"].map(bool_value)].copy()
        if not ready.empty:
            valid = ready

    valid = valid.sort_values(
        [
            "method_group",
            "mol_id",
            "target_window_success",
            "relaxed_guardrail_pass",
            "auxiliary_opposition_count",
            "final_edit_score",
        ],
        ascending=[True, True, False, False, True, False],
    )

    valid["rank_in_group_parent"] = (
        valid.groupby(["method_group", "mol_id"], dropna=False).cumcount() + 1
    )

    return valid[valid["rank_in_group_parent"] <= k].reset_index(drop=True)


def summary_row(name: str, df: pd.DataFrame, top_df: pd.DataFrame) -> dict[str, Any]:
    valid = df[df["edit_valid"].map(bool_value)].copy() if not df.empty else pd.DataFrame()
    scored = valid[valid["edited_main_pred"].notna()].copy() if not valid.empty and "edited_main_pred" in valid.columns else pd.DataFrame()

    return {
        "method_group": name,
        "candidate_attempts": int(len(df)),
        "valid_edits": int(len(valid)),
        "scored_edits": int(len(scored)),
        "covered_parents_scored": int(scored["mol_id"].nunique()) if not scored.empty else 0,
        "top5_rows": int(len(top_df)),
        "top5_parent_count": int(top_df["mol_id"].nunique()) if not top_df.empty else 0,
        "top5_mean_delta": float(top_df["main_delta_distance_edit"].mean()) if not top_df.empty else 0.0,
        "top5_median_delta": float(top_df["main_delta_distance_edit"].median()) if not top_df.empty else 0.0,
        "top5_target_window_rate": float(top_df["target_window_success"].mean()) if not top_df.empty else 0.0,
        "top5_guardrail_pass_rate": float(top_df["relaxed_guardrail_pass"].mean()) if not top_df.empty else 0.0,
        "top5_consensus_rate": float(top_df["consensus_success"].mean()) if not top_df.empty else 0.0,
        "top5_strict_rate": float(top_df["strict_success"].mean()) if not top_df.empty else 0.0,
        "top5_ood_rate": float(top_df["ood_risk"].mean()) if not top_df.empty else 0.0,
        "top5_mean_parent_tanimoto": float(top_df["parent_tanimoto"].mean()) if not top_df.empty else 0.0,
        "top5_mean_delta_qed": float(top_df["delta_qed"].mean()) if not top_df.empty else 0.0,
        "top5_mean_delta_sa": float(top_df["delta_sa_score"].mean()) if not top_df.empty else 0.0,
    }


def summarize_method_groups(scored: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHOD_ORDER:
        rows.append(
            summary_row(
                m,
                scored[scored["method_group"] == m],
                top[top["method_group"] == m],
            )
        )
    return pd.DataFrame(rows)


def paired_moltoken_vs_baselines(top: pd.DataFrame) -> pd.DataFrame:
    if top.empty:
        return pd.DataFrame()

    metrics = [
        "main_delta_distance_edit",
        "target_window_success",
        "strict_success",
        "consensus_success",
        "relaxed_guardrail_pass",
        "final_edit_score",
    ]

    parent_group = (
        top.groupby(["mol_id", "method_group"])[metrics]
        .mean()
        .reset_index()
    )

    ours = parent_group[parent_group["method_group"] == "moltoken_editor"].copy()

    rows = []
    for baseline in [m for m in METHOD_ORDER if m != "moltoken_editor"]:
        base = parent_group[parent_group["method_group"] == baseline].copy()
        merged = ours.merge(base, on="mol_id", suffixes=("_moltoken", "_baseline"))

        if merged.empty:
            continue

        row = {
            "baseline": baseline,
            "paired_parent_count": int(len(merged)),
        }

        for metric in metrics:
            diff = merged[f"{metric}_moltoken"] - merged[f"{metric}_baseline"]
            row[f"{metric}_moltoken_mean"] = float(merged[f"{metric}_moltoken"].mean())
            row[f"{metric}_baseline_mean"] = float(merged[f"{metric}_baseline"].mean())
            row[f"{metric}_mean_diff"] = float(diff.mean())
            row[f"{metric}_median_diff"] = float(diff.median())
            row[f"{metric}_moltoken_win_rate"] = float((diff > 0).mean())

        rows.append(row)

    return pd.DataFrame(rows)


def action_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()

    rows = []
    for (group, action), g in scored.groupby(["method_group", "edit_action"], dropna=False):
        valid = g[g["edit_valid"].map(bool_value)].copy()
        scored_valid = valid[valid["edited_main_pred"].notna()].copy() if "edited_main_pred" in valid.columns else pd.DataFrame()

        rows.append({
            "method_group": group,
            "edit_action": action,
            "candidate_attempts": int(len(g)),
            "valid_edits": int(len(valid)),
            "scored_edits": int(len(scored_valid)),
            "covered_parents_scored": int(scored_valid["mol_id"].nunique()) if not scored_valid.empty else 0,
            "mean_delta": float(scored_valid["main_delta_distance_edit"].mean()) if not scored_valid.empty else 0.0,
            "target_window_rate": float(scored_valid["target_window_success"].mean()) if not scored_valid.empty else 0.0,
            "strict_rate": float(scored_valid["strict_success"].mean()) if not scored_valid.empty else 0.0,
            "guardrail_pass_rate": float(scored_valid["relaxed_guardrail_pass"].mean()) if not scored_valid.empty else 0.0,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(TABLES / "phase19b_common_budget2_candidates_all.csv"))
    parser.add_argument("--comparison-name", default="common_budget2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_path}")

    candidates = pd.read_csv(candidates_path, low_memory=False)
    phase10d = import_phase10d()

    print("[INFO] candidates:", len(candidates))
    print(candidates["method_group"].value_counts())
    print(pd.crosstab(candidates["method_group"], candidates["edit_action"]))

    scored = phase10d.score_candidates(
        candidates,
        force_predict=args.force_predict,
        skip_predict=args.skip_predict,
    )

    scored_path = TABLES / f"phase19c_{args.comparison_name}_scored_all.csv"
    scored.to_csv(scored_path, index=False)

    top = top_by_group_parent(scored, k=args.top_k)
    top_path = TABLES / f"phase19c_{args.comparison_name}_top_by_parent.csv"
    top.to_csv(top_path, index=False)

    comparison = summarize_method_groups(scored, top)
    comparison_path = TABLES / f"phase19c_{args.comparison_name}_method_group_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    paired = paired_moltoken_vs_baselines(top)
    paired_path = TABLES / f"phase19c_{args.comparison_name}_moltoken_vs_baseline_paired.csv"
    paired.to_csv(paired_path, index=False)

    action = action_summary(scored)
    action_path = TABLES / f"phase19c_{args.comparison_name}_action_summary.csv"
    action.to_csv(action_path, index=False)

    report = {
        "phase": "19C_baseline_candidate_scoring",
        "status": "pass" if len(scored) else "failed",
        "comparison_name": args.comparison_name,
        "candidate_attempts": int(len(candidates)),
        "valid_edits": int(scored["edit_valid"].map(bool_value).sum()) if not scored.empty else 0,
        "scored_edits": int(scored["edited_main_pred"].notna().sum()) if "edited_main_pred" in scored.columns else 0,
        "top_rows": int(len(top)),
        "method_group_comparison": comparison.to_dict(orient="records"),
        "paired_moltoken_vs_baseline": paired.to_dict(orient="records") if len(paired) else [],
        "outputs": {
            "scored_all": str(scored_path),
            "top_by_parent": str(top_path),
            "method_group_comparison": str(comparison_path),
            "paired": str(paired_path),
            "action_summary": str(action_path),
        },
    }

    report_path = REPORTS / f"phase19c_{args.comparison_name}_self_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Method group comparison ===")
    print(comparison.to_string(index=False))

    print("\n=== Paired MolToken vs baseline ===")
    print(paired.to_string(index=False) if len(paired) else "EMPTY")

    print("\n[WROTE]", scored_path)
    print("[WROTE]", top_path)
    print("[WROTE]", comparison_path)
    print("[WROTE]", paired_path)
    print("[WROTE]", action_path)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
