#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 19-A3: Build fair all-parent and common-parent clean target sets.

Input:
  phase19_all_baseline_targets_v2.csv

Output:
  1. phase19_all_clean_targets_v2.csv
  2. phase19_all_clean_target_summary_v2.csv
  3. phase19_common_parent_set_v2.csv
  4. phase19_common_clean_targets_v2.csv
  5. phase19_common_clean_target_summary_v2.csv
  6. phase19a3_self_check.json

Purpose:
  Make baseline comparison fair by:
  - removing ring-cut targets
  - keeping the same parent set across methods for main comparison
  - keeping all-parent results for supplementary analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "localization/tables"
REPORT_DIR = ROOT / "localization/reports"


METHOD_ORDER = [
    "moltoken_editor",
    "brics_edit",
    "recap_edit",
    "rgroup_enumeration",
    "random_action_matched",
]


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def summarize(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []

    for method, g in df.groupby("baseline_method_group"):
        item = {
            "method_group": method,
            "target_rows": len(g),
            "parent_count": g["mol_id"].nunique(),
            "mean_targets_per_parent": len(g) / max(g["mol_id"].nunique(), 1),
            "mean_atom_count": pd.to_numeric(g["baseline_atom_count"], errors="coerce").mean(),
            "median_atom_count": pd.to_numeric(g["baseline_atom_count"], errors="coerce").median(),
            "mean_attachment_count": pd.to_numeric(g["baseline_attachment_count"], errors="coerce").mean(),
            "median_attachment_count": pd.to_numeric(g["baseline_attachment_count"], errors="coerce").median(),
            "ring_cut_rate": to_bool_series(g["baseline_cuts_ring"]).mean()
            if "baseline_cuts_ring" in g.columns else None,
        }
        rows.append(item)

    summary = pd.DataFrame(rows)

    if len(summary):
        summary["method_order"] = summary["method_group"].map(
            {m: i for i, m in enumerate(METHOD_ORDER)}
        )
        summary = summary.sort_values(["method_order", "method_group"]).drop(columns=["method_order"])

    summary.to_csv(out_path, index=False)
    return summary


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTDIR / "phase19_all_baseline_targets_v2.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path, low_memory=False)

    required = ["mol_id", "baseline_method_group", "baseline_cuts_ring"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # 1. Clean all-parent set: remove ring-cut targets
    clean = df[~to_bool_series(df["baseline_cuts_ring"])].copy()

    # Keep only expected methods, in case there are extras
    clean = clean[clean["baseline_method_group"].isin(METHOD_ORDER)].copy()

    all_clean_path = OUTDIR / "phase19_all_clean_targets_v2.csv"
    clean.to_csv(all_clean_path, index=False)

    all_summary_path = OUTDIR / "phase19_all_clean_target_summary_v2.csv"
    all_summary = summarize(clean, all_summary_path)

    # 2. Common parent set: parents that have at least one clean target for every method
    parent_sets = {}
    for m in METHOD_ORDER:
        parent_sets[m] = set(clean.loc[clean["baseline_method_group"] == m, "mol_id"].astype(str))

    common_parents = set.intersection(*parent_sets.values())

    common_parent_df = pd.DataFrame({"mol_id": sorted(common_parents)})
    common_parent_path = OUTDIR / "phase19_common_parent_set_v2.csv"
    common_parent_df.to_csv(common_parent_path, index=False)

    common = clean[clean["mol_id"].astype(str).isin(common_parents)].copy()

    common_path = OUTDIR / "phase19_common_clean_targets_v2.csv"
    common.to_csv(common_path, index=False)

    common_summary_path = OUTDIR / "phase19_common_clean_target_summary_v2.csv"
    common_summary = summarize(common, common_summary_path)

    # 3. Parent coverage matrix
    coverage_rows = []
    all_parent_union = sorted(set(clean["mol_id"].astype(str)))
    for pid in all_parent_union:
        row = {"mol_id": pid}
        for m in METHOD_ORDER:
            row[m] = pid in parent_sets[m]
        row["in_common_parent_set"] = pid in common_parents
        coverage_rows.append(row)

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = OUTDIR / "phase19_parent_coverage_matrix_v2.csv"
    coverage_df.to_csv(coverage_path, index=False)

    report = {
        "phase": "19A3_common_clean_target_set",
        "status": "pass",
        "input": str(in_path),
        "all_clean_targets": int(len(clean)),
        "all_clean_parent_union": int(len(all_parent_union)),
        "common_parent_count": int(len(common_parents)),
        "common_clean_targets": int(len(common)),
        "method_order": METHOD_ORDER,
        "all_clean_summary": all_summary.to_dict(orient="records"),
        "common_clean_summary": common_summary.to_dict(orient="records"),
        "outputs": {
            "all_clean_targets": str(all_clean_path),
            "all_clean_summary": str(all_summary_path),
            "common_parent_set": str(common_parent_path),
            "common_clean_targets": str(common_path),
            "common_clean_summary": str(common_summary_path),
            "parent_coverage_matrix": str(coverage_path),
        },
    }

    report_path = REPORT_DIR / "phase19a3_self_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== all-clean target summary ===")
    print(all_summary.to_string(index=False))

    print("\n=== common-clean target summary ===")
    print(common_summary.to_string(index=False))

    print("\n[INFO] common_parent_count:", len(common_parents))
    print("[WROTE]", all_clean_path)
    print("[WROTE]", all_summary_path)
    print("[WROTE]", common_parent_path)
    print("[WROTE]", common_path)
    print("[WROTE]", common_summary_path)
    print("[WROTE]", coverage_path)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
