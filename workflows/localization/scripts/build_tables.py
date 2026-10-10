#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE19 = ROOT / "localization"
TABLES = PHASE19 / "tables"
REPORTS = PHASE19 / "reports"

METHOD_ORDER = [
    "moltoken_editor",
    "brics_edit",
    "recap_edit",
    "rgroup_enumeration",
    "random_action_matched",
]

METHOD_LABELS = {
    "moltoken_editor": "MolToken-Editor",
    "brics_edit": "BRICS",
    "recap_edit": "RECAP",
    "rgroup_enumeration": "Bemis–Murcko side-chain",
    "random_action_matched": "Action-matched random",
}

METRIC_LABELS = {
    "main_delta_distance_edit": "Mean Δdistance",
    "target_window_success": "Target-window success",
    "strict_success": "Strict success",
    "final_edit_score": "Final edit score",
}

RATE_METRICS = {
    "target_window_success",
    "strict_success",
    "consensus_success",
    "relaxed_guardrail_pass",
}


def pct(x: float, digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "NA"


def num(x: float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"


def p_value(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "NA"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def ci_cell(metric: str, mean: float, low: float, high: float) -> str:
    if metric in RATE_METRICS:
        return f"{pct(mean)} ({pct(low)}–{pct(high)})"
    return f"{num(mean)} ({num(low)}–{num(high)})"


def diff_cell(metric: str, diff: float, low: float, high: float, p: float) -> str:
    if metric in RATE_METRICS:
        return f"{diff * 100:+.1f} pp ({low * 100:+.1f} to {high * 100:+.1f}); p={p_value(p)}"
    return f"{diff:+.3f} ({low:+.3f} to {high:+.3f}); p={p_value(p)}"


def safe_int(x) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return "NA"


def write_markdown(df: pd.DataFrame, path: Path, title: str, note: str) -> None:
    lines = [f"# {title}", "", note, ""]
    try:
        lines.append(df.to_markdown(index=False))
    except Exception:
        lines.append(df.to_csv(index=False))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_method_table(c_table: pd.DataFrame, d_method: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in METHOD_ORDER:
        c = c_table[c_table["method_group"] == method]
        if c.empty:
            continue
        c = c.iloc[0]

        d = d_method[d_method["method_group"] == method].copy()

        def get_metric(metric: str) -> str:
            g = d[d["metric"] == metric]
            if g.empty:
                return "NA"
            g = g.iloc[0]
            return ci_cell(metric, g["mean"], g["ci95_low"], g["ci95_high"])

        row = {
            "Method": METHOD_LABELS.get(method, method),
            "Candidate attempts": safe_int(c.get("candidate_attempts")),
            "Valid edits": safe_int(c.get("valid_edits")),
            "Scored edits": safe_int(c.get("scored_edits")),
            "Scored parents": safe_int(c.get("covered_parents_scored")),
            "Parent-level n": safe_int(d["n"].max()) if not d.empty else "NA",
            "Mean Δdistance (95% CI)": get_metric("main_delta_distance_edit"),
            "Target-window success (95% CI)": get_metric("target_window_success"),
            "Strict success (95% CI)": get_metric("strict_success"),
            "Final edit score (95% CI)": get_metric("final_edit_score"),
            "Consensus success": pct(c.get("top5_consensus_rate")),
            "Guardrail pass": pct(c.get("top5_guardrail_pass_rate")),
            "OOD risk": pct(c.get("top5_ood_rate")),
            "Parent Tanimoto": num(c.get("top5_mean_parent_tanimoto")),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_paired_table(d_paired: pd.DataFrame) -> pd.DataFrame:
    baseline_order = [
        "brics_edit",
        "recap_edit",
        "rgroup_enumeration",
        "random_action_matched",
    ]

    rows = []

    for baseline in baseline_order:
        g = d_paired[d_paired["baseline"] == baseline].copy()
        if g.empty:
            continue

        def metric_cell(metric: str) -> str:
            r = g[g["metric"] == metric]
            if r.empty:
                return "NA"
            r = r.iloc[0]
            return diff_cell(
                metric,
                r["mean_diff"],
                r["ci95_low"],
                r["ci95_high"],
                r["p_bootstrap_two_sided"],
            )

        n = g["paired_parent_count"].max()

        row = {
            "Baseline": METHOD_LABELS.get(baseline, baseline),
            "Paired parents": safe_int(n),
            "Δdistance advantage": metric_cell("main_delta_distance_edit"),
            "Target-window advantage": metric_cell("target_window_success"),
            "Strict-success advantage": metric_cell("strict_success"),
            "Final-score advantage": metric_cell("final_edit_score"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase19c-method",
        default=str(TABLES / "phase19c_common_budget2_safe_method_group_comparison.csv"),
    )
    parser.add_argument(
        "--phase19d-method-ci",
        default=str(TABLES / "phase19d_common_budget2_safe_method_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--phase19d-paired-ci",
        default=str(TABLES / "phase19d_common_budget2_safe_paired_bootstrap_ci.csv"),
    )
    parser.add_argument("--comparison-name", default="common_budget2_safe")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    c_path = Path(args.phase19c_method)
    d_method_path = Path(args.phase19d_method_ci)
    d_paired_path = Path(args.phase19d_paired_ci)

    for p in [c_path, d_method_path, d_paired_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input table: {p}")

    c_table = pd.read_csv(c_path, low_memory=False)
    d_method = pd.read_csv(d_method_path, low_memory=False)
    d_paired = pd.read_csv(d_paired_path, low_memory=False)

    method_table = build_method_table(c_table, d_method)
    paired_table = build_paired_table(d_paired)

    method_csv = TABLES / f"phase19e_{args.comparison_name}_table_method_level_baseline_manuscript.csv"
    method_md = TABLES / f"phase19e_{args.comparison_name}_table_method_level_baseline_manuscript.md"

    paired_csv = TABLES / f"phase19e_{args.comparison_name}_table_paired_moltoken_vs_baselines_manuscript.csv"
    paired_md = TABLES / f"phase19e_{args.comparison_name}_table_paired_moltoken_vs_baselines_manuscript.md"

    method_table.to_csv(method_csv, index=False)
    paired_table.to_csv(paired_csv, index=False)

    method_note = (
        "Values are parent-level Top-5 means. Confidence intervals are bootstrap 95% CIs. "
        "Candidate attempts, valid edits, scored edits, consensus success, guardrail pass, "
        "OOD risk, and parent Tanimoto are taken from Phase 19-C scoring. "
        "Bootstrap CIs are taken from Phase 19-D."
    )

    paired_note = (
        "Values report MolToken-Editor minus baseline. Positive values indicate an advantage "
        "of MolToken-Editor. Percentage-point differences are shown for rate metrics. "
        "p-values are two-sided bootstrap p-values."
    )

    write_markdown(
        method_table,
        method_md,
        "Table Sx. Method-level baseline comparison with bootstrap 95% CI",
        method_note,
    )

    write_markdown(
        paired_table,
        paired_md,
        "Table Sy. Paired bootstrap comparison of MolToken-Editor against baselines",
        paired_note,
    )

    report = {
        "phase": "19E_manuscript_ready_baseline_tables",
        "status": "pass",
        "comparison_name": args.comparison_name,
        "inputs": {
            "phase19c_method_group_comparison": str(c_path),
            "phase19d_method_bootstrap_ci": str(d_method_path),
            "phase19d_paired_bootstrap_ci": str(d_paired_path),
        },
        "outputs": {
            "method_table_csv": str(method_csv),
            "method_table_md": str(method_md),
            "paired_table_csv": str(paired_csv),
            "paired_table_md": str(paired_md),
        },
        "method_table_rows": int(len(method_table)),
        "paired_table_rows": int(len(paired_table)),
    }

    report_path = REPORTS / f"phase19e_{args.comparison_name}_self_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Manuscript-ready Table Sx ===")
    print(method_table.to_string(index=False))

    print("\n=== Manuscript-ready Table Sy ===")
    print(paired_table.to_string(index=False))

    print("\n[WROTE]", method_csv)
    print("[WROTE]", method_md)
    print("[WROTE]", paired_csv)
    print("[WROTE]", paired_md)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
