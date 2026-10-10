#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    "random_action_matched": "Random",
}

METRICS = {
    "main_delta_distance_edit": "Mean Δdistance",
    "target_window_success": "Target-window success",
    "strict_success": "Strict success",
    "final_edit_score": "Final edit score",
}


def bool_value(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "pass"}


def metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric in {"target_window_success", "strict_success", "consensus_success", "relaxed_guardrail_pass"}:
        return df[metric].map(bool_value).astype(float)
    return pd.to_numeric(df[metric], errors="coerce")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def bootstrap_diff_ci(diff: np.ndarray, n_boot: int, seed: int) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]

    if len(diff) == 0:
        return {
            "paired_parent_count": 0,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "win_rate": np.nan,
            "p_bootstrap_two_sided": np.nan,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)

    p_left = float((boot <= 0).mean())
    p_right = float((boot >= 0).mean())
    p_two = min(1.0, 2.0 * min(p_left, p_right))

    return {
        "paired_parent_count": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "win_rate": float((diff > 0).mean()),
        "p_bootstrap_two_sided": float(p_two),
    }


def make_parent_level_table(top: pd.DataFrame) -> pd.DataFrame:
    work = top.copy()

    for metric in METRICS:
        work[metric] = metric_series(work, metric)

    parent_level = (
        work.groupby(["mol_id", "method_group"], as_index=False)
        .agg({metric: "mean" for metric in METRICS})
    )

    return parent_level


def method_bootstrap(parent_level: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []

    for method in METHOD_ORDER:
        g = parent_level[parent_level["method_group"] == method].copy()

        for i, metric in enumerate(METRICS):
            stats = bootstrap_mean_ci(g[metric].to_numpy(dtype=float), n_boot=n_boot, seed=seed + i)
            row = {
                "method_group": method,
                "method_label": METHOD_LABELS.get(method, method),
                "metric": metric,
                "metric_label": METRICS[metric],
                **stats,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def paired_bootstrap(parent_level: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    ours = parent_level[parent_level["method_group"] == "moltoken_editor"].copy()
    rows = []

    for baseline in [m for m in METHOD_ORDER if m != "moltoken_editor"]:
        base = parent_level[parent_level["method_group"] == baseline].copy()
        merged = ours.merge(base, on="mol_id", suffixes=("_moltoken", "_baseline"))

        for i, metric in enumerate(METRICS):
            moltoken_col = f"{metric}_moltoken"
            baseline_col = f"{metric}_baseline"
            diff = merged[moltoken_col].to_numpy(dtype=float) - merged[baseline_col].to_numpy(dtype=float)

            stats = bootstrap_diff_ci(diff, n_boot=n_boot, seed=seed + 100 + i)
            row = {
                "baseline": baseline,
                "baseline_label": METHOD_LABELS.get(baseline, baseline),
                "metric": metric,
                "metric_label": METRICS[metric],
                "moltoken_mean": float(np.nanmean(merged[moltoken_col].to_numpy(dtype=float))) if len(merged) else np.nan,
                "baseline_mean": float(np.nanmean(merged[baseline_col].to_numpy(dtype=float))) if len(merged) else np.nan,
                **stats,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-by-parent",
        default=str(TABLES / "phase19c_common_budget2_safe_top_by_parent.csv"),
    )
    parser.add_argument("--comparison-name", default="common_budget2_safe")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    top_path = Path(args.top_by_parent)
    if not top_path.exists():
        raise FileNotFoundError(f"Missing top-by-parent table: {top_path}")

    top = pd.read_csv(top_path, low_memory=False)

    parent_level = make_parent_level_table(top)
    method_ci = method_bootstrap(parent_level, n_boot=args.n_bootstrap, seed=args.seed)
    paired_ci = paired_bootstrap(parent_level, n_boot=args.n_bootstrap, seed=args.seed)

    parent_path = TABLES / f"phase19d_{args.comparison_name}_parent_level_metrics.csv"
    method_path = TABLES / f"phase19d_{args.comparison_name}_method_bootstrap_ci.csv"
    paired_path = TABLES / f"phase19d_{args.comparison_name}_paired_bootstrap_ci.csv"

    parent_level.to_csv(parent_path, index=False)
    method_ci.to_csv(method_path, index=False)
    paired_ci.to_csv(paired_path, index=False)

    report = {
        "phase": "19D_bootstrap_confidence_intervals",
        "status": "pass",
        "comparison_name": args.comparison_name,
        "n_bootstrap": int(args.n_bootstrap),
        "top_by_parent_rows": int(len(top)),
        "parent_level_rows": int(len(parent_level)),
        "method_count": int(parent_level["method_group"].nunique()),
        "parent_count_total": int(parent_level["mol_id"].nunique()),
        "outputs": {
            "parent_level_metrics": str(parent_path),
            "method_bootstrap_ci": str(method_path),
            "paired_bootstrap_ci": str(paired_path),
        },
    }

    report_path = REPORTS / f"phase19d_{args.comparison_name}_self_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Method bootstrap CI ===")
    print(method_ci.to_string(index=False))

    print("\n=== Paired bootstrap CI ===")
    print(paired_ci.to_string(index=False))

    print("\n[WROTE]", parent_path)
    print("[WROTE]", method_path)
    print("[WROTE]", paired_path)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
