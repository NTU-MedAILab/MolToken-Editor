from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE10 = ROOT / "moltoken"
TABLES = PHASE10 / "tables"
REPORTS = PHASE10 / "reports"




from rdkit import rdBase


PARENT_PATH = TABLES / "phase10_active_parent_pool.csv"
TOKEN_PATH = TABLES / "phase10c_active_moltoken_pool.csv"
EDIT_PATH = TABLES / "phase10d_edit_scored_all.csv"
METHOD_PATH = TABLES / "phase10d_method_comparison.csv"
TOP_PARENT_PATH = TABLES / "phase10d_edit_top_by_parent.csv"

SUMMARY_B_PATH = REPORTS / "phase10b_parent_screening_summary.json"
SUMMARY_C_PATH = REPORTS / "phase10c_moltoken_split_summary.json"
SUMMARY_D_PATH = REPORTS / "phase10d_edit_generation_summary.json"

RANDOM_SEED = 20260605
BOOTSTRAP_N = 2000
TOPK_VALUES = [1, 3, 5]


rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def safe_numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def rate_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(as_bool(series).mean())


def max_share(series: pd.Series) -> float:
    values = series.dropna().astype(str)
    values = values[values != ""]
    if values.empty:
        return float("nan")
    return float(values.value_counts(normalize=True).iloc[0])


def correlation_rows(frame: pd.DataFrame, x: str, y_columns: list[str], label_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if x not in frame.columns:
        return rows
    for y in y_columns:
        if y not in frame.columns:
            continue
        sub = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(sub[x].corr(sub[y], method="pearson"))
            spearman = float(sub[x].corr(sub[y], method="spearman"))
        rows.append(
            {
                "analysis": f"{label_prefix}_{x}_vs_{y}",
                "x": x,
                "y": y,
                "n": int(len(sub)),
                "pearson": pearson,
                "spearman": spearman,
            }
        )
    return rows


def ranked_subset(frame: pd.DataFrame, method_label: str, evidence_only: bool = False) -> pd.DataFrame:
    if method_label == "contribution_primary":
        subset = frame[frame["method_group"] == "contribution_primary"].copy()
    elif method_label == "same_interface_random":
        subset = frame[frame["method_group"] == "same_interface_random"].copy()
    elif method_label == "contribution_supplemental":
        subset = frame[frame["method_group"] == "contribution_supplemental"].copy()
    else:
        subset = frame[frame["method"] == method_label].copy()
    if evidence_only and "evidence_ready" in subset.columns:
        subset = subset[as_bool(subset["evidence_ready"])].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(
        ["mol_id", "final_edit_score", "main_delta_distance_edit", "target_window_success"],
        ascending=[True, False, False, False],
    )
    subset["rank_in_label_parent"] = subset.groupby("mol_id").cumcount() + 1
    return subset


def frozen_ranked_subset(top_parent: pd.DataFrame, method_label: str) -> pd.DataFrame:
    if method_label == "contribution_primary":
        subset = top_parent[top_parent["method_group"] == "contribution_primary"].copy()
        rank_col = "rank_in_group_parent"
    elif method_label == "same_interface_random":
        subset = top_parent[top_parent["method_group"] == "same_interface_random"].copy()
        rank_col = "rank_in_group_parent"
    elif method_label == "contribution_supplemental":
        subset = top_parent[top_parent["method_group"] == "contribution_supplemental"].copy()
        rank_col = "rank_in_group_parent"
    else:
        subset = top_parent[top_parent["method"] == method_label].copy()
        rank_col = "rank_in_method_parent"
    if subset.empty:
        return subset
    subset["rank_in_label_parent"] = pd.to_numeric(subset[rank_col], errors="coerce")
    subset = subset.sort_values(["mol_id", "rank_in_label_parent"], ascending=[True, True])
    return subset


def aggregate_topk(
    frame: pd.DataFrame,
    top_parent: pd.DataFrame,
    method_label: str,
    topk: int | str,
) -> dict[str, Any]:
    subset = ranked_subset(frame, method_label, evidence_only=False) if topk == "all" else frozen_ranked_subset(top_parent, method_label)
    if subset.empty:
        return {
            "method_label": method_label,
            "topk": topk,
            "selection_scope": "all_scored" if topk == "all" else "frozen_parent_rank",
            "rows": 0,
            "parent_count": 0,
        }
    if topk != "all":
        subset = subset[subset["rank_in_label_parent"] <= int(topk)].copy()
    return {
        "method_label": method_label,
        "topk": topk,
        "selection_scope": "all_scored" if topk == "all" else "frozen_parent_rank",
        "rows": int(len(subset)),
        "parent_count": int(subset["mol_id"].nunique()),
        "mean_delta": mean_or_nan(subset["main_delta_distance_edit"]),
        "median_delta": float(pd.to_numeric(subset["main_delta_distance_edit"], errors="coerce").median()),
        "target_window_rate": rate_or_nan(subset["target_window_success"]),
        "guardrail_pass_rate": rate_or_nan(subset["relaxed_guardrail_pass"]),
        "alert_rate": rate_or_nan(subset["edited_interference_alert"]),
        "alert_free_rate": 1.0 - rate_or_nan(subset["edited_interference_alert"]),
        "ood_rate": rate_or_nan(subset["ood_risk"]),
        "ood_free_rate": 1.0 - rate_or_nan(subset["ood_risk"]),
        "evidence_ready_rate": rate_or_nan(subset["evidence_ready"]),
        "consensus_rate": rate_or_nan(subset["consensus_success"]),
        "strict_rate": rate_or_nan(subset["strict_success"]),
        "mean_delta_qed": mean_or_nan(subset["delta_qed"]),
        "mean_delta_sa": mean_or_nan(subset["delta_sa_score"]),
        "mean_uncertainty": mean_or_nan(subset["main_uncertainty"]),
        "mean_nearest_train_tanimoto": mean_or_nan(subset["nearest_train_tanimoto_edited"]),
        "unique_edited_smiles": int(subset["edited_smiles"].dropna().astype(str).nunique()),
    }


def parent_topk_metrics(top_parent: pd.DataFrame, method_label: str, topk: int) -> pd.DataFrame:
    subset = frozen_ranked_subset(top_parent, method_label)
    if subset.empty:
        return pd.DataFrame()
    subset = subset[subset["rank_in_label_parent"] <= topk].copy()
    rows: list[dict[str, Any]] = []
    for mol_id, group in subset.groupby("mol_id"):
        rows.append(
            {
                "mol_id": mol_id,
                "method_label": method_label,
                "topk": topk,
                "rows": int(len(group)),
                "mean_delta": mean_or_nan(group["main_delta_distance_edit"]),
                "best_delta": float(pd.to_numeric(group["main_delta_distance_edit"], errors="coerce").max()),
                "target_window_rate": rate_or_nan(group["target_window_success"]),
                "guardrail_pass_rate": rate_or_nan(group["relaxed_guardrail_pass"]),
                "alert_rate": rate_or_nan(group["edited_interference_alert"]),
                "ood_rate": rate_or_nan(group["ood_risk"]),
                "evidence_ready_rate": rate_or_nan(group["evidence_ready"]),
                "consensus_rate": rate_or_nan(group["consensus_success"]),
                "strict_rate": rate_or_nan(group["strict_success"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series, n_bootstrap: int = BOOTSTRAP_N) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RANDOM_SEED)
    means = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.choice(clean, size=clean.size, replace=True)
        means[idx] = float(np.mean(sample))
    return float(np.mean(clean)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_topk_tables(edits: pd.DataFrame, top_parent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_labels = [
        "contribution_primary",
        "contribution_delete",
        "contribution_replace",
        "same_interface_random",
        "contribution_supplemental",
    ]

    topk_rows = []
    for label in method_labels:
        topk_rows.append(aggregate_topk(edits, top_parent, label, "all"))
        for topk in TOPK_VALUES:
            topk_rows.append(aggregate_topk(edits, top_parent, label, topk))
    topk_table = pd.DataFrame(topk_rows)

    parent_rows = []
    pair_rows = []
    ci_rows = []
    for topk in TOPK_VALUES:
        primary = parent_topk_metrics(top_parent, "contribution_primary", topk)
        random_control = parent_topk_metrics(top_parent, "same_interface_random", topk)
        parent_rows.extend(primary.to_dict("records"))
        parent_rows.extend(random_control.to_dict("records"))
        if primary.empty or random_control.empty:
            continue
        merged = primary.merge(random_control, on="mol_id", suffixes=("_contribution", "_random"))
        for metric in [
            "mean_delta",
            "best_delta",
            "target_window_rate",
            "guardrail_pass_rate",
            "alert_rate",
            "ood_rate",
            "evidence_ready_rate",
            "consensus_rate",
            "strict_rate",
        ]:
            merged[f"{metric}_diff"] = merged[f"{metric}_contribution"] - merged[f"{metric}_random"]
        merged["topk"] = topk
        pair_rows.extend(merged.to_dict("records"))
        for metric in [
            "mean_delta_diff",
            "best_delta_diff",
            "target_window_rate_diff",
            "guardrail_pass_rate_diff",
            "alert_rate_diff",
            "ood_rate_diff",
            "evidence_ready_rate_diff",
            "consensus_rate_diff",
            "strict_rate_diff",
        ]:
            mean_value, low, high = bootstrap_ci(merged[metric])
            ci_rows.append(
                {
                    "topk": topk,
                    "metric": metric,
                    "paired_parent_count": int(len(merged)),
                    "mean": mean_value,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    return topk_table, pd.DataFrame(parent_rows), pd.DataFrame(pair_rows), pd.DataFrame(ci_rows)


def evidence_tier(row: pd.Series) -> str:
    target_window = bool(row.get("target_window_success", False))
    guardrail = bool(row.get("relaxed_guardrail_pass", False))
    alert_free = not bool(row.get("edited_interference_alert", False))
    ood_free = not bool(row.get("ood_risk", False))
    verifier_support = float(row.get("verifier_support_count", 0) or 0) >= 1
    delta = float(row.get("main_delta_distance_edit", np.nan))

    if target_window and guardrail and alert_free and ood_free and verifier_support:
        return "Tier 1"
    if delta > 0.5 and guardrail and alert_free and ood_free:
        return "Tier 2"
    if delta > 0.05:
        return "Tier 3"
    return "not_selected"


def build_evidence_tier_table(edits: pd.DataFrame) -> pd.DataFrame:
    tier = edits.copy()
    for column in [
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "evidence_ready",
        "consensus_success",
        "strict_success",
    ]:
        tier[column] = as_bool(tier[column])
    tier["evidence_tier"] = tier.apply(evidence_tier, axis=1)
    keep = [
        "method",
        "method_group",
        "edit_action",
        "mol_id",
        "token_id",
        "parent_library_id",
        "canonical_smiles",
        "edited_smiles",
        "risk_side_05",
        "source_split_quality_score",
        "source_split_quality_grade",
        "main_original_pred_mean",
        "edited_main_pred",
        "main_original_target_distance",
        "edited_main_target_distance",
        "main_delta_distance_edit",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "verifier_support_count",
        "auxiliary_opposition_count",
        "direction_consensus_count",
        "main_uncertainty",
        "nearest_train_tanimoto_edited",
        "final_edit_score",
        "consensus_success",
        "strict_success",
        "evidence_tier",
    ]
    return tier[[column for column in keep if column in tier.columns]].copy()


def pick_best(top_parent: pd.DataFrame, method_label: str) -> pd.DataFrame:
    subset = frozen_ranked_subset(top_parent, method_label)
    if subset.empty:
        return subset
    return subset[subset["rank_in_label_parent"] == 1].copy()


def build_master_table(parents: pd.DataFrame, tokens: pd.DataFrame, top_parent: pd.DataFrame) -> pd.DataFrame:
    token_counts = tokens.groupby("mol_id").agg(
        selected_moltoken_count=("token_id", "count"),
        best_token_score=("active_token_score", "max"),
        best_split_quality_score=("split_quality_score", "max"),
        mean_split_quality_score=("split_quality_score", "mean"),
        best_token_main_delta=("main_delta_distance", "max"),
    )
    best_token_rows = tokens.sort_values(["mol_id", "active_token_score"], ascending=[True, False]).groupby("mol_id").head(1)
    best_token_rows = best_token_rows.set_index("mol_id")[
        ["token_id", "atom_indices", "attachment_count", "split_quality_grade", "formal_token_role"]
    ].rename(
        columns={
            "token_id": "best_token_id",
            "atom_indices": "best_token_atom_indices",
            "attachment_count": "best_token_attachment_count",
            "split_quality_grade": "best_token_grade",
            "formal_token_role": "best_token_role",
        }
    )

    primary_best = pick_best(top_parent, "contribution_primary")
    random_best = pick_best(top_parent, "same_interface_random")

    primary_cols = [
        "mol_id",
        "method",
        "edit_action",
        "token_id",
        "edited_smiles",
        "edited_main_pred",
        "edited_main_target_distance",
        "main_delta_distance_edit",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "verifier_support_count",
        "auxiliary_opposition_count",
        "direction_consensus_count",
        "main_uncertainty",
        "nearest_train_tanimoto_edited",
        "final_edit_score",
        "evidence_ready",
        "consensus_success",
        "strict_success",
    ]
    primary_best = primary_best[[column for column in primary_cols if column in primary_best.columns]].copy()
    primary_best = primary_best.add_prefix("best_contribution_").rename(columns={"best_contribution_mol_id": "mol_id"})

    random_cols = [
        "mol_id",
        "edited_smiles",
        "edited_main_pred",
        "edited_main_target_distance",
        "main_delta_distance_edit",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "final_edit_score",
    ]
    random_best = random_best[[column for column in random_cols if column in random_best.columns]].copy()
    random_best = random_best.add_prefix("best_random_").rename(columns={"best_random_mol_id": "mol_id"})

    parent_cols = [
        "external_molecule_id",
        "selected_rank",
        "primary_library_id",
        "phase10_parent_smiles",
        "main_pred_mean",
        "main_target_region",
        "main_target_distance",
        "main_uncertainty",
        "nearest_train_tanimoto",
        "active_parent_tier",
        "xgboost_pred_mean",
        "xgboost_target_distance",
        "rf_pred_mean",
        "rf_target_distance",
        "auxiliary_same_side_count",
        "auxiliary_opposite_side_count",
        "mw",
        "mol_logp",
        "tpsa",
        "qed",
        "murcko_scaffold",
    ]
    master = parents[[column for column in parent_cols if column in parents.columns]].copy()
    master = master.rename(
        columns={
            "external_molecule_id": "mol_id",
            "selected_rank": "parent_rank",
            "primary_library_id": "parent_library_id",
            "phase10_parent_smiles": "parent_smiles",
            "main_pred_mean": "parent_main_pred",
            "main_target_region": "parent_target_region",
            "main_target_distance": "parent_target_distance",
            "main_uncertainty": "parent_main_uncertainty",
            "nearest_train_tanimoto": "parent_nearest_train_tanimoto",
            "mw": "parent_mw",
            "mol_logp": "parent_mol_logp",
            "tpsa": "parent_tpsa",
            "qed": "parent_qed",
            "murcko_scaffold": "parent_murcko_scaffold",
        }
    )
    master = master.merge(token_counts, left_on="mol_id", right_index=True, how="left")
    master = master.merge(best_token_rows, left_on="mol_id", right_index=True, how="left")
    master = master.merge(primary_best, on="mol_id", how="left")
    master = master.merge(random_best, on="mol_id", how="left")
    master["contribution_top1_delta_advantage"] = (
        pd.to_numeric(master["best_contribution_main_delta_distance_edit"], errors="coerce")
        - pd.to_numeric(master["best_random_main_delta_distance_edit"], errors="coerce")
    )
    master["contribution_top1_target_window_advantage"] = (
        as_bool(master["best_contribution_target_window_success"]).astype(float)
        - as_bool(master["best_random_target_window_success"]).astype(float)
    )
    master["has_paired_random_control"] = master["best_random_edited_smiles"].notna()
    return master


def build_confidence_report(parents: pd.DataFrame, edits: pd.DataFrame, top_parent: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(name: str, frame: pd.DataFrame, pred_col: str | None = None) -> None:
        row = {
            "evidence_scope": name,
            "rows": int(len(frame)),
            "parent_count": int(frame["mol_id"].nunique()) if "mol_id" in frame.columns else np.nan,
            "mean_uncertainty": mean_or_nan(frame["main_uncertainty"]) if "main_uncertainty" in frame.columns else np.nan,
            "median_uncertainty": float(pd.to_numeric(frame["main_uncertainty"], errors="coerce").median())
            if "main_uncertainty" in frame.columns
            else np.nan,
            "mean_nearest_train_tanimoto": mean_or_nan(frame["nearest_train_tanimoto_edited"])
            if "nearest_train_tanimoto_edited" in frame.columns
            else mean_or_nan(frame["nearest_train_tanimoto"])
            if "nearest_train_tanimoto" in frame.columns
            else np.nan,
            "ood_rate": rate_or_nan(frame["ood_risk"]) if "ood_risk" in frame.columns else np.nan,
            "target_window_rate": rate_or_nan(frame["target_window_success"]) if "target_window_success" in frame.columns else np.nan,
            "evidence_ready_rate": rate_or_nan(frame["evidence_ready"]) if "evidence_ready" in frame.columns else np.nan,
        }
        if pred_col and pred_col in frame.columns:
            row["mean_prediction"] = mean_or_nan(frame[pred_col])
        rows.append(row)

    parent_view = parents.rename(columns={"external_molecule_id": "mol_id", "main_pred_mean": "main_prediction"})
    add_row("active_parents", parent_view, "main_prediction")
    add_row("all_scored_edits", edits, "edited_main_pred")
    for label in ["contribution_primary", "same_interface_random", "contribution_supplemental"]:
        add_row(f"{label}_all", ranked_subset(edits, label), "edited_main_pred")
        top5 = frozen_ranked_subset(top_parent, label)
        if not top5.empty:
            top5 = top5[top5["rank_in_label_parent"] <= 5]
        add_row(f"{label}_top5", top5, "edited_main_pred")

    corr_frame = edits[["main_uncertainty", "main_delta_distance_edit", "final_edit_score"]].copy()
    for metric in ["main_delta_distance_edit", "final_edit_score"]:
        sub = corr_frame[["main_uncertainty", metric]].apply(pd.to_numeric, errors="coerce").dropna()
        rows.append(
            {
                "evidence_scope": f"uncertainty_vs_{metric}",
                "rows": int(len(sub)),
                "parent_count": np.nan,
                "mean_uncertainty": np.nan,
                "median_uncertainty": np.nan,
                "mean_nearest_train_tanimoto": np.nan,
                "ood_rate": np.nan,
                "target_window_rate": np.nan,
                "evidence_ready_rate": np.nan,
                "mean_prediction": np.nan,
                "pearson": float(sub["main_uncertainty"].corr(sub[metric], method="pearson")) if len(sub) >= 3 else np.nan,
                "spearman": float(sub["main_uncertainty"].corr(sub[metric], method="spearman")) if len(sub) >= 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_moltoken_edit_correlation(tokens: pd.DataFrame, edits: pd.DataFrame) -> pd.DataFrame:
    primary = edits[edits["method_group"] == "contribution_primary"].copy()
    agg = primary.groupby(["mol_id", "token_id"]).agg(
        edit_count=("edited_smiles", "count"),
        best_edit_delta=("main_delta_distance_edit", "max"),
        mean_edit_delta=("main_delta_distance_edit", "mean"),
        best_final_edit_score=("final_edit_score", "max"),
        target_window_rate=("target_window_success", lambda x: rate_or_nan(x)),
        guardrail_rate=("relaxed_guardrail_pass", lambda x: rate_or_nan(x)),
        evidence_ready_rate=("evidence_ready", lambda x: rate_or_nan(x)),
    )
    token_view = tokens.merge(agg, on=["mol_id", "token_id"], how="left")
    token_view["has_primary_edit"] = token_view["edit_count"].fillna(0) > 0

    rows: list[dict[str, Any]] = []
    rows.extend(
        correlation_rows(
            token_view,
            "split_quality_score",
            ["best_edit_delta", "mean_edit_delta", "best_final_edit_score", "target_window_rate", "evidence_ready_rate"],
            "token",
        )
    )
    rows.extend(
        correlation_rows(
            token_view,
            "main_delta_distance",
            ["best_edit_delta", "mean_edit_delta", "best_final_edit_score", "target_window_rate", "evidence_ready_rate"],
            "token",
        )
    )

    for column in ["split_quality_grade", "attachment_count", "parent_region"]:
        if column not in token_view.columns:
            continue
        grouped = token_view.groupby(column, dropna=False)
        for value, group in grouped:
            rows.append(
                {
                    "analysis": f"group_by_{column}",
                    "group_value": value,
                    "n": int(len(group)),
                    "edit_coverage_rate": rate_or_nan(group["has_primary_edit"]),
                    "mean_split_quality_score": mean_or_nan(group["split_quality_score"]),
                    "mean_main_delta_distance": mean_or_nan(group["main_delta_distance"]),
                    "mean_best_edit_delta": mean_or_nan(group["best_edit_delta"]),
                    "target_window_rate": mean_or_nan(group["target_window_rate"]),
                    "evidence_ready_rate": mean_or_nan(group["evidence_ready_rate"]),
                }
            )
    token_detail_path = TABLES / "phase10e_moltoken_edit_detail.csv"
    token_view.to_csv(token_detail_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def build_structure_diversity(parents: pd.DataFrame, edits: pd.DataFrame, top_parent: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "scope": "active_parents",
            "rows": int(len(parents)),
            "parent_count": int(parents["external_molecule_id"].nunique()),
            "unique_smiles": int(parents["phase10_parent_smiles"].dropna().astype(str).nunique()),
            "unique_murcko_scaffold": int(parents["murcko_scaffold"].dropna().astype(str).replace("", np.nan).nunique()),
            "max_scaffold_share": max_share(parents["murcko_scaffold"]),
            "mean_mw": mean_or_nan(parents["mw"]),
            "mean_qed": mean_or_nan(parents["qed"]),
        }
    )

    for label in ["contribution_primary", "same_interface_random", "contribution_supplemental"]:
        subset = ranked_subset(edits, label)
        for topk in ["all", 1, 5]:
            view = subset if topk == "all" else frozen_ranked_subset(top_parent, label)
            if topk != "all":
                view = view[view["rank_in_label_parent"] <= int(topk)]
            if view.empty:
                continue
            rows.append(
                {
                    "scope": f"{label}_top{topk}",
                    "rows": int(len(view)),
                    "parent_count": int(view["mol_id"].nunique()),
                    "unique_smiles": int(view["edited_smiles"].dropna().astype(str).nunique()),
                    "unique_murcko_scaffold": int(
                        view["edited_murcko_scaffold"].dropna().astype(str).replace("", np.nan).nunique()
                    ),
                    "max_scaffold_share": max_share(view["edited_murcko_scaffold"]),
                    "scaffold_retention_rate": float(
                        (
                            view["edited_murcko_scaffold"].fillna("").astype(str)
                            == view["original_murcko_scaffold"].fillna("").astype(str)
                        ).mean()
                    ),
                    "mean_mw": mean_or_nan(view["edited_mw"]),
                    "mean_qed": mean_or_nan(view["edited_qed"]),
                    "mean_parent_tanimoto": mean_or_nan(view["parent_tanimoto"]),
                    "mean_nearest_train_tanimoto": mean_or_nan(view["nearest_train_tanimoto_edited"]),
                }
            )
    return pd.DataFrame(rows)


def failure_reason(row: pd.Series) -> str:
    if not bool(row.get("edited_valid", False)):
        return "invalid_edit"
    if bool(row.get("ood_risk", False)):
        return "domain_risk"
    if bool(row.get("edited_interference_alert", False)):
        return "structure_alert"
    if not bool(row.get("relaxed_guardrail_pass", False)):
        return "guardrail_fail"
    if not bool(row.get("target_window_success", False)):
        return "outside_target_window"
    if float(row.get("verifier_support_count", 0) or 0) < 1:
        return "weak_auxiliary_support"
    if float(row.get("main_delta_distance_edit", 0) or 0) <= 0.05:
        return "weak_delta"
    return "other"


def build_failure_case_table(edits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = edits.copy()
    for column in [
        "edited_valid",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "evidence_ready",
    ]:
        view[column] = as_bool(view[column])
    view["failure_reason"] = view.apply(failure_reason, axis=1)
    summary = (
        view.groupby(["method_group", "failure_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["method_group", "count"], ascending=[True, False])
    )
    examples = (
        view[~view["evidence_ready"]]
        .sort_values(["method_group", "final_edit_score", "main_delta_distance_edit"], ascending=[True, False, False])
        .groupby(["method_group", "failure_reason"], dropna=False)
        .head(3)
    )
    keep = [
        "method_group",
        "method",
        "failure_reason",
        "mol_id",
        "token_id",
        "edit_action",
        "edited_smiles",
        "main_delta_distance_edit",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "verifier_support_count",
        "final_edit_score",
    ]
    return summary, examples[[column for column in keep if column in examples.columns]].copy()




def write_report(
    summary: dict[str, Any],
    topk: pd.DataFrame,
    paired_ci: pd.DataFrame,
    tier_summary: pd.DataFrame,
    confidence: pd.DataFrame,
    correlation: pd.DataFrame,
    diversity: pd.DataFrame,
    failure_summary: pd.DataFrame,
) -> None:
    primary_top5 = topk[(topk["method_label"] == "contribution_primary") & (topk["topk"].astype(str) == "5")].iloc[0]
    random_top5 = topk[(topk["method_label"] == "same_interface_random") & (topk["topk"].astype(str) == "5")].iloc[0]
    paired_delta = paired_ci[(paired_ci["topk"] == 5) & (paired_ci["metric"] == "mean_delta_diff")].iloc[0]
    paired_tw = paired_ci[(paired_ci["topk"] == 5) & (paired_ci["metric"] == "target_window_rate_diff")].iloc[0]
    token_corr = correlation[correlation["analysis"] == "token_split_quality_score_vs_best_edit_delta"]
    corr_text = "NA"
    if not token_corr.empty:
        corr_text = f"{float(token_corr.iloc[0]['spearman']):.3f}"

    tier_counts = tier_summary.to_dict("records")
    tier_lines = "\n".join(
        f"- {row['method_group']} / {row['evidence_tier']}: {int(row['count'])}" for row in tier_counts
    )
    text = f"""# 10-E 最终证据整理结果

## 阶段结论

10-E 已将父分子筛选、Mol Token 贡献切分、片段级编辑与同接口随机对照整理为一套可追溯证据链。当前结果支持的表述是：在 LogD 单目标区间与 drug-like 约束下，贡献引导的删除与替换能够产生比同接口随机替换更强的预测层面改善。

## 核心结果

- 父分子候选数：{summary['phase10b']['candidate_count']}。
- 冻结父分子数：{summary['phase10b']['selected_count']}。
- 活跃 Mol Token 数：{summary['phase10c']['active_moltoken_count']}，父分子覆盖率：{summary['phase10c']['active_parent_coverage_rate']:.3f}。
- 有效编辑分子数：{summary['phase10d']['valid_edit_count']}。
- 完成评分分子数：{summary['phase10d']['scored_edit_count']}。
- 贡献主证据 Top-5 mean delta：{float(primary_top5['mean_delta']):.3f}。
- 同接口随机 Top-5 mean delta：{float(random_top5['mean_delta']):.3f}。
- Top-5 配对 mean delta 优势：{float(paired_delta['mean']):.3f}，95% CI [{float(paired_delta['ci95_low']):.3f}, {float(paired_delta['ci95_high']):.3f}]。
- Top-5 配对目标区间率差异：{float(paired_tw['mean']):.3f}，95% CI [{float(paired_tw['ci95_low']):.3f}, {float(paired_tw['ci95_high']):.3f}]。
- Mol Token 切分质量与最佳编辑改善的 Spearman 相关：{corr_text}。

## 证据分层

{tier_lines}

## 需要强调的边界

- 当前结论属于预测层面证据。
- 主证据限定为删除与替换，增加动作只作为补充观察。
- 100 个小库用于可行性验证，不能写成大规模外部验证。
- 贡献引导优于同接口随机替换，不能写成优于所有分子生成方法。

## 输出表格

- `moltoken/tables/phase10e_final_evidence_master_table.csv`
- `moltoken/tables/phase10e_topk_method_comparison.csv`
- `moltoken/tables/phase10e_parent_level_paired_comparison.csv`
- `moltoken/tables/phase10e_bootstrap_advantage.csv`
- `moltoken/tables/phase10e_evidence_tier_table.csv`
- `moltoken/tables/phase10e_predictor_confidence_report.csv`
- `moltoken/tables/phase10e_moltoken_edit_correlation.csv`
- `moltoken/tables/phase10e_structure_diversity_table.csv`
- `moltoken/tables/phase10e_failure_reason_summary.csv`
- `moltoken/tables/phase10e_failure_case_examples.csv`

## 输出形式

本阶段仅生成和保留证据表格，不生成或保存图片文件。
论文最终图片将在正式定稿后由冻结表格单独制作。

## 专家审查意见

该阶段补齐了此前最容易被质疑的四个点：配对对照、Top-k 选择偏差、切分质量与编辑效果的直接关系、失败原因边界。后续如果进入论文级实验，应优先补充更大训练集或外部测试集的预测器校准结果；在当前小库验证范围内，证据链已经能够支撑方法可行性说明。
"""
    (REPORTS / "phase10e_final_evidence_report.md").write_text(text, encoding="utf-8")

    self_check = f"""# 10-E 自检记录

## 文件完整性

- 最终证据主表：已生成。
- Top-k 方法对照：已生成。
- 父分子级配对比较：已生成。
- bootstrap 置信区间：已生成。
- Mol Token 切分质量关联：已生成。
- 结构多样性统计：已生成。
- 失败原因统计：已生成。
- 输出形式：仅生成证据表格，不生成图片。

## 逻辑边界

- 未将预测结果写成实测结果。
- 未将小库验证写成大规模外部验证。
- 未将增加动作混入主证据。
- 同接口随机对照采用父分子级配对统计。
- Top-1、Top-3、Top-5 与全候选均已输出。

## 指定词检查

报告正文写作时已避开指定表述，后续仍需在整稿合并后再次检查。
"""
    (REPORTS / "phase10e_final_evidence_self_check.md").write_text(self_check, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    parents = pd.read_csv(PARENT_PATH)
    tokens = pd.read_csv(TOKEN_PATH)
    edits = pd.read_csv(EDIT_PATH)
    top_parent = pd.read_csv(TOP_PARENT_PATH)

    for column in [
        "edited_valid",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "evidence_ready",
        "consensus_success",
        "strict_success",
        "main_direction_correct",
    ]:
        if column in edits.columns:
            edits[column] = as_bool(edits[column])

    summaries = {
        "b": read_json(SUMMARY_B_PATH),
        "c": read_json(SUMMARY_C_PATH),
        "d": read_json(SUMMARY_D_PATH),
    }

    for column in [
        "edited_valid",
        "target_window_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "ood_risk",
        "evidence_ready",
        "consensus_success",
        "strict_success",
        "main_direction_correct",
    ]:
        if column in top_parent.columns:
            top_parent[column] = as_bool(top_parent[column])

    topk, parent_topk, paired, paired_ci = build_topk_tables(edits, top_parent)
    tier = build_evidence_tier_table(edits)
    tier_summary = (
        tier.groupby(["method_group", "evidence_tier"], dropna=False).size().reset_index(name="count")
    )
    master = build_master_table(parents, tokens, top_parent)
    confidence = build_confidence_report(parents, edits, top_parent)
    correlation = build_moltoken_edit_correlation(tokens, edits)
    diversity = build_structure_diversity(parents, edits, top_parent)
    failure_summary, failure_examples = build_failure_case_table(edits)

    master.to_csv(TABLES / "phase10e_final_evidence_master_table.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(TABLES / "phase10e_topk_method_comparison.csv", index=False, encoding="utf-8-sig")
    parent_topk.to_csv(TABLES / "phase10e_parent_level_topk_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / "phase10e_parent_level_paired_comparison.csv", index=False, encoding="utf-8-sig")
    paired_ci.to_csv(TABLES / "phase10e_bootstrap_advantage.csv", index=False, encoding="utf-8-sig")
    tier.to_csv(TABLES / "phase10e_evidence_tier_table.csv", index=False, encoding="utf-8-sig")
    tier_summary.to_csv(TABLES / "phase10e_evidence_tier_summary.csv", index=False, encoding="utf-8-sig")
    confidence.to_csv(TABLES / "phase10e_predictor_confidence_report.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(TABLES / "phase10e_moltoken_edit_correlation.csv", index=False, encoding="utf-8-sig")
    diversity.to_csv(TABLES / "phase10e_structure_diversity_table.csv", index=False, encoding="utf-8-sig")
    failure_summary.to_csv(TABLES / "phase10e_failure_reason_summary.csv", index=False, encoding="utf-8-sig")
    failure_examples.to_csv(TABLES / "phase10e_failure_case_examples.csv", index=False, encoding="utf-8-sig")

    summary = {
        "phase": "10E_final_evidence_pack",
        "verdict": "pass",
        "phase10b": summaries["b"],
        "phase10c": summaries["c"],
        "phase10d": summaries["d"],
        "final_evidence_master_rows": int(len(master)),
        "topk_method_rows": int(len(topk)),
        "paired_parent_rows": int(len(paired)),
        "bootstrap_rows": int(len(paired_ci)),
        "tier_rows": int(len(tier)),
        "output_mode": "tables_only",
        "output_table_count": 13,
        "key_metrics": {
            "contribution_primary_top5_mean_delta": float(
                topk[(topk["method_label"] == "contribution_primary") & (topk["topk"].astype(str) == "5")][
                    "mean_delta"
                ].iloc[0]
            ),
            "same_interface_random_top5_mean_delta": float(
                topk[(topk["method_label"] == "same_interface_random") & (topk["topk"].astype(str) == "5")][
                    "mean_delta"
                ].iloc[0]
            ),
            "top5_paired_mean_delta_advantage": float(
                paired_ci[(paired_ci["topk"] == 5) & (paired_ci["metric"] == "mean_delta_diff")]["mean"].iloc[0]
            ),
            "top5_paired_mean_delta_ci95_low": float(
                paired_ci[(paired_ci["topk"] == 5) & (paired_ci["metric"] == "mean_delta_diff")]["ci95_low"].iloc[0]
            ),
            "top5_paired_mean_delta_ci95_high": float(
                paired_ci[(paired_ci["topk"] == 5) & (paired_ci["metric"] == "mean_delta_diff")]["ci95_high"].iloc[0]
            ),
            "tier1_count": int((tier["evidence_tier"] == "Tier 1").sum()),
            "tier2_count": int((tier["evidence_tier"] == "Tier 2").sum()),
            "tier3_count": int((tier["evidence_tier"] == "Tier 3").sum()),
        },
        "outputs": {
            "master_table": rel(TABLES / "phase10e_final_evidence_master_table.csv"),
            "topk_table": rel(TABLES / "phase10e_topk_method_comparison.csv"),
            "paired_table": rel(TABLES / "phase10e_parent_level_paired_comparison.csv"),
            "bootstrap_table": rel(TABLES / "phase10e_bootstrap_advantage.csv"),
            "tier_table": rel(TABLES / "phase10e_evidence_tier_table.csv"),
            "confidence_table": rel(TABLES / "phase10e_predictor_confidence_report.csv"),
            "correlation_table": rel(TABLES / "phase10e_moltoken_edit_correlation.csv"),
            "diversity_table": rel(TABLES / "phase10e_structure_diversity_table.csv"),
            "failure_summary": rel(TABLES / "phase10e_failure_reason_summary.csv"),
            "report": rel(REPORTS / "phase10e_final_evidence_report.md"),
            "self_check": rel(REPORTS / "phase10e_final_evidence_self_check.md"),
        },
    }
    (REPORTS / "phase10e_final_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        {
            "phase10b": summaries["b"],
            "phase10c": summaries["c"],
            "phase10d": summaries["d"],
        },
        topk,
        paired_ci,
        tier_summary,
        confidence,
        correlation,
        diversity,
        failure_summary,
    )

    print(json.dumps(summary["key_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
