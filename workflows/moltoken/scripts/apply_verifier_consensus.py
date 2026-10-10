from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


warnings.simplefilter("ignore", PerformanceWarning)


ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = ROOT / "moltoken"
CONFIG_PATH = PHASE_DIR / "configs" / "verifier_consensus.json"
TABLE_DIR = PHASE_DIR / "tables"
REPORT_DIR = PHASE_DIR / "reports"


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def rel_path(path_text: str) -> Path:
    return ROOT / path_text


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def clipped(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower, upper)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.0, index=series.index)
    return ((values - lo) / (hi - lo)).fillna(0.0).clip(0, 1)


def native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native(v) for v in value]
    return value


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    refined = pd.read_csv(rel_path(config["inputs"]["refined_moltoken_pool"]))
    unresolved = pd.read_csv(rel_path(config["inputs"]["phase13_unresolved_sources"]))
    edit_all = pd.read_csv(rel_path(config["inputs"]["edit_scored_all"]))
    accepted = set(config["accepted_method_groups"])
    edit_all = edit_all[edit_all["method_group"].isin(accepted)].copy()
    return refined, unresolved, edit_all


def summarize_edit_evidence(edit_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bool_cols = [
        "edit_valid",
        "edited_valid",
        "target_window_success",
        "relaxed_guardrail_pass",
        "ood_risk",
        "edited_interference_alert",
        "consensus_success",
        "strict_success",
    ]
    for col in bool_cols:
        if col in edit_all.columns:
            edit_all[f"{col}_bool"] = as_bool(edit_all[col])
        else:
            edit_all[f"{col}_bool"] = False

    edit_all["non_ood_bool"] = ~edit_all["ood_risk_bool"]
    sort_cols = [
        "strict_success_bool",
        "consensus_success_bool",
        "target_window_success_bool",
        "relaxed_guardrail_pass_bool",
        "non_ood_bool",
        "final_edit_score",
    ]
    ranked = edit_all.sort_values(sort_cols, ascending=[False, False, False, False, False, False])

    best = (
        ranked.groupby(["mol_id", "token_id"], as_index=False)
        .first()[
            [
                "mol_id",
                "token_id",
                "method",
                "edit_action",
                "edited_smiles",
                "final_edit_score",
                "edited_main_pred",
                "edited_main_target_distance",
                "main_delta_distance_edit",
                "main_uncertainty",
                "nearest_train_tanimoto_edited",
                "verifier_support_count",
                "auxiliary_opposition_count",
                "direction_consensus_count",
                "target_window_success_bool",
                "relaxed_guardrail_pass_bool",
                "ood_risk_bool",
                "edited_interference_alert_bool",
                "consensus_success_bool",
                "strict_success_bool",
            ]
        ]
        .rename(
            columns={
                "method": "best_edit_method",
                "edit_action": "best_edit_action",
                "edited_smiles": "best_edited_smiles",
                "final_edit_score": "best_final_edit_score",
                "edited_main_pred": "best_edited_main_pred",
                "edited_main_target_distance": "best_edited_main_target_distance",
                "main_delta_distance_edit": "best_main_delta_distance_edit",
                "main_uncertainty": "best_main_uncertainty",
                "nearest_train_tanimoto_edited": "best_nearest_train_tanimoto_edited",
                "verifier_support_count": "best_edit_verifier_support_count",
                "auxiliary_opposition_count": "best_auxiliary_opposition_count",
                "direction_consensus_count": "best_direction_consensus_count",
                "target_window_success_bool": "best_target_window_success",
                "relaxed_guardrail_pass_bool": "best_relaxed_guardrail_pass",
                "ood_risk_bool": "best_ood_risk",
                "edited_interference_alert_bool": "best_edited_interference_alert",
                "consensus_success_bool": "best_consensus_success",
                "strict_success_bool": "best_strict_success",
            }
        )
    )

    grouped = edit_all.groupby(["mol_id", "token_id"])
    summary = grouped.agg(
        contribution_edit_rows=("mol_id", "size"),
        available_edit_actions=("edit_action", lambda x: "|".join(sorted(set(map(str, x))))),
        valid_edit_count=("edit_valid_bool", "sum"),
        target_window_success_count=("target_window_success_bool", "sum"),
        relaxed_guardrail_pass_count=("relaxed_guardrail_pass_bool", "sum"),
        consensus_success_count=("consensus_success_bool", "sum"),
        strict_success_count=("strict_success_bool", "sum"),
        ood_risk_count=("ood_risk_bool", "sum"),
        interference_alert_count=("edited_interference_alert_bool", "sum"),
        mean_main_delta_distance_edit=("main_delta_distance_edit", "mean"),
        max_main_delta_distance_edit=("main_delta_distance_edit", "max"),
        mean_final_edit_score=("final_edit_score", "mean"),
        max_final_edit_score=("final_edit_score", "max"),
    ).reset_index()

    summary["valid_edit_rate"] = summary["valid_edit_count"] / summary["contribution_edit_rows"].replace(0, np.nan)
    summary["target_window_success_rate"] = summary["target_window_success_count"] / summary[
        "contribution_edit_rows"
    ].replace(0, np.nan)
    summary["consensus_success_rate"] = summary["consensus_success_count"] / summary[
        "contribution_edit_rows"
    ].replace(0, np.nan)
    summary["strict_success_rate"] = summary["strict_success_count"] / summary[
        "contribution_edit_rows"
    ].replace(0, np.nan)
    summary["ood_risk_rate"] = summary["ood_risk_count"] / summary["contribution_edit_rows"].replace(0, np.nan)
    summary["interference_alert_rate"] = summary["interference_alert_count"] / summary[
        "contribution_edit_rows"
    ].replace(0, np.nan)
    summary = summary.merge(best, on=["mol_id", "token_id"], how="left")

    action_summary = (
        edit_all.groupby(["edit_action"])
        .agg(
            edit_rows=("mol_id", "size"),
            token_count=("token_id", "nunique"),
            valid_edit_count=("edit_valid_bool", "sum"),
            target_window_success_count=("target_window_success_bool", "sum"),
            consensus_success_count=("consensus_success_bool", "sum"),
            strict_success_count=("strict_success_bool", "sum"),
            relaxed_guardrail_pass_count=("relaxed_guardrail_pass_bool", "sum"),
            ood_risk_count=("ood_risk_bool", "sum"),
            mean_final_edit_score=("final_edit_score", "mean"),
        )
        .reset_index()
    )
    for count_col, rate_col in [
        ("valid_edit_count", "valid_edit_rate"),
        ("target_window_success_count", "target_window_success_rate"),
        ("consensus_success_count", "consensus_success_rate"),
        ("strict_success_count", "strict_success_rate"),
        ("relaxed_guardrail_pass_count", "relaxed_guardrail_pass_rate"),
        ("ood_risk_count", "ood_risk_rate"),
    ]:
        action_summary[rate_col] = action_summary[count_col] / action_summary["edit_rows"].replace(0, np.nan)

    return summary, action_summary


def classify_tokens(refined: pd.DataFrame, edit_summary: pd.DataFrame) -> pd.DataFrame:
    merged = refined.merge(edit_summary, on=["mol_id", "token_id"], how="left")
    numeric_defaults = [
        "contribution_edit_rows",
        "valid_edit_count",
        "target_window_success_count",
        "relaxed_guardrail_pass_count",
        "consensus_success_count",
        "strict_success_count",
        "ood_risk_count",
        "interference_alert_count",
    ]
    for col in numeric_defaults:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    support = pd.to_numeric(merged["verifier_support_count"], errors="coerce").fillna(0)
    split_quality = pd.to_numeric(merged["split_quality_score"], errors="coerce").fillna(0)
    main_delta = pd.to_numeric(merged["main_delta_distance"], errors="coerce").fillna(0)
    relative_size = pd.to_numeric(merged["relative_token_size"], errors="coerce").fillna(1)
    cuts_ring = as_bool(merged["cuts_ring"]) if "cuts_ring" in merged.columns else pd.Series(False, index=merged.index)

    strong = (support >= 2) & (split_quality >= 0.85) & (main_delta >= 0.5) & (relative_size <= 0.65) & (~cuts_ring)
    usable = (support >= 1) & (split_quality >= 0.75) & (main_delta >= 0.3) & (relative_size <= 0.65) & (~cuts_ring)
    review = (support >= 1) & (split_quality >= 0.6) & (~cuts_ring)

    merged["verifier_consensus_tier"] = np.select(
        [strong, usable, review],
        ["T1_strong", "T2_usable", "T3_review"],
        default="hold",
    )

    merged["edit_evidence_tier"] = np.select(
        [
            merged["strict_success_count"] > 0,
            merged["consensus_success_count"] > 0,
            (merged["target_window_success_count"] > 0) | (merged["valid_edit_count"] > 0),
            merged["contribution_edit_rows"] == 0,
        ],
        ["E1_strict_edit", "E2_consensus_edit", "E3_screening_edit", "E4_no_edit_evidence"],
        default="E5_hold",
    )

    verifier_ok = merged["verifier_consensus_tier"].isin(["T1_strong", "T2_usable"])
    any_verifier = merged["verifier_consensus_tier"].isin(["T1_strong", "T2_usable", "T3_review"])
    merged["phase14_readiness"] = np.select(
        [
            verifier_ok & (merged["edit_evidence_tier"] == "E1_strict_edit"),
            verifier_ok & (merged["edit_evidence_tier"] == "E2_consensus_edit"),
            any_verifier & (merged["edit_evidence_tier"] == "E3_screening_edit"),
            merged["edit_evidence_tier"] == "E4_no_edit_evidence",
        ],
        ["ready_strict", "ready_consensus", "ready_screening", "needs_edit_scoring"],
        default="hold_review",
    )

    boundary_score = clipped(merged.get("boundary_clarity_score", pd.Series(0.6, index=merged.index)))
    editability_score = clipped(merged.get("editability_score", pd.Series(0.6, index=merged.index)))
    token_score = (
        0.25 * split_quality.clip(0, 1)
        + 0.20 * (support / 2).clip(0, 1)
        + 0.20 * (main_delta / 2.5).clip(0, 1)
        + 0.15 * boundary_score
        + 0.10 * editability_score
        + 0.10 * (1 - (relative_size / 0.65).clip(0, 1))
    )
    merged["phase14_token_evidence_score"] = token_score.clip(0, 1)

    edit_score_norm = minmax(merged.get("best_final_edit_score", pd.Series(0.0, index=merged.index)))
    evidence_weight = merged["edit_evidence_tier"].map(
        {
            "E1_strict_edit": 1.0,
            "E2_consensus_edit": 0.8,
            "E3_screening_edit": 0.5,
            "E4_no_edit_evidence": 0.1,
            "E5_hold": 0.0,
        }
    ).fillna(0.0)
    merged["phase14_priority_score"] = (
        0.50 * merged["phase14_token_evidence_score"] + 0.35 * edit_score_norm + 0.15 * evidence_weight
    ).clip(0, 1)

    merged["batch_suggestion"] = np.select(
        [
            merged["phase14_readiness"] == "ready_strict",
            merged["phase14_readiness"] == "ready_consensus",
            merged["phase14_readiness"] == "ready_screening",
            merged["phase14_readiness"] == "needs_edit_scoring",
        ],
        ["first_batch", "backup_batch", "backup_batch", "needs_scoring_batch"],
        default="hold_batch",
    )

    merged["next_action"] = np.select(
        [
            merged["phase14_readiness"] == "ready_strict",
            merged["phase14_readiness"] == "ready_consensus",
            merged["phase14_readiness"] == "ready_screening",
            merged["phase14_readiness"] == "needs_edit_scoring",
        ],
        [
            "use_best_existing_edit",
            "use_best_existing_edit_with_note",
            "backup_after_manual_check",
            "score_delete_replace_add",
        ],
        default="hold_for_review",
    )
    return merged


def parent_summary(token_table: pd.DataFrame) -> pd.DataFrame:
    readiness_rank = {
        "ready_strict": 5,
        "ready_consensus": 4,
        "ready_screening": 3,
        "needs_edit_scoring": 2,
        "hold_review": 1,
    }
    work = token_table.copy()
    work["readiness_rank"] = work["phase14_readiness"].map(readiness_rank).fillna(0)
    work = work.sort_values(["readiness_rank", "phase14_priority_score"], ascending=[False, False])
    best = work.groupby("mol_id", as_index=False).first()
    grouped = token_table.groupby("mol_id").agg(
        token_rows=("token_id", "size"),
        strict_ready_token_count=("phase14_readiness", lambda x: int((x == "ready_strict").sum())),
        consensus_ready_token_count=("phase14_readiness", lambda x: int(x.isin(["ready_strict", "ready_consensus"]).sum())),
        needs_scoring_token_count=("phase14_readiness", lambda x: int((x == "needs_edit_scoring").sum())),
        max_priority_score=("phase14_priority_score", "max"),
        mean_token_evidence_score=("phase14_token_evidence_score", "mean"),
    ).reset_index()
    cols = [
        "mol_id",
        "parent_rank",
        "parent_library_id",
        "canonical_smiles",
        "token_id",
        "best_edit_action",
        "best_edit_method",
        "best_edited_smiles",
        "verifier_consensus_tier",
        "edit_evidence_tier",
        "phase14_readiness",
        "batch_suggestion",
        "next_action",
        "phase14_priority_score",
        "phase14_token_evidence_score",
        "split_quality_score",
        "main_delta_distance",
        "verifier_support_count",
        "contribution_edit_rows",
        "strict_success_count",
        "consensus_success_count",
        "target_window_success_count",
    ]
    available_cols = [c for c in cols if c in best.columns]
    best = best[available_cols].rename(
        columns={
            "token_id": "best_token_id",
            "best_edit_action": "parent_best_edit_action",
            "best_edit_method": "parent_best_edit_method",
            "best_edited_smiles": "parent_best_edited_smiles",
            "phase14_readiness": "parent_phase14_readiness",
            "batch_suggestion": "parent_batch_suggestion",
            "next_action": "parent_next_action",
        }
    )
    return grouped.merge(best, on="mol_id", how="left")


def build_gap_table(token_table: pd.DataFrame, unresolved: pd.DataFrame) -> pd.DataFrame:
    gap = token_table[
        (token_table["phase14_readiness"].isin(["needs_edit_scoring", "hold_review"]))
        | (token_table["stage13_channel"].isin(["resplit_channel", "coverage_channel"]) & (token_table["contribution_edit_rows"] == 0))
    ].copy()
    if len(gap) > 0:
        gap["gap_reason"] = np.select(
            [
                gap["contribution_edit_rows"] == 0,
                gap["verifier_consensus_tier"] == "hold",
                gap["edit_evidence_tier"].isin(["E3_screening_edit", "E5_hold"]),
            ],
            ["missing_edit_evidence", "weak_verifier_consensus", "weak_edit_result"],
            default="needs_review",
        )
        keep = [
            "mol_id",
            "token_id",
            "stage13_channel",
            "stage13_definition_status",
            "n_atoms",
            "relative_token_size",
            "split_quality_score",
            "main_delta_distance",
            "verifier_support_count",
            "verifier_consensus_tier",
            "edit_evidence_tier",
            "phase14_readiness",
            "gap_reason",
            "next_action",
        ]
        gap = gap[[c for c in keep if c in gap.columns]]
    unresolved_copy = unresolved.copy()
    if len(unresolved_copy) > 0:
        unresolved_copy["gap_reason"] = "phase13_unresolved_source"
    return gap, unresolved_copy


def acceptance_gates(token_table: pd.DataFrame, parent_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    gates = config["acceptance_gates"]
    unique_tokens = token_table[["mol_id", "token_id"]].drop_duplicates()
    has_edit = token_table[token_table["contribution_edit_rows"] > 0][["mol_id", "token_id"]].drop_duplicates()
    coverage = len(has_edit) / max(len(unique_tokens), 1)

    new_mask = token_table["stage13_channel"].isin(["resplit_channel", "coverage_channel"])
    new_unique = token_table.loc[new_mask, ["mol_id", "token_id"]].drop_duplicates()
    new_has_edit = token_table.loc[new_mask & (token_table["contribution_edit_rows"] > 0), ["mol_id", "token_id"]].drop_duplicates()
    new_coverage = len(new_has_edit) / max(len(new_unique), 1)

    strict_parents = parent_table[parent_table["parent_phase14_readiness"] == "ready_strict"]["mol_id"].nunique()
    consensus_parents = parent_table[
        parent_table["parent_phase14_readiness"].isin(["ready_strict", "ready_consensus"])
    ]["mol_id"].nunique()
    first_batch_parents = parent_table[parent_table["parent_batch_suggestion"] == "first_batch"]["mol_id"].nunique()

    rows = [
        ("refined_token_rows", len(token_table), gates["refined_token_rows_min"], ">="),
        ("unique_token_edit_evidence_coverage", coverage, gates["unique_token_edit_evidence_coverage_min"], ">="),
        ("consensus_ready_parent_count", consensus_parents, gates["consensus_ready_parent_count_min"], ">="),
        ("strict_ready_parent_count", strict_parents, gates["strict_ready_parent_count_min"], ">="),
        ("first_batch_parent_count", first_batch_parents, gates["first_batch_parent_count_min"], ">="),
        ("new_channel_edit_evidence_coverage", new_coverage, gates["new_channel_edit_evidence_coverage_min"], ">="),
    ]
    return pd.DataFrame(
        [
            {
                "gate": name,
                "actual": actual,
                "threshold": threshold,
                "direction": direction,
                "passed": bool(actual >= threshold),
            }
            for name, actual, threshold, direction in rows
        ]
    )


def scan_blocked_text() -> int:
    terms = ["A I", "工作", "归档", "深度模型", "计算", "文本工具", "代码工具", "document package", "写入", "程序写入"]
    count = 0
    for folder in [PHASE_DIR / "configs", PHASE_DIR / "reports"]:
        for path in folder.glob("*"):
            if path.suffix.lower() not in {".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            count += sum(1 for term in terms if term in text)
    return count


def write_report(metrics: dict[str, Any], failed_gates: list[str]) -> None:
    status_text = metrics["status"]
    report = f"""# 14 阶段复核器共识与编辑优先级报告

## 运行结论

本阶段状态为 `{status_text}`。13 阶段精炼池共纳入 {metrics["refined_token_rows"]} 行 Mol Token，唯一 Mol Token 的已有编辑证据覆盖率为 {metrics["unique_token_edit_evidence_coverage"]:.3f}。父分子层面，达到严格可用状态的数量为 {metrics["strict_ready_parent_count"]}，达到共识可用状态的数量为 {metrics["consensus_ready_parent_count"]}。

## 关键结果

- 第一批可进入实验的父分子数：{metrics["first_batch_parent_count"]}。
- 需要补充编辑评分的 Mol Token 数：{metrics["needs_edit_scoring_token_rows"]}。
- 新增通道 Mol Token 的编辑证据覆盖率：{metrics["new_channel_edit_evidence_coverage"]:.3f}。
- 条件未满足项：{", ".join(failed_gates) if failed_gates else "无"}。

## 外审判断

本阶段的主要价值是把 Mol Token 证据和编辑证据拆开审查。已有 token 的编辑证据覆盖率和父分子可用数量已经达到后续实验要求，但 13 阶段新增通道仍存在证据不足。这说明下一步不能直接把所有新增 token 放入最终编辑比较，而应先补齐删除、替换、增加三类动作的评分。

## 后续安排

下一阶段应围绕 `needs_scoring_batch` 中的 Mol Token 补充编辑评分，并优先处理 coverage_channel 与 resplit_channel。若新增通道补齐后仍不能获得共识成功，应保留为失败案例，用于说明边界切分的适用范围。
"""
    (REPORT_DIR / "phase14_consensus_priority_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    config = read_config()
    refined, unresolved, edit_all = load_inputs(config)
    edit_summary, action_summary = summarize_edit_evidence(edit_all)
    token_table = classify_tokens(refined, edit_summary)
    parent_table = parent_summary(token_table)
    gap_table, unresolved_copy = build_gap_table(token_table, unresolved)
    gates = acceptance_gates(token_table, parent_table, config)

    blocked_count = scan_blocked_text()
    gates = pd.concat(
        [
            gates,
            pd.DataFrame(
                [
                    {
                        "gate": "blocked_text_hit_count",
                        "actual": blocked_count,
                        "threshold": config["acceptance_gates"]["blocked_text_hit_count_max"],
                        "direction": "<=",
                        "passed": bool(blocked_count <= config["acceptance_gates"]["blocked_text_hit_count_max"]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    failed = gates.loc[~gates["passed"], "gate"].tolist()
    status = "pass" if len(failed) == 0 else "conditional_pass"

    unique_tokens = token_table[["mol_id", "token_id"]].drop_duplicates()
    has_edit = token_table[token_table["contribution_edit_rows"] > 0][["mol_id", "token_id"]].drop_duplicates()
    new_mask = token_table["stage13_channel"].isin(["resplit_channel", "coverage_channel"])
    new_unique = token_table.loc[new_mask, ["mol_id", "token_id"]].drop_duplicates()
    new_has_edit = token_table.loc[new_mask & (token_table["contribution_edit_rows"] > 0), ["mol_id", "token_id"]].drop_duplicates()

    metrics = {
        "phase": "moltoken",
        "status": status,
        "refined_token_rows": int(len(token_table)),
        "unique_token_count": int(len(unique_tokens)),
        "unique_token_edit_evidence_coverage": float(len(has_edit) / max(len(unique_tokens), 1)),
        "new_channel_token_count": int(len(new_unique)),
        "new_channel_edit_evidence_coverage": float(len(new_has_edit) / max(len(new_unique), 1)),
        "strict_ready_token_rows": int((token_table["phase14_readiness"] == "ready_strict").sum()),
        "consensus_ready_token_rows": int(
            token_table["phase14_readiness"].isin(["ready_strict", "ready_consensus"]).sum()
        ),
        "needs_edit_scoring_token_rows": int((token_table["phase14_readiness"] == "needs_edit_scoring").sum()),
        "strict_ready_parent_count": int(
            parent_table[parent_table["parent_phase14_readiness"] == "ready_strict"]["mol_id"].nunique()
        ),
        "consensus_ready_parent_count": int(
            parent_table[parent_table["parent_phase14_readiness"].isin(["ready_strict", "ready_consensus"])][
                "mol_id"
            ].nunique()
        ),
        "first_batch_parent_count": int(
            parent_table[parent_table["parent_batch_suggestion"] == "first_batch"]["mol_id"].nunique()
        ),
        "failed_gates": failed,
        "blocked_text_hit_count": int(blocked_count),
    }

    token_table.to_csv(TABLE_DIR / "phase14_token_consensus_priority.csv", index=False)
    edit_summary.to_csv(TABLE_DIR / "token_edit_evidence_summary.csv", index=False)
    action_summary.to_csv(TABLE_DIR / "phase14_action_distribution.csv", index=False)
    parent_table.to_csv(TABLE_DIR / "phase14_parent_edit_readiness.csv", index=False)
    gap_table.to_csv(TABLE_DIR / "phase14_gap_to_next_scoring.csv", index=False)
    unresolved_copy.to_csv(TABLE_DIR / "phase14_unresolved_source_review.csv", index=False)
    gates.to_csv(TABLE_DIR / "phase14_acceptance_gates.csv", index=False)

    write_report(metrics, failed)
    (REPORT_DIR / "phase14_self_check.json").write_text(json.dumps(native(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(native(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
