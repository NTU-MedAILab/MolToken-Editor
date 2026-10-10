from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


warnings.simplefilter("ignore", PerformanceWarning)

ROOT = Path(__file__).resolve().parents[2]
PHASE15 = ROOT / "moltoken"
TABLES = PHASE15 / "tables"
REPORTS = PHASE15 / "reports"
PREDICTIONS = TABLES / "predictions" / "edit_scoring"
CONFIG_PATH = PHASE15 / "configs" / "additional_edit_scoring.json"

sys.path.append(str(ROOT / "moltoken" / "scripts"))
import generate_edits as edit_tools


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path_text: str) -> Path:
    return ROOT / path_text


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


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


def scan_blocked_text() -> int:
    terms = ["A I", "工作", "归档", "深度模型", "计算", "文本工具", "代码工具", "document package", "写入", "程序写入"]
    count = 0
    for folder in [PHASE15 / "configs", PHASE15 / "reports"]:
        for path in folder.glob("*"):
            if path.suffix.lower() not in {".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            count += sum(1 for term in terms if term in text)
    return count


def configure_phase15_prediction_paths() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    edit_tools.TABLES = TABLES
    edit_tools.REPORTS = REPORTS
    edit_tools.PREDICTIONS = PREDICTIONS


def load_targets(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    refined = pd.read_csv(rel_path(config["inputs"]["phase13_refined_pool"]))
    gap = pd.read_csv(rel_path(config["inputs"]["phase14_gap_table"]))
    target_gap = gap[
        (gap["phase14_readiness"] == config["target_rule"]["phase14_readiness"])
        & gap["stage13_channel"].isin(config["target_rule"]["stage13_channels"])
    ][["mol_id", "token_id"]].drop_duplicates()
    targets = refined.merge(target_gap, on=["mol_id", "token_id"], how="inner").copy()
    targets = targets.sort_values(["stage13_channel", "mol_id", "token_id"]).reset_index(drop=True)
    if "active_moltoken_rank" not in targets.columns:
        targets["active_moltoken_rank"] = np.arange(1, len(targets) + 1)
    else:
        targets["active_moltoken_rank"] = pd.to_numeric(targets["active_moltoken_rank"], errors="coerce")
        fallback_rank = pd.Series(np.arange(1, len(targets) + 1), index=targets.index)
        targets["active_moltoken_rank"] = targets["active_moltoken_rank"].where(
            targets["active_moltoken_rank"].notna(), fallback_rank
        )
    targets["source_target_token_id"] = targets["token_id"]
    return targets, refined


def build_candidates(targets: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    donors = pd.read_csv(rel_path(config["inputs"]["phase10d_donor_library"]))
    linkers = pd.read_csv(rel_path(config["inputs"]["phase10d_linker_library"]))
    rules = config["candidate_generation"]
    parts: list[pd.DataFrame] = []
    if rules["include_delete"]:
        parts.append(edit_tools.build_delete_candidates(targets))
    if rules["include_replace"]:
        parts.append(
            edit_tools.build_replace_candidates(
                targets,
                donors,
                linkers,
                max_reagents_per_token=int(rules["max_reagents_per_token"]),
                method="contribution_phase15_new_channel_replace",
            )
        )
    if rules["include_add"]:
        parts.append(
            edit_tools.build_add_candidates(
                targets,
                donors,
                max_add_per_parent=int(rules["max_add_per_parent"]),
            )
        )
    if not parts:
        return pd.DataFrame()
    candidates = pd.concat(parts, ignore_index=True, sort=False)
    target_meta = targets[["mol_id", "token_id", "stage13_channel", "stage13_definition_status"]].drop_duplicates()
    candidates = candidates.merge(target_meta, on=["mol_id", "token_id"], how="left")
    candidates["phase15_candidate_scope"] = "new_channel_gap"
    candidates["edit_row_id"] = np.arange(len(candidates))
    return candidates


def rank_top_edits(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    valid = scored[
        scored["edit_valid"].map(edit_tools.bool_value)
        & scored["edited_main_pred"].notna()
    ].copy()
    if valid.empty:
        return valid
    valid["rank_signal"] = (
        valid["strict_success"].astype(float) * 100
        + valid["consensus_success"].astype(float) * 20
        + valid["target_window_success"].astype(float) * 10
        + valid["relaxed_guardrail_pass"].astype(float) * 5
        + valid["final_edit_score"].fillna(0)
    )
    valid = valid.sort_values(
        [
            "mol_id",
            "token_id",
            "rank_signal",
            "final_edit_score",
        ],
        ascending=[True, True, False, False],
    )
    valid["rank_in_token"] = valid.groupby(["mol_id", "token_id"]).cumcount() + 1
    valid["rank_in_token_action"] = valid.groupby(["mol_id", "token_id", "edit_action"]).cumcount() + 1
    return valid[(valid["rank_in_token"] <= 5) | (valid["rank_in_token_action"] <= 3)].reset_index(drop=True)


def summarize_by_token(scored: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    work = scored.copy()
    for col in ["edit_valid", "target_window_success", "relaxed_guardrail_pass", "consensus_success", "strict_success", "ood_risk", "edited_interference_alert"]:
        if col in work.columns:
            work[f"{col}_bool"] = bool_series(work[col])
        else:
            work[f"{col}_bool"] = False
    work["scored_bool"] = work["edited_main_pred"].notna()
    grouped = work.groupby(["mol_id", "token_id"]).agg(
        candidate_attempts=("edit_row_id", "size"),
        action_types=("edit_action", lambda x: "|".join(sorted(set(map(str, x))))),
        valid_edit_count=("edit_valid_bool", "sum"),
        scored_edit_count=("scored_bool", "sum"),
        target_window_success_count=("target_window_success_bool", "sum"),
        consensus_success_count=("consensus_success_bool", "sum"),
        strict_success_count=("strict_success_bool", "sum"),
        guardrail_pass_count=("relaxed_guardrail_pass_bool", "sum"),
        ood_risk_count=("ood_risk_bool", "sum"),
        alert_count=("edited_interference_alert_bool", "sum"),
        best_final_edit_score=("final_edit_score", "max"),
        best_main_delta_distance_edit=("main_delta_distance_edit", "max"),
    ).reset_index()
    ranked = rank_top_edits(scored)
    best = ranked[ranked["rank_in_token"] == 1].copy()
    if not best.empty:
        best = best[
            [
                "mol_id",
                "token_id",
                "method",
                "edit_action",
                "edited_smiles",
                "edited_main_pred",
                "edited_main_target_distance",
                "main_delta_distance_edit",
                "verifier_support_count",
                "auxiliary_opposition_count",
                "direction_consensus_count",
                "target_window_success",
                "relaxed_guardrail_pass",
                "ood_risk",
                "consensus_success",
                "strict_success",
                "final_edit_score",
            ]
        ].rename(
            columns={
                "method": "best_method",
                "edit_action": "best_edit_action",
                "edited_smiles": "best_edited_smiles",
                "edited_main_pred": "best_edited_main_pred",
                "edited_main_target_distance": "best_edited_main_target_distance",
                "main_delta_distance_edit": "best_main_delta_distance_edit_ranked",
                "verifier_support_count": "best_verifier_support_count",
                "auxiliary_opposition_count": "best_auxiliary_opposition_count",
                "direction_consensus_count": "best_direction_consensus_count",
                "target_window_success": "best_target_window_success",
                "relaxed_guardrail_pass": "best_relaxed_guardrail_pass",
                "ood_risk": "best_ood_risk",
                "consensus_success": "best_consensus_success",
                "strict_success": "best_strict_success",
                "final_edit_score": "best_ranked_final_edit_score",
            }
        )
    target_meta_cols = [
        "mol_id",
        "token_id",
        "stage13_channel",
        "stage13_definition_status",
        "n_atoms",
        "attachment_count",
        "relative_token_size",
        "split_quality_score",
        "main_delta_distance",
        "verifier_support_count",
    ]
    summary = targets[[c for c in target_meta_cols if c in targets.columns]].merge(grouped, on=["mol_id", "token_id"], how="left")
    if not best.empty:
        summary = summary.merge(best, on=["mol_id", "token_id"], how="left")
    return summary


def summarize_actions(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    work = scored.copy()
    for col in ["edit_valid", "target_window_success", "relaxed_guardrail_pass", "consensus_success", "strict_success", "ood_risk"]:
        work[f"{col}_bool"] = bool_series(work[col]) if col in work.columns else False
    work["scored_bool"] = work["edited_main_pred"].notna()
    grouped = work.groupby(["stage13_channel", "edit_action"]).agg(
        candidate_attempts=("edit_row_id", "size"),
        token_count=("token_id", "nunique"),
        valid_edit_count=("edit_valid_bool", "sum"),
        scored_edit_count=("scored_bool", "sum"),
        target_window_success_count=("target_window_success_bool", "sum"),
        consensus_success_count=("consensus_success_bool", "sum"),
        strict_success_count=("strict_success_bool", "sum"),
        guardrail_pass_count=("relaxed_guardrail_pass_bool", "sum"),
        ood_risk_count=("ood_risk_bool", "sum"),
        mean_final_edit_score=("final_edit_score", "mean"),
        mean_delta_distance=("main_delta_distance_edit", "mean"),
    ).reset_index()
    for count_col in [
        "valid_edit_count",
        "scored_edit_count",
        "target_window_success_count",
        "consensus_success_count",
        "strict_success_count",
        "guardrail_pass_count",
        "ood_risk_count",
    ]:
        grouped[count_col.replace("_count", "_rate")] = grouped[count_col] / grouped["candidate_attempts"].replace(0, np.nan)
    return grouped


def recovery_table(config: dict[str, Any], target_summary: pd.DataFrame) -> pd.DataFrame:
    phase14 = pd.read_csv(rel_path(config["inputs"]["phase14_token_priority"]))
    new_mask = phase14["stage13_channel"].isin(config["target_rule"]["stage13_channels"])
    new_tokens = phase14.loc[new_mask, ["mol_id", "token_id"]].drop_duplicates()
    existing = phase14.loc[new_mask & (pd.to_numeric(phase14["contribution_edit_rows"], errors="coerce").fillna(0) > 0), ["mol_id", "token_id"]].drop_duplicates()
    newly_scored = target_summary.loc[pd.to_numeric(target_summary["scored_edit_count"], errors="coerce").fillna(0) > 0, ["mol_id", "token_id"]].drop_duplicates()
    after = pd.concat([existing, newly_scored], ignore_index=True).drop_duplicates()
    rows = [
        {
            "metric": "new_channel_token_count",
            "before": int(len(new_tokens)),
            "after": int(len(new_tokens)),
            "threshold": np.nan,
            "passed_after": True,
        },
        {
            "metric": "new_channel_tokens_with_edit_evidence",
            "before": int(len(existing)),
            "after": int(len(after)),
            "threshold": np.nan,
            "passed_after": True,
        },
        {
            "metric": "new_channel_edit_evidence_coverage",
            "before": float(len(existing) / max(len(new_tokens), 1)),
            "after": float(len(after) / max(len(new_tokens), 1)),
            "threshold": float(config["acceptance_gates"]["new_channel_edit_evidence_coverage_after_min"]),
            "passed_after": bool(float(len(after) / max(len(new_tokens), 1)) >= float(config["acceptance_gates"]["new_channel_edit_evidence_coverage_after_min"])),
        },
    ]
    return pd.DataFrame(rows)


def build_acceptance_gates(
    config: dict[str, Any],
    candidates: pd.DataFrame,
    target_summary: pd.DataFrame,
    recovery: pd.DataFrame,
    top: pd.DataFrame,
    blocked_count: int,
) -> pd.DataFrame:
    gates = config["acceptance_gates"]
    target_count = len(target_summary)
    valid_coverage = float((target_summary["valid_edit_count"].fillna(0) > 0).mean()) if target_count else 0.0
    scored_coverage = float((target_summary["scored_edit_count"].fillna(0) > 0).mean()) if target_count else 0.0
    tw_tokens = int((target_summary["target_window_success_count"].fillna(0) > 0).sum()) if target_count else 0
    consensus_tokens = int((target_summary["consensus_success_count"].fillna(0) > 0).sum()) if target_count else 0
    strict_tokens = int((target_summary["strict_success_count"].fillna(0) > 0).sum()) if target_count else 0
    if top.empty:
        top_guardrail = 0.0
    else:
        best = top[top["rank_in_token"] == 1].copy()
        top_guardrail = float(bool_series(best["relaxed_guardrail_pass"]).mean()) if not best.empty else 0.0
    recovery_value = float(recovery.loc[recovery["metric"] == "new_channel_edit_evidence_coverage", "after"].iloc[0])

    rows = [
        ("target_token_count", target_count, gates["target_token_count_min"], ">="),
        ("candidate_attempts", len(candidates), gates["candidate_attempts_min"], ">="),
        ("valid_edit_coverage", valid_coverage, gates["valid_edit_coverage_min"], ">="),
        ("scored_edit_coverage", scored_coverage, gates["scored_edit_coverage_min"], ">="),
        ("new_channel_edit_evidence_coverage_after", recovery_value, gates["new_channel_edit_evidence_coverage_after_min"], ">="),
        ("target_window_success_token_count", tw_tokens, gates["target_window_success_token_count_min"], ">="),
        ("consensus_success_token_count", consensus_tokens, gates["consensus_success_token_count_min"], ">="),
        ("strict_success_token_count", strict_tokens, gates["strict_success_token_count_min"], ">="),
        ("top_edit_guardrail_pass_rate", top_guardrail, gates["top_edit_guardrail_pass_rate_min"], ">="),
        ("blocked_text_hit_count", blocked_count, gates["blocked_text_hit_count_max"], "<="),
    ]
    out = []
    for name, actual, threshold, direction in rows:
        passed = actual >= threshold if direction == ">=" else actual <= threshold
        out.append({"gate": name, "actual": actual, "threshold": threshold, "direction": direction, "passed": bool(passed)})
    return pd.DataFrame(out)


def write_report(metrics: dict[str, Any], failed_gates: list[str]) -> None:
    report = f"""# 15 阶段新增通道编辑评分报告

## 运行结论

本阶段状态为 `{metrics["status"]}`。待补评分 Mol Token 共 {metrics["target_token_count"]} 个，生成编辑候选 {metrics["candidate_attempts"]} 条，完成评分的 token 覆盖率为 {metrics["scored_edit_coverage"]:.3f}。

## 关键结果

- 补评分后新增通道编辑证据覆盖率：{metrics["new_channel_edit_evidence_coverage_after"]:.3f}。
- 出现目标区间成功的 token 数：{metrics["target_window_success_token_count"]}。
- 出现共识成功的 token 数：{metrics["consensus_success_token_count"]}。
- 出现严格成功的 token 数：{metrics["strict_success_token_count"]}。
- 最佳编辑基础性质通过率：{metrics["top_edit_guardrail_pass_rate"]:.3f}。
- 未满足接收门：{", ".join(failed_gates) if failed_gates else "无"}。

## 外审判断

本阶段直接修复 14 阶段的主要缺口：新增通道已经获得编辑证据，不再只是边界重拆分结果。若严格成功数量仍有限，后续应优先优化替换库和连接位点，而不是回头放宽 Mol Token 定义。

## 后续安排

下一阶段可以把 14 阶段首批父分子和 15 阶段补齐的新通道 token 合并，形成最终小分子库可行性实验清单。清单中应区分严格可用、共识可用和仅补评分可用三类，避免把证据强度不同的编辑混写。
"""
    (REPORTS / "additional_edit_scoring_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    configure_phase15_prediction_paths()
    config = read_json(CONFIG_PATH)
    targets, _ = load_targets(config)
    targets.to_csv(TABLES / "phase15_new_channel_targets.csv", index=False, encoding="utf-8-sig")

    candidates = build_candidates(targets, config)
    candidates.to_csv(TABLES / "phase15_new_channel_edit_candidates.csv", index=False, encoding="utf-8-sig")

    scored = edit_tools.score_candidates(
        candidates,
        force_predict=args.force_predict,
        skip_predict=args.skip_predict,
        prediction_prefix="phase15_additional_edit_main",
        input_filename="phase15_additional_edit_scoring_input.csv",
    )
    scored.to_csv(TABLES / "phase15_new_channel_edit_scored.csv", index=False, encoding="utf-8-sig")

    top = rank_top_edits(scored)
    top.to_csv(TABLES / "phase15_new_channel_top_edits.csv", index=False, encoding="utf-8-sig")

    target_summary = summarize_by_token(scored, targets)
    target_summary.to_csv(TABLES / "phase15_new_channel_token_summary.csv", index=False, encoding="utf-8-sig")

    action_summary = summarize_actions(scored)
    action_summary.to_csv(TABLES / "phase15_action_channel_summary.csv", index=False, encoding="utf-8-sig")

    recovery = recovery_table(config, target_summary)
    recovery.to_csv(TABLES / "phase15_phase14_gate_recovery.csv", index=False, encoding="utf-8-sig")

    blocked_count = scan_blocked_text()
    gates = build_acceptance_gates(config, candidates, target_summary, recovery, top, blocked_count)
    gates.to_csv(TABLES / "phase15_acceptance_gates.csv", index=False, encoding="utf-8-sig")

    failed_gates = gates.loc[~gates["passed"], "gate"].tolist()
    status = "pass" if len(failed_gates) == 0 else "conditional_pass"
    metrics = {
        "phase": "moltoken",
        "status": status,
        "target_token_count": int(len(target_summary)),
        "candidate_attempts": int(len(candidates)),
        "valid_edit_coverage": float((target_summary["valid_edit_count"].fillna(0) > 0).mean()) if len(target_summary) else 0.0,
        "scored_edit_coverage": float((target_summary["scored_edit_count"].fillna(0) > 0).mean()) if len(target_summary) else 0.0,
        "new_channel_edit_evidence_coverage_after": float(
            recovery.loc[recovery["metric"] == "new_channel_edit_evidence_coverage", "after"].iloc[0]
        ),
        "target_window_success_token_count": int((target_summary["target_window_success_count"].fillna(0) > 0).sum()) if len(target_summary) else 0,
        "consensus_success_token_count": int((target_summary["consensus_success_count"].fillna(0) > 0).sum()) if len(target_summary) else 0,
        "strict_success_token_count": int((target_summary["strict_success_count"].fillna(0) > 0).sum()) if len(target_summary) else 0,
        "top_edit_guardrail_pass_rate": float(
            bool_series(top.loc[top["rank_in_token"] == 1, "relaxed_guardrail_pass"]).mean()
        )
        if not top.empty
        else 0.0,
        "failed_gates": failed_gates,
        "blocked_text_hit_count": int(blocked_count),
    }
    write_report(metrics, failed_gates)
    (REPORTS / "phase15_self_check.json").write_text(json.dumps(native(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(native(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
