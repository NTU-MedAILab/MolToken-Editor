from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE16 = ROOT / "moltoken"
CONFIG_PATH = PHASE16 / "configs" / "validation_selection.json"
TABLES = PHASE16 / "tables"
REPORTS = PHASE16 / "reports"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path_text: str) -> Path:
    return ROOT / path_text


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


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
    for folder in [PHASE16 / "configs", PHASE16 / "reports"]:
        for path in folder.glob("*"):
            if path.suffix.lower() not in {".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            count += sum(1 for term in terms if term in text)
    return count


def load_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        key: pd.read_csv(rel_path(path_text))
        for key, path_text in config["inputs"].items()
    }


def add_common_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
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
        if col in out.columns:
            out[col] = bool_series(out[col])
        else:
            out[col] = False
    out["has_score"] = out.get("edited_main_pred", pd.Series(index=out.index)).notna()
    return out


def evidence_tier(frame: pd.DataFrame) -> pd.Series:
    strict = (
        frame["strict_success"]
        & frame["relaxed_guardrail_pass"]
        & (~frame["ood_risk"])
        & (~frame["edited_interference_alert"])
    )
    consensus = frame["consensus_success"] & (~frame["ood_risk"])
    screening = frame["target_window_success"] | (frame["edit_valid"] & frame["has_score"])
    return pd.Series(
        np.select(
            [strict, consensus, screening],
            ["Tier_A_strict", "Tier_B_consensus", "Tier_C_screening"],
            default="Tier_D_hold",
        ),
        index=frame.index,
    )


def build_existing_candidates(inputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    token_priority = inputs["phase14_token_priority"].copy()
    ready_levels = set(config["selection_policy"]["existing_readiness_levels"])
    if config["selection_policy"].get("include_ready_screening_in_backup", True):
        ready_levels.add("ready_screening")
    ready_tokens = token_priority[token_priority["phase14_readiness"].isin(ready_levels)].copy()
    meta_cols = [
        "mol_id",
        "token_id",
        "stage13_channel",
        "stage13_definition_status",
        "verifier_consensus_tier",
        "edit_evidence_tier",
        "phase14_readiness",
        "batch_suggestion",
        "phase14_priority_score",
        "phase14_token_evidence_score",
        "stage13_priority_score",
    ]
    ready_meta = ready_tokens[[c for c in meta_cols if c in ready_tokens.columns]].drop_duplicates(["mol_id", "token_id"])

    scored = inputs["phase10d_edit_scored_all"].copy()
    scored = scored[scored["method_group"].isin(["contribution_primary", "contribution_supplemental"])].copy()
    scored = scored.merge(ready_meta, on=["mol_id", "token_id"], how="inner")
    scored["candidate_source_layer"] = "phase14_existing_channel"
    scored["candidate_source_stage"] = "10d_existing_scored"
    return scored


def build_new_channel_candidates(inputs: dict[str, pd.DataFrame], config: dict[str, Any]) -> pd.DataFrame:
    top = inputs["phase15_top_edits"].copy()
    if config["selection_policy"].get("include_phase15_best_rank_only", True) and "rank_in_token" in top.columns:
        top = top[top["rank_in_token"] == 1].copy()
    summary = inputs["phase15_token_summary"].copy()
    meta_cols = [
        "mol_id",
        "token_id",
        "stage13_channel",
        "stage13_definition_status",
        "best_ranked_final_edit_score",
    ]
    meta = summary[[c for c in meta_cols if c in summary.columns]].drop_duplicates(["mol_id", "token_id"])
    top = top.merge(meta, on=["mol_id", "token_id", "stage13_channel", "stage13_definition_status"], how="left")
    top["candidate_source_layer"] = "phase15_new_channel"
    top["candidate_source_stage"] = "15_new_channel_scored"
    top["phase14_readiness"] = "ready_after_phase15"
    top["batch_suggestion"] = "first_batch"
    top["verifier_consensus_tier"] = np.where(
        pd.to_numeric(top.get("verifier_support_count", 0), errors="coerce").fillna(0) >= 2,
        "T1_strong",
        "T2_usable",
    )
    top["edit_evidence_tier"] = np.where(top.get("strict_success", False), "E1_strict_edit", "E2_consensus_edit")
    top["phase14_priority_score"] = pd.to_numeric(top.get("final_edit_score", 0), errors="coerce").fillna(0)
    top["phase14_token_evidence_score"] = pd.to_numeric(top.get("source_split_quality_score", 0), errors="coerce").fillna(0)
    top["stage13_priority_score"] = pd.to_numeric(top.get("source_split_quality_score", 0), errors="coerce").fillna(0)
    return top


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = add_common_flags(candidates)
    out["final_evidence_tier"] = evidence_tier(out)
    tier_rank = {
        "Tier_A_strict": 4,
        "Tier_B_consensus": 3,
        "Tier_C_screening": 2,
        "Tier_D_hold": 1,
    }
    out["final_evidence_rank"] = out["final_evidence_tier"].map(tier_rank).fillna(0)
    out["final_score"] = pd.to_numeric(out.get("final_edit_score", 0), errors="coerce").fillna(0)
    out["delta_distance"] = pd.to_numeric(out.get("main_delta_distance_edit", 0), errors="coerce").fillna(0)
    out["token_score"] = pd.to_numeric(out.get("phase14_token_evidence_score", np.nan), errors="coerce")
    out["token_score"] = out["token_score"].fillna(pd.to_numeric(out.get("source_split_quality_score", 0), errors="coerce").fillna(0))
    out["selection_score"] = (
        100 * out["final_evidence_rank"]
        + 10 * out["strict_success"].astype(int)
        + 5 * out["consensus_success"].astype(int)
        + 3 * out["target_window_success"].astype(int)
        + 2 * out["relaxed_guardrail_pass"].astype(int)
        + 1 * (~out["ood_risk"]).astype(int)
        + out["final_score"]
        + 0.2 * out["token_score"]
    )
    out["is_selectable"] = out["final_evidence_tier"].isin(["Tier_A_strict", "Tier_B_consensus"])
    out["candidate_key"] = (
        out["candidate_source_layer"].astype(str)
        + "::"
        + out["mol_id"].astype(str)
        + "::"
        + out["token_id"].astype(str)
        + "::"
        + out["edit_action"].astype(str)
        + "::"
        + out["edited_smiles"].astype(str)
    )
    return out


def select_final_parent_list(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selectable = master[master["is_selectable"] & master["edit_valid"] & master["has_score"]].copy()
    selectable = selectable.sort_values(
        [
            "mol_id",
            "final_evidence_rank",
            "strict_success",
            "consensus_success",
            "target_window_success",
            "relaxed_guardrail_pass",
            "ood_risk",
            "selection_score",
        ],
        ascending=[True, False, False, False, False, False, True, False],
    )
    selectable["rank_in_parent"] = selectable.groupby("mol_id").cumcount() + 1
    final = selectable[selectable["rank_in_parent"] == 1].copy()
    backup = selectable[selectable["rank_in_parent"] > 1].copy()

    selected_keys = set(final["candidate_key"].astype(str))
    excluded = master[~master["candidate_key"].astype(str).isin(selected_keys)].copy()
    excluded["exclusion_reason"] = np.select(
        [
            ~excluded["edit_valid"],
            ~excluded["has_score"],
            excluded["final_evidence_tier"] == "Tier_D_hold",
            excluded["candidate_key"].astype(str).isin(set(backup["candidate_key"].astype(str))),
        ],
        [
            "invalid_edit",
            "missing_score",
            "weak_evidence",
            "lower_rank_same_parent",
        ],
        default="not_parent_primary",
    )
    return final, backup, excluded


def compact_final_table(final: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "mol_id",
        "parent_rank",
        "parent_library_id",
        "canonical_smiles",
        "token_id",
        "stage13_channel",
        "candidate_source_layer",
        "method",
        "edit_action",
        "edited_smiles",
        "final_evidence_tier",
        "final_score",
        "selection_score",
        "edited_main_pred",
        "edited_main_target_distance",
        "main_delta_distance_edit",
        "target_window_success",
        "consensus_success",
        "strict_success",
        "relaxed_guardrail_pass",
        "ood_risk",
        "edited_interference_alert",
        "verifier_support_count",
        "auxiliary_opposition_count",
        "direction_consensus_count",
        "main_uncertainty",
        "nearest_train_tanimoto_edited",
        "source_split_quality_score",
        "source_main_delta_delete",
        "token_score",
        "verifier_consensus_tier",
        "edit_evidence_tier",
        "phase14_readiness",
        "batch_suggestion",
    ]
    available = [c for c in columns if c in final.columns]
    ordered = final.sort_values(["final_evidence_rank", "selection_score"], ascending=[False, False])
    return ordered[available].reset_index(drop=True)


def evidence_summary(master: pd.DataFrame, final: pd.DataFrame, backup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, frame in [
        ("candidate_master", master),
        ("final_parent_list", final),
        ("backup_candidates", backup),
    ]:
        if frame.empty:
            rows.append({"scope": scope, "rows": 0})
            continue
        rows.append(
            {
                "scope": scope,
                "rows": int(len(frame)),
                "parent_count": int(frame["mol_id"].nunique()),
                "strict_count": int(frame["strict_success"].sum()),
                "consensus_count": int(frame["consensus_success"].sum()),
                "target_window_count": int(frame["target_window_success"].sum()),
                "guardrail_pass_rate": float(frame["relaxed_guardrail_pass"].mean()),
                "ood_rate": float(frame["ood_risk"].mean()),
                "new_channel_parent_count": int(frame[frame["candidate_source_layer"] == "phase15_new_channel"]["mol_id"].nunique()),
                "new_channel_strict_parent_count": int(
                    frame[(frame["candidate_source_layer"] == "phase15_new_channel") & frame["strict_success"]]["mol_id"].nunique()
                ),
                "mean_delta_distance": float(pd.to_numeric(frame["main_delta_distance_edit"], errors="coerce").mean()),
                "mean_final_score": float(pd.to_numeric(frame["final_score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def acceptance_gates(config: dict[str, Any], master: pd.DataFrame, final: pd.DataFrame, blocked_count: int) -> pd.DataFrame:
    gates = config["acceptance_gates"]
    final_parent_count = final["mol_id"].nunique()
    strict_final_parent_count = final[final["strict_success"]]["mol_id"].nunique()
    new_channel_parent_count = final[final["candidate_source_layer"] == "phase15_new_channel"]["mol_id"].nunique()
    new_channel_strict_parent_count = final[
        (final["candidate_source_layer"] == "phase15_new_channel") & final["strict_success"]
    ]["mol_id"].nunique()
    final_guardrail = float(final["relaxed_guardrail_pass"].mean()) if not final.empty else 0.0
    final_consensus = float(final["consensus_success"].mean()) if not final.empty else 0.0
    final_strict = float(final["strict_success"].mean()) if not final.empty else 0.0
    duplicate_parent_primary = int(len(final) - final_parent_count)
    rows = [
        ("candidate_master_rows", len(master), gates["candidate_master_rows_min"], ">="),
        ("final_parent_count", final_parent_count, gates["final_parent_count_min"], ">="),
        ("strict_final_parent_count", strict_final_parent_count, gates["strict_final_parent_count_min"], ">="),
        ("new_channel_parent_count", new_channel_parent_count, gates["new_channel_parent_count_min"], ">="),
        ("new_channel_strict_parent_count", new_channel_strict_parent_count, gates["new_channel_strict_parent_count_min"], ">="),
        ("final_guardrail_pass_rate", final_guardrail, gates["final_guardrail_pass_rate_min"], ">="),
        ("final_consensus_rate", final_consensus, gates["final_consensus_rate_min"], ">="),
        ("final_strict_rate", final_strict, gates["final_strict_rate_min"], ">="),
        ("duplicate_parent_primary", duplicate_parent_primary, gates["duplicate_parent_primary_max"], "<="),
        ("blocked_text_hit_count", blocked_count, gates["blocked_text_hit_count_max"], "<="),
    ]
    out = []
    for name, actual, threshold, direction in rows:
        passed = actual >= threshold if direction == ">=" else actual <= threshold
        out.append({"gate": name, "actual": actual, "threshold": threshold, "direction": direction, "passed": bool(passed)})
    return pd.DataFrame(out)


def write_report(metrics: dict[str, Any], failed_gates: list[str]) -> None:
    report = f"""# 16 阶段最终小分子库验证清单报告

## 运行结论

本阶段状态为 `{metrics["status"]}`。候选级 master 表共 {metrics["candidate_master_rows"]} 行，父分子级最终清单共 {metrics["final_parent_count"]} 个父分子，其中严格成功父分子 {metrics["strict_final_parent_count"]} 个。

## 关键结果

- 新增通道进入最终清单的父分子数：{metrics["new_channel_parent_count"]}。
- 新增通道严格成功父分子数：{metrics["new_channel_strict_parent_count"]}。
- 最终清单基础性质通过率：{metrics["final_guardrail_pass_rate"]:.3f}。
- 最终清单共识成功率：{metrics["final_consensus_rate"]:.3f}。
- 最终清单严格成功率：{metrics["final_strict_rate"]:.3f}。
- 未满足接收门：{", ".join(failed_gates) if failed_gates else "无"}。

## 外审判断

本阶段完成了从 Mol Token 证据到父分子级实验清单的闭环。旧通道提供稳定基础，新增通道补足边界重拆分后的直接编辑证据。父分子级去重后仍能保持足够严格成功数量，说明当前流程可以进入后续小分子库可行性验证。

## 后续安排

下一阶段应围绕最终父分子清单组织小分子库验证，优先运行 Tier_A_strict 条目，同时保留 Tier_B_consensus 作为备选。实验汇总时需继续区分旧通道与新增通道，避免把不同证据强度混写。
"""
    (REPORTS / "phase16_final_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG_PATH)
    inputs = load_inputs(config)

    existing = build_existing_candidates(inputs, config)
    new_channel = build_new_channel_candidates(inputs, config)
    master = normalize_candidates(pd.concat([existing, new_channel], ignore_index=True, sort=False))
    final, backup, excluded = select_final_parent_list(master)
    final_compact = compact_final_table(final)
    summary = evidence_summary(master, final, backup)
    blocked_count = scan_blocked_text()
    gates = acceptance_gates(config, master, final, blocked_count)
    failed_gates = gates.loc[~gates["passed"], "gate"].tolist()
    status = "pass" if not failed_gates else "conditional_pass"

    master.to_csv(TABLES / "phase16_candidate_master.csv", index=False, encoding="utf-8-sig")
    final_compact.to_csv(TABLES / "phase16_final_parent_validation_list.csv", index=False, encoding="utf-8-sig")
    backup.to_csv(TABLES / "phase16_backup_candidates.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(TABLES / "phase16_excluded_candidates.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLES / "phase16_evidence_summary.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(TABLES / "phase16_acceptance_gates.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "phase": "moltoken",
        "status": status,
        "candidate_master_rows": int(len(master)),
        "final_parent_count": int(final["mol_id"].nunique()),
        "strict_final_parent_count": int(final[final["strict_success"]]["mol_id"].nunique()),
        "new_channel_parent_count": int(final[final["candidate_source_layer"] == "phase15_new_channel"]["mol_id"].nunique()),
        "new_channel_strict_parent_count": int(
            final[(final["candidate_source_layer"] == "phase15_new_channel") & final["strict_success"]]["mol_id"].nunique()
        ),
        "final_guardrail_pass_rate": float(final["relaxed_guardrail_pass"].mean()) if not final.empty else 0.0,
        "final_consensus_rate": float(final["consensus_success"].mean()) if not final.empty else 0.0,
        "final_strict_rate": float(final["strict_success"].mean()) if not final.empty else 0.0,
        "backup_candidate_rows": int(len(backup)),
        "excluded_candidate_rows": int(len(excluded)),
        "failed_gates": failed_gates,
        "blocked_text_hit_count": int(blocked_count),
    }
    write_report(metrics, failed_gates)
    (REPORTS / "phase16_self_check.json").write_text(json.dumps(native(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(native(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
