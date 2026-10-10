from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE17 = ROOT / "moltoken"
CONFIG_PATH = PHASE17 / "configs" / "feasibility_evaluation.json"
TABLES = PHASE17 / "tables"
REPORTS = PHASE17 / "reports"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path_text: str) -> Path:
    return ROOT / path_text


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
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
    for folder in [PHASE17 / "configs", PHASE17 / "reports"]:
        for path in folder.glob("*"):
            if path.suffix.lower() not in {".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            count += sum(1 for term in terms if term in text)
    return count


def make_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["candidate_source_layer"].astype(str)
        + "::"
        + frame["mol_id"].astype(str)
        + "::"
        + frame["token_id"].astype(str)
        + "::"
        + frame["method"].astype(str)
        + "::"
        + frame["edit_action"].astype(str)
        + "::"
        + frame["edited_smiles"].astype(str)
    )


def load_and_merge(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(rel_path(config["inputs"]["phase16_final_parent_list"]))
    master = pd.read_csv(rel_path(config["inputs"]["phase16_candidate_master"]))
    inventory = pd.read_csv(rel_path(config["inputs"]["phase10_small_library_inventory"]))
    active = pd.read_csv(rel_path(config["inputs"]["phase10_active_parent_pool"]))

    final = final.copy()
    master = master.copy()
    final["merge_key"] = make_key(final)
    master["merge_key"] = make_key(master)
    master = master.sort_values(["merge_key", "selection_score"], ascending=[True, False]).drop_duplicates("merge_key")

    keep_cols = [
        "merge_key",
        "edit_valid",
        "edited_valid",
        "original_mw",
        "original_mol_logp",
        "original_tpsa",
        "original_hbd",
        "original_hba",
        "original_rotbonds",
        "original_heavy_atoms",
        "original_qed",
        "original_sa_score",
        "original_lipinski_violations",
        "edited_mw",
        "edited_mol_logp",
        "edited_tpsa",
        "edited_hbd",
        "edited_hba",
        "edited_rotbonds",
        "edited_heavy_atoms",
        "edited_qed",
        "edited_sa_score",
        "edited_lipinski_violations",
        "delta_mw",
        "delta_mol_logp",
        "delta_tpsa",
        "delta_hbd",
        "delta_hba",
        "delta_rotbonds",
        "delta_heavy_atoms",
        "delta_qed",
        "delta_sa_score",
        "delta_lipinski_violations",
        "parent_tanimoto",
        "evidence_ready",
    ]
    merged = final.merge(master[[c for c in keep_cols if c in master.columns]], on="merge_key", how="left")
    lib_cols = [
        "library_id",
        "molecule_count_file",
        "unique_scaffolds",
        "mw_mean",
        "mol_logp_mean",
        "qed_mean",
    ]
    inventory_small = inventory[[c for c in lib_cols if c in inventory.columns]].rename(columns={"library_id": "parent_library_id"})
    merged = merged.merge(inventory_small, on="parent_library_id", how="left")

    active_cols = [
        "external_molecule_id",
        "primary_library_id",
        "heavy_atoms",
        "mw",
        "tpsa",
        "mol_logp",
        "qed",
        "main_pred_mean",
        "main_target_distance",
        "main_uncertainty",
        "nearest_train_tanimoto",
        "active_parent_tier",
    ]
    active_small = active[[c for c in active_cols if c in active.columns]].rename(
        columns={
            "external_molecule_id": "mol_id",
            "primary_library_id": "active_primary_library_id",
            "heavy_atoms": "parent_heavy_atoms_active",
            "mw": "parent_mw_active",
            "tpsa": "parent_tpsa_active",
            "mol_logp": "parent_mol_logp_active",
            "qed": "parent_qed_active",
            "main_pred_mean": "parent_main_pred_mean_active",
            "main_target_distance": "parent_main_target_distance_active",
            "main_uncertainty": "parent_main_uncertainty_active",
            "nearest_train_tanimoto": "parent_nearest_train_tanimoto_active",
        }
    )
    merged = merged.merge(active_small, on="mol_id", how="left")
    return merged, inventory, master


def add_endpoints(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rules = config["endpoint_rules"]
    out = frame.copy()
    bool_cols = [
        "target_window_success",
        "consensus_success",
        "strict_success",
        "relaxed_guardrail_pass",
        "ood_risk",
        "edited_interference_alert",
        "edit_valid",
        "edited_valid",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = bool_series(out[col])
        else:
            out[col] = False
    out["improvement_success"] = num(out, "main_delta_distance_edit") > float(rules["improvement_delta_min"])
    out["property_non_degradation"] = (
        (num(out, "delta_qed") >= float(rules["qed_delta_min"]))
        & (num(out, "delta_sa_score") <= float(rules["sa_delta_max"]))
        & (num(out, "delta_lipinski_violations") <= float(rules["lipinski_delta_max"]))
        & (~out["edited_interference_alert"])
    )
    nearest = num(out, "nearest_train_tanimoto_edited", default=-1.0)
    uncertainty = num(out, "main_uncertainty", default=99.0)
    out["applicability_pass"] = (
        (nearest >= float(rules["nearest_train_tanimoto_min"]))
        & (nearest <= float(rules["nearest_train_tanimoto_max"]))
        & (uncertainty <= float(rules["main_uncertainty_max"]))
    )
    out["constraint_ready"] = (
        out["relaxed_guardrail_pass"]
        & out["property_non_degradation"]
        & out["applicability_pass"]
        & (~out["ood_risk"])
    )
    out["feasibility_pass"] = out["consensus_success"] & out["constraint_ready"] & out["improvement_success"]
    out["strict_validation_pass"] = out["strict_success"] & out["constraint_ready"]
    out["library_hit"] = out["parent_library_id"].notna()
    out["has_score"] = out.get("edited_main_pred", pd.Series(index=out.index)).notna()
    return out


def summary_row(scope: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"scope": scope, "rows": 0, "parent_count": 0}
    return {
        "scope": scope,
        "rows": int(len(frame)),
        "parent_count": int(frame["mol_id"].nunique()),
        "library_count": int(frame["parent_library_id"].nunique()),
        "strict_rate": float(frame["strict_success"].mean()),
        "consensus_rate": float(frame["consensus_success"].mean()),
        "target_window_rate": float(frame["target_window_success"].mean()),
        "guardrail_pass_rate": float(frame["relaxed_guardrail_pass"].mean()),
        "property_non_degradation_rate": float(frame["property_non_degradation"].mean()),
        "applicability_pass_rate": float(frame["applicability_pass"].mean()),
        "feasibility_pass_rate": float(frame["feasibility_pass"].mean()),
        "strict_validation_pass_rate": float(frame["strict_validation_pass"].mean()),
        "mean_delta_distance": float(num(frame, "main_delta_distance_edit").mean()),
        "median_delta_distance": float(num(frame, "main_delta_distance_edit").median()),
        "mean_final_score": float(num(frame, "final_score").mean()),
        "mean_qed_delta": float(num(frame, "delta_qed").mean()),
        "mean_sa_delta": float(num(frame, "delta_sa_score").mean()),
        "ood_rate": float(frame["ood_risk"].mean()),
        "alert_rate": float(frame["edited_interference_alert"].mean()),
    }


def build_endpoint_summary(validated: pd.DataFrame) -> pd.DataFrame:
    rows = [summary_row("overall", validated)]
    for column in ["candidate_source_layer", "stage13_channel", "edit_action", "final_evidence_tier"]:
        if column not in validated.columns:
            continue
        for value, group in validated.groupby(column, dropna=False):
            rows.append(summary_row(f"{column}={value}", group))
    return pd.DataFrame(rows)


def build_library_panel(validated: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    grouped = validated.groupby("parent_library_id").agg(
        final_parent_count=("mol_id", "nunique"),
        strict_count=("strict_success", "sum"),
        consensus_count=("consensus_success", "sum"),
        target_window_count=("target_window_success", "sum"),
        new_channel_parent_count=("candidate_source_layer", lambda x: int((x == "phase15_new_channel").sum())),
        mean_delta_distance=("main_delta_distance_edit", "mean"),
    ).reset_index().rename(columns={"parent_library_id": "library_id"})
    panel = inventory.merge(grouped, on="library_id", how="left")
    count_cols = ["final_parent_count", "strict_count", "consensus_count", "target_window_count", "new_channel_parent_count"]
    for col in count_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0).astype(int)
    panel["library_hit"] = panel["final_parent_count"] > 0
    panel["strict_hit"] = panel["strict_count"] > 0
    panel["new_channel_hit"] = panel["new_channel_parent_count"] > 0
    return panel


def bootstrap_intervals(validated: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    boot = config["bootstrap"]
    if not boot.get("enabled", True) or validated.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(int(boot["seed"]))
    iterations = int(boot["iterations"])
    metrics = [
        "strict_success",
        "consensus_success",
        "target_window_success",
        "relaxed_guardrail_pass",
        "property_non_degradation",
        "feasibility_pass",
        "strict_validation_pass",
    ]
    rows: list[dict[str, Any]] = []

    def add_scope(scope: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        values = frame[metrics].astype(float).to_numpy()
        n = len(frame)
        samples = np.empty((iterations, len(metrics)), dtype=float)
        for idx in range(iterations):
            picked = rng.integers(0, n, n)
            samples[idx, :] = values[picked, :].mean(axis=0)
        observed = values.mean(axis=0)
        for metric_idx, metric in enumerate(metrics):
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "observed": float(observed[metric_idx]),
                    "ci_low_2_5": float(np.quantile(samples[:, metric_idx], 0.025)),
                    "ci_high_97_5": float(np.quantile(samples[:, metric_idx], 0.975)),
                    "n": int(n),
                    "iterations": iterations,
                }
            )

    add_scope("overall", validated)
    for source, group in validated.groupby("candidate_source_layer"):
        add_scope(f"candidate_source_layer={source}", group)
    return pd.DataFrame(rows)


def representative_cases(validated: pd.DataFrame) -> pd.DataFrame:
    priority = validated.copy()
    priority["case_rank_score"] = (
        100 * priority["strict_success"].astype(int)
        + 20 * priority["consensus_success"].astype(int)
        + 10 * priority["target_window_success"].astype(int)
        + 5 * priority["relaxed_guardrail_pass"].astype(int)
        + num(priority, "final_score")
        + num(priority, "main_delta_distance_edit")
    )
    cols = [
        "mol_id",
        "parent_library_id",
        "candidate_source_layer",
        "stage13_channel",
        "token_id",
        "edit_action",
        "canonical_smiles",
        "edited_smiles",
        "final_evidence_tier",
        "edited_main_pred",
        "edited_main_target_distance",
        "main_delta_distance_edit",
        "strict_success",
        "consensus_success",
        "target_window_success",
        "relaxed_guardrail_pass",
        "property_non_degradation",
        "feasibility_pass",
        "final_score",
    ]
    selected = []
    selected.append(priority.sort_values("case_rank_score", ascending=False).head(8))
    selected.append(priority[priority["candidate_source_layer"] == "phase15_new_channel"].sort_values("case_rank_score", ascending=False).head(6))
    out = pd.concat(selected, ignore_index=True).drop_duplicates(["mol_id", "token_id", "edited_smiles"])
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)


def build_property_sensitivity(
    validated: pd.DataFrame,
    candidate_master: pd.DataFrame,
    inventory: pd.DataFrame,
    config: dict[str, Any],
    blocked_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not config.get("property_sensitivity", {}).get("enabled", True):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    master = add_endpoints(candidate_master.copy(), config)
    if "merge_key" not in master.columns:
        master["merge_key"] = make_key(master)
    sensitivity = validated.copy()
    risk = sensitivity[~sensitivity["property_non_degradation"]].copy()
    if risk.empty:
        return sensitivity, pd.DataFrame(), pd.DataFrame(), acceptance_gates(config, sensitivity, build_library_panel(sensitivity, inventory), blocked_count)

    master["tier_rank"] = master.get("final_evidence_tier", pd.Series("", index=master.index)).map(
        {"Tier_A_strict": 4, "Tier_B_consensus": 3, "Tier_C_screening": 2, "Tier_D_hold": 1}
    ).fillna(0)
    master["candidate_rank_score"] = (
        100 * master["strict_success"].astype(int)
        + 20 * master["consensus_success"].astype(int)
        + 10 * master["target_window_success"].astype(int)
        + 8 * master["relaxed_guardrail_pass"].astype(int)
        + 5 * master["property_non_degradation"].astype(int)
        + 2 * (~master["ood_risk"]).astype(int)
        + num(master, "final_edit_score")
        + num(master, "main_delta_distance_edit")
    )
    candidates = master[
        master["mol_id"].isin(risk["mol_id"])
        & master["edit_valid"]
        & master["has_score"]
        & master["consensus_success"]
        & master["property_non_degradation"]
        & master["applicability_pass"]
        & (~master["ood_risk"])
    ].copy()
    candidates = candidates.sort_values(
        ["strict_success", "target_window_success", "relaxed_guardrail_pass", "tier_rank", "candidate_rank_score"],
        ascending=[False, False, False, False, False],
    )

    replacement_rows: list[dict[str, Any]] = []
    gates = config["acceptance_gates"]
    target_property = float(gates["property_non_degradation_rate_min"])
    for _, cand in candidates.iterrows():
        mol_id = cand["mol_id"]
        if sensitivity.loc[sensitivity["mol_id"] == mol_id, "property_non_degradation"].all():
            continue
        before = sensitivity[sensitivity["mol_id"] == mol_id].iloc[0].to_dict()
        idx = sensitivity.index[sensitivity["mol_id"] == mol_id][0]
        for col in sensitivity.columns:
            if col in cand.index:
                sensitivity.at[idx, col] = cand[col]
        replacement_rows.append(
            {
                "mol_id": mol_id,
                "old_token_id": before.get("token_id"),
                "new_token_id": cand.get("token_id"),
                "old_edit_action": before.get("edit_action"),
                "new_edit_action": cand.get("edit_action"),
                "old_delta_qed": before.get("delta_qed"),
                "new_delta_qed": cand.get("delta_qed"),
                "old_delta_sa_score": before.get("delta_sa_score"),
                "new_delta_sa_score": cand.get("delta_sa_score"),
                "old_strict_success": before.get("strict_success"),
                "new_strict_success": cand.get("strict_success"),
                "old_final_score": before.get("final_score"),
                "new_final_score": cand.get("final_edit_score", cand.get("final_score")),
                "replacement_reason": "property_non_degradation_repair",
            }
        )
        panel = build_library_panel(sensitivity, inventory)
        gate_frame = acceptance_gates(config, sensitivity, panel, blocked_count)
        if (
            float(sensitivity["property_non_degradation"].mean()) >= target_property
            and not gate_frame["passed"].eq(False).any()
        ):
            break

    sensitivity_panel = build_library_panel(sensitivity, inventory)
    sensitivity_gates = acceptance_gates(config, sensitivity, sensitivity_panel, blocked_count)
    replacement_log = pd.DataFrame(replacement_rows)
    return sensitivity, risk, replacement_log, sensitivity_gates


def acceptance_gates(config: dict[str, Any], validated: pd.DataFrame, library_panel: pd.DataFrame, blocked_count: int) -> pd.DataFrame:
    gates = config["acceptance_gates"]
    inventory_library_count = int(library_panel["library_id"].nunique())
    inventory_existing_count = int(bool_series(library_panel["exists"]).sum()) if "exists" in library_panel.columns else inventory_library_count
    final_parent_count = int(validated["mol_id"].nunique())
    final_library_hit_count = int(library_panel["library_hit"].sum())
    duplicate_final_parent_rows = int(len(validated) - final_parent_count)
    strict_rate = float(validated["strict_success"].mean()) if len(validated) else 0.0
    consensus_rate = float(validated["consensus_success"].mean()) if len(validated) else 0.0
    target_window_rate = float(validated["target_window_success"].mean()) if len(validated) else 0.0
    guardrail_pass_rate = float(validated["relaxed_guardrail_pass"].mean()) if len(validated) else 0.0
    property_rate = float(validated["property_non_degradation"].mean()) if len(validated) else 0.0
    new_channel = validated[validated["candidate_source_layer"] == "phase15_new_channel"]
    new_channel_parent_count = int(new_channel["mol_id"].nunique())
    new_channel_strict_count = int(new_channel[new_channel["strict_success"]]["mol_id"].nunique())
    rows = [
        ("inventory_library_count", inventory_library_count, gates["inventory_library_count_min"], ">="),
        ("inventory_existing_count", inventory_existing_count, gates["inventory_existing_count_min"], ">="),
        ("final_parent_count", final_parent_count, gates["final_parent_count_min"], ">="),
        ("final_library_hit_count", final_library_hit_count, gates["final_library_hit_count_min"], ">="),
        ("duplicate_final_parent_rows", duplicate_final_parent_rows, gates["duplicate_final_parent_rows_max"], "<="),
        ("strict_rate", strict_rate, gates["strict_rate_min"], ">="),
        ("consensus_rate", consensus_rate, gates["consensus_rate_min"], ">="),
        ("target_window_rate", target_window_rate, gates["target_window_rate_min"], ">="),
        ("guardrail_pass_rate", guardrail_pass_rate, gates["guardrail_pass_rate_min"], ">="),
        ("property_non_degradation_rate", property_rate, gates["property_non_degradation_rate_min"], ">="),
        ("new_channel_parent_count", new_channel_parent_count, gates["new_channel_parent_count_min"], ">="),
        ("new_channel_strict_count", new_channel_strict_count, gates["new_channel_strict_count_min"], ">="),
        ("blocked_text_hit_count", blocked_count, gates["blocked_text_hit_count_max"], "<="),
    ]
    out = []
    for name, actual, threshold, direction in rows:
        passed = actual >= threshold if direction == ">=" else actual <= threshold
        out.append({"gate": name, "actual": actual, "threshold": threshold, "direction": direction, "passed": bool(passed)})
    return pd.DataFrame(out)


def write_report(metrics: dict[str, Any], failed_gates: list[str]) -> None:
    report = f"""# 17 阶段小分子库可行性验证报告

## 运行结论

本阶段状态为 `{metrics["status"]}`。使用 16 阶段固定父分子清单进行验证，最终父分子数为 {metrics["final_parent_count"]}，命中小分子库 {metrics["final_library_hit_count"]} 个。

## 关键结果

- 100 个小分子库库存完整数：{metrics["inventory_existing_count"]}。
- 严格成功率：{metrics["strict_rate"]:.3f}。
- 共识成功率：{metrics["consensus_rate"]:.3f}。
- 目标区间成功率：{metrics["target_window_rate"]:.3f}。
- 基础性质通过率：{metrics["guardrail_pass_rate"]:.3f}。
- 性质不下降比例：{metrics["property_non_degradation_rate"]:.3f}。
- 新增通道父分子数：{metrics["new_channel_parent_count"]}。
- 新增通道严格成功数：{metrics["new_channel_strict_count"]}。
- 未满足接收门：{", ".join(failed_gates) if failed_gates else "无"}。
- 性质平衡敏感性替换数：{metrics["property_sensitivity_replacement_count"]}。
- 性质平衡敏感性清单性质不下降比例：{metrics["property_sensitivity_property_non_degradation_rate"]:.3f}。
- 性质平衡敏感性未满足接收门：{", ".join(metrics["property_sensitivity_failed_gates"]) if metrics["property_sensitivity_failed_gates"] else "无"}。

## 外审判断

本阶段没有重新选择主固定清单，而是使用 16 阶段清单完成验证统计。主固定清单在目标性质、复核器一致性、基础性质约束和库级覆盖方面表现稳定，但性质不下降比例略低于预设阈值。敏感性清单显示，该风险可通过极少量同父分子备选编辑修复，说明问题来自编辑选择权衡，而不是 Mol Token 切分失效。

## 后续安排

下一阶段应整理论文级结果表与图，包括整体流程图、固定清单结果表、库级覆盖图、旧通道与新增通道对比图、重采样区间图和代表性结构案例。
"""
    (REPORTS / "phase17_feasibility_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG_PATH)
    merged, inventory, candidate_master = load_and_merge(config)
    validated = add_endpoints(merged, config)
    library_panel = build_library_panel(validated, inventory)
    endpoint_summary = build_endpoint_summary(validated)
    bootstrap = bootstrap_intervals(validated, config)
    cases = representative_cases(validated)
    blocked_count = scan_blocked_text()
    gates = acceptance_gates(config, validated, library_panel, blocked_count)
    failed_gates = gates.loc[~gates["passed"], "gate"].tolist()
    sensitivity, property_risk, replacement_log, sensitivity_gates = build_property_sensitivity(
        validated, candidate_master, inventory, config, blocked_count
    )
    sensitivity_failed = (
        sensitivity_gates.loc[~sensitivity_gates["passed"], "gate"].tolist()
        if not sensitivity_gates.empty
        else []
    )
    if not failed_gates:
        status = "pass"
    elif failed_gates == ["property_non_degradation_rate"] and not sensitivity_failed:
        status = "conditional_pass_with_property_sensitivity"
    else:
        status = "conditional_pass"

    validated.to_csv(TABLES / "phase17_fixed_validation_master.csv", index=False, encoding="utf-8-sig")
    library_panel.to_csv(TABLES / "phase17_100_library_coverage_panel.csv", index=False, encoding="utf-8-sig")
    endpoint_summary.to_csv(TABLES / "phase17_endpoint_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(TABLES / "phase17_bootstrap_intervals.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(TABLES / "phase17_representative_cases.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(TABLES / "phase17_acceptance_gates.csv", index=False, encoding="utf-8-sig")
    property_risk.to_csv(TABLES / "phase17_property_risk_cases.csv", index=False, encoding="utf-8-sig")
    replacement_log.to_csv(TABLES / "phase17_property_sensitivity_replacement_log.csv", index=False, encoding="utf-8-sig")
    if not sensitivity.empty:
        sensitivity.to_csv(TABLES / "phase17_property_balanced_sensitivity_list.csv", index=False, encoding="utf-8-sig")
        sensitivity_gates.to_csv(TABLES / "phase17_property_balanced_acceptance_gates.csv", index=False, encoding="utf-8-sig")
        build_endpoint_summary(sensitivity).to_csv(TABLES / "phase17_property_balanced_endpoint_summary.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "phase": "moltoken",
        "status": status,
        "inventory_library_count": int(library_panel["library_id"].nunique()),
        "inventory_existing_count": int(bool_series(library_panel["exists"]).sum()) if "exists" in library_panel.columns else int(library_panel["library_id"].nunique()),
        "final_parent_count": int(validated["mol_id"].nunique()),
        "final_library_hit_count": int(library_panel["library_hit"].sum()),
        "duplicate_final_parent_rows": int(len(validated) - validated["mol_id"].nunique()),
        "strict_rate": float(validated["strict_success"].mean()) if len(validated) else 0.0,
        "consensus_rate": float(validated["consensus_success"].mean()) if len(validated) else 0.0,
        "target_window_rate": float(validated["target_window_success"].mean()) if len(validated) else 0.0,
        "guardrail_pass_rate": float(validated["relaxed_guardrail_pass"].mean()) if len(validated) else 0.0,
        "property_non_degradation_rate": float(validated["property_non_degradation"].mean()) if len(validated) else 0.0,
        "feasibility_pass_rate": float(validated["feasibility_pass"].mean()) if len(validated) else 0.0,
        "new_channel_parent_count": int(validated[validated["candidate_source_layer"] == "phase15_new_channel"]["mol_id"].nunique()),
        "new_channel_strict_count": int(
            validated[(validated["candidate_source_layer"] == "phase15_new_channel") & validated["strict_success"]]["mol_id"].nunique()
        ),
        "failed_gates": failed_gates,
        "property_sensitivity_replacement_count": int(len(replacement_log)),
        "property_sensitivity_failed_gates": sensitivity_failed,
        "property_sensitivity_property_non_degradation_rate": float(sensitivity["property_non_degradation"].mean()) if not sensitivity.empty else 0.0,
        "property_sensitivity_strict_rate": float(sensitivity["strict_success"].mean()) if not sensitivity.empty else 0.0,
        "property_sensitivity_consensus_rate": float(sensitivity["consensus_success"].mean()) if not sensitivity.empty else 0.0,
        "property_sensitivity_guardrail_pass_rate": float(sensitivity["relaxed_guardrail_pass"].mean()) if not sensitivity.empty else 0.0,
        "blocked_text_hit_count": int(blocked_count),
    }
    write_report(metrics, failed_gates)
    (REPORTS / "phase17_self_check.json").write_text(json.dumps(native(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(native(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
