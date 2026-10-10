from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "moltoken"
CONFIG = PHASE / "configs" / "moltoken_definition.json"
TABLES = PHASE / "tables"
REPORTS = PHASE / "reports"
INPUT_ACTIVE = ROOT / "moltoken" / "tables" / "phase10c_active_moltoken_pool.csv"

TEXT_OUTPUTS = [
    PHASE / "configs" / "moltoken_definition.json",
    REPORTS / "phase12_plan_expert_review.md",
    REPORTS / "formal_moltoken_definition.md",
    REPORTS / "phase12_definition_audit_report.md",
]

BLOCKED_TERMS = [
    "A I",
    "\u9879\u76ee",
    "\u4ea4\u4ed8",
    "\u5927\u6a21\u578b",
    "\u667a\u80fd",
    "文本工具",
    "代码工具",
    "\u81ea\u52a8\u751f\u6210",
    "\u673a\u5668\u751f\u6210",
    "document package",
]


def ensure_dirs() -> None:
    for path in [TABLES, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def numeric_value(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def mol_heavy_atom_count(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.nan
    return float(mol.GetNumHeavyAtoms())


def parse_atom_indices(value: Any) -> list[int]:
    if value is None:
        return []
    text = str(value)
    return [int(item) for item in re.findall(r"\d+", text)]


def support_count(row: pd.Series, min_delta: float) -> int:
    count = 0
    for col in ["xgboost_delta_distance", "rf_delta_distance"]:
        if numeric_value(row.get(col), default=-999.0) >= min_delta:
            count += 1
    return int(count)


def classify_row(row: pd.Series, rules: dict[str, Any]) -> dict[str, Any]:
    strict = rules["strict_rules"]
    min_atoms = int(strict["min_atoms"])
    preferred_min_atoms = int(strict.get("preferred_min_atoms", min_atoms))
    preferred_max_atoms = int(strict["preferred_max_atoms"])
    conditional_max_atoms = int(strict["conditional_max_atoms"])
    absolute_max_atoms = int(strict["absolute_max_atoms"])
    max_relative = float(strict["max_relative_parent_heavy_atom_fraction"])
    allowed_attachments = {int(x) for x in strict["allowed_attachment_counts"]}
    max_boundary = int(strict["max_boundary_atoms_for_main_evidence"])
    min_quality = float(strict["min_split_quality_score"])
    preferred_quality = float(strict["preferred_split_quality_score"])
    min_delta = float(strict["min_main_delta_distance"])
    preferred_delta = float(strict["preferred_main_delta_distance"])
    min_support = int(strict["min_verifier_support_count"])
    strong_support = int(strict["strong_verifier_support_count"])

    n_atoms = numeric_value(row.get("n_atoms"), default=np.nan)
    parent_heavy = numeric_value(row.get("parent_heavy_atoms"), default=np.nan)
    relative_size = n_atoms / parent_heavy if parent_heavy and not math.isnan(parent_heavy) else np.nan
    attachment = int(numeric_value(row.get("attachment_count"), default=-1))
    boundary = int(numeric_value(row.get("n_boundary_atoms"), default=-1))
    split_quality = numeric_value(row.get("split_quality_score"), default=0.0)
    main_delta = numeric_value(row.get("main_delta_distance"), default=-999.0)
    verifier_count = support_count(row, min_delta=min_delta)

    connected_pass = bool_value(row.get("is_connected", True))
    ring_pass = not bool_value(row.get("cuts_ring", False))
    counterfactual_pass = bool_value(row.get("counterfactual_usable_for_scoring", False))
    target_role_pass = str(row.get("formal_token_status", "")) == "retain_target" and str(
        row.get("formal_token_role", "")
    ) == "target"
    atom_min_pass = n_atoms >= min_atoms
    atom_preferred_min_pass = n_atoms >= preferred_min_atoms
    atom_absolute_pass = n_atoms <= absolute_max_atoms
    relative_pass = (not math.isnan(relative_size)) and relative_size <= max_relative
    attachment_pass = attachment in allowed_attachments
    boundary_pass = boundary <= max_boundary
    quality_pass = split_quality >= min_quality
    main_delta_pass = main_delta >= min_delta
    verifier_pass = verifier_count >= min_support
    preferred_size_pass = atom_preferred_min_pass and n_atoms <= preferred_max_atoms
    conditional_size_band = preferred_max_atoms < n_atoms <= conditional_max_atoms
    preferred_quality_pass = split_quality >= preferred_quality
    preferred_delta_pass = main_delta >= preferred_delta
    strong_support_pass = verifier_count >= strong_support

    hard_fail_reasons: list[str] = []
    if not connected_pass:
        hard_fail_reasons.append("not_connected")
    if not ring_pass:
        hard_fail_reasons.append("ring_cut")
    if not counterfactual_pass:
        hard_fail_reasons.append("counterfactual_not_usable")
    if not target_role_pass:
        hard_fail_reasons.append("not_target_role")
    if not atom_min_pass:
        hard_fail_reasons.append("too_small")
    if not atom_absolute_pass:
        hard_fail_reasons.append("too_large")
    if not attachment_pass:
        hard_fail_reasons.append("attachment_not_allowed")

    review_reasons: list[str] = []
    if not relative_pass:
        review_reasons.append("relative_size_high")
    if not boundary_pass:
        review_reasons.append("boundary_complex")
    if conditional_size_band:
        review_reasons.append("conditional_size_band")
    if atom_min_pass and not atom_preferred_min_pass:
        review_reasons.append("micro_token")
    if not quality_pass:
        review_reasons.append("split_quality_low")
    if not main_delta_pass:
        review_reasons.append("main_delta_low")
    if not verifier_pass:
        review_reasons.append("verifier_support_low")

    if hard_fail_reasons:
        status = "exclude"
    elif not relative_pass or not boundary_pass:
        status = "resplit_required"
    elif atom_min_pass and not atom_preferred_min_pass:
        status = "supplement_only"
    elif preferred_size_pass and quality_pass and main_delta_pass and verifier_pass:
        status = "primary_keep"
    elif (
        conditional_size_band
        and relative_pass
        and boundary_pass
        and preferred_quality_pass
        and preferred_delta_pass
        and verifier_pass
    ):
        status = "conditional_keep"
    elif conditional_size_band:
        status = "resplit_required"
    elif quality_pass and main_delta_pass:
        status = "supplement_only"
    else:
        status = "supplement_only"

    if status == "primary_keep" and strong_support_pass and preferred_quality_pass:
        evidence_tier = "T1_high_confidence"
    elif status in {"primary_keep", "conditional_keep"}:
        evidence_tier = "T2_main_usable"
    elif status == "resplit_required":
        evidence_tier = "needs_resplit"
    elif status == "supplement_only":
        evidence_tier = "T3_supplement"
    else:
        evidence_tier = "exclude"

    return {
        "relative_token_size": relative_size,
        "verifier_support_count": verifier_count,
        "connected_pass": connected_pass,
        "ring_pass": ring_pass,
        "counterfactual_pass": counterfactual_pass,
        "target_role_pass": target_role_pass,
        "atom_min_pass": atom_min_pass,
        "atom_preferred_min_pass": atom_preferred_min_pass,
        "atom_preferred_size_pass": preferred_size_pass,
        "atom_absolute_pass": atom_absolute_pass,
        "relative_size_pass": relative_pass,
        "attachment_pass": attachment_pass,
        "boundary_pass": boundary_pass,
        "split_quality_pass": quality_pass,
        "main_delta_pass": main_delta_pass,
        "verifier_pass": verifier_pass,
        "strong_verifier_pass": strong_support_pass,
        "definition_status": status,
        "evidence_tier": evidence_tier,
        "hard_fail_reasons": ";".join(hard_fail_reasons),
        "review_reasons": ";".join(review_reasons),
        "needs_stage13_resplit": status == "resplit_required",
    }


def build_audit(active: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    active = active.copy()
    active["parent_heavy_atoms"] = active["canonical_smiles"].map(mol_heavy_atom_count)
    audit_rows = []
    for _, row in active.iterrows():
        audit_rows.append(classify_row(row, rules))
    audit = pd.concat([active.reset_index(drop=True), pd.DataFrame(audit_rows)], axis=1)
    key_cols = [
        "mol_id",
        "parent_rank",
        "parent_library_id",
        "risk_side",
        "token_id",
        "canonical_smiles",
        "atom_indices",
        "n_atoms",
        "parent_heavy_atoms",
        "relative_token_size",
        "n_boundary_atoms",
        "attachment_count",
        "split_quality_score",
        "split_quality_grade",
        "main_delta_distance",
        "xgboost_delta_distance",
        "rf_delta_distance",
        "verifier_support_count",
        "predictor_support_level",
        "active_token_score",
        "definition_status",
        "evidence_tier",
        "hard_fail_reasons",
        "review_reasons",
        "needs_stage13_resplit",
    ]
    ordered = [col for col in key_cols if col in audit.columns]
    rest = [col for col in audit.columns if col not in ordered]
    return audit[ordered + rest]


def rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def build_summary(audit: pd.DataFrame, rules: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    strict = rules["strict_rules"]
    gates = rules["acceptance_gates"]
    status_counts = audit["definition_status"].value_counts().to_dict()
    tier_counts = audit["evidence_tier"].value_counts().to_dict()
    main_view = audit[audit["definition_status"].isin(["primary_keep", "conditional_keep"])].copy()
    primary = audit[audit["definition_status"] == "primary_keep"].copy()
    resplit = audit[audit["definition_status"] == "resplit_required"].copy()
    exclude = audit[audit["definition_status"] == "exclude"].copy()

    parent_count = int(audit["mol_id"].nunique())
    main_parent_coverage = float(main_view["mol_id"].nunique() / parent_count) if parent_count else 0.0
    uncovered_parents = sorted(set(audit["mol_id"].astype(str)) - set(main_view["mol_id"].astype(str)))
    primary_support_rate = rate(primary["verifier_pass"]) if "verifier_pass" in primary.columns else 0.0

    summary = {
        "phase": "moltoken",
        "verdict": "pass",
        "input_active_token_count": int(len(audit)),
        "parent_count": parent_count,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "evidence_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "primary_keep_count": int(len(primary)),
        "conditional_keep_count": int((audit["definition_status"] == "conditional_keep").sum()),
        "resplit_required_count": int(len(resplit)),
        "exclude_count": int(len(exclude)),
        "main_or_conditional_parent_coverage": main_parent_coverage,
        "uncovered_parent_count": int(len(uncovered_parents)),
        "uncovered_parents": uncovered_parents,
        "mean_n_atoms": float(audit["n_atoms"].mean()),
        "median_n_atoms": float(audit["n_atoms"].median()),
        "max_n_atoms": float(audit["n_atoms"].max()),
        "mean_relative_token_size": float(audit["relative_token_size"].mean()),
        "max_relative_token_size": float(audit["relative_token_size"].max()),
        "primary_mean_split_quality": float(primary["split_quality_score"].mean()) if len(primary) else 0.0,
        "primary_verifier_support_rate": primary_support_rate,
        "rules_used": strict,
    }

    gate_rows = []
    gate_specs = [
        ("audited_token_count", len(audit), gates["audited_token_count_min"], ">="),
        ("primary_keep_count", len(primary), gates["primary_keep_count_min"], ">="),
        (
            "main_or_conditional_parent_coverage",
            main_parent_coverage,
            gates["main_or_conditional_parent_coverage_min"],
            ">=",
        ),
        ("resplit_required_count", len(resplit), gates["resplit_required_count_max"], "<="),
        ("exclude_count", len(exclude), gates["exclude_count_max"], "<="),
        ("mean_relative_size", summary["mean_relative_token_size"], gates["mean_relative_size_max"], "<="),
        ("median_n_atoms", summary["median_n_atoms"], gates["median_n_atoms_max"], "<="),
        ("primary_mean_split_quality", summary["primary_mean_split_quality"], gates["primary_mean_split_quality_min"], ">="),
        ("primary_verifier_support_rate", primary_support_rate, gates["primary_verifier_support_rate_min"], ">="),
    ]
    verdict = "pass"
    for name, actual, threshold, direction in gate_specs:
        passed = bool(actual >= threshold) if direction == ">=" else bool(actual <= threshold)
        if not passed:
            verdict = "conditional_pass"
        gate_rows.append(
            {
                "gate": name,
                "actual": float(actual),
                "threshold": float(threshold),
                "direction": direction,
                "passed": passed,
            }
        )
    summary["verdict"] = verdict
    return summary, pd.DataFrame(gate_rows)


def build_coverage_recovery_candidates(audit: pd.DataFrame) -> pd.DataFrame:
    main_parents = set(
        audit[audit["definition_status"].isin(["primary_keep", "conditional_keep"])]["mol_id"].astype(str)
    )
    recovery = audit[
        (~audit["mol_id"].astype(str).isin(main_parents))
        & (audit["definition_status"].isin(["resplit_required", "supplement_only"]))
    ].copy()
    if recovery.empty:
        return recovery
    recovery["recovery_priority_score"] = (
        recovery["split_quality_score"].fillna(0.0)
        + 0.35 * recovery["main_delta_distance"].fillna(0.0).clip(lower=0.0)
        + 0.15 * recovery["verifier_support_count"].fillna(0.0).clip(lower=0.0, upper=2.0)
        - 0.10 * recovery["relative_token_size"].fillna(1.0).clip(lower=0.0)
    )
    recovery = recovery.sort_values(
        ["mol_id", "recovery_priority_score", "split_quality_score", "main_delta_distance"],
        ascending=[True, False, False, False],
    )
    recovery["coverage_recovery_rank"] = recovery.groupby("mol_id").cumcount() + 1
    keep_cols = [
        "mol_id",
        "token_id",
        "definition_status",
        "evidence_tier",
        "coverage_recovery_rank",
        "recovery_priority_score",
        "n_atoms",
        "parent_heavy_atoms",
        "relative_token_size",
        "n_boundary_atoms",
        "attachment_count",
        "split_quality_score",
        "main_delta_distance",
        "verifier_support_count",
        "review_reasons",
        "atom_indices",
        "canonical_smiles",
    ]
    return recovery[[col for col in keep_cols if col in recovery.columns]]


def write_definition_doc(rules: dict[str, Any]) -> None:
    strict = rules["strict_rules"]
    lines = [
        "# Mol Token 正式定义",
        "",
        "Mol Token 定义为目标性质贡献关系筛选后的可编辑连通单元。它不是固定官能团，也不是预设片段词典，而是在给定 LogD 目标区间下，由候选连通子图经过反事实贡献、边界可编辑性、尺寸约束和复核器支持共同筛选得到的局部单元。",
        "",
        "## 四级概念",
        "",
        "- 候选连通子图：从分子中枚举得到的化学连通区域，只代表可能的局部结构。",
        "- 候选 Mol Token：通过基础尺寸、边界、不切环和反事实可用性检查的连通区域。",
        "- 活跃 Mol Token：对目标性质偏离具有正向反事实贡献，并具备可编辑接口的候选单元。",
        "- 高可信 Mol Token：同时满足严格尺寸、边界、切分质量和复核器支持的活跃单元。",
        "",
        "## 主证据规则",
        "",
        f"- 原子数范围：{strict['preferred_min_atoms']}-{strict['preferred_max_atoms']} 个原子为主证据优选范围。",
        f"- 微小单元：{strict['min_atoms']} 个原子可保留为补充观察，但不直接进入主证据。",
        f"- 条件尺寸范围：{strict['preferred_max_atoms'] + 1}-{strict['conditional_max_atoms']} 个原子必须进入更严格复核。",
        f"- 相对尺寸：不超过父分子重原子数的 {strict['max_relative_parent_heavy_atom_fraction']:.2f}。",
        "- 边界接口：attachment 为 1 或 2，边界原子数不超过 2。",
        "- 环规则：不允许切断环系。",
        f"- 切分质量：split quality 不低于 {strict['min_split_quality_score']:.2f}。",
        f"- 贡献证据：主预测器反事实改善不低于 {strict['min_main_delta_distance']:.2f}。",
        f"- 复核证据：至少 {strict['min_verifier_support_count']} 个复核器支持贡献方向。",
        "",
        "## 状态解释",
        "",
        "- primary_keep：满足主证据规则，可进入后续主结果重建。",
        "- conditional_keep：尺寸偏大但贡献和复核支持很强，后续需单独说明。",
        "- resplit_required：需要在下一阶段进行二级拆分或边界收紧。",
        "- supplement_only：仅适合补充观察，不支撑强结论。",
        "- exclude：不满足硬性化学或有效性要求。",
        "",
        "## 写作边界",
        "",
        "后续结果应写成预测层面证据。主结论只围绕 LogD 目标区间优化和约束筛选后的删除、替换动作展开；增加动作和补充层 Token 只用于说明扩展空间。",
    ]
    (REPORTS / "formal_moltoken_definition.md").write_text("\n".join(lines), encoding="utf-8")


def write_audit_report(summary: dict[str, Any], gates: pd.DataFrame) -> None:
    lines = [
        "# 12 阶段 Mol Token 定义审计报告",
        "",
        "## 审计结论",
        "",
        f"- 审计状态：{summary['verdict']}",
        f"- 活跃 Mol Token 数：{summary['input_active_token_count']}",
        f"- 覆盖父分子数：{summary['parent_count']}",
        f"- primary_keep：{summary['primary_keep_count']}",
        f"- conditional_keep：{summary['conditional_keep_count']}",
        f"- resplit_required：{summary['resplit_required_count']}",
        f"- exclude：{summary['exclude_count']}",
        f"- 未覆盖父分子数：{summary['uncovered_parent_count']}",
        f"- 主证据或条件保留父分子覆盖率：{summary['main_or_conditional_parent_coverage']:.3f}",
        f"- 平均原子数：{summary['mean_n_atoms']:.3f}",
        f"- 中位原子数：{summary['median_n_atoms']:.3f}",
        f"- 最大原子数：{summary['max_n_atoms']:.0f}",
        f"- 平均相对尺寸：{summary['mean_relative_token_size']:.3f}",
        f"- 最大相对尺寸：{summary['max_relative_token_size']:.3f}",
        f"- primary_keep 平均 split quality：{summary['primary_mean_split_quality']:.3f}",
        "",
        "## 状态分布",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key, value in summary["status_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 证据层分布", "", "| tier | count |", "|---|---:|"])
    for key, value in summary["evidence_tier_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 验收门槛", "", "| gate | actual | threshold | direction | passed |", "|---|---:|---:|---|---|"])
    for row in gates.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.actual:.3f} | {row.threshold:.3f} | {row.direction} | {row.passed} |")
    lines.extend(
        [
            "",
            "## 审查意见",
            "",
            "本阶段将 Mol Token 从宽松活跃池收紧为可审查的主证据候选。primary_keep 可进入后续最终证据重建；resplit_required 应在 13 阶段进行二级拆分，不能直接作为强结论中的局部编辑单元。",
            "",
            "覆盖率未完全达标时，不建议放宽证据门槛。更合适的处理是进入 13 阶段，对未覆盖父分子中的高优先级候选执行二级拆分、边界重排或复核器重排。",
        ]
    )
    (REPORTS / "phase12_definition_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def scan_text_outputs() -> int:
    hits = 0
    for path in TEXT_OUTPUTS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in BLOCKED_TERMS:
            hits += text.count(term)
    return int(hits)


def main() -> None:
    ensure_dirs()
    rules = json.loads(CONFIG.read_text(encoding="utf-8"))
    active = pd.read_csv(INPUT_ACTIVE)
    audit = build_audit(active, rules)
    summary, gates = build_summary(audit, rules)

    audit.to_csv(TABLES / "moltoken_definition_audit.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(TABLES / "definition_acceptance_gates.csv", index=False, encoding="utf-8-sig")
    audit[audit["definition_status"] == "resplit_required"].to_csv(
        TABLES / "resplit_required_moltokens.csv", index=False, encoding="utf-8-sig"
    )
    coverage_recovery = build_coverage_recovery_candidates(audit)
    coverage_recovery.to_csv(TABLES / "coverage_recovery_candidates.csv", index=False, encoding="utf-8-sig")
    audit[["definition_status", "evidence_tier", "mol_id"]].groupby(
        ["definition_status", "evidence_tier"], as_index=False
    ).agg(token_count=("mol_id", "count"), parent_count=("mol_id", "nunique")).to_csv(
        TABLES / "moltoken_status_summary.csv", index=False, encoding="utf-8-sig"
    )

    write_definition_doc(rules)
    write_audit_report(summary, gates)

    blocked_hits = scan_text_outputs()
    self_check = {
        "phase": "moltoken",
        "status": summary["verdict"],
        "blocked_text_hit_count": blocked_hits,
        "output_files": {
            "audit_table": str(TABLES / "moltoken_definition_audit.csv"),
            "resplit_table": str(TABLES / "resplit_required_moltokens.csv"),
            "status_summary": str(TABLES / "moltoken_status_summary.csv"),
            "coverage_recovery": str(TABLES / "coverage_recovery_candidates.csv"),
            "definition_doc": str(REPORTS / "formal_moltoken_definition.md"),
            "audit_report": str(REPORTS / "phase12_definition_audit_report.md"),
        },
        "key_counts": {
            "input_active_token_count": summary["input_active_token_count"],
            "primary_keep_count": summary["primary_keep_count"],
            "conditional_keep_count": summary["conditional_keep_count"],
            "resplit_required_count": summary["resplit_required_count"],
            "exclude_count": summary["exclude_count"],
            "uncovered_parent_count": summary["uncovered_parent_count"],
        },
    }
    if blocked_hits:
        self_check["status"] = "conditional_pass"
    (REPORTS / "phase12_self_check.json").write_text(
        json.dumps(self_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(self_check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"phase12 audit failed: {exc}", file=sys.stderr)
        raise
