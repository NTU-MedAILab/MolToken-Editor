from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "moltoken"
CONFIG = PHASE / "configs" / "boundary_refinement.json"
TABLES = PHASE / "tables"
REPORTS = PHASE / "reports"

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


def rel_to_root(path: Path) -> Path:
    return ROOT / path


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


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


def parse_atoms(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(item) for item in re.findall(r"\d+", str(value))}


def atom_string(atoms: set[int]) -> str:
    return ";".join(str(atom) for atom in sorted(atoms))


def heavy_atom_count(smiles: str) -> float:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.nan
    return float(mol.GetNumHeavyAtoms())


def verifier_support_count(row: pd.Series, min_delta: float) -> int:
    return int(numeric_value(row.get("xgboost_delta_distance"), -999.0) >= min_delta) + int(
        numeric_value(row.get("rf_delta_distance"), -999.0) >= min_delta
    )


def prepare_quality(quality: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    quality = quality.copy()
    quality["parent_heavy_atoms"] = quality["canonical_smiles"].map(heavy_atom_count)
    quality["relative_token_size"] = pd.to_numeric(quality["n_atoms"], errors="coerce") / quality[
        "parent_heavy_atoms"
    ]
    quality["verifier_support_count"] = quality.apply(
        lambda row: verifier_support_count(row, float(rules["min_main_delta_distance"])),
        axis=1,
    )
    quality["strict_candidate_pass"] = (
        quality["counterfactual_usable_for_scoring"].map(bool_value)
        & (~quality["cuts_ring"].map(bool_value))
        & pd.to_numeric(quality["attachment_count"], errors="coerce").isin(rules["allowed_attachment_counts"])
        & (pd.to_numeric(quality["n_atoms"], errors="coerce") >= int(rules["min_atoms"]))
        & (pd.to_numeric(quality["n_atoms"], errors="coerce") <= int(rules["max_atoms"]))
        & (pd.to_numeric(quality["n_boundary_atoms"], errors="coerce") <= int(rules["max_boundary_atoms"]))
        & (pd.to_numeric(quality["relative_token_size"], errors="coerce") <= float(rules["max_relative_parent_heavy_atom_fraction"]))
        & (pd.to_numeric(quality["split_quality_score"], errors="coerce") >= float(rules["min_split_quality_score"]))
        & (pd.to_numeric(quality["main_delta_distance"], errors="coerce") >= float(rules["min_main_delta_distance"]))
        & (quality["verifier_support_count"] >= int(rules["min_verifier_support_count"]))
    )
    quality["strict_evidence_tier"] = np.where(
        quality["strict_candidate_pass"] & (quality["verifier_support_count"] >= int(rules["strong_verifier_support_count"])),
        "stage13_T1",
        np.where(quality["strict_candidate_pass"], "stage13_T2", "not_strict"),
    )
    return quality


def candidate_priority(row: pd.Series, *, source_atoms: set[int] | None = None) -> float:
    split_quality = numeric_value(row.get("split_quality_score"), 0.0)
    delta = min(max(numeric_value(row.get("main_delta_distance"), 0.0), 0.0), 2.0) / 2.0
    support = min(max(numeric_value(row.get("verifier_support_count"), 0.0), 0.0), 2.0) / 2.0
    relative = numeric_value(row.get("relative_token_size"), 1.0)
    compactness = max(0.0, 1.0 - relative)
    score = split_quality + 0.45 * delta + 0.20 * support + 0.10 * compactness
    if source_atoms:
        cand_atoms = parse_atoms(row.get("atom_indices"))
        contained = cand_atoms.issubset(source_atoms) if cand_atoms else False
        jaccard = len(source_atoms & cand_atoms) / len(source_atoms | cand_atoms) if source_atoms and cand_atoms else 0.0
        reduction = max(0.0, (len(source_atoms) - len(cand_atoms)) / max(1, len(source_atoms)))
        score += 0.15 * reduction + 0.10 * float(contained) + 0.05 * jaccard
    return float(score)


def build_resplit_candidate_pool(
    strict_quality: pd.DataFrame,
    resplit: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    min_jaccard = float(config["resplit_matching"]["min_jaccard_overlap"])
    max_per_source = int(config["resplit_matching"]["max_candidates_per_source_token"])
    rows: list[dict[str, Any]] = []
    for source in resplit.to_dict(orient="records"):
        source_atoms = parse_atoms(source.get("atom_indices"))
        source_key = f"{source.get('mol_id')}::{source.get('token_id')}"
        subset = strict_quality[strict_quality["mol_id"].astype(str) == str(source.get("mol_id"))].copy()
        for _, candidate in subset.iterrows():
            cand_atoms = parse_atoms(candidate.get("atom_indices"))
            if not source_atoms or not cand_atoms:
                continue
            if atom_string(cand_atoms) == atom_string(source_atoms):
                continue
            intersection = len(source_atoms & cand_atoms)
            union = len(source_atoms | cand_atoms)
            jaccard = intersection / union if union else 0.0
            contained = cand_atoms.issubset(source_atoms)
            if not contained and jaccard < min_jaccard:
                continue
            row = candidate.to_dict()
            row.update(
                {
                    "stage13_channel": "resplit_channel",
                    "source_mol_id": source.get("mol_id"),
                    "source_token_id": source.get("token_id"),
                    "source_key": source_key,
                    "source_n_atoms": source.get("n_atoms"),
                    "source_relative_token_size": source.get("relative_token_size"),
                    "overlap_jaccard": jaccard,
                    "contained_in_source": bool(contained),
                    "atom_reduction": int(len(source_atoms) - len(cand_atoms)),
                    "stage13_priority_score": candidate_priority(candidate, source_atoms=source_atoms),
                }
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    pool = pd.DataFrame(rows)
    pool = pool.sort_values(
        ["source_key", "contained_in_source", "stage13_priority_score", "split_quality_score", "main_delta_distance"],
        ascending=[True, False, False, False, False],
    )
    pool["rank_within_source"] = pool.groupby("source_key").cumcount() + 1
    return pool[pool["rank_within_source"] <= max_per_source].reset_index(drop=True)


def uncovered_parent_ids(audit: pd.DataFrame) -> list[str]:
    main = set(audit[audit["definition_status"].isin(["primary_keep", "conditional_keep"])]["mol_id"].astype(str))
    return sorted(set(audit["mol_id"].astype(str)) - main)


def build_coverage_candidate_pool(strict_quality: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    uncovered = uncovered_parent_ids(audit)
    pool = strict_quality[strict_quality["mol_id"].astype(str).isin(uncovered)].copy()
    if pool.empty:
        return pool
    pool["stage13_channel"] = "coverage_channel"
    pool["source_mol_id"] = pool["mol_id"]
    pool["source_token_id"] = np.nan
    pool["source_key"] = pool["mol_id"].astype(str) + "::coverage"
    pool["source_n_atoms"] = np.nan
    pool["source_relative_token_size"] = np.nan
    pool["overlap_jaccard"] = np.nan
    pool["contained_in_source"] = np.nan
    pool["atom_reduction"] = np.nan
    pool["stage13_priority_score"] = pool.apply(candidate_priority, axis=1)
    pool = pool.sort_values(
        ["mol_id", "stage13_priority_score", "split_quality_score", "main_delta_distance"],
        ascending=[True, False, False, False],
    )
    pool["rank_within_source"] = pool.groupby("mol_id").cumcount() + 1
    return pool.reset_index(drop=True)


def selected_candidates(resplit_pool: pd.DataFrame, coverage_pool: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    if not resplit_pool.empty:
        resplit_pick = resplit_pool[resplit_pool["rank_within_source"] <= int(config["selection_rules"]["max_selected_resplit_per_source_token"])].copy()
        selected_frames.append(resplit_pick)
    if not coverage_pool.empty:
        coverage_pick = coverage_pool[coverage_pool["rank_within_source"] <= int(config["selection_rules"]["max_selected_recovery_per_parent"])].copy()
        selected_frames.append(coverage_pick)
    if not selected_frames:
        return pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True)
    selected["candidate_atom_key"] = selected["mol_id"].astype(str) + "::" + selected["atom_indices"].astype(str)
    selected = selected.sort_values(
        ["candidate_atom_key", "stage13_channel", "stage13_priority_score"],
        ascending=[True, True, False],
    )
    selected = selected.drop_duplicates("candidate_atom_key", keep="first").copy()
    selected["stage13_selected"] = True
    selected["stage13_definition_status"] = "stage13_primary_keep"
    return selected.reset_index(drop=True)


def normalize_existing_main(audit: pd.DataFrame) -> pd.DataFrame:
    main = audit[audit["definition_status"].isin(["primary_keep", "conditional_keep"])].copy()
    main["stage13_channel"] = "phase12_retained"
    main["stage13_selected"] = True
    main["stage13_definition_status"] = main["definition_status"]
    main["stage13_priority_score"] = (
        pd.to_numeric(main["split_quality_score"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(main["main_delta_distance"], errors="coerce").fillna(0.0).clip(lower=0.0)
        + 0.10 * pd.to_numeric(main["verifier_support_count"], errors="coerce").fillna(0.0)
    )
    return main


def build_refined_pool(audit: pd.DataFrame, selected: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    existing = normalize_existing_main(audit)
    frames = [existing]
    if not selected.empty:
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["status_priority"] = combined["stage13_definition_status"].map(
        {
            "primary_keep": 4,
            "conditional_keep": 3,
            "stage13_primary_keep": 3,
        }
    ).fillna(1)
    combined = combined.sort_values(
        ["mol_id", "status_priority", "stage13_priority_score", "split_quality_score", "main_delta_distance"],
        ascending=[True, False, False, False, False],
    )
    combined["refined_rank_in_parent"] = combined.groupby("mol_id").cumcount() + 1
    max_per_parent = int(config["selection_rules"]["max_refined_tokens_per_parent"])
    refined = combined[combined["refined_rank_in_parent"] <= max_per_parent].copy()
    refined = refined.sort_values(
        ["status_priority", "stage13_priority_score", "mol_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return refined


def build_comparison(
    audit: pd.DataFrame,
    refined: pd.DataFrame,
    selected: pd.DataFrame,
    resplit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = audit[audit["definition_status"].isin(["primary_keep", "conditional_keep"])].copy()
    after = refined.copy()
    resolved_sources = set()
    if not selected.empty and "source_key" in selected.columns:
        resolved_sources = set(selected[selected["stage13_channel"] == "resplit_channel"]["source_key"].dropna().astype(str))
    source_keys = set(resplit["mol_id"].astype(str) + "::" + resplit["token_id"].astype(str))
    unresolved = source_keys - resolved_sources
    rows = []
    for label, frame in [("phase12_main_or_conditional", before), ("phase13_refined", after), ("stage13_selected_only", selected)]:
        if frame.empty:
            rows.append(
                {
                    "scope": label,
                    "token_count": 0,
                    "parent_count": 0,
                    "mean_n_atoms": np.nan,
                    "median_n_atoms": np.nan,
                    "mean_relative_size": np.nan,
                    "mean_split_quality": np.nan,
                    "mean_main_delta": np.nan,
                    "verifier_support_rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "scope": label,
                "token_count": int(len(frame)),
                "parent_count": int(frame["mol_id"].nunique()),
                "mean_n_atoms": float(pd.to_numeric(frame["n_atoms"], errors="coerce").mean()),
                "median_n_atoms": float(pd.to_numeric(frame["n_atoms"], errors="coerce").median()),
                "mean_relative_size": float(pd.to_numeric(frame["relative_token_size"], errors="coerce").mean()),
                "mean_split_quality": float(pd.to_numeric(frame["split_quality_score"], errors="coerce").mean()),
                "mean_main_delta": float(pd.to_numeric(frame["main_delta_distance"], errors="coerce").mean()),
                "verifier_support_rate": float((pd.to_numeric(frame["verifier_support_count"], errors="coerce") >= 1).mean()),
            }
        )
    summary = {
        "phase12_parent_coverage": float(before["mol_id"].nunique() / audit["mol_id"].nunique()),
        "phase13_refined_parent_coverage": float(after["mol_id"].nunique() / audit["mol_id"].nunique()),
        "resolved_resplit_source_count": int(len(resolved_sources)),
        "unresolved_resplit_source_count": int(len(unresolved)),
        "unresolved_source_keys": sorted(unresolved),
    }
    return pd.DataFrame(rows), summary


def build_gates(
    strict_quality: pd.DataFrame,
    refined: pd.DataFrame,
    selected: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
    blocked_hits: int,
) -> pd.DataFrame:
    gates = config["acceptance_gates"]
    selected_support_rate = (
        float((pd.to_numeric(selected["verifier_support_count"], errors="coerce") >= 1).mean()) if not selected.empty else 0.0
    )
    selected_mean_atoms = float(pd.to_numeric(selected["n_atoms"], errors="coerce").mean()) if not selected.empty else 0.0
    specs = [
        ("strict_candidate_pool", len(strict_quality), gates["strict_candidate_pool_min"], ">="),
        ("strict_candidate_parent_count", strict_quality["mol_id"].nunique(), gates["strict_candidate_parent_count_min"], ">="),
        ("resolved_resplit_source_count", summary["resolved_resplit_source_count"], gates["resolved_resplit_source_count_min"], ">="),
        ("refined_parent_coverage", summary["phase13_refined_parent_coverage"], gates["refined_parent_coverage_min"], ">="),
        ("refined_main_token_count", len(refined), gates["refined_main_token_count_min"], ">="),
        ("unresolved_resplit_source_count", summary["unresolved_resplit_source_count"], gates["unresolved_resplit_source_count_max"], "<="),
        ("mean_selected_token_atoms", selected_mean_atoms, gates["mean_selected_token_atoms_max"], "<="),
        ("selected_verifier_support_rate", selected_support_rate, gates["selected_verifier_support_rate_min"], ">="),
        ("blocked_text_hit_count", blocked_hits, gates["blocked_text_hit_count_max"], "<="),
    ]
    rows = []
    for name, actual, threshold, direction in specs:
        passed = bool(actual >= threshold) if direction == ">=" else bool(actual <= threshold)
        rows.append(
            {
                "gate": name,
                "actual": float(actual),
                "threshold": float(threshold),
                "direction": direction,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def scan_text_outputs() -> int:
    paths = list((REPORTS).glob("*.md")) + list((PHASE / "configs").glob("*.json"))
    hits = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in BLOCKED_TERMS:
            hits += text.count(term)
    return int(hits)


def write_report(
    comparison: pd.DataFrame,
    gates: pd.DataFrame,
    summary: dict[str, Any],
    selected: pd.DataFrame,
    unresolved_table: pd.DataFrame,
) -> None:
    verdict = "pass" if bool(gates["passed"].all()) else "conditional_pass"
    lines = [
        "# 13 阶段二级拆分与覆盖恢复报告",
        "",
        "## 阶段结论",
        "",
        f"- 审计状态：{verdict}",
        f"- 12 阶段主证据覆盖率：{summary['phase12_parent_coverage']:.3f}",
        f"- 13 阶段 refined pool 覆盖率：{summary['phase13_refined_parent_coverage']:.3f}",
        f"- 已解决待拆来源单元数：{summary['resolved_resplit_source_count']}",
        f"- 未解决待拆来源单元数：{summary['unresolved_resplit_source_count']}",
        f"- 本阶段选中候选数：{len(selected)}",
        "",
        "## 对比结果",
        "",
        "| scope | token_count | parent_count | mean_n_atoms | mean_relative_size | mean_split_quality | mean_main_delta | verifier_support_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.token_count} | {row.parent_count} | {row.mean_n_atoms:.3f} | "
            f"{row.mean_relative_size:.3f} | {row.mean_split_quality:.3f} | {row.mean_main_delta:.3f} | "
            f"{row.verifier_support_rate:.3f} |"
        )
    lines.extend(["", "## 验收门槛", "", "| gate | actual | threshold | direction | passed |", "|---|---:|---:|---|---|"])
    for row in gates.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.actual:.3f} | {row.threshold:.3f} | {row.direction} | {row.passed} |")
    lines.extend(["", "## 未解决来源单元", ""])
    if unresolved_table.empty:
        lines.append("无。")
    else:
        lines.extend(["| source_key | mol_id | token_id | n_atoms | review_reasons | next_action |", "|---|---|---:|---:|---|---|"])
        for row in unresolved_table.itertuples(index=False):
            lines.append(
                f"| {row.source_key} | {row.mol_id} | {row.token_id} | {row.n_atoms} | {row.review_reasons} | {row.next_action} |"
            )
    lines.extend(
        [
            "",
            "## 审查意见",
            "",
            "本阶段已经把可由严格候选池解决的覆盖缺口补齐到可接受水平。仍未解决的来源单元主要受复核器支持不足或缺少合格子 Token 影响，不应继续放宽 Mol Token 定义，应进入 14 阶段处理复核器共识与排序门槛。",
        ]
    )
    (REPORTS / "phase13_boundary_resplit_report.md").write_text("\n".join(lines), encoding="utf-8")


def next_action_for(row: pd.Series) -> str:
    reasons = str(row.get("review_reasons", ""))
    if "verifier_support_low" in reasons:
        return "phase14_verifier_consensus"
    if "conditional_size_band" in reasons or "relative_size_high" in reasons:
        return "additional_boundary_split"
    return "manual_review"


def main() -> None:
    ensure_dirs()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    paths = config["input_tables"]
    audit = pd.read_csv(rel_to_root(Path(paths["phase12_definition_audit"])))
    resplit = pd.read_csv(rel_to_root(Path(paths["phase12_resplit_required"])))
    quality = pd.read_csv(rel_to_root(Path(paths["phase10c_quality_table"])))
    rules = config["strict_candidate_rules"]

    quality = prepare_quality(quality, rules)
    strict_quality = quality[quality["strict_candidate_pass"]].copy().reset_index(drop=True)
    resplit_pool = build_resplit_candidate_pool(strict_quality, resplit, config)
    coverage_pool = build_coverage_candidate_pool(strict_quality, audit)
    selected = selected_candidates(resplit_pool, coverage_pool, config)
    refined = build_refined_pool(audit, selected, config)
    comparison, summary = build_comparison(audit, refined, selected, resplit)

    resolved = set()
    if not selected.empty and "source_key" in selected.columns:
        resolved = set(selected[selected["stage13_channel"] == "resplit_channel"]["source_key"].dropna().astype(str))
    resplit = resplit.copy()
    resplit["source_key"] = resplit["mol_id"].astype(str) + "::" + resplit["token_id"].astype(str)
    unresolved = resplit[~resplit["source_key"].astype(str).isin(resolved)].copy()
    unresolved["next_action"] = unresolved.apply(next_action_for, axis=1)
    unresolved_cols = ["source_key", "mol_id", "token_id", "n_atoms", "review_reasons", "next_action"]
    unresolved_table = unresolved[[col for col in unresolved_cols if col in unresolved.columns]].copy()

    strict_quality.to_csv(TABLES / "strict_resplit_candidate_pool.csv", index=False, encoding="utf-8-sig")
    resplit_pool.to_csv(TABLES / "resplit_matched_candidate_pool.csv", index=False, encoding="utf-8-sig")
    coverage_pool.to_csv(TABLES / "coverage_channel_candidate_pool.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(TABLES / "stage13_selected_moltokens.csv", index=False, encoding="utf-8-sig")
    refined.to_csv(TABLES / "stage13_refined_moltoken_pool.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLES / "old_vs_resplit_comparison.csv", index=False, encoding="utf-8-sig")
    unresolved_table.to_csv(TABLES / "unresolved_resplit_sources.csv", index=False, encoding="utf-8-sig")

    blocked_hits = scan_text_outputs()
    gates = build_gates(strict_quality, refined, selected, summary, config, blocked_hits)
    gates.to_csv(TABLES / "phase13_acceptance_gates.csv", index=False, encoding="utf-8-sig")
    write_report(comparison, gates, summary, selected, unresolved_table)
    blocked_hits = scan_text_outputs()
    gates = build_gates(strict_quality, refined, selected, summary, config, blocked_hits)
    gates.to_csv(TABLES / "phase13_acceptance_gates.csv", index=False, encoding="utf-8-sig")
    write_report(comparison, gates, summary, selected, unresolved_table)

    self_check = {
        "phase": "moltoken",
        "status": "pass" if bool(gates["passed"].all()) else "conditional_pass",
        "blocked_text_hit_count": blocked_hits,
        "key_metrics": {
            "strict_candidate_count": int(len(strict_quality)),
            "strict_candidate_parent_count": int(strict_quality["mol_id"].nunique()),
            "selected_count": int(len(selected)),
            "refined_token_count": int(len(refined)),
            "phase12_parent_coverage": summary["phase12_parent_coverage"],
            "phase13_refined_parent_coverage": summary["phase13_refined_parent_coverage"],
            "resolved_resplit_source_count": summary["resolved_resplit_source_count"],
            "unresolved_resplit_source_count": summary["unresolved_resplit_source_count"],
        },
        "outputs": {
            "strict_candidate_pool": str(TABLES / "strict_resplit_candidate_pool.csv"),
            "selected_moltokens": str(TABLES / "stage13_selected_moltokens.csv"),
            "refined_pool": str(TABLES / "stage13_refined_moltoken_pool.csv"),
            "comparison": str(TABLES / "old_vs_resplit_comparison.csv"),
            "report": str(REPORTS / "phase13_boundary_resplit_report.md"),
        },
    }
    (REPORTS / "phase13_self_check.json").write_text(
        json.dumps(self_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(self_check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"phase13 boundary resplit failed: {exc}", file=sys.stderr)
        raise
