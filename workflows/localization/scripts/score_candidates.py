#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE19 = ROOT / "localization"
TABLES = PHASE19 / "tables"
REPORTS = PHASE19 / "reports"
PREDICTIONS = PHASE19 / "predictions"

BASE_SCORING_SCRIPT = PHASE19 / "scripts/base_scoring.py"


def bool_value(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "pass"}


def import_base_phase19c():
    if not BASE_SCORING_SCRIPT.exists():
        raise FileNotFoundError(f"Missing base scoring script: {BASE_SCORING_SCRIPT}")

    sys.path.insert(0, str(BASE_SCORING_SCRIPT.parent))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path.cwd()))

    spec = importlib.util.spec_from_file_location("phase19c_base", BASE_SCORING_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def mark_failed_chunk(chunk: pd.DataFrame, error_text: str) -> pd.DataFrame:
    out = chunk.copy()

    if "edit_reason" not in out.columns:
        out["edit_reason"] = ""

    out["phase19c_score_failed"] = True
    out["phase19c_score_error"] = error_text[:500]
    out["edit_valid"] = False
    out["edit_reason"] = (
        out["edit_reason"].astype(str).replace("nan", "")
        + "|chemprop_score_failed"
    )

    return out


def score_chunk_recursive(
    phase10d,
    chunk: pd.DataFrame,
    force_predict: bool,
    skip_predict: bool,
    min_chunk_size: int,
    failed_records: list[dict],
    depth: int = 0,
) -> pd.DataFrame:
    chunk = chunk.copy()

    valid_mask = (
        chunk["edit_valid"].map(bool_value)
        & chunk["counterfactual_smiles"].notna()
    )

    if valid_mask.sum() == 0:
        chunk["phase19c_score_failed"] = False
        return chunk

    try:
        scored = phase10d.score_candidates(
            chunk,
            force_predict=force_predict,
            skip_predict=skip_predict,
        )
        scored["phase19c_score_failed"] = False
        return scored

    except Exception as e:
        err = repr(e)
        print(f"[WARN] score failed: rows={len(chunk)} valid={int(valid_mask.sum())} depth={depth} error={err}")

        if len(chunk) <= min_chunk_size:
            failed_records.append({
                "rows": int(len(chunk)),
                "valid_rows": int(valid_mask.sum()),
                "depth": int(depth),
                "error": err,
                "mol_ids": sorted(chunk.get("mol_id", pd.Series(dtype=str)).astype(str).unique().tolist())[:20],
                "example_smiles": chunk.loc[valid_mask, "counterfactual_smiles"].astype(str).head(10).tolist(),
            })
            return mark_failed_chunk(chunk, err)

        mid = len(chunk) // 2
        left = chunk.iloc[:mid].copy()
        right = chunk.iloc[mid:].copy()

        scored_left = score_chunk_recursive(
            phase10d,
            left,
            force_predict=force_predict,
            skip_predict=skip_predict,
            min_chunk_size=min_chunk_size,
            failed_records=failed_records,
            depth=depth + 1,
        )

        scored_right = score_chunk_recursive(
            phase10d,
            right,
            force_predict=force_predict,
            skip_predict=skip_predict,
            min_chunk_size=min_chunk_size,
            failed_records=failed_records,
            depth=depth + 1,
        )

        return pd.concat([scored_left, scored_right], ignore_index=True, sort=False)


def score_batched(
    phase10d,
    candidates: pd.DataFrame,
    batch_size: int,
    min_chunk_size: int,
    force_predict: bool,
    skip_predict: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    parts = []
    failed_records = []

    n = len(candidates)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = candidates.iloc[start:end].copy()

        valid_count = int(chunk["edit_valid"].map(bool_value).sum())
        print(f"[BATCH] {start}:{end} rows={len(chunk)} valid={valid_count}")

        scored = score_chunk_recursive(
            phase10d,
            chunk,
            force_predict=force_predict,
            skip_predict=skip_predict,
            min_chunk_size=min_chunk_size,
            failed_records=failed_records,
            depth=0,
        )
        parts.append(scored)

    scored_all = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return scored_all, failed_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(TABLES / "phase19b_common_budget2_candidates_chemprop_safe.csv"))
    parser.add_argument("--comparison-name", default="common_budget2_safe_batched")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--min-chunk-size", type=int, default=1)
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    base = import_base_phase19c()
    phase10d = base.import_phase10d()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_path}")

    candidates = pd.read_csv(candidates_path, low_memory=False)

    print("[INFO] candidates:", len(candidates))
    print("[INFO] valid candidates:", int(candidates["edit_valid"].map(bool_value).sum()))
    print("[INFO] method/action:")
    print(pd.crosstab(candidates["method_group"], candidates["edit_action"]))

    scored, failed_records = score_batched(
        phase10d,
        candidates,
        batch_size=args.batch_size,
        min_chunk_size=args.min_chunk_size,
        force_predict=args.force_predict,
        skip_predict=args.skip_predict,
    )

    expected_cols = [
        "edited_main_pred",
        "main_delta_distance_edit",
        "target_window_success",
        "relaxed_guardrail_pass",
        "consensus_success",
        "strict_success",
        "ood_risk",
        "parent_tanimoto",
        "delta_qed",
        "delta_sa_score",
        "final_edit_score",
        "auxiliary_opposition_count",
    ]
    for c in expected_cols:
        if c not in scored.columns:
            scored[c] = pd.NA

    scored_path = TABLES / f"phase19c_{args.comparison_name}_scored_all.csv"
    scored.to_csv(scored_path, index=False)

    top = base.top_by_group_parent(scored, k=args.top_k)
    top_path = TABLES / f"phase19c_{args.comparison_name}_top_by_parent.csv"
    top.to_csv(top_path, index=False)

    comparison = base.summarize_method_groups(scored, top)
    comparison_path = TABLES / f"phase19c_{args.comparison_name}_method_group_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    paired = base.paired_moltoken_vs_baselines(top)
    paired_path = TABLES / f"phase19c_{args.comparison_name}_moltoken_vs_baseline_paired.csv"
    paired.to_csv(paired_path, index=False)

    action = base.action_summary(scored)
    action_path = TABLES / f"phase19c_{args.comparison_name}_action_summary.csv"
    action.to_csv(action_path, index=False)

    failed_path = REPORTS / f"phase19c_{args.comparison_name}_failed_chunks.json"
    failed_path.write_text(json.dumps(failed_records, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "phase": "19C_baseline_candidate_scoring_batched",
        "status": "pass" if len(scored) else "failed",
        "comparison_name": args.comparison_name,
        "candidate_attempts": int(len(candidates)),
        "valid_edits_after_failure_downgrade": int(scored["edit_valid"].map(bool_value).sum()) if not scored.empty else 0,
        "scored_edits": int(scored["edited_main_pred"].notna().sum()) if "edited_main_pred" in scored.columns else 0,
        "top_rows": int(len(top)),
        "failed_chunk_count": int(len(failed_records)),
        "method_group_comparison": comparison.to_dict(orient="records"),
        "paired_moltoken_vs_baseline": paired.to_dict(orient="records") if len(paired) else [],
        "outputs": {
            "scored_all": str(scored_path),
            "top_by_parent": str(top_path),
            "method_group_comparison": str(comparison_path),
            "paired": str(paired_path),
            "action_summary": str(action_path),
            "failed_chunks": str(failed_path),
        },
    }

    report_path = REPORTS / f"phase19c_{args.comparison_name}_self_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Method group comparison ===")
    print(comparison.to_string(index=False))

    print("\n=== Paired MolToken vs baseline ===")
    print(paired.to_string(index=False) if len(paired) else "EMPTY")

    print("\n[INFO] failed_chunk_count:", len(failed_records))
    print("[WROTE]", scored_path)
    print("[WROTE]", top_path)
    print("[WROTE]", comparison_path)
    print("[WROTE]", paired_path)
    print("[WROTE]", action_path)
    print("[WROTE]", failed_path)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
