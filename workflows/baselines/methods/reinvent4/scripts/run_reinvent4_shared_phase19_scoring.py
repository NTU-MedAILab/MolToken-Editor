#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true", "1", "1.0", "yes", "y", "pass"
    }


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mark_failed(chunk: pd.DataFrame, error: str) -> pd.DataFrame:
    out = chunk.copy()
    out["shared_phase19_score_failed"] = True
    out["shared_phase19_score_error"] = error[:1000]
    out["edit_valid"] = False
    if "edit_reason" not in out:
        out["edit_reason"] = ""
    out["edit_reason"] = (
        out["edit_reason"].fillna("").astype(str)
        + "|shared_phase19_score_failed"
    )
    return out


def score_recursive(
    scorer,
    chunk: pd.DataFrame,
    force_predict: bool,
    skip_predict: bool,
    min_chunk_size: int,
    failures: list[dict],
    depth: int = 0,
) -> pd.DataFrame:
    try:
        scored = scorer.score_candidates(
            chunk,
            force_predict=force_predict,
            skip_predict=skip_predict,
        )
        scored["shared_phase19_score_failed"] = False
        scored["shared_phase19_score_error"] = ""
        return scored
    except Exception as error:
        message = repr(error)
        print(
            "[WARN] shared score failure:",
            f"rows={len(chunk)} depth={depth} error={message}",
            flush=True,
        )
        if len(chunk) <= min_chunk_size:
            failures.append(
                {
                    "rows": int(len(chunk)),
                    "depth": int(depth),
                    "error": message,
                    "mol_ids":
                        chunk["mol_id"].astype(str).unique().tolist()[:20],
                    "candidate_smiles":
                        chunk["counterfactual_smiles"]
                        .astype(str)
                        .head(10)
                        .tolist(),
                }
            )
            return mark_failed(chunk, message)

        middle = len(chunk) // 2
        left = score_recursive(
            scorer,
            chunk.iloc[:middle].copy(),
            force_predict,
            skip_predict,
            min_chunk_size,
            failures,
            depth + 1,
        )
        right = score_recursive(
            scorer,
            chunk.iloc[middle:].copy(),
            force_predict,
            skip_predict,
            min_chunk_size,
            failures,
            depth + 1,
        )
        return pd.concat([left, right], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--min-chunk-size", type=int, default=1)
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--expected-candidates", type=int, default=1741)
    parser.add_argument("--expected-parents", type=int, default=88)
    args = parser.parse_args()

    if args.force_predict and args.skip_predict:
        raise ValueError(
            "--force-predict and --skip-predict cannot be used together"
        )
    if not args.force_predict and not args.skip_predict:
        raise ValueError(
            "Formal scoring requires --force-predict so that cached "
            "Phase10-D predictions cannot be reused accidentally"
        )

    started = datetime.now(timezone.utc)
    project_root = Path(args.project_root).expanduser().resolve()
    candidates_path = Path(args.candidates).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_script = (
        project_root / "workflows" /
        "localization" / "scripts" /
        "base_scoring.py"
    )
    phase10d_script = (
        project_root / "workflows" /
        "moltoken" / "scripts" /
        "generate_edits.py"
    )
    required = [candidates_path, base_script, phase10d_script]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    source_root = (
        project_root / "workflows"
    )
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(base_script.parent))
    sys.path.insert(0, str(phase10d_script.parent))

    base = import_file("phase24_shared_phase19_base", base_script)
    base.ROOT = source_root
    base.PHASE10D_SCRIPT = phase10d_script
    base.TABLES = output_dir
    base.REPORTS = reports_dir
    base.PREDICTIONS = predictions_dir

    scorer = base.import_phase10d()
    scorer.TABLES = output_dir
    scorer.REPORTS = reports_dir
    scorer.PREDICTIONS = predictions_dir

    candidates = pd.read_csv(candidates_path, low_memory=False)
    required_columns = [
        "method",
        "method_group",
        "mol_id",
        "token_id",
        "canonical_smiles",
        "counterfactual_smiles",
        "edited_smiles",
        "edit_valid",
        "replace_direction",
        "main_original_pred_mean",
        "main_original_target_distance",
        "xgboost_original_pred_mean",
        "xgboost_original_target_distance",
        "rf_original_pred_mean",
        "rf_original_target_distance",
    ]
    missing = [
        name for name in required_columns if name not in candidates.columns
    ]
    if missing:
        raise KeyError(f"Candidate input is missing columns: {missing}")
    if len(candidates) != args.expected_candidates:
        raise ValueError(
            f"Candidate count {len(candidates)} != "
            f"{args.expected_candidates}"
        )
    if candidates["mol_id"].astype(str).nunique() != args.expected_parents:
        raise ValueError(
            "Candidate parent coverage "
            f"{candidates['mol_id'].astype(str).nunique()} != "
            f"{args.expected_parents}"
        )
    if not candidates["edit_valid"].map(bool_value).all():
        raise ValueError(
            "Unique-valid scoring input contains edit_valid=False rows"
        )

    failures: list[dict] = []
    parts: list[pd.DataFrame] = []

    for start in range(0, len(candidates), args.batch_size):
        end = min(start + args.batch_size, len(candidates))
        print(
            f"[BATCH] {start}:{end} rows={end-start}",
            flush=True,
        )
        part = score_recursive(
            scorer,
            candidates.iloc[start:end].copy(),
            args.force_predict,
            args.skip_predict,
            args.min_chunk_size,
            failures,
        )
        parts.append(part)

    scored = pd.concat(parts, ignore_index=True, sort=False)
    scored_path = output_dir / "reinvent4_phase19_scored_unique.csv"
    scored.to_csv(scored_path, index=False)

    scored_main = (
        scored["edited_main_pred"].notna()
        if "edited_main_pred" in scored
        else pd.Series(False, index=scored.index)
    )
    evidence = (
        scored["evidence_ready"].map(bool_value)
        if "evidence_ready" in scored
        else pd.Series(False, index=scored.index)
    )
    strict = (
        scored["strict_success"].map(bool_value)
        if "strict_success" in scored
        else pd.Series(False, index=scored.index)
    )
    finished = datetime.now(timezone.utc)

    checks = {
        "input_rows": len(candidates) == args.expected_candidates,
        "output_rows": len(scored) == args.expected_candidates,
        "input_parent_coverage":
            candidates["mol_id"].astype(str).nunique()
            == args.expected_parents,
        "scored_main_nonempty": int(scored_main.sum()) > 0,
        "failed_leaf_count": len(failures) == 0,
    }
    manifest = {
        "phase": "phase24_reinvent4_shared_phase19_scoring",
        "status": "completed" if all(checks.values()) else "needs_review",
        "checks": checks,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "force_predict": bool(args.force_predict),
        "skip_predict": bool(args.skip_predict),
        "batch_size": args.batch_size,
        "counts": {
            "input_rows": len(candidates),
            "output_rows": len(scored),
            "parents": scored["mol_id"].astype(str).nunique(),
            "main_scored_rows": int(scored_main.sum()),
            "evidence_ready_rows": int(evidence.sum()),
            "evidence_ready_parents":
                scored.loc[evidence, "mol_id"].astype(str).nunique(),
            "strict_success_rows": int(strict.sum()),
            "failed_leaf_count": len(failures),
        },
        "inputs": {
            "candidates": str(candidates_path),
            "candidates_sha256": sha256(candidates_path),
            "phase19_base_script": str(base_script),
            "phase19_base_script_sha256": sha256(base_script),
            "phase10d_script": str(phase10d_script),
            "phase10d_script_sha256": sha256(phase10d_script),
        },
        "outputs": {
            "scored": str(scored_path),
            "scored_sha256": sha256(scored_path),
        },
        "failed_leaves": failures,
    }
    manifest_path = output_dir / "scoring_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "completed":
        raise SystemExit(
            "Shared scoring completed with failed integrity checks"
        )


if __name__ == "__main__":
    main()
