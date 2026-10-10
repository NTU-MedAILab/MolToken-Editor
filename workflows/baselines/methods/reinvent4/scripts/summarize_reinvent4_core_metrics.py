#!/usr/bin/env python3
"""Summarize the locked Phase24 REINVENT4 single-target logD run.

This script intentionally reports only metrics that can be computed from the
REINVENT output and the locked parent table.  MolToken-specific composite
metrics (for example ``final_edit_score`` or MMP site recovery) remain the job
of the shared downstream evaluator and must not be inferred here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


TOP_KS = (1, 3, 5, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def target_distance(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, default=88)
    parser.add_argument("--expected-attempts", type=int, default=4400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    parents = pd.read_csv(args.parents)

    required_candidate_columns = {
        "raw_attempt_index",
        "candidate_smiles",
        "SMILES_state",
        "logd_window_reward",
        "predicted_logd",
        "step",
        "parent_id",
        "parent_smiles",
        "chemically_valid",
        "reward_eligible",
        "duplicate_within_batch",
    }
    required_parent_columns = {
        "parent_id",
        "parent_smiles",
        "original_main_pred_logd",
        "optimization_direction",
        "target_low",
        "target_high",
    }
    missing_candidates = required_candidate_columns - set(candidates.columns)
    missing_parents = required_parent_columns - set(parents.columns)
    if missing_candidates or missing_parents:
        raise ValueError(
            f"Missing columns: candidates={sorted(missing_candidates)}, "
            f"parents={sorted(missing_parents)}"
        )
    if len(candidates) != args.expected_attempts:
        raise ValueError(
            f"Expected {args.expected_attempts} attempts, found {len(candidates)}"
        )
    if len(parents) != args.expected_parents or parents["parent_id"].nunique() != args.expected_parents:
        raise ValueError(
            f"Expected {args.expected_parents} unique parents, found "
            f"{len(parents)} rows/{parents['parent_id'].nunique()} IDs"
        )
    observed_parent_ids = set(candidates["parent_id"].dropna().astype(str))
    locked_parent_ids = set(parents["parent_id"].astype(str))
    if observed_parent_ids != locked_parent_ids:
        raise ValueError(
            "Candidate/parent ID sets differ: "
            f"missing={sorted(locked_parent_ids - observed_parent_ids)}, "
            f"extra={sorted(observed_parent_ids - locked_parent_ids)}"
        )

    # State 1 is the set actually scored by REINVENT.  State 2 is a chemically
    # valid within-batch duplicate and is counted in chemical validity only.
    eligible = candidates.loc[
        candidates["reward_eligible"].astype(bool) & (candidates["SMILES_state"] == 1)
    ].copy()
    eligible["canonical_candidate_smiles"] = eligible["candidate_smiles"].map(canonicalize)
    if eligible["canonical_candidate_smiles"].isna().any():
        raise ValueError("An allegedly reward-eligible molecule failed RDKit canonicalization")

    # One candidate may recur at later RL steps.  Keep its best score, then its
    # earliest appearance, to avoid counting the same molecule repeatedly.
    eligible = eligible.sort_values(
        ["parent_id", "canonical_candidate_smiles", "logd_window_reward", "step", "raw_attempt_index"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    unique = eligible.drop_duplicates(
        ["parent_id", "canonical_candidate_smiles"], keep="first"
    ).copy()
    unique = unique.sort_values(
        ["parent_id", "logd_window_reward", "step", "canonical_candidate_smiles"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    unique["rank_within_parent"] = unique.groupby("parent_id").cumcount() + 1

    parent_lookup = parents.set_index("parent_id")
    unique["parent_predicted_logd"] = unique["parent_id"].map(
        parent_lookup["original_main_pred_logd"]
    )
    unique["optimization_direction"] = unique["parent_id"].map(
        parent_lookup["optimization_direction"]
    )
    unique["target_low"] = unique["parent_id"].map(parent_lookup["target_low"])
    unique["target_high"] = unique["parent_id"].map(parent_lookup["target_high"])
    unique["parent_target_distance"] = unique.apply(
        lambda row: target_distance(
            float(row["parent_predicted_logd"]),
            float(row["target_low"]),
            float(row["target_high"]),
        ),
        axis=1,
    )
    unique["candidate_target_distance"] = unique.apply(
        lambda row: target_distance(
            float(row["predicted_logd"]),
            float(row["target_low"]),
            float(row["target_high"]),
        ),
        axis=1,
    )
    unique["delta_target_distance"] = (
        unique["parent_target_distance"] - unique["candidate_target_distance"]
    )
    unique["target_window_hit"] = unique["candidate_target_distance"] == 0.0
    unique["improved_vs_parent"] = unique["delta_target_distance"] > 0.0

    if unique["parent_id"].nunique() != args.expected_parents:
        raise ValueError("At least one locked parent has no reward-eligible unique candidate")

    top1 = unique.loc[unique["rank_within_parent"] == 1].copy()
    per_parent_counts = unique.groupby("parent_id").size()

    top_k_metrics: dict[str, dict[str, float | int]] = {}
    for k in TOP_KS:
        subset = unique.loc[unique["rank_within_parent"] <= k]
        any_hit = subset.groupby("parent_id")["target_window_hit"].any().reindex(
            parents["parent_id"], fill_value=False
        )
        top_k_metrics[str(k)] = {
            "parents_with_target_window_hit": int(any_hit.sum()),
            "target_window_success_rate": float(any_hit.mean()),
        }

    direction_metrics: dict[str, dict[str, float | int]] = {}
    for direction, frame in top1.groupby("optimization_direction", sort=True):
        direction_metrics[str(direction)] = {
            "parent_count": int(len(frame)),
            "top1_target_window_hits": int(frame["target_window_hit"].sum()),
            "top1_target_window_success_rate": float(frame["target_window_hit"].mean()),
            "top1_mean_delta_target_distance": float(frame["delta_target_distance"].mean()),
        }

    states = candidates["SMILES_state"].value_counts().sort_index()
    chemically_valid_count = int(candidates["chemically_valid"].astype(bool).sum())
    reward_eligible_count = int(candidates["reward_eligible"].astype(bool).sum())
    metrics: dict[str, object] = {
        "schema_version": 1,
        "experiment": "Phase24 REINVENT4-Mol2Mol single-target predicted-logD primary run",
        "metric_scope": (
            "REINVENT-only core metrics; not the shared MolToken composite/structural evaluator"
        ),
        "ranking_rule": (
            "SMILES_state=1 reward-eligible candidates; RDKit canonical deduplication "
            "within parent; descending logD target-window reward; earliest step as tie-break"
        ),
        "target_window": {"low": 1.0, "high": 3.4, "inclusive": True},
        "counts": {
            "locked_parents": int(len(parents)),
            "raw_attempts": int(len(candidates)),
            "chemically_valid_including_state2_duplicates": chemically_valid_count,
            "chemically_valid_rate_including_state2_duplicates": chemically_valid_count / len(candidates),
            "reward_eligible_state1": reward_eligible_count,
            "reward_eligible_rate": reward_eligible_count / len(candidates),
            "invalid_state0": int(states.get(0, 0)),
            "duplicate_state2": int(states.get(2, 0)),
            "unique_reward_eligible_parent_candidate_pairs": int(len(unique)),
        },
        "unique_candidates_per_parent": {
            "mean": float(per_parent_counts.mean()),
            "median": float(per_parent_counts.median()),
            "min": int(per_parent_counts.min()),
            "max": int(per_parent_counts.max()),
        },
        "top_k": top_k_metrics,
        "top1": {
            "parents": int(len(top1)),
            "target_window_hits": int(top1["target_window_hit"].sum()),
            "target_window_success_rate": float(top1["target_window_hit"].mean()),
            "improved_vs_parent_count": int(top1["improved_vs_parent"].sum()),
            "improved_vs_parent_rate": float(top1["improved_vs_parent"].mean()),
            "mean_delta_target_distance": float(top1["delta_target_distance"].mean()),
            "predicted_logd_mean": float(top1["predicted_logd"].mean()),
            "predicted_logd_std_sample": float(top1["predicted_logd"].std(ddof=1)),
            "predicted_logd_min": float(top1["predicted_logd"].min()),
            "predicted_logd_max": float(top1["predicted_logd"].max()),
        },
        "top1_by_direction": direction_metrics,
        "input_files": {
            "candidates": str(args.candidates),
            "candidates_sha256": sha256(args.candidates),
            "parents": str(args.parents),
            "parents_sha256": sha256(args.parents),
        },
        "limitations": [
            "The two molecules excluded by the locked protocol-A vocabulary rule are not evaluated.",
            "These are predicted-logD oracle results, not wet-lab measured logD.",
            "final_edit_score, strict_success_rate, novelty, structural filters, and MMP site recovery require the shared downstream evaluator.",
            "Direct MolToken-versus-REINVENT claims require rerunning MolToken on the exact same 88-parent set and evaluator.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = args.output_dir / "ranked_unique_candidates.csv"
    top1_path = args.output_dir / "top1_parent_metrics.csv"
    metrics_path = args.output_dir / "core_metrics.json"
    unique.to_csv(ranked_path, index=False)
    top1.to_csv(top1_path, index=False)
    metrics["output_files"] = {
        "ranked_unique_candidates": str(ranked_path),
        "ranked_unique_candidates_sha256": sha256(ranked_path),
        "top1_parent_metrics": str(top1_path),
        "top1_parent_metrics_sha256": sha256(top1_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
