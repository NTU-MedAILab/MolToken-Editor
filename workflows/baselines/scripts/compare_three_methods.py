#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REFERENCE_METHOD = "moltoken_editor"
BASELINE_METHODS = [
    "reinvent4_mol2mol",
    "mmp_transformer_base",
]
TOP_K = [1, 3, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MolToken-Editor, REINVENT4, and base-only "
            "MMP-Transformer under an oracle-independent B=5 candidate cap."
        )
    )
    parser.add_argument("--p26-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--expected-parents", type=int, default=88)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_col(
    frame: pd.DataFrame,
    choices: list[str],
    required: bool = True,
) -> str | None:
    for column in choices:
        if column in frame.columns:
            return column
    if required:
        raise KeyError(f"Missing required column; tried: {choices}")
    return None


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "1.0", "yes", "y", "pass"})
    )


def canonicalize(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def prepare_method(
    path: Path,
    method: str,
    budget_cap: int,
    expected_parents: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path, low_memory=False)

    parent_col = find_col(raw, ["mol_id", "parent_id"])
    parent_ids = raw[parent_col].astype(str).str.strip()

    if int(parent_ids.nunique()) != expected_parents:
        raise ValueError(
            f"{method}: expected {expected_parents} parents, "
            f"found {int(parent_ids.nunique())}"
        )

    selected_counts = (
        parent_ids.value_counts()
        .sort_index()
        .astype(int)
    )
    if selected_counts.lt(1).any():
        raise ValueError(
            f"{method}: parent has no selected candidate: "
            f"{selected_counts[selected_counts < 1].to_dict()}"
        )
    if selected_counts.gt(budget_cap).any():
        raise ValueError(
            f"{method}: selected count exceeds B={budget_cap}: "
            f"{selected_counts[selected_counts > budget_cap].to_dict()}"
        )

    maximum_rows = expected_parents * budget_cap
    if len(raw) > maximum_rows:
        raise ValueError(
            f"{method}: {len(raw)} rows exceed cap total {maximum_rows}"
        )

    evidence_col = find_col(raw, ["evidence_ready"], required=False)
    valid_col = find_col(raw, ["edit_valid"], required=False)
    main_pred_col = find_col(
        raw,
        [
            "edited_main_pred",
            "edited_main_pred_mean",
            "main_edited_pred_mean",
            "main_pred",
        ],
        required=False,
    )
    target_col = find_col(
        raw,
        [
            "target_window_success",
            "target_success",
            "in_target_window",
            "window_success",
        ],
    )
    guardrail_col = find_col(
        raw,
        [
            "relaxed_guardrail_pass",
            "guardrail_pass",
            "strict_guardrail_pass",
        ],
    )
    strict_col = find_col(
        raw,
        [
            "strict_success",
            "strict_endpoint_success",
            "strict_validation_success",
        ],
        required=False,
    )
    opposition_col = find_col(
        raw,
        [
            "auxiliary_opposition_count",
            "aux_opposition_count",
            "opposition_count",
        ],
        required=False,
    )
    final_score_col = find_col(
        raw,
        [
            "final_edit_score",
            "final_score",
            "edit_score",
        ],
    )
    smiles_col = find_col(
        raw,
        [
            "_candidate_canonical",
            "_candidate_canonical_smiles",
            "edited_smiles",
            "counterfactual_smiles",
            "candidate_smiles",
        ],
    )
    tanimoto_col = find_col(
        raw,
        [
            "parent_tanimoto",
            "tanimoto_to_parent",
            "parent_similarity",
            "similarity_to_parent",
        ],
        required=False,
    )
    direction_col = find_col(
        raw,
        [
            "direction",
            "optimization_direction",
            "task_direction",
            "target_direction",
            "replace_direction",
        ],
        required=False,
    )
    original_distance_col = find_col(
        raw,
        [
            "main_original_target_distance",
            "original_main_target_distance",
            "parent_target_distance",
        ],
        required=False,
    )
    edited_distance_col = find_col(
        raw,
        [
            "main_edited_target_distance",
            "edited_main_target_distance",
            "edited_target_distance",
            "main_target_distance",
        ],
        required=False,
    )
    delta_distance_col = find_col(
        raw,
        [
            "main_delta_target_distance",
            "delta_target_distance",
            "target_distance_improvement",
            "distance_improvement",
        ],
        required=False,
    )

    evidence = (
        as_bool(raw[evidence_col])
        if evidence_col is not None
        else pd.Series(True, index=raw.index)
    )
    if valid_col is not None:
        evidence &= as_bool(raw[valid_col])
    if main_pred_col is not None:
        evidence &= pd.to_numeric(
            raw[main_pred_col],
            errors="coerce",
        ).notna()

    target = as_bool(raw[target_col])
    guardrail = as_bool(raw[guardrail_col])
    strict = (
        as_bool(raw[strict_col])
        if strict_col is not None
        else target & guardrail
    )
    opposition = (
        pd.to_numeric(
            raw[opposition_col],
            errors="coerce",
        ).fillna(999.0)
        if opposition_col is not None
        else pd.Series(0.0, index=raw.index)
    )
    final_score = pd.to_numeric(
        raw[final_score_col],
        errors="coerce",
    ).fillna(-np.inf)
    tanimoto = (
        pd.to_numeric(
            raw[tanimoto_col],
            errors="coerce",
        )
        if tanimoto_col is not None
        else pd.Series(np.nan, index=raw.index)
    )

    if delta_distance_col is not None:
        delta_distance = pd.to_numeric(
            raw[delta_distance_col],
            errors="coerce",
        )
    elif original_distance_col and edited_distance_col:
        delta_distance = (
            pd.to_numeric(
                raw[original_distance_col],
                errors="coerce",
            )
            - pd.to_numeric(
                raw[edited_distance_col],
                errors="coerce",
            )
        )
    else:
        delta_distance = pd.Series(np.nan, index=raw.index)

    direction = (
        raw[direction_col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        if direction_col is not None
        else pd.Series("unknown", index=raw.index)
    )

    derived = pd.DataFrame(
        {
            "_phase26_method": method,
            "_phase26_mol_id": parent_ids,
            "_evidence_ready": evidence.astype(bool),
            "_target_success": target.astype(bool),
            "_guardrail_pass": guardrail.astype(bool),
            "_strict_success": strict.astype(bool),
            "_opposition_count": opposition,
            "_final_edit_score": final_score,
            "_canonical_smiles": raw[smiles_col].map(canonicalize),
            "_parent_tanimoto": tanimoto,
            "_delta_distance": delta_distance,
            "_direction": direction,
        },
        index=raw.index,
    )
    frame = pd.concat(
        [
            raw.reset_index(drop=True),
            derived.reset_index(drop=True),
        ],
        axis=1,
    )

    deficient = {
        str(parent): int(count)
        for parent, count in selected_counts.items()
        if int(count) < budget_cap
    }
    missing_slots = int(
        (budget_cap - selected_counts)
        .clip(lower=0)
        .sum()
    )

    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "parents": int(parent_ids.nunique()),
        "budget_cap_per_parent": int(budget_cap),
        "minimum_selected_per_parent": int(selected_counts.min()),
        "median_selected_per_parent": float(selected_counts.median()),
        "maximum_selected_per_parent": int(selected_counts.max()),
        "missing_generation_slots": missing_slots,
        "deficient_parents": deficient,
        "columns_used": {
            "parent": parent_col,
            "target_success": target_col,
            "guardrail": guardrail_col,
            "strict_success": (
                strict_col
                if strict_col is not None
                else "<target_success AND guardrail_pass>"
            ),
            "opposition": opposition_col,
            "final_score": final_score_col,
            "evidence_ready": evidence_col,
            "tanimoto": tanimoto_col,
            "direction": direction_col,
        },
    }
    return frame, metadata


def parent_context(
    frame: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, str]]:
    selected_counts = (
        frame.groupby("_phase26_mol_id")
        .size()
        .astype(int)
        .to_dict()
    )
    directions = (
        frame.groupby("_phase26_mol_id")["_direction"]
        .first()
        .astype(str)
        .to_dict()
    )
    return selected_counts, directions


def continuous_means(group: pd.DataFrame | None) -> dict[str, float]:
    if group is None or group.empty:
        return {
            "mean_final_edit_score": np.nan,
            "mean_delta_distance": np.nan,
            "mean_parent_tanimoto": np.nan,
        }
    return {
        "mean_final_edit_score": float(
            group["_final_edit_score"]
            .replace([-np.inf, np.inf], np.nan)
            .mean()
        ),
        "mean_delta_distance": float(
            group["_delta_distance"].mean()
        ),
        "mean_parent_tanimoto": float(
            group["_parent_tanimoto"].mean()
        ),
    }


def parent_any_hit(
    frame: pd.DataFrame,
    parents: list[str],
    method: str,
    budget: int,
) -> pd.DataFrame:
    eligible = frame[frame["_evidence_ready"]].copy()
    grouped = {
        key: value
        for key, value in eligible.groupby("_phase26_mol_id")
    }
    selected_counts, directions = parent_context(frame)

    rows: list[dict[str, Any]] = []
    for mol_id in parents:
        group = grouped.get(mol_id)
        evidence_count = 0 if group is None else int(len(group))
        selected_count = int(selected_counts[mol_id])
        target_successes = (
            0
            if group is None
            else int(group["_target_success"].sum())
        )
        strict_successes = (
            0
            if group is None
            else int(group["_strict_success"].sum())
        )

        row = {
            "method_group": method,
            "protocol": "oracle_independent_B5_any_hit",
            "k": int(budget),
            "mol_id": mol_id,
            "selected_candidates": selected_count,
            "missing_generation_slots": int(budget - selected_count),
            "available_ranked_candidates": evidence_count,
            "unfilled_evidence_slots_at_k": int(
                budget - min(evidence_count, budget)
            ),
            "parent_target_hit": bool(target_successes > 0),
            "parent_strict_hit": bool(strict_successes > 0),
            "target_success_count": target_successes,
            "strict_success_count": strict_successes,
            "candidate_target_rate": float(
                target_successes / budget
            ),
            "candidate_strict_rate": float(
                strict_successes / budget
            ),
            "candidate_rate_denominator": int(budget),
            "direction": directions.get(mol_id, "unknown"),
        }
        row.update(continuous_means(group))
        rows.append(row)

    return pd.DataFrame(rows)


def rank_by_shared_score(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    eligible = frame[frame["_evidence_ready"]].copy()
    eligible = eligible.sort_values(
        [
            "_phase26_mol_id",
            "_final_edit_score",
            "_guardrail_pass",
            "_opposition_count",
            "_canonical_smiles",
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    eligible["_shared_rank"] = (
        eligible.groupby("_phase26_mol_id").cumcount() + 1
    )
    return eligible


def parent_topk(
    ranked: pd.DataFrame,
    source_frame: pd.DataFrame,
    parents: list[str],
    method: str,
    k: int,
) -> pd.DataFrame:
    top = ranked[ranked["_shared_rank"].le(k)].copy()
    grouped = {
        key: value
        for key, value in top.groupby("_phase26_mol_id")
    }
    selected_counts, directions = parent_context(source_frame)

    rows: list[dict[str, Any]] = []
    for mol_id in parents:
        group = grouped.get(mol_id)
        available = 0 if group is None else int(len(group))
        target_successes = (
            0
            if group is None
            else int(group["_target_success"].sum())
        )
        strict_successes = (
            0
            if group is None
            else int(group["_strict_success"].sum())
        )

        row = {
            "method_group": method,
            "protocol": "shared_score_ranked_topk",
            "k": int(k),
            "mol_id": mol_id,
            "selected_candidates": int(selected_counts[mol_id]),
            "missing_generation_slots": int(
                max(0, k - min(selected_counts[mol_id], k))
            ),
            "available_ranked_candidates": available,
            "unfilled_evidence_slots_at_k": int(k - available),
            "parent_target_hit": bool(target_successes > 0),
            "parent_strict_hit": bool(strict_successes > 0),
            "target_success_count": target_successes,
            "strict_success_count": strict_successes,
            "candidate_target_rate": float(
                target_successes / k
            ),
            "candidate_strict_rate": float(
                strict_successes / k
            ),
            "candidate_rate_denominator": int(k),
            "direction": directions.get(mol_id, "unknown"),
        }
        row.update(continuous_means(group))
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(
    parent_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (method, protocol, k), group in parent_table.groupby(
        ["method_group", "protocol", "k"],
        dropna=False,
    ):
        rows.append(
            {
                "method_group": method,
                "protocol": protocol,
                "k": int(k),
                "parents_total": int(len(group)),
                "parents_with_ranked_candidates": int(
                    group["available_ranked_candidates"]
                    .gt(0)
                    .sum()
                ),
                "parent_target_hit_rate": float(
                    group["parent_target_hit"].mean()
                ),
                "parent_strict_hit_rate": float(
                    group["parent_strict_hit"].mean()
                ),
                "candidate_target_rate_fixed_k": float(
                    group["candidate_target_rate"].mean()
                ),
                "candidate_strict_rate_fixed_k": float(
                    group["candidate_strict_rate"].mean()
                ),
                "candidate_rate_denominator_per_parent": int(k),
                "mean_selected_candidates": float(
                    group["selected_candidates"].mean()
                ),
                "total_missing_generation_slots": int(
                    group["missing_generation_slots"].sum()
                ),
                "mean_available_ranked_candidates": float(
                    group["available_ranked_candidates"].mean()
                ),
                "mean_unfilled_evidence_slots_at_k": float(
                    group["unfilled_evidence_slots_at_k"].mean()
                ),
                "mean_final_edit_score_available_only": float(
                    pd.to_numeric(
                        group["mean_final_edit_score"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_delta_distance_available_only": float(
                    pd.to_numeric(
                        group["mean_delta_distance"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_parent_tanimoto_available_only": float(
                    pd.to_numeric(
                        group["mean_parent_tanimoto"],
                        errors="coerce",
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def exact_mcnemar_p(
    a: pd.Series,
    b: pd.Series,
) -> tuple[int, int, float]:
    a_values = a.astype(bool).to_numpy()
    b_values = b.astype(bool).to_numpy()
    reference_only = int(np.sum(a_values & ~b_values))
    baseline_only = int(np.sum(~a_values & b_values))
    discordant = reference_only + baseline_only

    if discordant == 0:
        return reference_only, baseline_only, 1.0

    from scipy.stats import binomtest

    p_value = float(
        binomtest(
            min(reference_only, baseline_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    return reference_only, baseline_only, p_value


def paired_bootstrap(
    a: pd.Series,
    b: pd.Series,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    difference = (
        a.astype(float).to_numpy()
        - b.astype(float).to_numpy()
    )
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    n = len(difference)
    estimates = np.empty(reps, dtype=float)

    for index in range(reps):
        sample = rng.integers(0, n, size=n)
        estimates[index] = difference[sample].mean()

    lower, upper = np.quantile(
        estimates,
        [0.025, 0.975],
    )
    return observed, float(lower), float(upper)


def paired_comparisons(
    parent_table: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    conditions = (
        parent_table[["protocol", "k"]]
        .drop_duplicates()
        .sort_values(["protocol", "k"])
    )

    for _, condition in conditions.iterrows():
        protocol = str(condition["protocol"])
        k = int(condition["k"])

        reference_table = parent_table[
            parent_table["method_group"].eq(REFERENCE_METHOD)
            & parent_table["protocol"].eq(protocol)
            & parent_table["k"].eq(k)
        ].set_index("mol_id")

        for baseline in BASELINE_METHODS:
            baseline_table = parent_table[
                parent_table["method_group"].eq(baseline)
                & parent_table["protocol"].eq(protocol)
                & parent_table["k"].eq(k)
            ].set_index("mol_id")

            joined = reference_table.join(
                baseline_table,
                how="inner",
                lsuffix="_reference",
                rsuffix="_baseline",
            )
            if len(joined) != len(reference_table):
                raise ValueError(
                    f"{baseline}/{protocol}/K={k}: "
                    f"paired parents {len(joined)} != "
                    f"{len(reference_table)}"
                )

            for endpoint in [
                "parent_target_hit",
                "parent_strict_hit",
            ]:
                a = joined[f"{endpoint}_reference"]
                b = joined[f"{endpoint}_baseline"]

                difference, lower, upper = paired_bootstrap(
                    a,
                    b,
                    reps=reps,
                    seed=seed + k,
                )
                reference_only, baseline_only, p_value = (
                    exact_mcnemar_p(a, b)
                )

                rows.append(
                    {
                        "reference_method": REFERENCE_METHOD,
                        "baseline_method": baseline,
                        "protocol": protocol,
                        "k": k,
                        "endpoint": endpoint,
                        "paired_parents": int(len(joined)),
                        "reference_rate": float(a.mean()),
                        "baseline_rate": float(b.mean()),
                        "difference_reference_minus_baseline": difference,
                        "bootstrap_ci95_lower": lower,
                        "bootstrap_ci95_upper": upper,
                        "reference_only_successes": reference_only,
                        "baseline_only_successes": baseline_only,
                        "mcnemar_exact_p": p_value,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    p26 = args.p26_root.expanduser().resolve()

    input_paths = {
        REFERENCE_METHOD: (
            p26
            / "inputs/budget_lock/moltoken_editor/"
              "moltoken_b5_selected_scored.csv"
        ),
        "reinvent4_mol2mol": (
            p26
            / f"inputs/budget_lock/reinvent4_seed_{args.seed}/"
              f"reinvent4_seed_{args.seed}_b5_selected_scored.csv"
        ),
        "mmp_transformer_base": (
            p26
            / f"inputs/budget_lock/"
              f"mmp_transformer_seed_{args.seed}_base/"
              f"mmp_transformer_seed_{args.seed}_base_b5_selected_scored.csv"
        ),
    }

    prepared: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}

    for method, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        prepared[method], metadata[method] = prepare_method(
            path=path,
            method=method,
            budget_cap=args.budget,
            expected_parents=args.expected_parents,
        )

    parent_sets = [
        set(frame["_phase26_mol_id"].unique())
        for frame in prepared.values()
    ]
    if not all(
        parent_set == parent_sets[0]
        for parent_set in parent_sets[1:]
    ):
        raise ValueError(
            "The three methods do not contain identical parent sets."
        )
    all_parents = sorted(parent_sets[0])

    parent_frames: list[pd.DataFrame] = []
    ranked_frames: list[pd.DataFrame] = []

    for method, frame in prepared.items():
        parent_frames.append(
            parent_any_hit(
                frame=frame,
                parents=all_parents,
                method=method,
                budget=args.budget,
            )
        )

        ranked = rank_by_shared_score(frame)
        ranked["_phase26_seed"] = args.seed
        ranked_frames.append(ranked)

        for k in TOP_K:
            parent_frames.append(
                parent_topk(
                    ranked=ranked,
                    source_frame=frame,
                    parents=all_parents,
                    method=method,
                    k=k,
                )
            )

    parent_all = pd.concat(
        parent_frames,
        ignore_index=True,
    )
    parent_all.insert(0, "seed", args.seed)

    ranked_all = pd.concat(
        ranked_frames,
        ignore_index=True,
    )

    summary = summarize(parent_all)
    summary.insert(0, "seed", args.seed)

    paired = paired_comparisons(
        parent_all,
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    paired.insert(0, "seed", args.seed)

    output_dir = (
        p26
        / f"runs/equal_budget/seed_{args.seed}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parent_path = (
        output_dir
        / "phase26_b5_parent_metrics.csv"
    )
    ranked_path = (
        output_dir
        / "phase26_b5_shared_score_ranked_candidates.csv"
    )
    summary_path = (
        output_dir
        / "phase26_b5_method_summary.csv"
    )
    paired_path = (
        output_dir
        / "phase26_b5_paired_comparisons.csv"
    )
    manifest_path = (
        output_dir
        / "phase26_b5_comparison_manifest.json"
    )

    parent_all.to_csv(parent_path, index=False)
    ranked_all.to_csv(ranked_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)

    manifest = {
        "phase": "phase26_common_budget_b5_cap_comparison",
        "status": "pass",
        "seed": args.seed,
        "parents": args.expected_parents,
        "budget_cap_per_parent": args.budget,
        "primary_protocol": {
            "name": "oracle_independent_B5_any_hit",
            "description": (
                "Parent-level success among up to five candidates "
                "locked before property-oracle ranking."
            ),
            "candidate_rate_denominator": (
                "Fixed requested B=5 per parent; non-evidence candidates "
                "and unavailable generation slots count as zero successes."
            ),
        },
        "secondary_protocol": {
            "name": "shared_score_ranked_topk",
            "description": (
                "Shared-evaluator-assisted recoverability sensitivity. "
                "Evidence-ready candidates are ranked by final_edit_score, "
                "guardrail, opposition count, and canonical SMILES."
            ),
            "interpretation_limit": (
                "This is not native/fair generator ranking because "
                "final_edit_score overlaps the evaluated endpoint."
            ),
            "candidate_rate_denominator": (
                "Fixed requested K per parent; unavailable slots count "
                "as zero successes."
            ),
        },
        "continuous_metrics": (
            "Means are calculated only over actually available "
            "evidence-ready candidates."
        ),
        "methods": metadata,
        "outputs": {
            "parent_metrics": {
                "path": str(parent_path),
                "sha256": sha256(parent_path),
            },
            "shared_score_ranked_candidates": {
                "path": str(ranked_path),
                "sha256": sha256(ranked_path),
            },
            "method_summary": {
                "path": str(summary_path),
                "sha256": sha256(summary_path),
            },
            "paired_comparisons": {
                "path": str(paired_path),
                "sha256": sha256(paired_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "method_group",
        "protocol",
        "k",
        "parents_with_ranked_candidates",
        "parent_target_hit_rate",
        "parent_strict_hit_rate",
        "candidate_target_rate_fixed_k",
        "candidate_strict_rate_fixed_k",
        "mean_selected_candidates",
        "total_missing_generation_slots",
    ]

    print("STATUS: pass")
    print("===== PHASE26 B=5 CAP SUMMARY =====")
    print(
        summary[display_columns].to_string(
            index=False,
        )
    )
    print()
    print("===== PAIRED PARENT-LEVEL COMPARISONS =====")
    print(
        paired[
            [
                "baseline_method",
                "protocol",
                "k",
                "endpoint",
                "difference_reference_minus_baseline",
                "bootstrap_ci95_lower",
                "bootstrap_ci95_upper",
                "mcnemar_exact_p",
            ]
        ].to_string(index=False)
    )
    print("MANIFEST:", manifest_path)
    print("PHASE26_B5_CAP_COMPARISON_COMPLETE")


if __name__ == "__main__":
    main()
