#!/usr/bin/env python3
"""Frozen-data ablation analysis for MolToken-Editor.

Candidate selection is reconstructed from the pre-score Phase19 candidate
pool.  Candidates are selected before shared outcome columns are consulted.
All variants are subsequently evaluated with the same frozen Phase19 scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


VARIANTS = {
    "full": ("moltoken_editor", None),
    "random_localization": ("random_action_matched", None),
    "without_delete": ("moltoken_editor", ("exclude", "delete")),
    "without_replace": ("moltoken_editor", ("exclude", "replace")),
    "without_add": ("moltoken_editor", ("exclude", "add")),
    "delete_only": ("moltoken_editor", ("only", "delete")),
    "replace_only": ("moltoken_editor", ("only", "replace")),
    "add_only": ("moltoken_editor", ("only", "add")),
}

PRIMARY_METRICS = [
    "parent_target_window_rate",
    "parent_strict_success_rate",
    "candidate_target_window_rate_fixed_budget",
    "candidate_strict_success_rate_fixed_budget",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-free frozen-candidate ablation of MolToken-Editor."
        )
    )
    parser.add_argument("repo", type=Path, help="MolToken-Editor repository")
    parser.add_argument(
        "--archive-root",
        type=Path,
        required=True,
        help="Directory containing frozen Phase19 and Phase10 archives",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--budget", type=int, default=5)
    return parser.parse_args()


def canonicalize(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_parent_key"] = result["canonical_smiles"].map(canonicalize)
    result["_candidate_key"] = result["edited_smiles"].map(canonicalize)
    return result


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SystemExit(f"[FAIL] {label} missing columns: {missing}")


def select_variant(
    raw: pd.DataFrame,
    variant: str,
    budget: int,
) -> pd.DataFrame:
    method_group, action_rule = VARIANTS[variant]
    frame = raw[raw["method_group"].astype(str) == method_group].copy()

    if action_rule is not None:
        operation, action = action_rule
        if operation == "exclude":
            frame = frame[frame["edit_action"].astype(str) != action].copy()
        elif operation == "only":
            frame = frame[frame["edit_action"].astype(str) == action].copy()
        else:  # pragma: no cover
            raise AssertionError(operation)

    frame = frame[frame["_candidate_key"].notna()].copy()
    frame = frame.sort_values("_raw_row_order", kind="mergesort")
    frame = frame.drop_duplicates(
        subset=["_parent_key", "_candidate_key"],
        keep="first",
    )
    frame["selection_rank"] = (
        frame.groupby("_parent_key", sort=False).cumcount() + 1
    )
    frame = frame[frame["selection_rank"] <= budget].copy()
    frame["variant"] = variant
    return frame


def attach_outcomes(selected: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    keys = ["method_group", "_parent_key", "_candidate_key"]
    outcome_columns = [
        "target_window_success",
        "evidence_ready",
        "strict_success",
        "consensus_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
        "delta_qed",
        "delta_sa_score",
        "parent_tanimoto",
        "final_edit_score",
    ]
    available = [column for column in outcome_columns if column in scored.columns]
    lookup = (
        scored[keys + available]
        .drop_duplicates(subset=keys, keep="first")
        .copy()
    )
    merged = selected.merge(
        lookup,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if merged["target_window_success"].isna().any():
        missing = int(merged["target_window_success"].isna().sum())
        raise SystemExit(f"[FAIL] {missing} selected candidates lack outcomes")
    for column in [
        "target_window_success",
        "evidence_ready",
        "strict_success",
        "consensus_success",
        "relaxed_guardrail_pass",
        "edited_interference_alert",
    ]:
        if column in merged.columns:
            merged[column] = as_bool(merged[column])
    # Phase26's reported target endpoint is an oracle-independent hit among
    # evidence-ready candidates.  The raw Phase19 target_window_success flag
    # alone is an internal-window diagnostic and is intentionally not used as
    # the paper endpoint.
    merged["phase26_target_hit"] = (
        merged["target_window_success"] & merged["evidence_ready"]
    )
    return merged


def build_parent_table(
    selected: pd.DataFrame,
    parent_order: list[str],
    budget: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = {key: group for key, group in selected.groupby("_parent_key")}
    variant = str(selected["variant"].iloc[0]) if len(selected) else "unknown"

    for parent_key in parent_order:
        group = grouped.get(parent_key, selected.iloc[0:0])
        target_count = int(group["phase26_target_hit"].sum())
        strict_count = int(group["strict_success"].sum())
        rows.append(
            {
                "variant": variant,
                "parent_key": parent_key,
                "selected_candidates": int(len(group)),
                "missing_slots": int(budget - len(group)),
                "parent_target_window": int(target_count > 0),
                "parent_strict_success": int(strict_count > 0),
                "candidate_target_window_rate_fixed_budget": target_count / budget,
                "candidate_strict_success_rate_fixed_budget": strict_count / budget,
            }
        )
    return pd.DataFrame(rows)


def summarize_variant(
    selected: pd.DataFrame,
    parent_table: pd.DataFrame,
    budget: int,
) -> dict[str, object]:
    total_slots = len(parent_table) * budget
    target_count = int(selected["phase26_target_hit"].sum())
    strict_count = int(selected["strict_success"].sum())
    row: dict[str, object] = {
        "variant": selected["variant"].iloc[0],
        "parents": len(parent_table),
        "budget_per_parent": budget,
        "selected_candidates": len(selected),
        "missing_slots": total_slots - len(selected),
        "parents_with_full_budget": int(
            (parent_table["selected_candidates"] == budget).sum()
        ),
        "parent_target_window_rate": parent_table[
            "parent_target_window"
        ].mean(),
        "parent_strict_success_rate": parent_table[
            "parent_strict_success"
        ].mean(),
        "candidate_target_window_rate_fixed_budget": target_count / total_slots,
        "candidate_strict_success_rate_fixed_budget": strict_count / total_slots,
        "candidate_target_window_rate_observed": (
            target_count / len(selected) if len(selected) else np.nan
        ),
        "candidate_strict_success_rate_observed": (
            strict_count / len(selected) if len(selected) else np.nan
        ),
    }
    optional_means = {
        "candidate_guardrail_pass_rate_observed": "relaxed_guardrail_pass",
        "candidate_consensus_rate_observed": "consensus_success",
        "mean_delta_qed": "delta_qed",
        "mean_delta_sa_score": "delta_sa_score",
        "mean_parent_tanimoto": "parent_tanimoto",
    }
    for output_column, source_column in optional_means.items():
        row[output_column] = (
            pd.to_numeric(selected[source_column], errors="coerce").mean()
            if source_column in selected.columns and len(selected)
            else np.nan
        )
    row["candidate_alert_free_rate_observed"] = (
        1.0 - selected["edited_interference_alert"].mean()
        if "edited_interference_alert" in selected.columns and len(selected)
        else np.nan
    )
    return row


def paired_bootstrap(
    parent_tables: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    metric_columns = {
        "parent_target_window_rate": "parent_target_window",
        "parent_strict_success_rate": "parent_strict_success",
        "candidate_target_window_rate_fixed_budget": (
            "candidate_target_window_rate_fixed_budget"
        ),
        "candidate_strict_success_rate_fixed_budget": (
            "candidate_strict_success_rate_fixed_budget"
        ),
    }
    full = parent_tables["full"].set_index("parent_key")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(full), size=(replicates, len(full)))
    rows: list[dict[str, object]] = []

    for variant, table in parent_tables.items():
        if variant == "full":
            continue
        aligned = table.set_index("parent_key").loc[full.index]
        for metric, column in metric_columns.items():
            differences = (
                aligned[column].to_numpy(dtype=float)
                - full[column].to_numpy(dtype=float)
            )
            boot = differences[indices].mean(axis=1)
            rows.append(
                {
                    "variant": variant,
                    "reference": "full",
                    "metric": metric,
                    "difference": differences.mean(),
                    "ci95_low": np.quantile(boot, 0.025),
                    "ci95_high": np.quantile(boot, 0.975),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    archive_root = args.archive_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "phase19_candidates": archive_root
        / "localization_frozen_all/tables/phase19b_common_budget2_candidates_all.csv",
        "phase19_scored": archive_root
        / "localization_frozen_all/tables/phase19c_common_budget2_safe_scored_all.csv",
        "official_selected_prescore": repo
        / "workflows/baselines/inputs/budget_lock/moltoken_editor/moltoken_b5_selected_prescore.csv",
        "official_selected_scored": repo
        / "workflows/baselines/inputs/budget_lock/moltoken_editor/moltoken_b5_selected_scored.csv",
        "exact88": repo
        / "workflows/baselines/inputs/common_exact88/phase26_locked_exact88.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing {label}: {path}")

    raw = add_keys(pd.read_csv(paths["phase19_candidates"], low_memory=False))
    scored = add_keys(pd.read_csv(paths["phase19_scored"], low_memory=False))
    official = add_keys(
        pd.read_csv(paths["official_selected_prescore"], low_memory=False)
    )
    exact88 = pd.read_csv(paths["exact88"], low_memory=False)

    require_columns(
        raw,
        ["method_group", "edit_action", "canonical_smiles", "edited_smiles"],
        "Phase19 candidate pool",
    )
    require_columns(
        scored,
        ["target_window_success", "strict_success"],
        "Phase19 scored pool",
    )

    parent_order = [canonicalize(value) for value in exact88["parent_smiles"]]
    if any(value is None for value in parent_order) or len(set(parent_order)) != 88:
        raise SystemExit("[FAIL] Exact88 parent structures are not 88 unique molecules")
    parent_set = set(parent_order)
    raw = raw[raw["_parent_key"].isin(parent_set)].copy()
    scored = scored[scored["_parent_key"].isin(parent_set)].copy()
    raw["_raw_row_order"] = np.arange(len(raw), dtype=int)

    selections: dict[str, pd.DataFrame] = {}
    parent_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for variant in VARIANTS:
        selected = select_variant(raw, variant, args.budget)
        selected = attach_outcomes(selected, scored)
        parent_table = build_parent_table(selected, parent_order, args.budget)
        selections[variant] = selected
        parent_tables[variant] = parent_table
        summaries.append(summarize_variant(selected, parent_table, args.budget))

    official_pairs = set(zip(official["_parent_key"], official["_candidate_key"]))
    reconstructed_pairs = set(
        zip(selections["full"]["_parent_key"], selections["full"]["_candidate_key"])
    )
    if official_pairs != reconstructed_pairs or len(reconstructed_pairs) != 440:
        raise SystemExit(
            "[FAIL] reconstructed full selection does not match all 440 frozen candidates"
        )

    summary = pd.DataFrame(summaries)
    full = summary.set_index("variant").loc["full"]
    expected = {
        "parent_target_window_rate": 0.7954545454545454,
        "parent_strict_success_rate": 0.5795454545454546,
        "candidate_target_window_rate_fixed_budget": 0.40454545454545454,
        "candidate_strict_success_rate_fixed_budget": 0.29772727272727273,
    }
    for metric, expected_value in expected.items():
        actual = float(full[metric])
        if not np.isclose(actual, expected_value, atol=1e-12):
            raise SystemExit(
                f"[FAIL] full {metric}: expected {expected_value}, observed {actual}"
            )

    effects = paired_bootstrap(parent_tables, args.bootstrap, args.seed)
    selected_long = pd.concat(selections.values(), ignore_index=True)
    parents_long = pd.concat(parent_tables.values(), ignore_index=True)

    output_files = {
        "ablation_summary.csv": summary,
        "ablation_effects_vs_full.csv": effects,
        "ablation_parent_level.csv": parents_long,
        "ablation_selected_candidates.csv": selected_long,
    }
    for name, frame in output_files.items():
        frame.to_csv(output_dir / name, index=False)

    manifest = {
        "status": "pass",
        "protocol": {
            "population": "Exact88",
            "candidate_budget_per_parent": args.budget,
            "selection": (
                "stable original pre-score row order after canonical candidate "
                "deduplication; no shared outcome used for selection"
            ),
            "missing_candidate_slots": "counted as failures in fixed-budget rates",
            "evaluation": "same frozen Phase19 shared scorer for every variant",
            "target_endpoint": (
                "target_window_success AND evidence_ready, matching the "
                "Phase26 oracle-independent target-hit definition"
            ),
            "bootstrap": {
                "unit": "parent molecule",
                "paired": True,
                "replicates": args.bootstrap,
                "seed": args.seed,
            },
        },
        "variants": VARIANTS,
        "validation": {
            "official_full_candidate_set_matched": True,
            "official_full_candidate_count": 440,
            "published_metrics_reproduced": expected,
        },
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    package_path = output_dir / "moltoken_ablation_results.tar.gz"
    package_names = list(output_files) + ["manifest.json"]
    with tarfile.open(package_path, "w:gz") as archive:
        for name in package_names:
            archive.add(output_dir / name, arcname=name)

    print(summary.to_string(index=False))
    print()
    print("[PASS] reconstructed frozen full selection: 440/440")
    print("[PASS] reproduced all four published full-model endpoints")
    print("[OUTPUT]", output_dir)
    print("[PACKAGE]", package_path)
    print("[SHA256]", sha256(package_path))


if __name__ == "__main__":
    main()
