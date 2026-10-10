#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEEDS = [20260703, 20260704, 20260705]
METHODS = [
    "moltoken_editor",
    "reinvent4_mol2mol",
    "mmp_transformer_base",
]
METHOD_LABELS = {
    "moltoken_editor": "MolToken-Editor",
    "reinvent4_mol2mol": "REINVENT4",
    "mmp_transformer_base": "MMP-Transformer",
}
PRIMARY_PROTOCOL = "oracle_independent_B5_any_hit"
SECONDARY_PROTOCOL = "shared_score_ranked_topk"
PRIMARY_K = 5
ENDPOINTS = [
    "parent_target_hit",
    "parent_strict_hit",
    "candidate_target_rate",
    "candidate_strict_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the canonical Phase26 B=5 cap comparison across "
            "three seeds and add manuscript-ready tables to the final package."
        )
    )
    parser.add_argument("--p26-root", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--expected-parents", type=int, default=88)
    parser.add_argument("--budget", type=int, default=5)
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
        raise KeyError(f"Missing required column; tried {choices}")
    return None


def load_direction_map(
    p26: Path,
    expected_parents: int,
) -> dict[str, str]:
    path = p26 / "inputs/common_exact88/phase26_locked_exact88.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, low_memory=False)
    parent_col = find_col(frame, ["parent_id", "mol_id"])
    direction_col = find_col(
        frame,
        [
            "optimization_direction",
            "replace_direction",
            "direction",
            "task_direction",
        ],
    )

    work = frame[[parent_col, direction_col]].copy()
    work.columns = ["mol_id", "direction"]
    work["mol_id"] = work["mol_id"].astype(str).str.strip()
    work["direction"] = work["direction"].astype(str).str.strip()
    work = work.drop_duplicates("mol_id", keep="first")

    if len(work) != expected_parents:
        raise ValueError(
            f"Locked parent table has {len(work)} unique parents, "
            f"expected {expected_parents}"
        )
    if work["direction"].eq("").any():
        raise ValueError("Locked parent table contains blank directions.")

    return dict(zip(work["mol_id"], work["direction"]))


def load_inputs(
    p26: Path,
    expected_parents: int,
    budget: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    direction_map = load_direction_map(p26, expected_parents)
    frames: list[pd.DataFrame] = []
    input_metadata: dict[str, Any] = {}

    for seed in SEEDS:
        root = p26 / f"runs/equal_budget/seed_{seed}"
        parent_path = root / "phase26_b5_parent_metrics.csv"
        manifest_path = root / "phase26_b5_comparison_manifest.json"

        for path in [parent_path, manifest_path]:
            if not path.exists():
                raise FileNotFoundError(path)

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.get("status") != "pass":
            raise ValueError(f"{manifest_path}: status is not pass")
        if manifest.get("phase") != (
            "phase26_common_budget_b5_cap_comparison"
        ):
            raise ValueError(
                f"{manifest_path}: unexpected phase "
                f"{manifest.get('phase')}"
            )
        if int(manifest.get("seed")) != seed:
            raise ValueError(f"{manifest_path}: seed mismatch")
        if int(manifest.get("budget_cap_per_parent")) != budget:
            raise ValueError(f"{manifest_path}: budget mismatch")

        frame = pd.read_csv(parent_path, low_memory=False)
        required = {
            "seed",
            "method_group",
            "protocol",
            "k",
            "mol_id",
            "selected_candidates",
            "missing_generation_slots",
            "available_ranked_candidates",
            "candidate_rate_denominator",
            "parent_target_hit",
            "parent_strict_hit",
            "candidate_target_rate",
            "candidate_strict_rate",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(
                f"{parent_path} missing columns: {sorted(missing)}"
            )

        frame["seed"] = pd.to_numeric(
            frame["seed"], errors="raise"
        ).astype(int)
        if not frame["seed"].eq(seed).all():
            raise ValueError(f"{parent_path}: row-level seed mismatch")

        frame["mol_id"] = frame["mol_id"].astype(str).str.strip()
        if frame["mol_id"].nunique() != expected_parents:
            raise ValueError(
                f"{parent_path}: parent count "
                f"{frame['mol_id'].nunique()} != {expected_parents}"
            )

        if set(frame["method_group"]) != set(METHODS):
            raise ValueError(
                f"{parent_path}: unexpected methods "
                f"{sorted(frame['method_group'].unique())}"
            )

        frame["direction"] = frame["mol_id"].map(direction_map)
        if frame["direction"].isna().any():
            missing_ids = sorted(
                frame.loc[frame["direction"].isna(), "mol_id"].unique()
            )
            raise ValueError(
                f"{parent_path}: missing directions for {missing_ids}"
            )

        for column in [
            "parent_target_hit",
            "parent_strict_hit",
        ]:
            if frame[column].dtype != bool:
                frame[column] = (
                    frame[column]
                    .fillna(False)
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"true", "1", "1.0", "yes", "pass"})
                )

        for column in [
            "candidate_target_rate",
            "candidate_strict_rate",
            "selected_candidates",
            "missing_generation_slots",
            "available_ranked_candidates",
            "candidate_rate_denominator",
        ]:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )

        expected_rows = len(METHODS) * expected_parents * 4
        if len(frame) != expected_rows:
            raise ValueError(
                f"{parent_path}: rows {len(frame)} != {expected_rows}"
            )

        duplicate_key = [
            "method_group",
            "protocol",
            "k",
            "mol_id",
        ]
        if frame.duplicated(duplicate_key).any():
            raise ValueError(
                f"{parent_path}: duplicate method/protocol/K/parent keys"
            )

        frames.append(frame)
        input_metadata[str(parent_path)] = {
            "sha256": sha256(parent_path),
            "rows": int(len(frame)),
        }
        input_metadata[str(manifest_path)] = {
            "sha256": sha256(manifest_path),
            "status": manifest.get("status"),
            "phase": manifest.get("phase"),
        }

    combined = pd.concat(frames, ignore_index=True)

    # MolToken is deterministic: the parent-level outputs must be identical.
    moltoken = combined[
        combined["method_group"].eq("moltoken_editor")
    ].copy()
    compare_columns = [
        "protocol",
        "k",
        "mol_id",
        "selected_candidates",
        "missing_generation_slots",
        "available_ranked_candidates",
        "candidate_rate_denominator",
        *ENDPOINTS,
        "direction",
    ]
    reference = (
        moltoken[moltoken["seed"].eq(SEEDS[0])][compare_columns]
        .sort_values(["protocol", "k", "mol_id"])
        .reset_index(drop=True)
    )
    for seed in SEEDS[1:]:
        current = (
            moltoken[moltoken["seed"].eq(seed)][compare_columns]
            .sort_values(["protocol", "k", "mol_id"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            reference,
            current,
            check_exact=False,
            atol=1e-12,
            rtol=0,
        )

    # Fixed-K denominator audit.
    expected_denominator = combined["k"].astype(int)
    if not np.array_equal(
        combined["candidate_rate_denominator"].astype(int).to_numpy(),
        expected_denominator.to_numpy(),
    ):
        raise ValueError("Candidate-rate denominator is not equal to fixed K.")

    # Primary protocol must be K=5 only.
    primary = combined[
        combined["protocol"].eq(PRIMARY_PROTOCOL)
    ]
    if set(primary["k"].astype(int)) != {PRIMARY_K}:
        raise ValueError("Primary protocol does not contain only K=5.")

    return combined, input_metadata


def summarize_seedwise(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "parent_target_hit",
        "parent_strict_hit",
        "candidate_target_rate",
        "candidate_strict_rate",
        "selected_candidates",
        "missing_generation_slots",
        "available_ranked_candidates",
    ]

    seed_rows: list[dict[str, Any]] = []
    for (
        seed,
        method,
        protocol,
        k,
    ), group in combined.groupby(
        ["seed", "method_group", "protocol", "k"],
        dropna=False,
    ):
        seed_rows.append(
            {
                "seed": int(seed),
                "method_group": method,
                "method_label": METHOD_LABELS[method],
                "protocol": protocol,
                "k": int(k),
                "parents": int(group["mol_id"].nunique()),
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
                "mean_selected_candidates": float(
                    group["selected_candidates"].mean()
                ),
                "total_missing_generation_slots": int(
                    group["missing_generation_slots"].sum()
                ),
                "mean_available_ranked_candidates": float(
                    group["available_ranked_candidates"].mean()
                ),
            }
        )

    return pd.DataFrame(seed_rows)


def aggregate_method_summary(
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    metrics = [
        "parent_target_hit_rate",
        "parent_strict_hit_rate",
        "candidate_target_rate_fixed_k",
        "candidate_strict_rate_fixed_k",
        "mean_selected_candidates",
        "total_missing_generation_slots",
        "mean_available_ranked_candidates",
    ]

    for (method, protocol, k), group in seed_summary.groupby(
        ["method_group", "protocol", "k"],
        dropna=False,
    ):
        deterministic = method == "moltoken_editor"
        row: dict[str, Any] = {
            "method_group": method,
            "method_label": METHOD_LABELS[method],
            "protocol": protocol,
            "k": int(k),
            "seed_treatment": (
                "fixed_deterministic"
                if deterministic
                else "three_seed_mean_sd"
            ),
            "n_seed_runs": 1 if deterministic else len(SEEDS),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="raise",
            ).to_numpy(dtype=float)

            if deterministic:
                if not np.allclose(
                    values,
                    values[0],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError(
                        f"MolToken deterministic mismatch for {metric}"
                    )
                mean = float(values[0])
                sd = np.nan
            else:
                mean = float(values.mean())
                sd = float(values.std(ddof=1))

            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_display"] = (
                f"{mean:.4f} (fixed)"
                if deterministic
                else f"{mean:.4f} ± {sd:.4f}"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    combined: pd.DataFrame,
    reps: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)

    conditions = (
        combined[["protocol", "k"]]
        .drop_duplicates()
        .sort_values(["protocol", "k"])
        .itertuples(index=False)
    )

    for condition_index, condition in enumerate(conditions):
        protocol = str(condition.protocol)
        k = int(condition.k)

        condition_frame = combined[
            combined["protocol"].eq(protocol)
            & combined["k"].eq(k)
        ].copy()

        parent_ids = sorted(condition_frame["mol_id"].unique())
        n_parents = len(parent_ids)

        moltoken = (
            condition_frame[
                condition_frame["method_group"].eq("moltoken_editor")
                & condition_frame["seed"].eq(SEEDS[0])
            ]
            .set_index("mol_id")
            .reindex(parent_ids)
        )

        for baseline_index, baseline in enumerate(
            ["reinvent4_mol2mol", "mmp_transformer_base"]
        ):
            baseline_frames = []
            for seed in SEEDS:
                current = (
                    condition_frame[
                        condition_frame["method_group"].eq(baseline)
                        & condition_frame["seed"].eq(seed)
                    ]
                    .set_index("mol_id")
                    .reindex(parent_ids)
                )
                if current.isna().all(axis=1).any():
                    raise ValueError(
                        f"{baseline}/{seed}/{protocol}/K={k}: missing parent"
                    )
                baseline_frames.append(current)

            for endpoint_index, endpoint in enumerate(ENDPOINTS):
                mt_values = moltoken[endpoint].astype(float).to_numpy()
                baseline_values = np.stack(
                    [
                        frame[endpoint].astype(float).to_numpy()
                        for frame in baseline_frames
                    ],
                    axis=0,
                )

                observed = float(
                    mt_values.mean()
                    - baseline_values.mean(axis=0).mean()
                )

                samples = np.empty(reps, dtype=float)
                local_rng = np.random.default_rng(
                    bootstrap_seed
                    + condition_index * 1000
                    + baseline_index * 100
                    + endpoint_index
                )

                for rep in range(reps):
                    parent_draw = local_rng.integers(
                        0,
                        n_parents,
                        size=n_parents,
                    )
                    seed_draw = local_rng.integers(
                        0,
                        len(SEEDS),
                        size=len(SEEDS),
                    )
                    sampled_mt = mt_values[parent_draw]
                    sampled_baseline = (
                        baseline_values[seed_draw][:, parent_draw]
                        .mean(axis=0)
                    )
                    samples[rep] = float(
                        sampled_mt.mean()
                        - sampled_baseline.mean()
                    )

                lower, upper = np.quantile(
                    samples,
                    [0.025, 0.975],
                )
                excludes_zero = bool(
                    lower > 0 or upper < 0
                )

                rows.append(
                    {
                        "reference_method": "moltoken_editor",
                        "reference_label": METHOD_LABELS[
                            "moltoken_editor"
                        ],
                        "baseline_method": baseline,
                        "baseline_label": METHOD_LABELS[baseline],
                        "protocol": protocol,
                        "k": k,
                        "endpoint": endpoint,
                        "parents": n_parents,
                        "stochastic_baseline_seeds": len(SEEDS),
                        "bootstrap_reps": reps,
                        "observed_difference": observed,
                        "ci95_lower": float(lower),
                        "ci95_upper": float(upper),
                        "ci_excludes_zero": excludes_zero,
                        "interpretation": (
                            "CI excludes zero"
                            if excludes_zero
                            else "CI crosses zero"
                        ),
                        "note": (
                            "Hierarchical bootstrap resampled parents and "
                            "baseline seeds. No bootstrap p-value is reported."
                        ),
                    }
                )

    return pd.DataFrame(rows)


def direction_summary(
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = combined[
        combined["protocol"].eq(PRIMARY_PROTOCOL)
        & combined["k"].eq(PRIMARY_K)
    ].copy()

    seed_rows: list[dict[str, Any]] = []
    for (
        seed,
        method,
        direction,
    ), group in primary.groupby(
        ["seed", "method_group", "direction"],
        dropna=False,
    ):
        seed_rows.append(
            {
                "seed": int(seed),
                "method_group": method,
                "method_label": METHOD_LABELS[method],
                "direction": str(direction),
                "parents": int(group["mol_id"].nunique()),
                "parent_target_hit_rate": float(
                    group["parent_target_hit"].mean()
                ),
                "parent_strict_hit_rate": float(
                    group["parent_strict_hit"].mean()
                ),
                "candidate_target_rate_fixed_b5": float(
                    group["candidate_target_rate"].mean()
                ),
                "candidate_strict_rate_fixed_b5": float(
                    group["candidate_strict_rate"].mean()
                ),
                "total_missing_generation_slots": int(
                    group["missing_generation_slots"].sum()
                ),
            }
        )

    seed_table = pd.DataFrame(seed_rows)
    rows: list[dict[str, Any]] = []

    metrics = [
        "parent_target_hit_rate",
        "parent_strict_hit_rate",
        "candidate_target_rate_fixed_b5",
        "candidate_strict_rate_fixed_b5",
        "total_missing_generation_slots",
    ]

    for (method, direction), group in seed_table.groupby(
        ["method_group", "direction"],
        dropna=False,
    ):
        deterministic = method == "moltoken_editor"
        row: dict[str, Any] = {
            "method_group": method,
            "method_label": METHOD_LABELS[method],
            "direction": direction,
            "parents": int(group["parents"].iloc[0]),
            "seed_treatment": (
                "fixed_deterministic"
                if deterministic
                else "three_seed_mean_sd"
            ),
        }

        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values[0] if deterministic else values.mean())
            sd = (
                np.nan
                if deterministic
                else float(values.std(ddof=1))
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd

        rows.append(row)

    aggregate = pd.DataFrame(rows)
    return seed_table, aggregate


def write_interpretation(
    path: Path,
    primary_summary: pd.DataFrame,
    primary_bootstrap: pd.DataFrame,
) -> None:
    lines = [
        "# Phase26 equal-budget B=5 interpretation",
        "",
        "- Primary endpoint: parent-level target-window and strict hit "
        "within up to five candidates locked before property-oracle ranking.",
        "- Candidate success rates use a fixed B=5 denominator; unavailable "
        "generation slots and non-evidence candidates contribute zero successes.",
        "- MolToken-Editor is deterministic and is reported as a fixed estimate, "
        "not as mean ± 0.00.",
        "- REINVENT4 and MMP-Transformer are reported as three-seed mean ± SD.",
        "- Hierarchical bootstrap intervals resample parents and stochastic "
        "baseline seeds. They are confidence intervals, not p-values.",
        "- Shared-score Top-K is secondary recoverability sensitivity because "
        "final_edit_score overlaps the evaluated endpoint.",
        "",
        "## Primary method summary",
        "",
        "```text",
        primary_summary.to_string(index=False),
        "```",
        "",
        "## Primary hierarchical comparisons",
        "",
        "```text",
        primary_bootstrap.to_string(index=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    p26 = args.p26_root.expanduser().resolve()

    output_dir = p26 / "runs/equal_budget/summary"
    package_dir = p26 / "runs/equal_budget/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    combined, input_metadata = load_inputs(
        p26=p26,
        expected_parents=args.expected_parents,
        budget=args.budget,
    )
    seed_summary = summarize_seedwise(combined)
    aggregate_summary = aggregate_method_summary(seed_summary)
    bootstrap = hierarchical_bootstrap(
        combined=combined,
        reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    direction_seed, direction_aggregate = direction_summary(combined)

    primary_summary = aggregate_summary[
        aggregate_summary["protocol"].eq(PRIMARY_PROTOCOL)
        & aggregate_summary["k"].eq(PRIMARY_K)
    ].copy()
    secondary_summary = aggregate_summary[
        aggregate_summary["protocol"].eq(SECONDARY_PROTOCOL)
    ].copy()
    primary_bootstrap = bootstrap[
        bootstrap["protocol"].eq(PRIMARY_PROTOCOL)
        & bootstrap["k"].eq(PRIMARY_K)
    ].copy()
    secondary_bootstrap = bootstrap[
        bootstrap["protocol"].eq(SECONDARY_PROTOCOL)
    ].copy()

    output_paths = {
        "parent_seed_long":
            output_dir / "phase26_b5_parent_seed_long.csv",
        "seed_summary":
            output_dir / "phase26_b5_seed_summary.csv",
        "aggregate_summary":
            output_dir / "phase26_b5_multiseed_summary.csv",
        "hierarchical_bootstrap":
            output_dir / "phase26_b5_hierarchical_bootstrap.csv",
        "direction_seed":
            output_dir / "phase26_b5_direction_seed_summary.csv",
        "direction_aggregate":
            output_dir / "phase26_b5_direction_multiseed_summary.csv",
        "interpretation":
            output_dir / "phase26_b5_interpretation_notes.md",
        "manifest":
            output_dir / "phase26_b5_multiseed_manifest.json",
    }

    combined.to_csv(output_paths["parent_seed_long"], index=False)
    seed_summary.to_csv(output_paths["seed_summary"], index=False)
    aggregate_summary.to_csv(
        output_paths["aggregate_summary"], index=False
    )
    bootstrap.to_csv(
        output_paths["hierarchical_bootstrap"], index=False
    )
    direction_seed.to_csv(
        output_paths["direction_seed"], index=False
    )
    direction_aggregate.to_csv(
        output_paths["direction_aggregate"], index=False
    )
    write_interpretation(
        output_paths["interpretation"],
        primary_summary,
        primary_bootstrap,
    )

    package_paths = {
        "main_table_4_equal_budget_b5":
            package_dir / "main_table_4_equal_budget_b5.csv",
        "main_table_4_equal_budget_b5_pairwise":
            package_dir / (
                "main_table_4_equal_budget_b5_pairwise.csv"
            ),
        "supp_table_b5_secondary_topk":
            package_dir / "supp_table_b5_secondary_topk.csv",
        "supp_table_b5_secondary_pairwise":
            package_dir / "supp_table_b5_secondary_pairwise.csv",
        "supp_table_b5_direction":
            package_dir / "supp_table_b5_direction.csv",
        "b5_addendum_manifest":
            package_dir / (
                "phase26_b5_final_package_addendum_manifest.json"
            ),
    }

    primary_summary.to_csv(
        package_paths["main_table_4_equal_budget_b5"],
        index=False,
    )
    primary_bootstrap.to_csv(
        package_paths["main_table_4_equal_budget_b5_pairwise"],
        index=False,
    )
    secondary_summary.to_csv(
        package_paths["supp_table_b5_secondary_topk"],
        index=False,
    )
    secondary_bootstrap.to_csv(
        package_paths["supp_table_b5_secondary_pairwise"],
        index=False,
    )
    direction_aggregate.to_csv(
        package_paths["supp_table_b5_direction"],
        index=False,
    )

    manifest = {
        "phase": "phase26_b5_three_seed_aggregation",
        "status": "pass",
        "seeds": SEEDS,
        "parents": args.expected_parents,
        "budget_cap_per_parent": args.budget,
        "bootstrap": {
            "type": "hierarchical_parent_and_baseline_seed",
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "p_value_reported": False,
        },
        "method_treatment": {
            "moltoken_editor": "fixed deterministic estimate",
            "reinvent4_mol2mol": "three-seed mean and sample SD",
            "mmp_transformer_base": "three-seed mean and sample SD",
        },
        "primary_protocol": PRIMARY_PROTOCOL,
        "secondary_protocol": SECONDARY_PROTOCOL,
        "inputs": input_metadata,
        "outputs": {},
    }

    for name, path in output_paths.items():
        if name == "manifest":
            continue
        manifest["outputs"][name] = {
            "path": str(path),
            "sha256": sha256(path),
        }

    output_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    addendum_manifest = {
        "phase": "phase26_b5_final_package_addendum",
        "status": "pass",
        "source_manifest": {
            "path": str(output_paths["manifest"]),
            "sha256": sha256(output_paths["manifest"]),
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in package_paths.items()
            if name != "b5_addendum_manifest"
        },
        "note": (
            "Adds the missing equal-budget B=5 three-seed tables to "
            "the existing final manuscript package without rewriting "
            "the pre-existing package manifest."
        ),
    }
    package_paths["b5_addendum_manifest"].write_text(
        json.dumps(
            addendum_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "method_label",
        "parent_target_hit_rate_display",
        "parent_strict_hit_rate_display",
        "candidate_target_rate_fixed_k_display",
        "candidate_strict_rate_fixed_k_display",
        "mean_selected_candidates_display",
        "total_missing_generation_slots_display",
    ]

    print("STATUS: pass")
    print("===== PRIMARY EQUAL-BUDGET B=5 SUMMARY =====")
    print(primary_summary[display_columns].to_string(index=False))
    print()
    print("===== PRIMARY HIERARCHICAL COMPARISONS =====")
    print(
        primary_bootstrap[
            [
                "baseline_label",
                "endpoint",
                "observed_difference",
                "ci95_lower",
                "ci95_upper",
                "interpretation",
            ]
        ].to_string(index=False)
    )
    print("OUTPUT_DIR:", output_dir)
    print("FINAL_PACKAGE_DIR:", package_dir)
    print("MANIFEST:", output_paths["manifest"])
    print("PHASE26_B5_THREE_SEED_AGGREGATION_COMPLETE")


if __name__ == "__main__":
    main()
