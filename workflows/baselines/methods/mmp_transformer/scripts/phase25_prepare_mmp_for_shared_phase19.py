#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


TARGET_LOW = 1.0
TARGET_HIGH = 3.4

OUTPUT_COLUMNS = [
    "method",
    "edit_action",
    "method_group",
    "mol_id",
    "token_id",
    "source_target_token_id",
    "parent_rank",
    "parent_library_id",
    "canonical_smiles",
    "counterfactual_smiles",
    "edited_smiles",
    "risk_side_05",
    "replace_direction",
    "main_original_pred_mean",
    "main_original_target_distance",
    "main_original_pred_std",
    "main_original_pred_split_std",
    "xgboost_original_pred_mean",
    "xgboost_original_target_distance",
    "rf_original_pred_mean",
    "rf_original_target_distance",
    "edit_valid",
    "edit_reason",
    "baseline_method_group",
    "phase19_baseline",
    "reagent_type",
    "reagent_role",
    "reagent_source",
    "chemprop_safe_smiles",
    "chemprop_safe_reason",
    "_raw_attempt_index",
    "_candidate_canonical_smiles",
    "_shared_reward_eligible",
    "step",
    "SMILES_state",
]

PARENT_METADATA_COLUMNS = [
    "mol_id",
    "parent_rank",
    "parent_library_id",
    "canonical_smiles",
    "risk_side_05",
    "replace_direction",
    "main_original_pred_mean",
    "main_original_target_distance",
    "main_original_pred_std",
    "main_original_pred_split_std",
    "xgboost_original_pred_mean",
    "xgboost_original_target_distance",
    "rf_original_pred_mean",
    "rf_original_target_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oracle-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase19-template",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lineage-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=4400,
    )
    parser.add_argument(
        "--expected-parents",
        type=int,
        default=88,
    )
    parser.add_argument(
        "--expected-per-parent",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--parent-prediction-tolerance",
        type=float,
        default=5e-6,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def parse_molecule(value: object) -> Chem.Mol | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return Chem.MolFromSmiles(text)


def canonicalize(value: object) -> str | None:
    mol = parse_molecule(value)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def connectivity_key(value: object) -> str | None:
    mol = parse_molecule(value)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def main() -> None:
    args = parse_args()

    oracle_path = args.oracle_input.expanduser().resolve()
    template_path = (
        args.phase19_template.expanduser().resolve()
    )
    output_path = args.output.expanduser().resolve()
    lineage_path = (
        args.lineage_output.expanduser().resolve()
    )
    manifest_path = args.manifest.expanduser().resolve()

    for path in [oracle_path, template_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    lineage_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    oracle = pd.read_csv(
        oracle_path,
        low_memory=False,
    )

    template = pd.read_csv(
        template_path,
        low_memory=False,
    )

    required_oracle = {
        "candidate_id",
        "parent_id",
        "candidate_rank",
        "canonical_generated_smiles",
        "Source_Mol_locked",
        "original_main_pred_logd",
        "optimization_direction",
        "source_run",
        "source_run_order",
        "source_rank",
        "model_input_normalization",
        "Delta_LogD",
        "condition_target_logd",
        "desired_delta_logd",
    }

    missing_oracle = sorted(
        required_oracle.difference(
            oracle.columns
        )
    )

    if missing_oracle:
        raise KeyError(
            f"Oracle input missing columns: "
            f"{missing_oracle}"
        )

    missing_template = sorted(
        set(PARENT_METADATA_COLUMNS).difference(
            template.columns
        )
    )

    if missing_template:
        raise KeyError(
            f"Phase19 template missing columns: "
            f"{missing_template}"
        )

    if len(oracle) != args.expected_candidates:
        raise ValueError(
            f"Oracle rows {len(oracle)} != "
            f"{args.expected_candidates}"
        )

    if (
        oracle["candidate_id"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate candidate IDs in oracle input."
        )

    oracle["parent_id"] = (
        oracle["parent_id"]
        .astype(str)
        .str.strip()
    )

    if (
        oracle["parent_id"].nunique()
        != args.expected_parents
    ):
        raise ValueError(
            "Unexpected oracle parent count."
        )

    per_parent = oracle.groupby(
        "parent_id"
    ).size()

    if not per_parent.eq(
        args.expected_per_parent
    ).all():
        raise ValueError(
            "Oracle candidates are not exactly "
            f"{args.expected_per_parent} per parent."
        )

    template["mol_id"] = (
        template["mol_id"]
        .astype(str)
        .str.strip()
    )

    # 同一父分子在Phase19模板中的元数据必须完全一致。
    inconsistent = {}

    for column in PARENT_METADATA_COLUMNS[1:]:
        counts = (
            template.groupby("mol_id")[column]
            .nunique(dropna=False)
        )

        bad = counts[counts > 1]

        if not bad.empty:
            inconsistent[column] = (
                bad.index.astype(str).tolist()
            )

    if inconsistent:
        raise ValueError(
            "Inconsistent parent metadata in "
            f"Phase19 template: {inconsistent}"
        )

    metadata = (
        template[PARENT_METADATA_COLUMNS]
        .drop_duplicates(
            "mol_id",
            keep="first",
        )
        .reset_index(drop=True)
    )

    if (
        metadata["mol_id"].nunique()
        != args.expected_parents
    ):
        raise ValueError(
            "Phase19 template does not contain "
            f"{args.expected_parents} parents."
        )

    oracle_parent_set = set(
        oracle["parent_id"]
    )
    template_parent_set = set(
        metadata["mol_id"]
    )

    if oracle_parent_set != template_parent_set:
        raise ValueError(
            "Parent sets differ between Phase25 "
            "oracle input and Phase19 template. "
            f"Only Phase25: "
            f"{sorted(oracle_parent_set - template_parent_set)}; "
            f"only template: "
            f"{sorted(template_parent_set - oracle_parent_set)}"
        )

    work = oracle.merge(
        metadata,
        left_on="parent_id",
        right_on="mol_id",
        how="left",
        validate="many_to_one",
    )

    work = work.sort_values(
        [
            "parent_rank",
            "parent_id",
            "candidate_rank",
        ],
        kind="stable",
    ).reset_index(drop=True)

    frozen_parent_pred = pd.to_numeric(
        work["original_main_pred_logd"],
        errors="coerce",
    )

    template_parent_pred = pd.to_numeric(
        work["main_original_pred_mean"],
        errors="coerce",
    )

    if (
        frozen_parent_pred.isna().any()
        or template_parent_pred.isna().any()
    ):
        raise ValueError(
            "Missing parent prediction values."
        )

    prediction_difference = (
        frozen_parent_pred
        - template_parent_pred
    ).abs()

    max_prediction_difference = float(
        prediction_difference.max()
    )

    if (
        max_prediction_difference
        > args.parent_prediction_tolerance
    ):
        raise ValueError(
            "Frozen parent prediction differs from "
            "Phase19 metadata beyond tolerance: "
            f"{max_prediction_difference}"
        )

    phase25_direction = (
        work["optimization_direction"]
        .astype(str)
        .str.strip()
    )

    phase19_direction = (
        work["replace_direction"]
        .astype(str)
        .str.strip()
    )

    direction_mismatch = (
        phase25_direction
        != phase19_direction
    )

    if direction_mismatch.any():
        bad = (
            work.loc[
                direction_mismatch,
                [
                    "parent_id",
                    "optimization_direction",
                    "replace_direction",
                ],
            ]
            .drop_duplicates()
            .to_dict(orient="records")
        )

        raise ValueError(
            f"Optimization direction mismatch: {bad}"
        )

    candidate_canonical = work[
        "canonical_generated_smiles"
    ].map(canonicalize)

    if candidate_canonical.isna().any():
        bad = work.loc[
            candidate_canonical.isna(),
            "candidate_id",
        ].tolist()

        raise ValueError(
            f"Invalid candidate SMILES: {bad[:20]}"
        )

    candidate_graph = work[
        "canonical_generated_smiles"
    ].map(connectivity_key)

    locked_parent_graph = work[
        "Source_Mol_locked"
    ].map(connectivity_key)

    template_parent_graph = work[
        "canonical_smiles"
    ].map(connectivity_key)

    if locked_parent_graph.isna().any():
        raise ValueError(
            "Invalid locked parent SMILES."
        )

    if template_parent_graph.isna().any():
        raise ValueError(
            "Invalid Phase19 parent SMILES."
        )

    parent_graph_mismatch = (
        locked_parent_graph
        != template_parent_graph
    )

    if parent_graph_mismatch.any():
        bad = (
            work.loc[
                parent_graph_mismatch,
                "parent_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Locked and Phase19 parent connectivity "
            f"differs for: {bad}"
        )

    same_as_parent = (
        candidate_graph
        == template_parent_graph
    )

    if same_as_parent.any():
        bad = work.loc[
            same_as_parent,
            "candidate_id",
        ].tolist()

        raise ValueError(
            "Candidate has same connectivity as "
            f"locked parent: {bad[:20]}"
        )

    duplicate_mask = pd.DataFrame(
        {
            "parent_id": work["parent_id"],
            "candidate": candidate_canonical,
        }
    ).duplicated(
        [
            "parent_id",
            "candidate",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        bad = work.loc[
            duplicate_mask,
            [
                "parent_id",
                "candidate_id",
            ],
        ].head(20)

        raise ValueError(
            "Duplicate candidates within a parent:\n"
            + bad.to_string(index=False)
        )

    method_name = (
        "mmp_transformer_single_logd"
    )

    adapted = pd.DataFrame(
        {
            "method": method_name,
            "edit_action": "generate",
            "method_group": method_name,
            "mol_id": work["parent_id"],
            "token_id":
                work["candidate_id"].astype(str),
            "source_target_token_id":
                work["candidate_id"].astype(str),
            "parent_rank": work["parent_rank"],
            "parent_library_id":
                work["parent_library_id"],
            # Phase19定义：canonical_smiles为原始父分子。
            "canonical_smiles":
                work["canonical_smiles"],
            "counterfactual_smiles":
                candidate_canonical,
            "edited_smiles":
                candidate_canonical,
            "risk_side_05":
                work["risk_side_05"],
            "replace_direction":
                work["replace_direction"],
            "main_original_pred_mean":
                work["main_original_pred_mean"],
            "main_original_target_distance":
                work["main_original_target_distance"],
            "main_original_pred_std":
                work["main_original_pred_std"],
            "main_original_pred_split_std":
                work[
                    "main_original_pred_split_std"
                ],
            "xgboost_original_pred_mean":
                work["xgboost_original_pred_mean"],
            "xgboost_original_target_distance":
                work[
                    "xgboost_original_target_distance"
                ],
            "rf_original_pred_mean":
                work["rf_original_pred_mean"],
            "rf_original_target_distance":
                work[
                    "rf_original_target_distance"
                ],
            "edit_valid": True,
            "edit_reason": "ok",
            "baseline_method_group":
                method_name,
            "phase19_baseline": True,
            "reagent_type":
                "external_single_target_generator",
            "reagent_role":
                work["replace_direction"],
            "reagent_source":
                (
                    "MMP-Transformer_He2021_"
                    "official_pretrained_"
                    "single_logD_condition"
                ),
            "chemprop_safe_smiles": True,
            "chemprop_safe_reason": "ok",
            "_raw_attempt_index":
                range(len(work)),
            "_candidate_canonical_smiles":
                candidate_canonical,
            "_shared_reward_eligible": True,
            "step":
                work["candidate_rank"].astype(int),
            "SMILES_state": 1,
        }
    )

    adapted = adapted[OUTPUT_COLUMNS]

    if adapted.isna().any().any():
        # parent_rank等理论上不应为空；错误时打印具体列。
        missing_counts = (
            adapted.isna().sum()
        )
        missing_counts = missing_counts[
            missing_counts > 0
        ]

        raise ValueError(
            "Missing values in adapted input:\n"
            + missing_counts.to_string()
        )

    if not adapted[
        "edit_valid"
    ].all():
        raise ValueError(
            "Adapted table contains edit_valid=False."
        )

    if not adapted[
        "_shared_reward_eligible"
    ].all():
        raise ValueError(
            "Adapted table contains "
            "reward-ineligible candidates."
        )

    adapted_counts = adapted.groupby(
        "mol_id"
    ).size()

    if not adapted_counts.eq(
        args.expected_per_parent
    ).all():
        raise ValueError(
            "Adapted candidates are not exactly "
            f"{args.expected_per_parent} per parent."
        )

    adapted.to_csv(
        output_path,
        index=False,
    )

    lineage_columns = [
        "candidate_id",
        "parent_id",
        "candidate_rank",
        "source_run",
        "source_run_order",
        "source_rank",
        "model_input_normalization",
        "Delta_LogD",
        "condition_target_logd",
        "desired_delta_logd",
    ]

    lineage = work[
        lineage_columns
    ].copy()

    lineage.insert(
        0,
        "token_id",
        work["candidate_id"].astype(str),
    )

    lineage.to_csv(
        lineage_path,
        index=False,
    )

    checks = {
        "candidate_rows":
            len(adapted)
            == args.expected_candidates,
        "parent_coverage":
            adapted["mol_id"].nunique()
            == args.expected_parents,
        "candidates_per_parent":
            bool(
                adapted_counts.eq(
                    args.expected_per_parent
                ).all()
            ),
        "token_ids_unique":
            not adapted[
                "token_id"
            ].duplicated().any(),
        "all_edit_valid":
            bool(
                adapted[
                    "edit_valid"
                ].all()
            ),
        "all_reward_eligible":
            bool(
                adapted[
                    "_shared_reward_eligible"
                ].all()
            ),
        "no_same_parent_connectivity":
            not bool(same_as_parent.any()),
        "no_parent_candidate_duplicates":
            not bool(duplicate_mask.any()),
        "parent_direction_match":
            not bool(
                direction_mismatch.any()
            ),
        "parent_connectivity_match":
            not bool(
                parent_graph_mismatch.any()
            ),
        "parent_prediction_match":
            (
                max_prediction_difference
                <= args.parent_prediction_tolerance
            ),
    }

    manifest = {
        "phase":
            "phase25_prepare_mmp_for_shared_phase19",
        "status":
            "pass"
            if all(checks.values())
            else "needs_review",
        "method":
            method_name,
        "target_window": [
            TARGET_LOW,
            TARGET_HIGH,
        ],
        "checks": checks,
        "counts": {
            "candidate_rows":
                int(len(adapted)),
            "parents":
                int(
                    adapted[
                        "mol_id"
                    ].nunique()
                ),
            "candidates_per_parent_min":
                int(
                    adapted_counts.min()
                ),
            "candidates_per_parent_max":
                int(
                    adapted_counts.max()
                ),
            "lineage_rows":
                int(len(lineage)),
        },
        "parent_prediction_validation": {
            "tolerance":
                args.parent_prediction_tolerance,
            "max_abs_difference":
                max_prediction_difference,
            "mean_abs_difference":
                float(
                    prediction_difference.mean()
                ),
        },
        "input": {
            "oracle_input":
                str(oracle_path),
            "oracle_input_sha256":
                sha256(oracle_path),
            "phase19_template":
                str(template_path),
            "phase19_template_sha256":
                sha256(template_path),
        },
        "output": {
            "candidates":
                str(output_path),
            "candidates_sha256":
                sha256(output_path),
            "lineage":
                str(lineage_path),
            "lineage_sha256":
                sha256(lineage_path),
        },
        "selection_policy": (
            "All 50 oracle-independent unique valid "
            "MMP-Transformer candidates per parent "
            "are retained in generation order."
        ),
        "parent_reference_policy": (
            "Shared Phase19 parent metadata and locked "
            "original parent structures are reused "
            "without modification."
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("===== ADAPTED FIRST ROW =====")
    print(
        adapted.head(1).to_string(
            index=False
        )
    )

    print()
    print(
        "PHASE25_MMP_SHARED_PHASE19_"
        "ADAPTATION_COMPLETE"
    )

    if manifest["status"] != "pass":
        raise SystemExit(
            "Adaptation integrity checks failed."
        )


if __name__ == "__main__":
    main()
