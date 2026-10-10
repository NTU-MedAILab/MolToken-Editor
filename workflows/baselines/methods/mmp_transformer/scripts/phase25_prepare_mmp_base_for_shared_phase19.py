#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--p24-root", type=Path, required=True)
    parser.add_argument("--p25-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-parents", type=int, default=88)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(smiles: Any, isomeric: bool = True) -> str:
    if pd.isna(smiles):
        return ""
    text = str(smiles).strip()
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return ""
    if len(Chem.GetMolFrags(mol)) != 1:
        return ""
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=isomeric,
    )


def find_col(
    frame: pd.DataFrame,
    choices: list[str],
    required: bool = True,
) -> str | None:
    for column in choices:
        if column in frame.columns:
            return column
    if required:
        raise KeyError(f"Missing one of required columns: {choices}")
    return None


def value_from_row(
    row: pd.Series,
    choices: list[str],
    default: Any = np.nan,
) -> Any:
    for column in choices:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return default


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    p24 = args.p24_root.expanduser().resolve()
    p25 = args.p25_root.expanduser().resolve()
    seed = args.seed

    generated_path = (
        p25
        / "external/deep-molecular-optimization/experiments"
        / f"phase25_exact88_single_logd_seed_{seed}"
        / "phase25_exact88_single_logd_model_input/evaluation_60"
        / "generated_molecules.csv"
    )
    model_input_path = (
        p25
        / "data/phase25_exact88_single_logd"
        / "phase25_exact88_single_logd_model_input.csv"
    )
    phase19_template_path = (
        p24
        / "outputs/reinvent4_seed_20260703_server/shared_phase19"
        / "reinvent4_phase19_candidates_unique_valid.csv"
    )

    for path in (
        generated_path,
        model_input_path,
        phase19_template_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    generated = pd.read_csv(generated_path, low_memory=False)
    model_input = pd.read_csv(model_input_path, low_memory=False)
    template = pd.read_csv(phase19_template_path, low_memory=False)

    generated_parent_col = find_col(
        generated,
        ["parent_id", "Phase25_parent_id", "mol_id"],
    )
    model_parent_col = find_col(
        model_input,
        ["parent_id", "Phase25_parent_id", "mol_id"],
    )
    template_parent_col = find_col(template, ["mol_id", "parent_id"])

    prediction_columns = sorted(
        [
            column
            for column in generated.columns
            if re.fullmatch(r"Predicted_smi_\d+", str(column))
        ],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )

    if len(generated) != args.expected_parents:
        raise ValueError(
            f"Expected {args.expected_parents} generated rows, "
            f"found {len(generated)}"
        )
    if generated[generated_parent_col].astype(str).nunique() != args.expected_parents:
        raise ValueError("Generated parent coverage is not 88.")
    if len(prediction_columns) != 50:
        raise ValueError(
            f"Expected 50 prediction columns, found {len(prediction_columns)}"
        )

    generated = generated.copy()
    model_input = model_input.copy()
    template = template.copy()

    generated["_parent_key"] = (
        generated[generated_parent_col].astype(str).str.strip()
    )
    model_input["_parent_key"] = (
        model_input[model_parent_col].astype(str).str.strip()
    )
    template["_parent_key"] = (
        template[template_parent_col].astype(str).str.strip()
    )

    if model_input["_parent_key"].duplicated().any():
        raise ValueError("Model-input parent IDs are not unique.")

    model_lookup = model_input.set_index("_parent_key", drop=False)
    template_lookup = (
        template.sort_index()
        .drop_duplicates("_parent_key", keep="first")
        .set_index("_parent_key", drop=False)
    )

    generated_parent_set = set(generated["_parent_key"])
    missing_model = sorted(generated_parent_set - set(model_lookup.index))
    missing_template = sorted(generated_parent_set - set(template_lookup.index))
    if missing_model:
        raise ValueError(f"Parents missing from model input: {missing_model}")
    if missing_template:
        raise ValueError(f"Parents missing from Phase19 template: {missing_template}")

    accepted_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    parent_summary_rows: list[dict[str, Any]] = []

    global_attempt_index = 0

    for parent_index, generated_row in generated.reset_index(drop=True).iterrows():
        parent_id = str(generated_row["_parent_key"])
        model_row = model_lookup.loc[parent_id]
        template_row = template_lookup.loc[parent_id]

        locked_parent_smiles = value_from_row(
            model_row,
            ["Source_Mol_locked", "Source_Mol", "canonical_smiles"],
            default=template_row.get("canonical_smiles", ""),
        )
        parent_connectivity = canonicalize(
            locked_parent_smiles,
            isomeric=False,
        )

        optimization_direction = str(
            value_from_row(
                model_row,
                ["optimization_direction", "replace_direction"],
                default=template_row.get("replace_direction", "unknown"),
            )
        ).strip()

        seen: set[str] = set()
        nonmissing = 0
        valid_single = 0
        same_as_parent = 0
        duplicates = 0
        accepted_for_parent = 0

        for prediction_column in prediction_columns:
            source_rank = int(prediction_column.rsplit("_", 1)[1])
            raw_smiles = generated_row[prediction_column]

            if pd.isna(raw_smiles) or not str(raw_smiles).strip():
                global_attempt_index += 1
                continue

            nonmissing += 1
            canonical = canonicalize(raw_smiles, isomeric=True)
            connectivity = canonicalize(raw_smiles, isomeric=False)

            if not canonical:
                global_attempt_index += 1
                continue

            valid_single += 1

            if connectivity and connectivity == parent_connectivity:
                same_as_parent += 1
                global_attempt_index += 1
                continue

            if canonical in seen:
                duplicates += 1
                global_attempt_index += 1
                continue

            seen.add(canonical)
            accepted_for_parent += 1

            token_id = (
                f"MMP25_S{seed}_{parent_id}_BASE_R{source_rank:02d}"
            )

            adapted = {
                column: template_row.get(column, np.nan)
                for column in template.columns
                if column != "_parent_key"
            }

            adapted.update(
                {
                    "method": "mmp_transformer_single_logd",
                    "edit_action": "generate",
                    "method_group": "mmp_transformer_single_logd",
                    "mol_id": parent_id,
                    "token_id": token_id,
                    "source_target_token_id": token_id,
                    "counterfactual_smiles": canonical,
                    "edited_smiles": canonical,
                    "replace_direction": optimization_direction,
                    "edit_valid": True,
                    "edit_reason": "ok",
                    "baseline_method_group": "mmp_transformer_single_logd",
                    "phase19_baseline": True,
                    "reagent_type": "external_single_target_generator",
                    "reagent_role": optimization_direction,
                    "reagent_source": (
                        "MMP-Transformer_He2021_"
                        "official_pretrained_single_logD_condition"
                    ),
                    "chemprop_safe_smiles": canonical,
                    "chemprop_safe_reason": "ok",
                    "_raw_attempt_index": global_attempt_index,
                    "_candidate_canonical_smiles": canonical,
                    "_shared_reward_eligible": True,
                    "step": source_rank,
                }
            )

            if "SMILES_state" in adapted:
                adapted["SMILES_state"] = 1

            accepted_rows.append(adapted)

            original_main_pred = value_from_row(
                template_row,
                ["main_original_pred_mean"],
                default=value_from_row(
                    model_row,
                    [
                        "original_main_pred_logd",
                        "parent_predicted_logd",
                        "main_original_pred_mean",
                    ],
                ),
            )

            oracle_rows.append(
                {
                    "candidate_id": token_id,
                    "smiles": canonical,
                    "Phase25_parent_index": parent_index + 1,
                    "parent_id": parent_id,
                    "candidate_rank": accepted_for_parent,
                    "generated_smiles": str(raw_smiles).strip(),
                    "canonical_generated_smiles": canonical,
                    "Source_Mol_locked": value_from_row(
                        model_row,
                        ["Source_Mol_locked"],
                        default=locked_parent_smiles,
                    ),
                    "Source_Mol": value_from_row(
                        model_row,
                        ["Source_Mol"],
                        default=locked_parent_smiles,
                    ),
                    "original_main_pred_logd": original_main_pred,
                    "optimization_direction": optimization_direction,
                    "target_low": 1.0,
                    "target_high": 3.4,
                    "condition_target_logd": value_from_row(
                        model_row,
                        ["condition_target_logd"],
                    ),
                    "desired_delta_logd": value_from_row(
                        model_row,
                        ["desired_delta_logd"],
                    ),
                    "Delta_LogD": value_from_row(
                        model_row,
                        ["Delta_LogD"],
                    ),
                    "model_input_normalization": value_from_row(
                        model_row,
                        ["model_input_normalization"],
                    ),
                    "source_run": "base",
                    "source_run_order": 0,
                    "source_rank": source_rank,
                    "rdkit_valid_pre_oracle": True,
                }
            )

            lineage_rows.append(
                {
                    "token_id": token_id,
                    "source_target_token_id": token_id,
                    "mol_id": parent_id,
                    "parent_id": parent_id,
                    "generated_smiles": str(raw_smiles).strip(),
                    "canonical_generated_smiles": canonical,
                    "source_run": "base",
                    "source_run_order": 0,
                    "source_rank": source_rank,
                    "_raw_attempt_index": global_attempt_index,
                }
            )

            global_attempt_index += 1

        parent_summary_rows.append(
            {
                "parent_id": parent_id,
                "nonmissing_candidates": nonmissing,
                "valid_single_component_candidates": valid_single,
                "same_connectivity_as_parent": same_as_parent,
                "duplicates_within_parent": duplicates,
                "accepted_unique_valid_candidates": accepted_for_parent,
            }
        )

    adapted_frame = pd.DataFrame(accepted_rows)
    oracle_frame = pd.DataFrame(oracle_rows)
    lineage_frame = pd.DataFrame(lineage_rows)
    parent_summary = pd.DataFrame(parent_summary_rows)

    if adapted_frame.empty:
        raise ValueError("No candidates survived base-run filtering.")

    if adapted_frame["mol_id"].nunique() != args.expected_parents:
        missing = sorted(
            generated_parent_set - set(adapted_frame["mol_id"].astype(str))
        )
        raise ValueError(
            "Not all parents retained a unique valid base candidate. "
            f"Missing parents: {missing}"
        )

    if adapted_frame["token_id"].duplicated().any():
        raise ValueError("Generated token IDs are not unique.")
    if lineage_frame["token_id"].duplicated().any():
        raise ValueError("Lineage token IDs are not unique.")
    if adapted_frame.duplicated(
        ["mol_id", "_candidate_canonical_smiles"]
    ).any():
        raise ValueError("Duplicate parent-candidate canonical pairs remain.")

    # Match the formal Phase19 input schema exactly.
    template_columns = [
        column
        for column in template.columns
        if column != "_parent_key"
    ]
    for column in template_columns:
        if column not in adapted_frame.columns:
            adapted_frame[column] = np.nan
    adapted_frame = adapted_frame[template_columns]

    output_input_dir = (
        p25 / f"inputs/shared_phase19/seed_{seed}"
    )
    output_table_dir = (
        p25 / f"tables/shared_phase19/seed_{seed}"
    )
    output_oracle_dir = (
        p25 / f"inputs/oracle/seed_{seed}"
    )
    native_dir = (
        p25 / f"tables/exact88/seed_{seed}/native_single_run"
    )

    for directory in (
        output_input_dir,
        output_table_dir,
        output_oracle_dir,
        native_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    adapted_path = (
        output_input_dir
        / "mmp_transformer_phase19_candidates_base_unique_valid.csv"
    )
    lineage_path = (
        output_table_dir
        / "mmp_transformer_phase19_base_candidate_lineage.csv"
    )
    oracle_path = (
        output_oracle_dir
        / f"phase25_seed_{seed}_oracle_input_base_only.csv"
    )
    parent_summary_path = (
        native_dir / "mmp_base_generation_parent_summary.csv"
    )
    manifest_path = (
        native_dir / "mmp_base_preparation_manifest.json"
    )

    adapted_frame.to_csv(adapted_path, index=False)
    lineage_frame.to_csv(lineage_path, index=False)
    oracle_frame.to_csv(oracle_path, index=False)
    parent_summary.to_csv(parent_summary_path, index=False)

    counts = (
        adapted_frame.groupby("mol_id")
        .size()
        .describe()
    )

    manifest = {
        "phase": "phase25_mmp_base_only_preparation",
        "status": "pass",
        "seed": seed,
        "parents": int(adapted_frame["mol_id"].nunique()),
        "raw_nonmissing_candidates": int(
            parent_summary["nonmissing_candidates"].sum()
        ),
        "unique_valid_candidates": int(len(adapted_frame)),
        "candidates_per_parent": {
            "min": int(counts["min"]),
            "median": float(counts["50%"]),
            "mean": float(counts["mean"]),
            "max": int(counts["max"]),
        },
        "selection_policy": (
            "Base generation only. Candidates are traversed in "
            "Predicted_smi_1..50 order; missing, RDKit-invalid, "
            "multi-component, dummy-atom, parent-connectivity-identical, "
            "and within-parent duplicate candidates are removed. "
            "No property-oracle value is used for selection."
        ),
        "inputs": {
            "generated": {
                "path": str(generated_path),
                "sha256": sha256(generated_path),
            },
            "model_input": {
                "path": str(model_input_path),
                "sha256": sha256(model_input_path),
            },
            "phase19_template": {
                "path": str(phase19_template_path),
                "sha256": sha256(phase19_template_path),
            },
        },
        "outputs": {
            "adapted_phase19_input": {
                "path": str(adapted_path),
                "sha256": sha256(adapted_path),
            },
            "lineage": {
                "path": str(lineage_path),
                "sha256": sha256(lineage_path),
            },
            "oracle_audit_input": {
                "path": str(oracle_path),
                "sha256": sha256(oracle_path),
            },
            "parent_summary": {
                "path": str(parent_summary_path),
                "sha256": sha256(parent_summary_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("STATUS: pass")
    print(f"SEED={seed}")
    print(
        "RAW_NONMISSING="
        f"{int(parent_summary['nonmissing_candidates'].sum())}"
    )
    print(f"UNIQUE_VALID={len(adapted_frame)}")
    print(f"PARENTS={adapted_frame['mol_id'].nunique()}")
    print(
        "PER_PARENT_MIN_MEDIAN_MEAN_MAX="
        f"{int(counts['min'])}/"
        f"{float(counts['50%']):.1f}/"
        f"{float(counts['mean']):.3f}/"
        f"{int(counts['max'])}"
    )
    print("ADAPTED_INPUT:", adapted_path)
    print("LINEAGE:", lineage_path)
    print("MANIFEST:", manifest_path)
    print(f"PHASE25_MMP_BASE_PREP_SEED_{seed}_COMPLETE")


if __name__ == "__main__":
    main()
