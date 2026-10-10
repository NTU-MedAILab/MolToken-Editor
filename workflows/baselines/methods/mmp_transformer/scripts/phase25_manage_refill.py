#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate base and refill MMP-Transformer outputs, "
            "deduplicate candidates across rounds, select the first "
            "50 unique strict candidates per parent, and prepare the "
            "next refill input."
        )
    )
    parser.add_argument("--p25", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--target-count", type=int, default=50)
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


def parse_smiles(value: object) -> dict:
    if pd.isna(value):
        return {
            "raw": "",
            "valid": False,
            "canonical": None,
            "graph_key": None,
            "single_component": False,
            "has_dummy": False,
            "missing": True,
        }

    text = str(value).strip()

    if not text:
        return {
            "raw": "",
            "valid": False,
            "canonical": None,
            "graph_key": None,
            "single_component": False,
            "has_dummy": False,
            "missing": True,
        }

    mol = Chem.MolFromSmiles(text)

    if mol is None:
        return {
            "raw": text,
            "valid": False,
            "canonical": None,
            "graph_key": None,
            "single_component": False,
            "has_dummy": False,
            "missing": False,
        }

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    graph_key = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )

    return {
        "raw": text,
        "valid": True,
        "canonical": canonical,
        "graph_key": graph_key,
        "single_component": "." not in canonical,
        "has_dummy": any(
            atom.GetAtomicNum() == 0
            for atom in mol.GetAtoms()
        ),
        "missing": False,
    }


def prediction_columns(data: pd.DataFrame) -> list[str]:
    columns = [
        str(column)
        for column in data.columns
        if re.fullmatch(
            r"Predicted_smi_\d+",
            str(column),
        )
    ]

    return sorted(
        columns,
        key=lambda value: int(
            value.rsplit("_", 1)[1]
        ),
    )


def main() -> None:
    args = parse_args()

    p25 = Path(args.p25).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    exact_dir = Path(args.exact_dir).expanduser().resolve()

    base_seed = int(args.base_seed)
    target_count = int(args.target_count)

    master_path = (
        exact_dir
        / "phase25_exact88_single_logd_model_input.csv"
    )

    vocab_path = exact_dir / "vocab.pkl"

    base_result = (
        repo
        / "experiments"
        / f"phase25_exact88_single_logd_seed_{base_seed}"
        / "phase25_exact88_single_logd_model_input"
        / "evaluation_60"
        / "generated_molecules.csv"
    )

    for path in [master_path, vocab_path, base_result]:
        if not path.exists():
            raise FileNotFoundError(path)

    state_dir = (
        p25
        / "tables"
        / "exact88"
        / f"seed_{base_seed}"
        / "refill_state"
    )
    state_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path)

    if len(master) != 88:
        raise RuntimeError(
            f"Expected 88 master parents, found {len(master)}."
        )

    if master["parent_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate parent IDs in master input."
        )

    master["parent_id"] = (
        master["parent_id"]
        .astype(str)
        .str.strip()
    )

    master_by_id = master.set_index(
        "parent_id",
        drop=False,
    )

    parent_order = master["parent_id"].tolist()

    # 发现已经完成的补充采样轮次。
    refill_pattern = (
        repo
        / "experiments"
        / (
            f"phase25_exact88_refill_seed_"
            f"{base_seed}_round_*"
        )
    )

    refill_results = []

    for experiment_dir in sorted(
        refill_pattern.parent.glob(
            refill_pattern.name
        )
    ):
        match = re.search(
            r"round_(\d+)$",
            experiment_dir.name,
        )

        if not match:
            continue

        round_number = int(match.group(1))

        result_path = (
            experiment_dir
            / f"refill_round_{round_number:02d}"
            / "evaluation_60"
            / "generated_molecules.csv"
        )

        if result_path.exists():
            refill_results.append(
                (
                    round_number,
                    result_path,
                )
            )

    refill_results.sort(key=lambda item: item[0])

    run_sources = [
        {
            "run_order": 0,
            "run_label": "base",
            "refill_round": 0,
            "path": base_result,
        }
    ]

    for round_number, result_path in refill_results:
        run_sources.append(
            {
                "run_order": round_number,
                "run_label":
                    f"refill_round_{round_number:02d}",
                "refill_round": round_number,
                "path": result_path,
            }
        )

    parent_state = {}

    for parent_id in parent_order:
        master_row = master_by_id.loc[parent_id]

        model_parent = parse_smiles(
            master_row["Source_Mol"]
        )
        locked_parent = parse_smiles(
            master_row["Source_Mol_locked"]
        )

        if not model_parent["valid"]:
            raise RuntimeError(
                f"Invalid model parent: {parent_id}"
            )

        if not locked_parent["valid"]:
            raise RuntimeError(
                f"Invalid locked parent: {parent_id}"
            )

        parent_state[parent_id] = {
            "seen": set(),
            "accepted": [],
            "model_parent_canonical":
                model_parent["canonical"],
            "locked_parent_graph":
                locked_parent["graph_key"],
        }

    attempt_records = []
    source_hashes = {}

    for run in run_sources:
        result_path = run["path"]
        source_hashes[run["run_label"]] = (
            sha256(result_path)
        )

        generated = pd.read_csv(result_path)

        if generated["parent_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate parent rows in {result_path}"
            )

        generated["parent_id"] = (
            generated["parent_id"]
            .astype(str)
            .str.strip()
        )

        columns = prediction_columns(generated)

        if not columns:
            raise RuntimeError(
                f"No prediction columns in {result_path}"
            )

        for _, row in generated.iterrows():
            parent_id = str(row["parent_id"])

            if parent_id not in parent_state:
                raise RuntimeError(
                    f"Unknown parent ID: {parent_id}"
                )

            state = parent_state[parent_id]

            for rank, column in enumerate(
                columns,
                start=1,
            ):
                parsed = parse_smiles(row[column])

                same_as_model_parent = bool(
                    parsed["canonical"] is not None
                    and parsed["canonical"]
                    == state["model_parent_canonical"]
                )

                same_locked_connectivity = bool(
                    parsed["graph_key"] is not None
                    and parsed["graph_key"]
                    == state["locked_parent_graph"]
                )

                duplicate_across_rounds = bool(
                    parsed["canonical"] is not None
                    and parsed["canonical"]
                    in state["seen"]
                )

                chemical_accept = bool(
                    not parsed["missing"]
                    and parsed["valid"]
                    and parsed["single_component"]
                    and not parsed["has_dummy"]
                    and not same_as_model_parent
                    and not same_locked_connectivity
                )

                accepted_new_unique = bool(
                    chemical_accept
                    and not duplicate_across_rounds
                )

                unique_index = None
                selected_for_evaluation = False

                if accepted_new_unique:
                    state["seen"].add(
                        parsed["canonical"]
                    )

                    unique_index = (
                        len(state["accepted"]) + 1
                    )

                    candidate_record = {
                        "parent_id": parent_id,
                        "unique_candidate_index":
                            unique_index,
                        "generated_smiles":
                            parsed["raw"],
                        "canonical_generated_smiles":
                            parsed["canonical"],
                        "source_run":
                            run["run_label"],
                        "source_run_order":
                            run["run_order"],
                        "source_rank":
                            rank,
                    }

                    state["accepted"].append(
                        candidate_record
                    )

                    selected_for_evaluation = (
                        unique_index <= target_count
                    )

                attempt_records.append(
                    {
                        "parent_id": parent_id,
                        "optimization_direction":
                            master_by_id.loc[
                                parent_id,
                                "optimization_direction",
                            ],
                        "source_run":
                            run["run_label"],
                        "source_run_order":
                            run["run_order"],
                        "source_rank":
                            rank,
                        "generated_smiles":
                            parsed["raw"],
                        "is_missing":
                            parsed["missing"],
                        "is_valid_rdkit":
                            parsed["valid"],
                        "is_single_component":
                            parsed["single_component"],
                        "has_dummy_atom":
                            parsed["has_dummy"],
                        "same_as_model_parent":
                            same_as_model_parent,
                        "same_connectivity_as_locked_parent":
                            same_locked_connectivity,
                        "duplicate_across_rounds":
                            duplicate_across_rounds,
                        "accepted_new_unique":
                            accepted_new_unique,
                        "unique_candidate_index":
                            unique_index,
                        "selected_for_evaluation":
                            selected_for_evaluation,
                    }
                )

    attempts = pd.DataFrame(attempt_records)

    unique_records = []

    for parent_id in parent_order:
        unique_records.extend(
            parent_state[parent_id]["accepted"]
        )

    unique_pool = pd.DataFrame(unique_records)

    if unique_pool.empty:
        raise RuntimeError(
            "No unique candidates were collected."
        )

    unique_pool["selected_for_evaluation"] = (
        unique_pool["unique_candidate_index"]
        <= target_count
    )

    selected_long = unique_pool[
        unique_pool["selected_for_evaluation"]
    ].copy()

    parent_records = []

    for parent_id in parent_order:
        count = len(
            parent_state[parent_id]["accepted"]
        )

        selected_count = min(
            count,
            target_count,
        )

        parent_records.append(
            {
                "parent_id": parent_id,
                "optimization_direction":
                    master_by_id.loc[
                        parent_id,
                        "optimization_direction",
                    ],
                "model_input_normalization":
                    master_by_id.loc[
                        parent_id,
                        "model_input_normalization",
                    ],
                "total_unique_candidates":
                    count,
                "selected_candidates":
                    selected_count,
                "deficit":
                    max(
                        0,
                        target_count - selected_count,
                    ),
                "complete":
                    selected_count == target_count,
            }
        )

    parent_status = pd.DataFrame(
        parent_records
    )

    selected_wide = master.copy()

    for rank in range(1, target_count + 1):
        values = []

        for parent_id in parent_order:
            candidates = parent_state[
                parent_id
            ]["accepted"]

            if len(candidates) >= rank:
                values.append(
                    candidates[rank - 1][
                        "generated_smiles"
                    ]
                )
            else:
                values.append("")

        selected_wide[
            f"Predicted_smi_{rank}"
        ] = values

    attempts_path = (
        state_dir
        / "all_candidate_attempts.csv"
    )
    unique_path = (
        state_dir
        / "unique_candidate_pool.csv"
    )
    selected_long_path = (
        state_dir
        / "selected_50_candidates_long.csv"
    )
    selected_wide_path = (
        state_dir
        / "selected_50_candidates_wide.csv"
    )
    parent_status_path = (
        state_dir
        / "parent_refill_status.csv"
    )
    summary_path = (
        state_dir
        / "refill_state_summary.json"
    )

    attempts.to_csv(
        attempts_path,
        index=False,
    )
    unique_pool.to_csv(
        unique_path,
        index=False,
    )
    selected_long.to_csv(
        selected_long_path,
        index=False,
    )
    selected_wide.to_csv(
        selected_wide_path,
        index=False,
    )
    parent_status.to_csv(
        parent_status_path,
        index=False,
    )

    incomplete = parent_status[
        ~parent_status["complete"]
    ].copy()

    completed_rounds = [
        round_number
        for round_number, _ in refill_results
    ]

    next_round = (
        max(completed_rounds) + 1
        if completed_rounds
        else 1
    )

    next_input_csv = None
    next_input_dir = None

    if not incomplete.empty:
        next_input_dir = (
            p25
            / "data"
            / (
                f"phase25_exact88_refill_seed_"
                f"{base_seed}_round_{next_round:02d}"
            )
        )

        next_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            vocab_path,
            next_input_dir / "vocab.pkl",
        )

        incomplete_ids = set(
            incomplete["parent_id"]
        )

        next_input = master[
            master["parent_id"].isin(
                incomplete_ids
            )
        ].copy()

        status_lookup = (
            parent_status
            .set_index("parent_id")
        )

        next_input["refill_base_seed"] = (
            base_seed
        )
        next_input["refill_round"] = (
            next_round
        )
        next_input[
            "refill_current_unique"
        ] = next_input["parent_id"].map(
            status_lookup[
                "selected_candidates"
            ]
        )
        next_input["refill_deficit"] = (
            next_input["parent_id"].map(
                status_lookup["deficit"]
            )
        )

        next_input_csv = (
            next_input_dir
            / f"refill_round_{next_round:02d}.csv"
        )

        next_input.to_csv(
            next_input_csv,
            index=False,
        )

    total_selected = int(
        parent_status[
            "selected_candidates"
        ].sum()
    )

    total_deficit = int(
        parent_status["deficit"].sum()
    )

    summary = {
        "phase":
            "phase25_exact88_refill_state",
        "base_seed": base_seed,
        "target_candidates_per_parent":
            target_count,
        "source_runs": [
            {
                "run_label":
                    run["run_label"],
                "refill_round":
                    run["refill_round"],
                "path":
                    str(run["path"].resolve()),
                "sha256":
                    source_hashes[
                        run["run_label"]
                    ],
            }
            for run in run_sources
        ],
        "completed_refill_rounds":
            completed_rounds,
        "parents": int(len(parent_status)),
        "parents_complete": int(
            parent_status["complete"].sum()
        ),
        "parents_incomplete": int(
            (~parent_status["complete"]).sum()
        ),
        "selected_candidates":
            total_selected,
        "target_candidate_slots":
            int(len(parent_status) * target_count),
        "remaining_deficit":
            total_deficit,
        "minimum_selected_per_parent":
            int(
                parent_status[
                    "selected_candidates"
                ].min()
            ),
        "median_selected_per_parent":
            float(
                parent_status[
                    "selected_candidates"
                ].median()
            ),
        "maximum_selected_per_parent":
            int(
                parent_status[
                    "selected_candidates"
                ].max()
            ),
        "status": (
            "complete"
            if incomplete.empty
            else "refill_required"
        ),
        "selection_policy": (
            "First 50 unique strict candidates "
            "in deterministic base-run, refill-round, "
            "and source-rank order. No property oracle "
            "is used during candidate selection."
        ),
        "next_refill_round": (
            None
            if incomplete.empty
            else next_round
        ),
        "next_refill_input_csv": (
            None
            if next_input_csv is None
            else str(
                next_input_csv.resolve()
            )
        ),
        "next_refill_input_rows": (
            0
            if incomplete.empty
            else int(len(incomplete))
        ),
        "all_attempts_csv":
            str(attempts_path.resolve()),
        "unique_pool_csv":
            str(unique_path.resolve()),
        "selected_long_csv":
            str(selected_long_path.resolve()),
        "selected_wide_csv":
            str(selected_wide_path.resolve()),
        "parent_status_csv":
            str(parent_status_path.resolve()),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("===== INCOMPLETE PARENTS =====")

    if incomplete.empty:
        print("ALL_PARENTS_COMPLETE")
    else:
        print(
            incomplete[
                [
                    "parent_id",
                    "selected_candidates",
                    "deficit",
                ]
            ]
            .sort_values(
                [
                    "selected_candidates",
                    "parent_id",
                ]
            )
            .head(30)
            .to_string(index=False)
        )

    print()
    print("REFILL_STATE_UPDATED")


if __name__ == "__main__":
    main()
