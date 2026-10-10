from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moltoken_common import TargetWindow, ensure_dirs, morgan_features, rdkit_descriptor_frame
from contribution_quality import assign_status, remove_atoms_counterfactual, score_rows


rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")

PYTHON = Path(sys.executable)
PHASE04 = ROOT / "predictors"
PHASE10 = ROOT / "moltoken"
CHEMPROP_CLI = PHASE04 / "scripts" / "chemprop_stable_cli.py"

PHASE10_CONFIG = PHASE10 / "configs" / "target_expansion.json"
QUALITY_CONFIG = PHASE10 / "configs" / "quality_scoring.json"
ACTIVE_PARENT_PATH = PHASE10 / "tables" / "phase10_active_parent_pool.csv"
TABLES = PHASE10 / "tables"
REPORTS = PHASE10 / "reports"
PREDICTIONS = TABLES / "predictions" / "moltoken_counterfactual"

SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
SPLIT_TYPES = ["random", "scaffold"]
N_BITS = 2048


@dataclass
class RegionRecord:
    atoms: tuple[int, ...]
    bonds: tuple[int, ...]
    bits: set[int]
    occurrence_count: int
    ambiguous_bit_count: int
    source: str


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def list_string(values: list[int] | tuple[int, ...] | set[int]) -> str:
    return ";".join(str(int(value)) for value in sorted(values))


def target_distance(target: TargetWindow, value: float) -> float:
    return float(target.distance([value])[0])


def region(target: TargetWindow, value: float) -> str:
    return target.region(float(value))


def model_paths(split_type: str) -> list[Path]:
    paths = [
        PHASE04
        / "models"
        / "chemprop_main"
        / "dmpnn_rdkit2d"
        / split_type
        / f"seed_{seed}"
        / "model_0"
        / "best.pt"
        for seed in SEEDS
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Chemprop models: {missing}")
    return paths


def atom_environment(mol: Chem.Mol, center: int, radius: int) -> tuple[set[int], set[int]]:
    if radius == 0:
        return {int(center)}, set()
    bond_ids = set(int(bid) for bid in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    atoms = {int(center)}
    for bond_id in bond_ids:
        bond = mol.GetBondWithIdx(int(bond_id))
        atoms.add(int(bond.GetBeginAtomIdx()))
        atoms.add(int(bond.GetEndAtomIdx()))
    return atoms, bond_ids


def cuts_ring(mol: Chem.Mol, atoms: set[int]) -> bool:
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    return any(0 < len(ring.intersection(atoms)) < len(ring) for ring in atom_rings)


def complete_intersected_rings(mol: Chem.Mol, atoms: set[int]) -> set[int]:
    expanded = set(atoms)
    changed = True
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    while changed:
        changed = False
        for ring in atom_rings:
            if ring.intersection(expanded) and not ring.issubset(expanded):
                before = len(expanded)
                expanded.update(ring)
                changed = changed or len(expanded) > before
    return expanded


def boundary_atoms(mol: Chem.Mol, atoms: set[int]) -> tuple[list[int], list[int]]:
    boundary: set[int] = set()
    external: set[int] = set()
    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        for neighbor in atom.GetNeighbors():
            n_idx = int(neighbor.GetIdx())
            if n_idx not in atoms:
                boundary.add(int(atom_idx))
                external.add(n_idx)
    return sorted(boundary), sorted(external)


def component_bonds(mol: Chem.Mol, atoms: set[int]) -> list[int]:
    bonds: list[int] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin in atoms and end in atoms:
            bonds.append(int(bond.GetIdx()))
    return sorted(bonds)


def connected_components_after_cuts(mol: Chem.Mol, removed_bonds: set[int]) -> list[set[int]]:
    adjacency: dict[int, list[int]] = {atom.GetIdx(): [] for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        if int(bond.GetIdx()) in removed_bonds:
            continue
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        adjacency[begin].append(end)
        adjacency[end].append(begin)

    remaining = set(adjacency)
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    comp.add(neighbor)
                    stack.append(neighbor)
        components.append(comp)
    return components


def single_bond_cut_candidates(mol: Chem.Mol) -> list[set[int]]:
    cuttable = [
        int(bond.GetIdx())
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()
    ]
    components: list[set[int]] = []
    for bond_id in cuttable:
        components.extend(connected_components_after_cuts(mol, {bond_id}))
    return components


def two_bond_cut_candidates(mol: Chem.Mol) -> list[set[int]]:
    cuttable = [
        int(bond.GetIdx())
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()
    ]
    components: list[set[int]] = []
    for left_idx, left in enumerate(cuttable):
        for right in cuttable[left_idx + 1 :]:
            comps = connected_components_after_cuts(mol, {left, right})
            for comp in comps:
                boundary, _ = boundary_atoms(mol, comp)
                if len(boundary) == 2:
                    components.append(comp)
    return components


def morgan_region_records(mol: Chem.Mol) -> list[RegionRecord]:
    bit_info: dict[int, tuple[tuple[int, int], ...]] = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS, bitInfo=bit_info)
    grouped: dict[tuple[int, ...], RegionRecord] = {}
    for bit, occurrences in bit_info.items():
        ambiguous = len(occurrences) > 1
        for center, radius in occurrences:
            if int(radius) == 0:
                continue
            atoms, bonds = atom_environment(mol, int(center), int(radius))
            atoms = complete_intersected_rings(mol, atoms)
            key = tuple(sorted(atoms))
            if key not in grouped:
                grouped[key] = RegionRecord(
                    atoms=key,
                    bonds=tuple(sorted(component_bonds(mol, atoms) or bonds)),
                    bits=set(),
                    occurrence_count=0,
                    ambiguous_bit_count=0,
                    source="morgan_radius",
                )
            rec = grouped[key]
            rec.bits.add(int(bit))
            rec.occurrence_count += 1
            rec.ambiguous_bit_count += int(ambiguous)
    return list(grouped.values())


def structural_confidence(
    *,
    parent_distance: float,
    boundary_count: int,
    ambiguous_fraction: float,
    nearest_train_tanimoto: float,
    ring_cut_flag: bool,
    n_atoms: int,
) -> float:
    strength = min(max(parent_distance, 0.0) / 1.0, 1.0)
    boundary = 1.0 if boundary_count in {1, 2} else 0.65 if boundary_count == 3 else 0.15
    ambiguity = max(0.0, 1.0 - min(ambiguous_fraction / 0.5, 1.0))
    domain = min(max(nearest_train_tanimoto, 0.0) / 0.5, 1.0)
    size = 1.0 if 3 <= n_atoms <= 22 else 0.65 if 2 <= n_atoms <= 25 else 0.1
    ring = 0.0 if ring_cut_flag else 1.0
    return float(0.25 * strength + 0.25 * boundary + 0.15 * ambiguity + 0.15 * domain + 0.15 * size + 0.05 * ring)


def add_candidate(
    rows: list[dict[str, object]],
    *,
    mol: Chem.Mol,
    parent: pd.Series,
    atoms: set[int],
    source: str,
    occurrence_count: int = 1,
    ambiguous_bit_count: int = 0,
    source_ids: str = "",
) -> None:
    atom_count = mol.GetNumAtoms()
    atoms = set(int(atom) for atom in atoms)
    if not atoms or atoms >= set(range(atom_count)):
        return
    atoms = complete_intersected_rings(mol, atoms)
    if not atoms or atoms >= set(range(atom_count)):
        return
    boundary, external = boundary_atoms(mol, atoms)
    if not boundary or len(boundary) > 3:
        return
    n_atoms = len(atoms)
    if n_atoms < 2 or n_atoms > 25:
        return
    ring_cut = cuts_ring(mol, atoms)
    if ring_cut:
        return
    bonds = component_bonds(mol, atoms)
    ambiguous_fraction = ambiguous_bit_count / max(1, occurrence_count)
    parent_distance = float(parent["main_target_distance"])
    confidence = structural_confidence(
        parent_distance=parent_distance,
        boundary_count=len(boundary),
        ambiguous_fraction=ambiguous_fraction,
        nearest_train_tanimoto=float(parent["nearest_train_tanimoto"]),
        ring_cut_flag=ring_cut,
        n_atoms=n_atoms,
    )
    rows.append(
        {
            "mol_id": str(parent["external_molecule_id"]),
            "parent_rank": int(parent["selected_rank"]),
            "parent_library_id": str(parent["primary_library_id"]),
            "parent_region": str(parent["main_target_region"]),
            "token_id": -1,
            "canonical_smiles": str(parent["phase10_parent_smiles"]),
            "atom_indices": list_string(atoms),
            "bond_indices": list_string(bonds),
            "boundary_atoms": list_string(boundary),
            "external_neighbor_atoms": list_string(external),
            "n_atoms": int(n_atoms),
            "n_bonds": int(len(bonds)),
            "n_boundary_atoms": int(len(boundary)),
            "source_region_ids": source_ids or source,
            "candidate_source": source,
            "risk_side": str(parent["main_target_region"]),
            "mean_combined_harm_score": 0.0,
            "max_combined_harm_score": 0.0,
            "mean_distance_harm_score": 0.0,
            "mean_region_harm_score": 0.0,
            "ambiguous_bit_fraction": float(ambiguous_fraction),
            "cuts_ring": bool(ring_cut),
            "nearest_train_tanimoto": float(parent["nearest_train_tanimoto"]),
            "token_confidence": confidence,
            "suggested_role": "replace_or_delete",
            "is_connected": True,
            "is_candidate_moltoken": bool(confidence >= 0.45),
        }
    )


def enumerate_candidates(parents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parent in parents.itertuples(index=False):
        parent_series = pd.Series(parent._asdict())
        mol = Chem.MolFromSmiles(str(parent_series["phase10_parent_smiles"]))
        if mol is None:
            continue

        for idx, record in enumerate(morgan_region_records(mol)):
            add_candidate(
                rows,
                mol=mol,
                parent=parent_series,
                atoms=set(record.atoms),
                source="morgan_radius",
                occurrence_count=record.occurrence_count,
                ambiguous_bit_count=record.ambiguous_bit_count,
                source_ids=f"morgan_{idx}",
            )

        for idx, atoms in enumerate(single_bond_cut_candidates(mol)):
            add_candidate(rows, mol=mol, parent=parent_series, atoms=atoms, source="single_cut", source_ids=f"cut1_{idx}")

        for idx, atoms in enumerate(two_bond_cut_candidates(mol)):
            add_candidate(rows, mol=mol, parent=parent_series, atoms=atoms, source="two_cut", source_ids=f"cut2_{idx}")

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    candidates["atom_key"] = candidates["atom_indices"]
    candidates["source_rank"] = candidates["candidate_source"].map({"single_cut": 0, "two_cut": 1, "morgan_radius": 2}).fillna(3)
    candidates = candidates.sort_values(
        ["mol_id", "atom_key", "is_candidate_moltoken", "source_rank", "token_confidence"],
        ascending=[True, True, False, True, False],
    )
    candidates = candidates.drop_duplicates(["mol_id", "atom_key"], keep="first").copy()
    candidates = candidates.sort_values(["parent_rank", "source_rank", "token_confidence"], ascending=[True, True, False])
    candidates["token_id"] = candidates.groupby("mol_id").cumcount()
    candidates["candidate_row_id"] = np.arange(len(candidates))
    return candidates.drop(columns=["atom_key", "source_rank"])


def build_counterfactual_table(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        atoms = [int(item) for item in str(row["atom_indices"]).split(";") if item != ""]
        cf_smiles, meta = remove_atoms_counterfactual(str(row["canonical_smiles"]), atoms)
        out = row.to_dict()
        out["cf_row_id"] = int(idx)
        out["counterfactual_smiles"] = cf_smiles
        out.update(meta)
        rows.append(out)
    counterfactuals = pd.DataFrame(rows)
    valid = counterfactuals[
        (counterfactuals["counterfactual_valid"] == True)
        & (counterfactuals["counterfactual_usable_for_scoring"] == True)
        & counterfactuals["counterfactual_smiles"].notna()
    ].copy()
    return counterfactuals, valid


def run_chemprop_counterfactual_predictions(input_csv: Path, force: bool) -> None:
    ensure_dirs(PREDICTIONS, REPORTS)
    for split_type in SPLIT_TYPES:
        output = PREDICTIONS / f"phase10c_counterfactual_main_{split_type}.csv"
        if output.exists() and not force:
            continue
        cmd = [
            str(PYTHON),
            str(CHEMPROP_CLI),
            "predict",
            "-i",
            str(input_csv),
            "-o",
            str(output),
            "-s",
            "smiles",
            "--model-paths",
            *[str(path) for path in model_paths(split_type)],
            "--batch-size",
            "64",
            "--num-workers",
            "0",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--molecule-featurizers",
            "rdkit_2d",
        ]
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        log_path = REPORTS / f"phase10c_counterfactual_main_{split_type}.log"
        log_path.write_text(
            "\n".join(
                [
                    f"returncode: {result.returncode}",
                    "",
                    "command:",
                    " ".join(cmd),
                    "",
                    "stdout:",
                    result.stdout,
                    "",
                    "stderr:",
                    result.stderr,
                ]
            ),
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Chemprop prediction failed for {split_type}; see {log_path}")


def read_main_counterfactual_predictions(valid: pd.DataFrame, target: TargetWindow) -> pd.DataFrame:
    rows = valid[["cf_row_id", "mol_id", "token_id"]].copy()
    pred_cols: list[pd.Series] = []
    for split_type in SPLIT_TYPES:
        pred_path = PREDICTIONS / f"phase10c_counterfactual_main_{split_type}.csv"
        pred_df = pd.read_csv(pred_path)
        if "exp" not in pred_df.columns:
            raise ValueError(f"missing exp prediction column in {pred_path}")
        rows[f"main_counterfactual_pred_{split_type}"] = pred_df["exp"].to_numpy(dtype=float)
        pred_cols.append(rows[f"main_counterfactual_pred_{split_type}"])

        individual_path = pred_path.with_name(f"{pred_path.stem}_individual{pred_path.suffix}")
        if individual_path.exists():
            individual = pd.read_csv(individual_path)
            numeric_cols = [col for col in individual.columns if col != "smiles"]
            if numeric_cols:
                rows[f"main_counterfactual_seed_std_{split_type}"] = individual[numeric_cols].std(axis=1).to_numpy(dtype=float)

    stacked = np.vstack([col.to_numpy(dtype=float) for col in pred_cols])
    rows["main_counterfactual_pred_mean"] = np.mean(stacked, axis=0)
    rows["main_counterfactual_pred_std_across_splits"] = np.std(stacked, axis=0)
    seed_std_cols = [col for col in rows.columns if col.startswith("main_counterfactual_seed_std_")]
    rows["main_counterfactual_seed_std"] = (
        rows[seed_std_cols].mean(axis=1) if seed_std_cols else rows["main_counterfactual_pred_std_across_splits"]
    )
    rows["main_counterfactual_target_distance"] = target.distance(rows["main_counterfactual_pred_mean"].to_numpy(dtype=float))
    return rows


def build_features_for_smiles(smiles_values: pd.Series) -> np.ndarray:
    temp = pd.DataFrame({"canonical_smiles": smiles_values.astype(str)})
    fp = morgan_features(temp["canonical_smiles"], radius=2, n_bits=N_BITS)
    desc = rdkit_descriptor_frame(temp, smiles_col="canonical_smiles").to_numpy(dtype=np.float32)
    return np.hstack([fp, desc]).astype(np.float32)


def predict_verifier_counterfactual(valid: pd.DataFrame, target: TargetWindow) -> pd.DataFrame:
    rows = valid[["cf_row_id", "mol_id", "token_id"]].copy()
    X = build_features_for_smiles(valid["counterfactual_smiles"])
    for split_type in SPLIT_TYPES:
        for model_key, prefix in [("xgboost", "xgboost"), ("random_forest", "rf")]:
            preds = []
            for seed in SEEDS:
                path = PHASE04 / "models" / "verifiers" / split_type / f"seed_{seed}" / f"{model_key}.joblib"
                model = joblib.load(path)
                preds.append(np.asarray(model.predict(X), dtype=float))
            arr = np.vstack(preds)
            rows[f"{prefix}_counterfactual_pred_{split_type}"] = arr.mean(axis=0)
            rows[f"{prefix}_counterfactual_seed_std_{split_type}"] = arr.std(axis=0)

    for prefix in ["xgboost", "rf"]:
        pred_cols = [f"{prefix}_counterfactual_pred_{split_type}" for split_type in SPLIT_TYPES]
        std_cols = [f"{prefix}_counterfactual_seed_std_{split_type}" for split_type in SPLIT_TYPES]
        rows[f"{prefix}_counterfactual_pred_mean"] = rows[pred_cols].mean(axis=1)
        rows[f"{prefix}_counterfactual_seed_std"] = rows[std_cols].mean(axis=1)
        rows[f"{prefix}_counterfactual_target_distance"] = target.distance(
            rows[f"{prefix}_counterfactual_pred_mean"].to_numpy(dtype=float)
        )
    return rows


def original_prediction_table(parents: pd.DataFrame, target: TargetWindow) -> pd.DataFrame:
    original = parents[
        [
            "external_molecule_id",
            "main_pred_mean",
            "main_seed_std",
            "main_split_std",
            "main_target_region",
            "main_target_distance",
            "nearest_train_tanimoto",
            "xgboost_pred_mean",
            "rf_pred_mean",
        ]
    ].copy()
    original = original.rename(
        columns={
            "external_molecule_id": "mol_id",
            "main_pred_mean": "main_original_pred_mean",
            "main_seed_std": "main_original_pred_std",
            "main_split_std": "main_original_pred_split_std",
            "main_target_region": "risk_side_05",
            "main_target_distance": "main_original_target_distance",
            "nearest_train_tanimoto": "nearest_train_tanimoto_main",
            "xgboost_pred_mean": "xgboost_original_pred_mean",
            "rf_pred_mean": "rf_original_pred_mean",
        }
    )
    original["exp"] = np.nan
    original["target_region_exp"] = ""
    original["target_distance_exp"] = np.nan
    original["xgboost_original_target_distance"] = target.distance(original["xgboost_original_pred_mean"].to_numpy(dtype=float))
    original["rf_original_target_distance"] = target.distance(original["rf_original_pred_mean"].to_numpy(dtype=float))
    return original


def prepare_quality_fields(df: pd.DataFrame, target: TargetWindow) -> pd.DataFrame:
    original_distance = df["main_original_target_distance"].to_numpy(dtype=float)
    cf_pred = df["main_counterfactual_pred_mean"].to_numpy(dtype=float)
    cf_distance = target.distance(cf_pred)
    cf_distance[np.isnan(cf_pred)] = np.nan
    delta = original_distance - cf_distance
    positive = np.clip(np.nan_to_num(delta, nan=0.0), 0.0, 1.0)
    df["mean_combined_harm_score"] = positive
    df["max_combined_harm_score"] = positive
    df["mean_distance_harm_score"] = positive
    df["mean_region_harm_score"] = positive
    df["token_confidence"] = np.maximum(df["token_confidence"].fillna(0.0), np.clip(0.45 + 0.45 * positive, 0.0, 1.0))
    return df


def select_active_moltokens(quality: pd.DataFrame, phase10_config: dict) -> pd.DataFrame:
    cfg = phase10_config["moltoken_screening"]
    allowed_attachments = set(int(value) for value in cfg["allowed_attachment_counts"])
    preferred_grades = set(cfg["prefer_split_quality_grades"])
    max_per_parent = int(cfg["max_tokens_per_parent"])

    eligible = quality[
        (quality["formal_token_status"] == "retain_target")
        & (quality["formal_token_role"] == "target")
        & (quality["attachment_count"].isin(allowed_attachments))
        & (quality["split_quality_score"] >= float(cfg["min_split_quality_score"]))
        & (quality["split_quality_grade"].isin(preferred_grades))
        & (~quality["cuts_ring"].fillna(False).astype(bool))
        & (quality["counterfactual_usable_for_scoring"].fillna(False).astype(bool))
    ].copy()

    eligible["active_token_score"] = (
        eligible["split_quality_score"].fillna(0.0)
        + 0.35 * eligible["main_delta_distance"].fillna(0.0).clip(lower=0.0)
        + 0.08 * eligible["direction_agreement_score"].fillna(0.0)
        - 0.08 * eligible["main_counterfactual_seed_std"].fillna(0.5)
    )
    eligible = eligible.sort_values(
        ["mol_id", "active_token_score", "split_quality_score", "main_delta_distance"],
        ascending=[True, False, False, False],
    )
    selected = eligible.groupby("mol_id", as_index=False, group_keys=False).head(max_per_parent).copy()
    selected = selected.sort_values(
        ["active_token_score", "split_quality_score", "main_delta_distance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected["active_moltoken_rank"] = np.arange(1, len(selected) + 1)
    selected["phase10c_active_moltoken"] = True
    return selected


def gate_status(name: str, actual: float, threshold: float, direction: str = ">=") -> dict:
    passed = actual >= threshold if direction == ">=" else actual <= threshold
    return {
        "gate": name,
        "actual": float(actual),
        "threshold": float(threshold),
        "direction": direction,
        "status": "pass" if passed else "needs_fix",
    }


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def write_reports(
    *,
    parents: pd.DataFrame,
    candidates: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    valid: pd.DataFrame,
    quality: pd.DataFrame,
    active: pd.DataFrame,
    phase10_config: dict,
) -> None:
    gates = phase10_config["acceptance_gates"]
    parent_count = int(len(parents))
    candidate_count = int(len(candidates))
    valid_count = int(len(valid))
    formal_target_count = int((quality["formal_token_status"] == "retain_target").sum())
    active_count = int(len(active))
    covered_parent_count = int(active["mol_id"].nunique()) if active_count else 0
    coverage = covered_parent_count / parent_count if parent_count else 0.0
    ring_cut_rate = float(active["cuts_ring"].mean()) if active_count else 0.0
    attachment_ok_rate = float(active["attachment_count"].isin([1, 2]).mean()) if active_count else 0.0
    mean_delta = finite_float(active["main_delta_distance"].mean()) if active_count else 0.0
    mean_score = finite_float(active["split_quality_score"].mean()) if active_count else 0.0

    gate_rows = [
        gate_status("active_moltoken_count", active_count, gates["active_moltoken_count_min"]),
        gate_status("active_moltoken_valid_edit_coverage", coverage, gates["active_moltoken_valid_edit_coverage_min"]),
        gate_status("active_moltoken_ring_cut_rate", ring_cut_rate, 0.0, direction="<="),
        gate_status("active_moltoken_attachment_ok_rate", attachment_ok_rate, 1.0),
    ]
    verdict = "pass" if all(row["status"] == "pass" for row in gate_rows) else "needs_fix"

    summary = {
        "phase": "10C_moltoken_contribution_split",
        "verdict": verdict,
        "parent_count": parent_count,
        "candidate_moltoken_count": candidate_count,
        "counterfactual_valid_count": int((counterfactuals["counterfactual_valid"] == True).sum()),
        "counterfactual_usable_count": valid_count,
        "formal_target_count": formal_target_count,
        "active_moltoken_count": active_count,
        "active_parent_coverage_count": covered_parent_count,
        "active_parent_coverage_rate": coverage,
        "active_mean_main_delta_distance": mean_delta,
        "active_mean_split_quality_score": mean_score,
        "active_region_counts": active["risk_side_05"].value_counts().to_dict() if active_count else {},
        "active_grade_counts": active["split_quality_grade"].value_counts().to_dict() if active_count else {},
        "active_attachment_counts": active["attachment_count"].value_counts().to_dict() if active_count else {},
        "formal_status_counts": quality["formal_token_status"].value_counts().to_dict(),
        "candidate_source_counts": candidates["candidate_source"].value_counts().to_dict() if candidate_count else {},
        "gates": gate_rows,
    }
    (REPORTS / "phase10c_moltoken_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# 10-C Mol Token 贡献切分报告",
        "",
        f"结论：{verdict}",
        "",
        "## 核心结果",
        "",
        f"- 冻结父分子：{parent_count}",
        f"- 候选 Mol Token：{candidate_count}",
        f"- 可用于贡献评分的反事实分子：{valid_count}",
        f"- retain_target 数量：{formal_target_count}",
        f"- 冻结可编辑 Mol Token：{active_count}",
        f"- 覆盖父分子数：{covered_parent_count}",
        f"- 覆盖率：{coverage:.3f}",
        f"- 平均贡献差分：{mean_delta:.3f}",
        f"- 平均切分质量：{mean_score:.3f}",
        "",
        "## 门控检查",
        "",
    ]
    for row in gate_rows:
        lines.append(
            f"- {row['gate']}: {row['actual']:.3f} {row['direction']} {row['threshold']:.3f} -> {row['status']}"
        )
    lines.extend(
        [
            "",
            "## 输出表",
            "",
            f"- `{rel(TABLES / 'phase10c_candidate_moltoken_regions.csv')}`",
            f"- `{rel(TABLES / 'phase10c_counterfactual_moltoken_smiles.csv')}`",
            f"- `{rel(TABLES / 'phase10c_contribution_split_quality_table.csv')}`",
            f"- `{rel(TABLES / 'phase10c_active_moltoken_pool.csv')}`",
            "",
            "## 后续动作",
            "",
            "- 若结论为 pass，下一步进入 10-D：基于冻结 Mol Token 做删除、替换和增加候选编辑。",
            "- 若结论为 needs_fix，优先检查父分子覆盖率、可评分反事实数量和低/高两侧 token 分布。",
            "",
        ]
    )
    (REPORTS / "phase10c_moltoken_contribution_split_report.md").write_text("\n".join(lines), encoding="utf-8")

    self_check_lines = [
        "# 10-C 自检记录",
        "",
        f"总体结论：{verdict}",
        "",
        "## 检查项",
        "",
        f"- 父分子输入数量：{parent_count}",
        f"- 候选 Mol Token 数量：{candidate_count}",
        f"- 反事实可评分数量：{valid_count}",
        f"- 冻结 Mol Token 数量：{active_count}",
        f"- 父分子覆盖率：{coverage:.3f}",
        f"- ring-cut 率：{ring_cut_rate:.3f}",
        f"- attachment 1/2 覆盖率：{attachment_ok_rate:.3f}",
        "",
        "## 门控",
        "",
    ]
    for row in gate_rows:
        self_check_lines.append(
            f"- {row['gate']}: {row['status']} ({row['actual']:.3f} {row['direction']} {row['threshold']:.3f})"
        )
    (REPORTS / "phase10c_moltoken_split_self_check.md").write_text(
        "\n".join(self_check_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-chemprop-predict", action="store_true")
    args = parser.parse_args()

    ensure_dirs(TABLES, REPORTS, PREDICTIONS)
    phase10_config = load_json(PHASE10_CONFIG)
    quality_config = load_json(QUALITY_CONFIG)
    target_cfg = phase10_config["predictor_screening"]
    target = TargetWindow(low=target_cfg["target_window_low"], high=target_cfg["target_window_high"])

    parents = pd.read_csv(ACTIVE_PARENT_PATH)
    parents = parents.sort_values("selected_rank").copy()
    candidates = enumerate_candidates(parents)
    candidate_path = TABLES / "phase10c_candidate_moltoken_regions.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    counterfactuals, valid = build_counterfactual_table(candidates)
    counterfactuals.to_csv(TABLES / "phase10c_counterfactual_moltoken_smiles.csv", index=False, encoding="utf-8-sig")

    chemprop_input = PREDICTIONS / "phase10c_counterfactual_chemprop_input.csv"
    valid[["counterfactual_smiles"]].rename(columns={"counterfactual_smiles": "smiles"}).to_csv(
        chemprop_input,
        index=False,
        encoding="utf-8",
    )
    if len(valid) and not args.skip_chemprop_predict:
        run_chemprop_counterfactual_predictions(chemprop_input, force=args.force_predict)

    main_cf = read_main_counterfactual_predictions(valid, target) if len(valid) else pd.DataFrame()
    verifier_cf = predict_verifier_counterfactual(valid, target) if len(valid) else pd.DataFrame()
    original = original_prediction_table(parents, target)

    quality = counterfactuals.merge(original, on="mol_id", how="left")
    if not main_cf.empty:
        quality = quality.merge(main_cf.drop(columns=["mol_id", "token_id"]), on="cf_row_id", how="left")
    if not verifier_cf.empty:
        quality = quality.merge(verifier_cf.drop(columns=["mol_id", "token_id"]), on="cf_row_id", how="left")
    quality = prepare_quality_fields(quality, target)
    quality = score_rows(quality, quality_config)
    quality = assign_status(quality, quality_config)
    quality.to_csv(TABLES / "phase10c_contribution_split_quality_table.csv", index=False, encoding="utf-8-sig")

    active = select_active_moltokens(quality, phase10_config)
    active.to_csv(TABLES / "phase10c_active_moltoken_pool.csv", index=False, encoding="utf-8-sig")
    write_reports(
        parents=parents,
        candidates=candidates,
        counterfactuals=counterfactuals,
        valid=valid,
        quality=quality,
        active=active,
        phase10_config=phase10_config,
    )


if __name__ == "__main__":
    main()
