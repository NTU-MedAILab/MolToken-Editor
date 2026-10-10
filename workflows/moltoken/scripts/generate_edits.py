from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, BRICS, Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from moltoken_common import TargetWindow, ensure_dirs, morgan_features, rdkit_descriptor_frame
import contribution_quality as quality_tools
import linker_replacement_utils as linker_tools
import sidechain_replacement_utils as sidechain_tools


rdBase.DisableLog("rdApp.*")

PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE04 = ROOT / "predictors"
EXTERNAL_LIBRARY_MEMBERSHIP = ROOT.parent / "data" / "external" / "library_membership.csv"
PHASE10 = ROOT / "moltoken"
TABLES = PHASE10 / "tables"
REPORTS = PHASE10 / "reports"
PREDICTIONS = TABLES / "predictions" / "edit_scoring"
ACTIVE_TOKEN_PATH = TABLES / "phase10c_active_moltoken_pool.csv"
QUALITY_TABLE_PATH = TABLES / "phase10c_contribution_split_quality_table.csv"
CONFIG_PATH = PHASE10 / "configs" / "target_expansion.json"

TARGET = TargetWindow(low=1.0, high=3.4)
SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
SPLIT_TYPES = ["random", "scaffold"]
N_BITS = 2048
RANDOM_SEED = 20260605


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def parse_ints(value: Any) -> list[int]:
    if value is None or pd.isna(value):
        return []
    return [int(part) for part in str(value).split(";") if part != ""]


def target_distance(value: float) -> float:
    return float(TARGET.distance([value])[0])


def target_region(value: float) -> str:
    return TARGET.region(float(value))


def target_direction(row: dict[str, Any]) -> str:
    risk = str(row.get("risk_side_05", row.get("risk_side", row.get("parent_region", ""))))
    if risk == "low":
        return "increase_logd"
    if risk == "high":
        return "decrease_logd"
    pred = row.get("main_original_pred_mean", np.nan)
    try:
        pred_region = target_region(float(pred))
    except (TypeError, ValueError):
        return "none"
    if pred_region == "low":
        return "increase_logd"
    if pred_region == "high":
        return "decrease_logd"
    return "none"


def canonical_smiles(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if smiles is None or pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return Chem.MolFromSmiles(text)


def count_dummy_atoms(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def build_filter_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


FILTER_CATALOG = build_filter_catalog()


def interference_alert(mol: Chem.Mol | None) -> tuple[bool, str]:
    if mol is None:
        return False, ""
    matches = FILTER_CATALOG.GetMatches(mol)
    if not matches:
        return False, ""
    names = sorted({match.GetDescription() for match in matches})
    return True, ";".join(names)


def murcko(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def descriptor_record(smiles: str) -> dict[str, Any]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "mw": np.nan,
            "mol_logp": np.nan,
            "tpsa": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "rotbonds": np.nan,
            "heavy_atoms": np.nan,
            "qed": np.nan,
            "sa_score": np.nan,
            "lipinski_violations": np.nan,
            "interference_alert": False,
            "interference_alert_names": "",
            "murcko_scaffold": "",
        }
    alert, names = interference_alert(mol)
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    return {
        "valid": True,
        "mw": mw,
        "mol_logp": logp,
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": hbd,
        "hba": hba,
        "rotbonds": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "qed": float(QED.qed(mol)),
        "sa_score": float(sascorer.calculateScore(mol)),
        "lipinski_violations": int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)),
        "interference_alert": bool(alert),
        "interference_alert_names": names,
        "murcko_scaffold": murcko(smiles),
    }


def fingerprint(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    fp_a = fingerprint(smiles_a)
    fp_b = fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def training_fingerprints() -> list:
    frames = []
    for path in [PHASE01 / "splits" / "train.csv", PHASE04 / "splits" / "scaffold_train.csv"]:
        df = pd.read_csv(path)
        col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        frames.append(df[col].dropna().astype(str))
    smiles_values = pd.concat(frames, ignore_index=True).drop_duplicates()
    return [fp for fp in (fingerprint(smiles) for smiles in smiles_values) if fp is not None]


def nearest_training_similarity(smiles: str, train_fps: list) -> float:
    fp = fingerprint(smiles)
    if fp is None or not train_fps:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return float(max(sims)) if sims else float("nan")


def canonical_donor_smiles(smiles: str) -> str | None:
    return sidechain_tools.canonical_donor_smiles(smiles)


def canonical_linker_smiles(smiles: str) -> str | None:
    return linker_tools.canonical_linker_smiles(smiles)


def augment_reagents_from_phase10_libraries(
    donors: pd.DataFrame,
    linkers: pd.DataFrame,
    max_external_per_role: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(EXTERNAL_LIBRARY_MEMBERSHIP)
    smiles_values = (
        frame["standardized_smiles"].dropna().astype(str).tolist()
        if "standardized_smiles" in frame.columns
        else []
    )

    donor_counts: dict[str, int] = {}
    linker_counts: dict[str, int] = {}
    for smiles in smiles_values:
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        try:
            fragments = BRICS.BRICSDecompose(mol, keepNonLeafNodes=False)
        except Exception:
            continue
        for fragment in fragments:
            donor = canonical_donor_smiles(fragment)
            if donor is not None:
                donor_counts[donor] = donor_counts.get(donor, 0) + 1
            linker = canonical_linker_smiles(fragment)
            if linker is not None:
                linker_counts[linker] = linker_counts.get(linker, 0) + 1

    donor_seen = set(donors["donor_smiles"].astype(str)) if not donors.empty else set()
    donor_rows: list[dict[str, Any]] = []
    for idx, (smiles, count) in enumerate(sorted(donor_counts.items(), key=lambda item: (-item[1], item[0]))):
        if smiles in donor_seen:
            continue
        desc = sidechain_tools.donor_descriptor(smiles)
        if desc is None or int(desc["heavy_atoms"]) > 14:
            continue
        role = sidechain_tools.infer_donor_role(desc)
        if role == "neutral":
            continue
        donor_rows.append(
            {
                "donor_id": f"phase10_library_brics_{idx:04d}",
                "donor_smiles": smiles,
                "donor_role": role,
                "donor_source": "phase10_library_brics",
                "train_occurrence": int(count),
                **{f"donor_{key}": value for key, value in desc.items()},
            }
        )

    linker_seen = set(linkers["linker_smiles"].astype(str)) if not linkers.empty else set()
    linker_rows: list[dict[str, Any]] = []
    for idx, (smiles, count) in enumerate(sorted(linker_counts.items(), key=lambda item: (-item[1], item[0]))):
        if smiles in linker_seen:
            continue
        desc = linker_tools.linker_descriptor(smiles)
        if desc is None or int(desc["heavy_atoms"]) > 18:
            continue
        role = linker_tools.infer_linker_role(desc)
        if role == "neutral":
            continue
        linker_rows.append(
            {
                "linker_id": f"phase10_library_linker_{idx:04d}",
                "linker_smiles": smiles,
                "linker_role": role,
                "linker_source": "phase10_library_brics",
                "train_occurrence": int(count),
                **{f"linker_{key}": value for key, value in desc.items()},
            }
        )

    extra_donors = []
    for role in ["increase_logd", "decrease_logd"]:
        role_rows = [row for row in donor_rows if row["donor_role"] == role]
        role_rows = sorted(role_rows, key=lambda row: (-int(row["train_occurrence"]), row["donor_smiles"]))
        extra_donors.extend(role_rows[:max_external_per_role])
    extra_linkers = []
    for role in ["increase_logd", "decrease_logd"]:
        role_rows = [row for row in linker_rows if row["linker_role"] == role]
        role_rows = sorted(role_rows, key=lambda row: (-int(row["train_occurrence"]), row["linker_smiles"]))
        extra_linkers.extend(role_rows[:max_external_per_role])

    if extra_donors:
        donors = pd.concat([donors, pd.DataFrame(extra_donors)], ignore_index=True, sort=False)
    if extra_linkers:
        linkers = pd.concat([linkers, pd.DataFrame(extra_linkers)], ignore_index=True, sort=False)
    donors = donors.sort_values(["donor_role", "donor_source", "train_occurrence"], ascending=[True, True, False])
    donors["donor_rank_in_role"] = donors.groupby("donor_role").cumcount() + 1
    linkers = linkers.sort_values(["linker_role", "linker_source", "train_occurrence"], ascending=[True, True, False])
    linkers["linker_rank_in_role"] = linkers.groupby("linker_role").cumcount() + 1
    return donors.reset_index(drop=True), linkers.reset_index(drop=True)


def base_candidate_row(target: dict[str, Any], method: str, action: str, edited_smiles: str | None) -> dict[str, Any]:
    direction = target_direction(target)
    if method in {"contribution_delete", "contribution_replace"}:
        method_group = "contribution_primary"
    elif method.startswith("contribution"):
        method_group = "contribution_supplemental"
    else:
        method_group = "same_interface_random"
    return {
        "method": method,
        "edit_action": action,
        "method_group": method_group,
        "mol_id": target["mol_id"],
        "token_id": target["token_id"],
        "source_target_token_id": target.get("source_target_token_id", target["token_id"]),
        "parent_rank": target.get("parent_rank", np.nan),
        "parent_library_id": target.get("parent_library_id", ""),
        "canonical_smiles": target["canonical_smiles"],
        "counterfactual_smiles": edited_smiles,
        "edited_smiles": edited_smiles,
        "atom_indices": target.get("atom_indices", ""),
        "boundary_atoms": target.get("boundary_atoms", ""),
        "external_neighbor_atoms": target.get("external_neighbor_atoms", ""),
        "n_atoms": target.get("n_atoms", np.nan),
        "n_boundary_atoms": target.get("n_boundary_atoms", target.get("attachment_count", np.nan)),
        "attachment_count": target.get("attachment_count", target.get("n_boundary_atoms", np.nan)),
        "risk_side_05": target.get("risk_side_05", target.get("risk_side", target.get("parent_region", ""))),
        "replace_direction": direction,
        "main_original_pred_mean": target.get("main_original_pred_mean", np.nan),
        "main_original_target_distance": target.get("main_original_target_distance", np.nan),
        "main_original_pred_std": target.get("main_original_pred_std", np.nan),
        "main_original_pred_split_std": target.get("main_original_pred_split_std", np.nan),
        "xgboost_original_pred_mean": target.get("xgboost_original_pred_mean", np.nan),
        "xgboost_original_target_distance": target.get("xgboost_original_target_distance", np.nan),
        "rf_original_pred_mean": target.get("rf_original_pred_mean", np.nan),
        "rf_original_target_distance": target.get("rf_original_target_distance", np.nan),
        "source_split_quality_score": target.get("split_quality_score", np.nan),
        "source_split_quality_grade": target.get("split_quality_grade", ""),
        "source_main_delta_delete": target.get("main_delta_distance", np.nan),
    }


def descriptor_delta(original_smiles: str, edited_smiles: str | None) -> dict[str, Any]:
    if edited_smiles is None:
        return {}
    original = descriptor_record(original_smiles)
    edited = descriptor_record(edited_smiles)
    if not edited.get("valid"):
        return {}
    out: dict[str, Any] = {}
    for name in ["mw", "qed", "tpsa", "hbd", "hba", "rotbonds", "mol_logp", "heavy_atoms", "sa_score"]:
        out[f"original_{name}"] = original.get(name, np.nan)
        out[f"edited_{name}"] = edited.get(name, np.nan)
        out[f"delta_{name}"] = float(edited.get(name, np.nan)) - float(original.get(name, np.nan))
    return out


def build_delete_candidates(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets.to_dict(orient="records"):
        edited = target.get("counterfactual_smiles", None)
        valid = bool_value(target.get("counterfactual_usable_for_scoring", False)) and isinstance(edited, str)
        row = base_candidate_row(target, "contribution_delete", "delete", edited if valid else None)
        row.update(
            {
                "reagent_id": "delete",
                "reagent_smiles": "",
                "reagent_role": row["replace_direction"],
                "reagent_source": "direct_counterfactual",
                "edit_valid": bool(valid),
                "edit_reason": "ok" if valid else "invalid_delete_counterfactual",
            }
        )
        row.update(descriptor_delta(str(target["canonical_smiles"]), row["edited_smiles"]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_replace_candidates(
    targets: pd.DataFrame,
    donors: pd.DataFrame,
    linkers: pd.DataFrame,
    max_reagents_per_token: int,
    method: str = "contribution_replace",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets.to_dict(orient="records"):
        direction = target_direction(target)
        if direction == "none":
            continue
        atoms = parse_ints(target.get("atom_indices", ""))
        attachment = int(float(target.get("attachment_count", target.get("n_boundary_atoms", 0))))
        if attachment == 1:
            reagents = donors[donors["donor_role"] == direction].head(max_reagents_per_token)
            for reagent in reagents.to_dict(orient="records"):
                edited, meta = sidechain_tools.replace_single_attachment(
                    str(target["canonical_smiles"]), atoms, str(reagent["donor_smiles"])
                )
                row = base_candidate_row(target, method, "replace", edited)
                key = (method, str(row["mol_id"]), str(edited))
                duplicate = edited is not None and key in seen
                if edited is not None and not duplicate:
                    seen.add(key)
                row.update(
                    {
                        "reagent_id": reagent["donor_id"],
                        "reagent_smiles": reagent["donor_smiles"],
                        "reagent_role": reagent["donor_role"],
                        "reagent_source": reagent["donor_source"],
                        "reagent_type": "donor",
                        "reagent_rank_in_role": reagent.get("donor_rank_in_role", np.nan),
                        "edit_valid": bool_value(meta.get("edit_valid", False)) and not duplicate,
                        "edit_reason": "duplicate_edited_smiles" if duplicate else meta.get("edit_reason", ""),
                        **meta,
                    }
                )
                row.update(descriptor_delta(str(target["canonical_smiles"]), row["edited_smiles"]))
                rows.append(row)
        elif attachment == 2:
            reagents = linkers[linkers["linker_role"] == direction].head(max_reagents_per_token)
            for reagent in reagents.to_dict(orient="records"):
                for orientation in ["forward", "reverse"]:
                    edited, meta = linker_tools.replace_two_attachment_once(
                        str(target["canonical_smiles"]), atoms, str(reagent["linker_smiles"]), orientation
                    )
                    row = base_candidate_row(target, method, "replace", edited)
                    key = (method, str(row["mol_id"]), str(edited))
                    duplicate = edited is not None and key in seen
                    if edited is not None and not duplicate:
                        seen.add(key)
                    row.update(
                        {
                            "reagent_id": reagent["linker_id"],
                            "reagent_smiles": reagent["linker_smiles"],
                            "reagent_role": reagent["linker_role"],
                            "reagent_source": reagent["linker_source"],
                            "reagent_type": "linker",
                            "reagent_rank_in_role": reagent.get("linker_rank_in_role", np.nan),
                            "orientation": orientation,
                            "edit_valid": bool_value(meta.get("edit_valid", False)) and not duplicate,
                            "edit_reason": "duplicate_edited_smiles" if duplicate else meta.get("edit_reason", ""),
                            **meta,
                        }
                    )
                    row.update(descriptor_delta(str(target["canonical_smiles"]), row["edited_smiles"]))
                    rows.append(row)
    return pd.DataFrame(rows)


def add_single_attachment(original_smiles: str, anchor_idx: int, donor_smiles: str) -> tuple[str | None, dict[str, Any]]:
    mol = mol_from_smiles(original_smiles)
    donor = mol_from_smiles(donor_smiles)
    if mol is None:
        return None, {"edit_valid": False, "edit_reason": "invalid_original"}
    if donor is None:
        return None, {"edit_valid": False, "edit_reason": "invalid_donor"}
    if anchor_idx < 0 or anchor_idx >= mol.GetNumAtoms():
        return None, {"edit_valid": False, "edit_reason": "invalid_anchor"}
    dummy_atoms = [atom.GetIdx() for atom in donor.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        return None, {"edit_valid": False, "edit_reason": "donor_dummy_count_not_one"}
    dummy_idx = dummy_atoms[0]
    neighbors = [nbr.GetIdx() for nbr in donor.GetAtomWithIdx(dummy_idx).GetNeighbors()]
    if len(neighbors) != 1:
        return None, {"edit_valid": False, "edit_reason": "donor_dummy_not_terminal"}
    donor_attach = neighbors[0]

    out = Chem.RWMol(mol)
    donor_to_out: dict[int, int] = {}
    for atom in donor.GetAtoms():
        if atom.GetIdx() == dummy_idx:
            continue
        donor_to_out[atom.GetIdx()] = out.AddAtom(Chem.Atom(atom))
    for bond in donor.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a == dummy_idx or b == dummy_idx:
            continue
        out.AddBond(donor_to_out[a], donor_to_out[b], bond.GetBondType())
    try:
        out.AddBond(int(anchor_idx), donor_to_out[donor_attach], Chem.BondType.SINGLE)
    except Exception:
        return None, {"edit_valid": False, "edit_reason": "bond_add_failed"}
    edited = out.GetMol()
    edited_smiles = canonical_smiles(edited)
    if edited_smiles is None:
        return None, {"edit_valid": False, "edit_reason": "sanitize_or_canonicalize_failed"}
    edited_mol = mol_from_smiles(edited_smiles)
    if edited_mol is None or len(Chem.GetMolFrags(edited_mol)) != 1:
        return None, {"edit_valid": False, "edit_reason": "edited_not_single_fragment"}
    if edited_smiles == Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True):
        return None, {"edit_valid": False, "edit_reason": "unchanged_smiles"}
    return edited_smiles, {
        "edit_valid": True,
        "edit_reason": "ok",
        "attachment_count": 1,
        "add_anchor_atom": int(anchor_idx),
        "edited_heavy_atoms": int(edited_mol.GetNumHeavyAtoms()),
    }


def build_add_candidates(targets: pd.DataFrame, donors: pd.DataFrame, max_add_per_parent: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_parent_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for target in targets.sort_values("active_moltoken_rank").to_dict(orient="records"):
        mol_id = str(target["mol_id"])
        if per_parent_counts.get(mol_id, 0) >= max_add_per_parent:
            continue
        direction = target_direction(target)
        if direction == "none":
            continue
        anchors = parse_ints(target.get("external_neighbor_atoms", ""))
        if not anchors:
            continue
        reagents = donors[donors["donor_role"] == direction].head(8)
        for anchor in anchors[:2]:
            for reagent in reagents.to_dict(orient="records"):
                if per_parent_counts.get(mol_id, 0) >= max_add_per_parent:
                    break
                edited, meta = add_single_attachment(str(target["canonical_smiles"]), anchor, str(reagent["donor_smiles"]))
                row = base_candidate_row(target, "contribution_add", "add", edited)
                key = (mol_id, str(edited))
                duplicate = edited is not None and key in seen
                if edited is not None and not duplicate:
                    seen.add(key)
                    per_parent_counts[mol_id] = per_parent_counts.get(mol_id, 0) + 1
                row.update(
                    {
                        "reagent_id": reagent["donor_id"],
                        "reagent_smiles": reagent["donor_smiles"],
                        "reagent_role": reagent["donor_role"],
                        "reagent_source": reagent["donor_source"],
                        "reagent_type": "donor_addition",
                        "reagent_rank_in_role": reagent.get("donor_rank_in_role", np.nan),
                        "edit_valid": bool_value(meta.get("edit_valid", False)) and not duplicate,
                        "edit_reason": "duplicate_edited_smiles" if duplicate else meta.get("edit_reason", ""),
                        **meta,
                    }
                )
                row.update(descriptor_delta(str(target["canonical_smiles"]), row["edited_smiles"]))
                rows.append(row)
    return pd.DataFrame(rows)


def build_random_control_targets(active: pd.DataFrame, quality: pd.DataFrame, max_targets: int = 58) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    quality = quality.copy()
    quality["attachment_count"] = numeric(quality, "attachment_count")
    quality["n_atoms"] = numeric(quality, "n_atoms")
    active_keys = set(zip(active["mol_id"].astype(str), active["atom_indices"].astype(str)))
    rows: list[pd.Series] = []
    for target in active.sort_values("active_moltoken_rank").to_dict(orient="records"):
        mol_id = str(target["mol_id"])
        attachment = float(target.get("attachment_count", np.nan))
        size = float(target.get("n_atoms", np.nan))
        subset = quality[
            (quality["mol_id"].astype(str) == mol_id)
            & (quality["attachment_count"] == attachment)
            & (quality["counterfactual_usable_for_scoring"].map(bool_value))
            & (~quality["cuts_ring"].map(bool_value))
            & (quality["atom_indices"].astype(str) != str(target["atom_indices"]))
        ].copy()
        if not math.isnan(size):
            subset = subset[(subset["n_atoms"] - size).abs() <= max(3.0, 0.5 * size)]
        subset = subset[
            ~subset.apply(lambda row: (str(row["mol_id"]), str(row["atom_indices"])) in active_keys, axis=1)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sample(frac=1.0, random_state=rng.randint(1, 10_000_000))
        picked = subset.iloc[0].copy()
        picked["source_target_token_id"] = target["token_id"]
        picked["parent_rank"] = target.get("parent_rank", np.nan)
        picked["parent_library_id"] = target.get("parent_library_id", "")
        rows.append(picked)
        if len(rows) >= max_targets:
            break
    return pd.DataFrame(rows)


def configure_prediction_paths() -> None:
    ensure_dirs(PREDICTIONS, REPORTS)
    quality_tools.PREDICTIONS = PREDICTIONS
    quality_tools.REPORTS = REPORTS


def build_features_for_smiles(smiles_values: pd.Series) -> np.ndarray:
    temp = pd.DataFrame({"canonical_smiles": smiles_values.astype(str)})
    fp = morgan_features(temp["canonical_smiles"], radius=2, n_bits=N_BITS)
    desc = rdkit_descriptor_frame(temp, smiles_col="canonical_smiles").to_numpy(dtype=np.float32)
    return np.hstack([fp, desc]).astype(np.float32)


def predict_auxiliary(valid: pd.DataFrame) -> pd.DataFrame:
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
        cols = [f"{prefix}_counterfactual_pred_{split_type}" for split_type in SPLIT_TYPES]
        std_cols = [f"{prefix}_counterfactual_seed_std_{split_type}" for split_type in SPLIT_TYPES]
        rows[f"{prefix}_counterfactual_pred_mean"] = rows[cols].mean(axis=1)
        rows[f"{prefix}_counterfactual_seed_std"] = rows[std_cols].mean(axis=1)
    return rows


def score_candidates(
    candidates: pd.DataFrame,
    force_predict: bool,
    skip_predict: bool,
    prediction_prefix: str = "phase10d_edit_main",
    input_filename: str = "phase10d_edit_scoring_input.csv",
) -> pd.DataFrame:
    configure_prediction_paths()
    if candidates.empty:
        return candidates
    candidates = candidates.copy()
    candidates["cf_row_id"] = np.arange(len(candidates))
    valid = candidates[(candidates["edit_valid"].map(bool_value)) & candidates["counterfactual_smiles"].notna()].copy()
    if valid.empty:
        return candidates
    chemprop_input = PREDICTIONS / input_filename
    valid[["counterfactual_smiles"]].rename(columns={"counterfactual_smiles": "smiles"}).to_csv(
        chemprop_input, index=False, encoding="utf-8"
    )
    if not skip_predict:
        quality_tools.run_chemprop_counterfactual_predictions(
            chemprop_input, force=force_predict, output_prefix=prediction_prefix
        )
    main_cf = quality_tools.read_main_counterfactual_predictions(valid, output_prefix=prediction_prefix)
    aux_cf = predict_auxiliary(valid)
    scored = candidates.merge(main_cf.drop(columns=["mol_id", "token_id"]), on="cf_row_id", how="left")
    scored = scored.merge(aux_cf.drop(columns=["mol_id", "token_id"]), on="cf_row_id", how="left")
    return add_scoring_columns(scored)


def add_descriptor_panel(scored: pd.DataFrame) -> pd.DataFrame:
    train_fps = training_fingerprints()
    original_cache: dict[str, dict[str, Any]] = {}
    edited_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for row in scored.to_dict(orient="records"):
        original_smiles = str(row.get("canonical_smiles", ""))
        edited_smiles = str(row.get("edited_smiles", ""))
        original = original_cache.setdefault(original_smiles, descriptor_record(original_smiles))
        edited = edited_cache.setdefault(edited_smiles, descriptor_record(edited_smiles))
        rec: dict[str, Any] = {}
        for key, value in edited.items():
            rec[f"edited_{key}"] = value
        for key, value in original.items():
            rec[f"original_{key}"] = value
        rec["parent_tanimoto"] = tanimoto(original_smiles, edited_smiles)
        rec["nearest_train_tanimoto_edited"] = nearest_training_similarity(edited_smiles, train_fps)
        for name in ["mw", "mol_logp", "tpsa", "hbd", "hba", "rotbonds", "heavy_atoms", "qed", "sa_score", "lipinski_violations"]:
            rec[f"delta_{name}"] = float(edited.get(name, np.nan)) - float(original.get(name, np.nan))
        records.append(rec)
    panel = pd.DataFrame(records)
    base = scored.reset_index(drop=True).drop(columns=[col for col in panel.columns if col in scored.columns], errors="ignore")
    return pd.concat([base, panel], axis=1)


def relaxed_guardrail_pass(row: pd.Series) -> bool:
    return bool(
        row.get("edited_valid", False)
        and row.get("edited_mw", 9999) <= 650
        and row.get("edited_tpsa", 9999) <= 160
        and row.get("edited_hbd", 9999) <= 5
        and row.get("edited_hba", 9999) <= 12
        and row.get("edited_rotbonds", 9999) <= 12
        and row.get("edited_qed", -1) >= 0.35
        and row.get("edited_sa_score", 9999) <= 6.0
        and not bool_value(row.get("edited_interference_alert", False))
        and row.get("parent_tanimoto", 0.0) >= 0.35
        and 0.30 <= row.get("nearest_train_tanimoto_edited", -1.0) <= 0.95
    )


def add_scoring_columns(scored: pd.DataFrame) -> pd.DataFrame:
    scored = add_descriptor_panel(scored)
    main_pred = scored["main_counterfactual_pred_mean"].to_numpy(dtype=float)
    scored["edited_main_pred"] = main_pred
    scored["edited_main_target_region"] = [target_region(value) if not np.isnan(value) else "invalid" for value in main_pred]
    scored["edited_main_target_distance"] = TARGET.distance(main_pred)
    scored.loc[np.isnan(main_pred), "edited_main_target_distance"] = np.nan
    scored["main_delta_distance_edit"] = scored["main_original_target_distance"] - scored["edited_main_target_distance"]
    scored["main_pred_shift"] = scored["edited_main_pred"] - scored["main_original_pred_mean"]
    scored["main_direction_correct"] = np.where(
        scored["replace_direction"] == "increase_logd",
        scored["main_pred_shift"] > 0.05,
        np.where(scored["replace_direction"] == "decrease_logd", scored["main_pred_shift"] < -0.05, False),
    )
    scored["main_uncertainty"] = np.sqrt(
        scored["main_counterfactual_seed_std"].fillna(0.5) ** 2
        + scored["main_counterfactual_pred_std_across_splits"].fillna(0.5) ** 2
    )

    for prefix in ["xgboost", "rf"]:
        pred_col = f"{prefix}_counterfactual_pred_mean"
        scored[f"{prefix}_edited_target_distance"] = TARGET.distance(scored[pred_col].to_numpy(dtype=float))
        scored[f"{prefix}_delta_distance_edit"] = scored[f"{prefix}_original_target_distance"] - scored[
            f"{prefix}_edited_target_distance"
        ]
        scored[f"{prefix}_pred_shift"] = scored[pred_col] - scored[f"{prefix}_original_pred_mean"]
        scored[f"{prefix}_direction_correct"] = np.where(
            scored["replace_direction"] == "increase_logd",
            scored[f"{prefix}_pred_shift"] > 0.02,
            np.where(scored["replace_direction"] == "decrease_logd", scored[f"{prefix}_pred_shift"] < -0.02, False),
        )
    scored["verifier_support_count"] = (
        (scored["xgboost_delta_distance_edit"].fillna(-999) > 0.02).astype(int)
        + (scored["rf_delta_distance_edit"].fillna(-999) > 0.02).astype(int)
    )
    scored["auxiliary_opposition_count"] = (
        (scored["xgboost_delta_distance_edit"].fillna(0) < -0.02).astype(int)
        + (scored["rf_delta_distance_edit"].fillna(0) < -0.02).astype(int)
    )
    scored["direction_consensus_count"] = (
        scored["main_direction_correct"].fillna(False).astype(int)
        + scored["xgboost_direction_correct"].fillna(False).astype(int)
        + scored["rf_direction_correct"].fillna(False).astype(int)
    )
    scored["target_window_success"] = scored["edited_main_target_region"] == "in_range"
    scored["relaxed_guardrail_pass"] = scored.apply(relaxed_guardrail_pass, axis=1)
    scored["ood_risk"] = (
        (scored["nearest_train_tanimoto_edited"].fillna(-1) < 0.30)
        | (scored["nearest_train_tanimoto_edited"].fillna(99) > 0.95)
        | (scored["main_uncertainty"].fillna(99) > 0.35)
    )
    scored["evidence_ready"] = (
        scored["relaxed_guardrail_pass"]
        & (~scored["edited_interference_alert"].map(bool_value))
        & (~scored["ood_risk"])
    )
    clipped_delta = scored["main_delta_distance_edit"].fillna(0.0).clip(lower=0.0, upper=1.5) / 1.5
    verifier_score = scored["verifier_support_count"].fillna(0).clip(lower=0, upper=2) / 2.0
    guardrail_score = scored["relaxed_guardrail_pass"].astype(float)
    uncertainty_score = (1.0 - scored["main_uncertainty"].fillna(0.5).clip(lower=0.0, upper=0.5) / 0.5).clip(0.0, 1.0)
    parent_similarity_score = scored["parent_tanimoto"].fillna(0.0).clip(0.0, 1.0)
    domain_score = (~scored["ood_risk"]).astype(float)
    scored["final_edit_score"] = (
        0.30 * clipped_delta
        + 0.35 * scored["target_window_success"].astype(float)
        + 0.10 * verifier_score
        + 0.12 * guardrail_score
        + 0.05 * uncertainty_score
        + 0.04 * parent_similarity_score
        + 0.04 * domain_score
    ).clip(0.0, 1.0)
    scored["consensus_success"] = (
        scored["edit_valid"].map(bool_value)
        & (scored["main_delta_distance_edit"].fillna(-999) > 0.05)
        & (scored["verifier_support_count"] >= 1)
        & (scored["auxiliary_opposition_count"] == 0)
    )
    scored["strict_success"] = (
        scored["consensus_success"]
        & scored["target_window_success"]
        & scored["relaxed_guardrail_pass"]
        & (scored["main_uncertainty"] <= 0.35)
    )
    return scored


def top_by_parent(scored: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    valid = scored[scored["edit_valid"].map(bool_value) & scored["edited_main_pred"].notna()].copy()
    if valid.empty:
        return valid
    valid = valid[valid["evidence_ready"].map(bool_value)].copy()
    if valid.empty:
        return valid
    valid = valid.sort_values(
        [
            "method_group",
            "method",
            "mol_id",
            "target_window_success",
            "relaxed_guardrail_pass",
            "auxiliary_opposition_count",
            "final_edit_score",
        ],
        ascending=[True, True, True, False, False, True, False],
    )
    valid["rank_in_method_parent"] = valid.groupby(["method_group", "method", "mol_id"], dropna=False).cumcount() + 1
    valid["rank_in_group_parent"] = valid.groupby(["method_group", "mol_id"], dropna=False).cumcount() + 1
    return valid[(valid["rank_in_group_parent"] <= k) | (valid["rank_in_method_parent"] <= k)].reset_index(drop=True)


def summarize_methods(scored: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = sorted(scored["method"].dropna().unique().tolist())
    primary_top = top[(top["method_group"] == "contribution_primary") & (top["rank_in_group_parent"] <= 5)]
    rows.append(
        summary_row(
            "contribution_primary",
            scored[scored["method_group"] == "contribution_primary"],
            primary_top,
        )
    )
    supplemental_top = top[(top["method_group"] == "contribution_supplemental") & (top["rank_in_group_parent"] <= 5)]
    rows.append(
        summary_row(
            "contribution_supplemental",
            scored[scored["method_group"] == "contribution_supplemental"],
            supplemental_top,
        )
    )
    random_group_top = top[(top["method_group"] == "same_interface_random") & (top["rank_in_group_parent"] <= 5)]
    rows.append(
        summary_row(
            "same_interface_random",
            scored[scored["method_group"] == "same_interface_random"],
            random_group_top,
        )
    )
    for method in methods:
        method_top = top[(top["method"] == method) & (top["rank_in_method_parent"] <= 5)]
        rows.append(summary_row(method, scored[scored["method"] == method], method_top))
    return pd.DataFrame(rows)


def summary_row(name: str, df: pd.DataFrame, top_df: pd.DataFrame) -> dict[str, Any]:
    valid = df[df["edit_valid"].map(bool_value)].copy() if not df.empty else pd.DataFrame()
    scored = valid[valid["edited_main_pred"].notna()].copy() if not valid.empty else pd.DataFrame()
    return {
        "method": name,
        "candidate_attempts": int(len(df)),
        "valid_edits": int(len(valid)),
        "scored_edits": int(len(scored)),
        "covered_parents": int(scored["mol_id"].nunique()) if not scored.empty else 0,
        "top5_rows": int(len(top_df)),
        "top5_parent_count": int(top_df["mol_id"].nunique()) if not top_df.empty else 0,
        "top5_mean_delta": float(top_df["main_delta_distance_edit"].mean()) if not top_df.empty else 0.0,
        "top5_target_window_rate": float(top_df["target_window_success"].mean()) if not top_df.empty else 0.0,
        "top5_guardrail_pass_rate": float(top_df["relaxed_guardrail_pass"].mean()) if not top_df.empty else 0.0,
        "top5_alert_rate": float(top_df["edited_interference_alert"].map(bool_value).mean()) if not top_df.empty else 0.0,
        "top5_ood_rate": float(top_df["ood_risk"].mean()) if not top_df.empty else 0.0,
        "top5_mean_delta_qed": float(top_df["delta_qed"].mean()) if not top_df.empty else 0.0,
        "top5_mean_delta_sa": float(top_df["delta_sa_score"].mean()) if not top_df.empty else 0.0,
        "top5_consensus_rate": float(top_df["consensus_success"].mean()) if not top_df.empty else 0.0,
        "top5_strict_rate": float(top_df["strict_success"].mean()) if not top_df.empty else 0.0,
    }


def gate_status(name: str, actual: float, threshold: float, direction: str = ">=") -> dict[str, Any]:
    passed = actual >= threshold if direction == ">=" else actual <= threshold
    return {
        "gate": name,
        "actual": float(actual),
        "threshold": float(threshold),
        "direction": direction,
        "status": "pass" if passed else "needs_fix",
    }


def build_gate_rows(scored: pd.DataFrame, top: pd.DataFrame, metrics: pd.DataFrame) -> list[dict[str, Any]]:
    metric = metrics.set_index("method")
    cg = metric.loc["contribution_primary"] if "contribution_primary" in metric.index else pd.Series(dtype=float)
    random_row = (
        metric.loc["random_same_interface_replace"]
        if "random_same_interface_replace" in metric.index
        else metric.loc["same_interface_random"]
        if "same_interface_random" in metric.index
        else pd.Series(dtype=float)
    )
    valid_count = int(scored["edit_valid"].map(bool_value).sum()) if not scored.empty else 0
    scored_count = int(scored["edited_main_pred"].notna().sum()) if "edited_main_pred" in scored.columns else 0
    paired = top[top["method_group"].isin(["contribution_primary", "same_interface_random"])].groupby("mol_id")[
        "method_group"
    ].nunique()
    paired_parent_count = int((paired >= 2).sum())
    advantage = float(cg.get("top5_mean_delta", 0.0)) - float(random_row.get("top5_mean_delta", 0.0))
    return [
        gate_status("valid_edit_count", valid_count, 300),
        gate_status("scored_edit_count", scored_count, 300),
        gate_status("contribution_top5_parent_count", float(cg.get("top5_parent_count", 0.0)), 20),
        gate_status("contribution_top5_mean_delta", float(cg.get("top5_mean_delta", 0.0)), 0.50),
        gate_status("contribution_top5_target_window_rate", float(cg.get("top5_target_window_rate", 0.0)), 0.60),
        gate_status("contribution_top5_guardrail_pass_rate", float(cg.get("top5_guardrail_pass_rate", 0.0)), 0.80),
        gate_status("contribution_top5_alert_rate", float(cg.get("top5_alert_rate", 1.0)), 0.0, direction="<="),
        gate_status("contribution_top5_ood_rate", float(cg.get("top5_ood_rate", 1.0)), 0.20, direction="<="),
        gate_status("contribution_top5_mean_delta_qed", float(cg.get("top5_mean_delta_qed", -999.0)), -0.05),
        gate_status("contribution_top5_mean_delta_sa", float(cg.get("top5_mean_delta_sa", 999.0)), 0.50, direction="<="),
        gate_status("paired_control_parent_count", paired_parent_count, 15),
        gate_status("top5_mean_delta_advantage", advantage, 0.05),
    ]


def write_reports(candidates: pd.DataFrame, scored: pd.DataFrame, top: pd.DataFrame, metrics: pd.DataFrame, gates: list[dict[str, Any]], donors: pd.DataFrame, linkers: pd.DataFrame) -> None:
    verdict = "pass" if all(row["status"] == "pass" for row in gates) else "needs_fix"
    summary = {
        "phase": "10D_edit_generation_scoring",
        "verdict": verdict,
        "candidate_attempts": int(len(candidates)),
        "valid_edit_count": int(scored["edit_valid"].map(bool_value).sum()) if not scored.empty else 0,
        "scored_edit_count": int(scored["edited_main_pred"].notna().sum()) if "edited_main_pred" in scored.columns else 0,
        "method_counts": candidates["method"].value_counts().to_dict() if not candidates.empty else {},
        "donor_count": int(len(donors)),
        "linker_count": int(len(linkers)),
        "donor_source_counts": donors["donor_source"].value_counts().to_dict() if not donors.empty else {},
        "linker_source_counts": linkers["linker_source"].value_counts().to_dict() if not linkers.empty else {},
        "gates": gates,
    }
    (REPORTS / "phase10d_edit_generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# 10-D Mol Token 编辑生成与筛选报告",
        "",
        f"结论：{verdict}",
        "",
        "## 核心结果",
        "",
        f"- 候选编辑尝试：{len(candidates)}",
        f"- 有效编辑分子：{summary['valid_edit_count']}",
        f"- 完成评分分子：{summary['scored_edit_count']}",
        f"- donor 数量：{len(donors)}",
        f"- linker 数量：{len(linkers)}",
        "",
        "## 方法对比",
        "",
        "| method | valid | scored | parents | top5 mean delta | TW rate | guardrail | alert | OOD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        lines.append(
            "| "
            f"{row['method']} | {int(row['valid_edits'])} | {int(row['scored_edits'])} | "
            f"{int(row['top5_parent_count'])} | {float(row['top5_mean_delta']):.3f} | "
            f"{float(row['top5_target_window_rate']):.3f} | {float(row['top5_guardrail_pass_rate']):.3f} | "
            f"{float(row['top5_alert_rate']):.3f} | {float(row['top5_ood_rate']):.3f} |"
        )
    lines.extend(["", "## 门控检查", ""])
    for row in gates:
        lines.append(
            f"- {row['gate']}: {row['actual']:.3f} {row['direction']} {row['threshold']:.3f} -> {row['status']}"
        )
    lines.extend(
        [
            "",
            "## 输出表",
            "",
            f"- `{rel(TABLES / 'phase10d_edit_candidate_library.csv')}`",
            f"- `{rel(TABLES / 'phase10d_edit_scored_all.csv')}`",
            f"- `{rel(TABLES / 'phase10d_edit_top_by_parent.csv')}`",
            f"- `{rel(TABLES / 'phase10d_method_comparison.csv')}`",
            f"- `{rel(TABLES / 'phase10d_guardrail_panel.csv')}`",
            "",
            "## 审查意见",
            "",
            "- 若结论为 pass，可进入 10-E 整理最终证据。",
            "- 若结论为 needs_fix，优先检查同接口随机对照、结构警示、适用域和增加动作噪声。",
            "- 本阶段结果仅作为预测层面的优化证据，不写成实测提升。",
            "",
        ]
    )
    (REPORTS / "phase10d_edit_generation_report.md").write_text("\n".join(lines), encoding="utf-8")

    self_lines = [
        "# 10-D 自检记录",
        "",
        f"总体结论：{verdict}",
        "",
        "## 检查项",
        "",
        f"- 候选编辑尝试：{len(candidates)}",
        f"- 有效编辑分子：{summary['valid_edit_count']}",
        f"- 完成评分分子：{summary['scored_edit_count']}",
        f"- 主贡献编辑覆盖父分子：{int(metrics.set_index('method').loc['contribution_primary', 'top5_parent_count']) if 'contribution_primary' in set(metrics['method']) else 0}",
        "",
        "## 门控",
        "",
    ]
    for row in gates:
        self_lines.append(
            f"- {row['gate']}: {row['status']} ({row['actual']:.3f} {row['direction']} {row['threshold']:.3f})"
        )
    (REPORTS / "phase10d_edit_generation_self_check.md").write_text("\n".join(self_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--max-reagents-per-token", type=int, default=20)
    parser.add_argument("--max-add-per-parent", type=int, default=20)
    parser.add_argument("--max-external-reagents-per-role", type=int, default=30)
    args = parser.parse_args()

    ensure_dirs(TABLES, REPORTS, PREDICTIONS)
    _ = load_json(CONFIG_PATH)
    active = pd.read_csv(ACTIVE_TOKEN_PATH)
    quality = pd.read_csv(QUALITY_TABLE_PATH)
    for frame in [active, quality]:
        if "attachment_count" in frame.columns:
            frame["attachment_count"] = pd.to_numeric(frame["attachment_count"], errors="coerce")
        if "n_boundary_atoms" in frame.columns:
            frame["n_boundary_atoms"] = pd.to_numeric(frame["n_boundary_atoms"], errors="coerce")

    donors = sidechain_tools.build_donor_library(max_brics_per_role=45)
    linkers = linker_tools.build_linker_library(max_brics_per_role=45)
    donors, linkers = augment_reagents_from_phase10_libraries(
        donors, linkers, max_external_per_role=args.max_external_reagents_per_role
    )
    donors.to_csv(TABLES / "phase10d_donor_library.csv", index=False, encoding="utf-8-sig")
    linkers.to_csv(TABLES / "phase10d_linker_library.csv", index=False, encoding="utf-8-sig")

    deletes = build_delete_candidates(active)
    replacements = build_replace_candidates(active, donors, linkers, max_reagents_per_token=args.max_reagents_per_token)
    additions = build_add_candidates(active, donors, max_add_per_parent=args.max_add_per_parent)
    random_targets = build_random_control_targets(active, quality, max_targets=len(active))
    random_targets.to_csv(TABLES / "phase10d_same_interface_random_targets.csv", index=False, encoding="utf-8-sig")
    random_replacements = build_replace_candidates(
        random_targets,
        donors,
        linkers,
        max_reagents_per_token=max(10, args.max_reagents_per_token // 2),
        method="random_same_interface_replace",
    )

    candidates = pd.concat([deletes, replacements, additions, random_replacements], ignore_index=True, sort=False)
    candidates["edit_row_id"] = np.arange(len(candidates))
    candidates.to_csv(TABLES / "phase10d_edit_candidate_library.csv", index=False, encoding="utf-8-sig")

    scored = score_candidates(candidates, force_predict=args.force_predict, skip_predict=args.skip_predict)
    scored.to_csv(TABLES / "phase10d_edit_scored_all.csv", index=False, encoding="utf-8-sig")
    guardrail_cols = [
        "edit_row_id",
        "method",
        "mol_id",
        "edited_smiles",
        "edited_valid",
        "edited_mw",
        "edited_tpsa",
        "edited_hbd",
        "edited_hba",
        "edited_rotbonds",
        "edited_qed",
        "edited_sa_score",
        "edited_interference_alert",
        "parent_tanimoto",
        "nearest_train_tanimoto_edited",
        "relaxed_guardrail_pass",
        "ood_risk",
    ]
    scored[[col for col in guardrail_cols if col in scored.columns]].to_csv(
        TABLES / "phase10d_guardrail_panel.csv", index=False, encoding="utf-8-sig"
    )
    top = top_by_parent(scored, k=5)
    top.to_csv(TABLES / "phase10d_edit_top_by_parent.csv", index=False, encoding="utf-8-sig")
    random_replacements.to_csv(TABLES / "phase10d_same_interface_random_control.csv", index=False, encoding="utf-8-sig")
    metrics = summarize_methods(scored, top)
    metrics.to_csv(TABLES / "phase10d_method_comparison.csv", index=False, encoding="utf-8-sig")
    gates = build_gate_rows(scored, top, metrics)
    write_reports(candidates, scored, top, metrics, gates, donors, linkers)


if __name__ == "__main__":
    main()
