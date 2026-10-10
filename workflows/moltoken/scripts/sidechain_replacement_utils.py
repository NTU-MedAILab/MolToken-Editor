from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


def _mol(smiles: str):
    return Chem.MolFromSmiles(str(smiles)) if smiles is not None else None


def _canon(mol) -> str | None:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _remove_atoms(mol, atoms: set[int]):
    rw = Chem.RWMol(mol)
    mapping: dict[int, int] = {}
    for old in range(mol.GetNumAtoms()):
        if old not in atoms:
            mapping[old] = old - sum(1 for a in atoms if a < old)
    for idx in sorted(atoms, reverse=True):
        rw.RemoveAtom(int(idx))
    return rw.GetMol(), mapping


def _attachment_atoms(mol, token_atoms: set[int]) -> list[int]:
    outs = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in token_atoms and b not in token_atoms:
            outs.append(b)
        elif b in token_atoms and a not in token_atoms:
            outs.append(a)
    return list(dict.fromkeys(outs))


def donor_descriptor(smiles: str) -> dict[str, Any] | None:
    mol = _mol(smiles)
    if mol is None:
        return None
    heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    try:
        return {
            "heavy_atoms": int(heavy),
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "rotbonds": int(Lipinski.NumRotatableBonds(mol)),
            "qed": float(QED.qed(mol)),
        }
    except Exception:
        return {"heavy_atoms": int(heavy), "logp": 0.0, "tpsa": 0.0, "hbd": 0, "hba": 0, "rotbonds": 0, "qed": 0.0, "mw": 0.0}


def infer_donor_role(desc: dict[str, Any]) -> str:
    logp = float(desc.get("logp", 0.0))
    tpsa = float(desc.get("tpsa", 0.0))
    hetero = float(desc.get("hbd", 0)) + float(desc.get("hba", 0))
    if logp >= 1.2 and tpsa <= 60 and hetero <= 3:
        return "increase_logd"
    if logp <= 0.6 or tpsa >= 45 or hetero >= 3:
        return "decrease_logd"
    return "neutral"


def canonical_donor_smiles(smiles: str) -> str | None:
    mol = _mol(smiles)
    if mol is None:
        return None
    dummy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy) != 1:
        return None
    if len(mol.GetAtomWithIdx(dummy[0]).GetNeighbors()) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_donor_library(max_brics_per_role: int = 45) -> pd.DataFrame:
    # Conservative one-attachment BRICS-style side-chain donors. Additional donors are mined from mini libraries in phase10d.
    seeds = [
        ("donor_inc_001", "[*]C", "increase_logd", "seed_sidechain"),
        ("donor_inc_002", "[*]CC", "increase_logd", "seed_sidechain"),
        ("donor_inc_003", "[*]CCC", "increase_logd", "seed_sidechain"),
        ("donor_inc_004", "[*]c1ccccc1", "increase_logd", "seed_sidechain"),
        ("donor_inc_005", "[*]Cl", "increase_logd", "seed_sidechain"),
        ("donor_dec_001", "[*]O", "decrease_logd", "seed_sidechain"),
        ("donor_dec_002", "[*]CO", "decrease_logd", "seed_sidechain"),
        ("donor_dec_003", "[*]N", "decrease_logd", "seed_sidechain"),
        ("donor_dec_004", "[*]C(=O)O", "decrease_logd", "seed_sidechain"),
        ("donor_dec_005", "[*]C#N", "decrease_logd", "seed_sidechain"),
    ]
    rows = []
    for did, smi, role, source in seeds:
        can = canonical_donor_smiles(smi)
        if can is None:
            continue
        desc = donor_descriptor(can) or {}
        rows.append({"donor_id": did, "donor_smiles": can, "donor_role": role, "donor_source": source, "train_occurrence": 1, **{f"donor_{k}": v for k, v in desc.items()}})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["donor_rank_in_role"] = out.groupby("donor_role").cumcount() + 1
    return out


def replace_single_attachment(parent_smiles: str, atom_indices: list[int], donor_smiles: str) -> tuple[str | None, dict[str, Any]]:
    meta: dict[str, Any] = {"edit_valid": False, "edit_reason": "uninitialized"}
    parent = _mol(parent_smiles)
    donor = _mol(donor_smiles)
    if parent is None or donor is None:
        meta["edit_reason"] = "invalid_parent_or_donor"
        return None, meta
    token_atoms = {int(x) for x in atom_indices}
    attachments = _attachment_atoms(parent, token_atoms)
    if len(attachments) != 1:
        meta["edit_reason"] = f"expected_one_attachment_found_{len(attachments)}"
        return None, meta
    dummy = [a.GetIdx() for a in donor.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy) != 1:
        meta["edit_reason"] = "donor_not_single_dummy"
        return None, meta
    d_nei = donor.GetAtomWithIdx(dummy[0]).GetNeighbors()
    if len(d_nei) != 1:
        meta["edit_reason"] = "donor_dummy_degree_not_one"
        return None, meta
    skeleton, parent_map = _remove_atoms(parent, token_atoms)
    if attachments[0] not in parent_map:
        meta["edit_reason"] = "attachment_removed"
        return None, meta
    donor_core, donor_map = _remove_atoms(donor, set(dummy))
    neighbor_old = d_nei[0].GetIdx()
    if neighbor_old not in donor_map:
        meta["edit_reason"] = "donor_neighbor_missing"
        return None, meta
    combo = Chem.RWMol(Chem.CombineMols(skeleton, donor_core))
    offset = skeleton.GetNumAtoms()
    combo.AddBond(parent_map[attachments[0]], offset + donor_map[neighbor_old], Chem.BondType.SINGLE)
    smi = _canon(combo.GetMol())
    if smi is None:
        meta["edit_reason"] = "sanitize_failed"
        return None, meta
    meta.update({"edit_valid": True, "edit_reason": "ok", "parent_attachment_atom": attachments[0]})
    return smi, meta
