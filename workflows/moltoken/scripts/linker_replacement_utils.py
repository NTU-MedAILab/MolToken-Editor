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


def linker_descriptor(smiles: str) -> dict[str, Any] | None:
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


def infer_linker_role(desc: dict[str, Any]) -> str:
    logp = float(desc.get("logp", 0.0))
    tpsa = float(desc.get("tpsa", 0.0))
    hetero = float(desc.get("hbd", 0)) + float(desc.get("hba", 0))
    if logp >= 1.0 and tpsa <= 70 and hetero <= 4:
        return "increase_logd"
    if logp <= 0.6 or tpsa >= 50 or hetero >= 3:
        return "decrease_logd"
    return "neutral"


def canonical_linker_smiles(smiles: str) -> str | None:
    mol = _mol(smiles)
    if mol is None:
        return None
    dummy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy) != 2:
        return None
    if any(len(mol.GetAtomWithIdx(i).GetNeighbors()) != 1 for i in dummy):
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_linker_library(max_brics_per_role: int = 45) -> pd.DataFrame:
    seeds = [
        ("linker_inc_001", "[*]C[*]", "increase_logd", "seed_linker"),
        ("linker_inc_002", "[*]CC[*]", "increase_logd", "seed_linker"),
        ("linker_inc_003", "[*]c1ccccc1[*]", "increase_logd", "seed_linker"),
        ("linker_dec_001", "[*]O[*]", "decrease_logd", "seed_linker"),
        ("linker_dec_002", "[*]CO[*]", "decrease_logd", "seed_linker"),
        ("linker_dec_003", "[*]N[*]", "decrease_logd", "seed_linker"),
        ("linker_dec_004", "[*]C(=O)N[*]", "decrease_logd", "seed_linker"),
    ]
    rows = []
    for lid, smi, role, source in seeds:
        can = canonical_linker_smiles(smi)
        if can is None:
            continue
        desc = linker_descriptor(can) or {}
        rows.append({"linker_id": lid, "linker_smiles": can, "linker_role": role, "linker_source": source, "train_occurrence": 1, **{f"linker_{k}": v for k, v in desc.items()}})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["linker_rank_in_role"] = out.groupby("linker_role").cumcount() + 1
    return out


def replace_two_attachment_once(parent_smiles: str, atom_indices: list[int], linker_smiles: str, orientation: str = "forward") -> tuple[str | None, dict[str, Any]]:
    meta: dict[str, Any] = {"edit_valid": False, "edit_reason": "uninitialized", "orientation": orientation}
    parent = _mol(parent_smiles)
    linker = _mol(linker_smiles)
    if parent is None or linker is None:
        meta["edit_reason"] = "invalid_parent_or_linker"
        return None, meta
    token_atoms = {int(x) for x in atom_indices}
    attachments = _attachment_atoms(parent, token_atoms)
    if len(attachments) != 2:
        meta["edit_reason"] = f"expected_two_attachments_found_{len(attachments)}"
        return None, meta
    dummy = [a.GetIdx() for a in linker.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy) != 2:
        meta["edit_reason"] = "linker_not_two_dummy"
        return None, meta
    neigh = []
    for d in dummy:
        ns = linker.GetAtomWithIdx(d).GetNeighbors()
        if len(ns) != 1:
            meta["edit_reason"] = "linker_dummy_degree_not_one"
            return None, meta
        neigh.append(ns[0].GetIdx())
    skeleton, parent_map = _remove_atoms(parent, token_atoms)
    if any(a not in parent_map for a in attachments):
        meta["edit_reason"] = "attachment_removed"
        return None, meta
    linker_core, linker_map = _remove_atoms(linker, set(dummy))
    if any(n not in linker_map for n in neigh):
        meta["edit_reason"] = "linker_neighbor_missing"
        return None, meta
    combo = Chem.RWMol(Chem.CombineMols(skeleton, linker_core))
    offset = skeleton.GetNumAtoms()
    order = [(0, 0), (1, 1)] if orientation != "reverse" else [(0, 1), (1, 0)]
    for parent_i, neigh_i in order:
        combo.AddBond(parent_map[attachments[parent_i]], offset + linker_map[neigh[neigh_i]], Chem.BondType.SINGLE)
    smi = _canon(combo.GetMol())
    if smi is None:
        meta["edit_reason"] = "sanitize_failed"
        return None, meta
    meta.update({"edit_valid": True, "edit_reason": "ok", "parent_attachment_atoms": ";".join(map(str, attachments))})
    return smi, meta
