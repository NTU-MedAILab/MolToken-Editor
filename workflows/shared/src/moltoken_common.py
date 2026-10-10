from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = ROOT

rdBase.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class TargetWindow:
    low: float = 1.0
    high: float = 3.4

    def distance(self, values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.where(arr < self.low, self.low - arr, np.where(arr > self.high, arr - self.high, 0.0))

    def region(self, value: float) -> str:
        if value < self.low:
            return "low"
        if value > self.high:
            return "high"
        return "in_range"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def morgan_features(smiles_list: Iterable[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    rows: list[np.ndarray] = []
    for smiles in smiles_list:
        mol = mol_from_smiles(smiles)
        arr = np.zeros((n_bits,), dtype=np.float32)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


def rdkit_descriptor_frame(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for smiles in df[smiles_col]:
        mol = mol_from_smiles(smiles)
        if mol is None:
            records.append({})
            continue
        records.append(
            {
                "MolWt": Descriptors.MolWt(mol),
                "MolLogP": Descriptors.MolLogP(mol),
                "TPSA": rdMolDescriptors.CalcTPSA(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "RotBonds": Lipinski.NumRotatableBonds(mol),
                "HeavyAtomCount": mol.GetNumHeavyAtoms(),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
                "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
                "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                "QED": QED.qed(mol),
            }
        )
    desc = pd.DataFrame.from_records(records)
    return desc.replace([np.inf, -np.inf], np.nan).fillna(desc.median(numeric_only=True))


def tanimoto_to_training(candidate_smiles: str, training_smiles: Iterable[str]) -> float:
    mol = mol_from_smiles(candidate_smiles)
    if mol is None:
        return float("nan")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    best = 0.0
    for train_smiles in training_smiles:
        tmol = mol_from_smiles(train_smiles)
        if tmol is None:
            continue
        tfp = AllChem.GetMorganFingerprintAsBitVect(tmol, 2, nBits=2048)
        best = max(best, DataStructs.TanimotoSimilarity(fp, tfp))
    return best
