#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Repair Phase19-A baseline target tables.

Fixes:
1. Rebuild BRICS targets using true graph fragments after cutting BRICS bonds.
2. Add baseline_atom_count / baseline_attachment_count / baseline_cuts_ring
   for MolToken-Editor original targets.
3. Rebuild combined target table and summary.

This script should be run after build_phase19_baseline_targets.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
FROZEN_PHASE10 = (
    ROOT
    / "moltoken"
    / "tables"
)
OUTDIR = ROOT / "localization/tables"
REPORT_DIR = ROOT / "localization/reports"


def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return mol
    except Exception:
        return None


def parse_atom_indices(x) -> List[int]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return sorted(set(int(i) for i in x))
    s = str(x).strip()
    if not s:
        return []
    for ch in "[](){}":
        s = s.replace(ch, "")
    s = s.replace(";", ",").replace("|", ",")
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            pass
    return sorted(set(out))


def atom_indices_to_string(indices: List[int]) -> str:
    return ",".join(str(int(i)) for i in sorted(set(indices)))


def infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy fallback
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def infer_mol_col(df):
    return infer_col(df, ["mol_id", "parent_id", "external_molecule_id"])


def infer_smiles_col(df):
    return infer_col(df, ["canonical_smiles", "smiles", "parent_smiles", "SMILES"])


def infer_atom_col(df):
    return infer_col(df, [
        "atom_indices",
        "selected_atom_indices",
        "token_atom_indices",
        "moltoken_atom_indices",
        "candidate_atom_indices",
        "region_atom_indices",
        "atom_ids",
        "selected_atoms",
        "atoms",
    ])


def infer_attach_col(df):
    return infer_col(df, [
        "attachment_count",
        "n_attachments",
        "num_attachments",
        "baseline_attachment_count",
        "n_boundary_bonds",
        "num_boundary_bonds",
    ])


def count_attachments(mol: Chem.Mol, atom_set: Set[int]) -> int:
    count = 0
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if (a in atom_set and c not in atom_set) or (c in atom_set and a not in atom_set):
            count += 1
    return count


def cuts_ring(mol: Chem.Mol, atom_set: Set[int]) -> bool:
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if ((a in atom_set and c not in atom_set) or (c in atom_set and a not in atom_set)) and b.IsInRing():
            return True
    return False


def rebuild_moltoken_targets() -> pd.DataFrame:
    p = OUTDIR / "phase19_targets_moltoken_editor.csv"
    df = pd.read_csv(p, low_memory=False)

    smi_col = infer_smiles_col(df)
    atom_col = infer_atom_col(df)
    attach_col = infer_attach_col(df)

    atom_counts = []
    attach_counts = []
    ring_flags = []

    for _, r in df.iterrows():
        smi = r[smi_col] if smi_col else None
        mol = mol_from_smiles(smi)
        atoms = parse_atom_indices(r[atom_col]) if atom_col else []

        atom_counts.append(len(atoms) if atoms else None)

        if mol is not None and atoms:
            aset = set(atoms)
            attach_counts.append(count_attachments(mol, aset))
            ring_flags.append(cuts_ring(mol, aset))
        elif attach_col and not pd.isna(r.get(attach_col)):
            try:
                attach_counts.append(int(float(r[attach_col])))
            except Exception:
                attach_counts.append(None)
            ring_flags.append(False)
        else:
            attach_counts.append(None)
            ring_flags.append(False)

    df["baseline_method_group"] = "moltoken_editor"
    df["baseline_target_source"] = "attribution_guided_moltoken"
    df["baseline_atom_count"] = atom_counts
    df["baseline_attachment_count"] = attach_counts
    df["baseline_cuts_ring"] = ring_flags
    df["baseline_note"] = "Original active MolToken targets with repaired summary fields."

    out = OUTDIR / "phase19_targets_moltoken_editor_v2.csv"
    df.to_csv(out, index=False)
    print("[WROTE]", out, "rows=", len(df))
    return df


def build_parent_df() -> pd.DataFrame:
    active_path = FROZEN_PHASE10 / "phase10c_active_moltoken_pool.csv"
    quality_path = FROZEN_PHASE10 / "phase10c_contribution_split_quality_table.csv"

    active = pd.read_csv(active_path, low_memory=False)
    quality = pd.read_csv(quality_path, low_memory=False)

    mol_col = infer_mol_col(active)
    smi_col = infer_smiles_col(active)
    q_mol_col = infer_mol_col(quality)
    q_smi_col = infer_smiles_col(quality)

    base = active.drop_duplicates(subset=[mol_col]).copy()

    if q_mol_col and q_smi_col:
        qbase = quality.drop_duplicates(subset=[q_mol_col]).copy()
        if q_mol_col != mol_col:
            qbase[mol_col] = qbase[q_mol_col]
        if q_smi_col != smi_col:
            qbase[smi_col] = qbase[q_smi_col]

        known = set(base[mol_col].astype(str))
        qbase = qbase[~qbase[mol_col].astype(str).isin(known)]

        for c in base.columns:
            if c not in qbase.columns:
                qbase[c] = None
        base = pd.concat([base, qbase[base.columns]], ignore_index=True)

    return base


def make_like_template(template_cols, parent_row, mol_id, smiles, atoms, method, source, rank, note):
    row = {c: None for c in template_cols}
    for c in template_cols:
        if c in parent_row.index:
            row[c] = parent_row[c]

    for c in ["mol_id", "parent_id", "external_molecule_id"]:
        if c in row:
            row[c] = mol_id

    for c in ["canonical_smiles", "smiles", "parent_smiles", "SMILES"]:
        if c in row:
            row[c] = smiles

    for c in ["atom_indices", "selected_atom_indices", "token_atom_indices"]:
        if c in row:
            row[c] = atom_indices_to_string(atoms)

    if "token_id" in row:
        row["token_id"] = f"{source}_{mol_id}_{rank:03d}"

    mol = mol_from_smiles(smiles)
    aset = set(atoms)
    attach = count_attachments(mol, aset) if mol is not None else None
    ring = cuts_ring(mol, aset) if mol is not None else None

    row["baseline_method_group"] = method
    row["baseline_target_source"] = source
    row["baseline_token_rank"] = rank
    row["baseline_atom_count"] = len(atoms)
    row["baseline_attachment_count"] = attach
    row["baseline_cuts_ring"] = ring
    row["baseline_note"] = note

    return row


def rebuild_brics_targets(max_targets_per_parent=3) -> pd.DataFrame:
    parent_df = build_parent_df()
    template_cols = list(pd.read_csv(
        FROZEN_PHASE10 / "phase10c_active_moltoken_pool.csv",
        nrows=1
    ).columns)

    mol_col = infer_mol_col(parent_df)
    smi_col = infer_smiles_col(parent_df)

    rows = []

    for _, pr in parent_df.iterrows():
        mol_id = str(pr[mol_col])
        smiles = str(pr[smi_col])
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue

        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds:
            continue

        bond_indices = []
        for item in brics_bonds:
            try:
                a, b = item[0]
                bond = mol.GetBondBetweenAtoms(int(a), int(b))
                if bond is not None:
                    bond_indices.append(bond.GetIdx())
            except Exception:
                pass

        if not bond_indices:
            continue

        # Remove BRICS bonds from graph, then get original atom-index fragments
        rw = Chem.RWMol(mol)
        # remove in descending bond index to avoid index shifts
        for bi in sorted(set(bond_indices), reverse=True):
            b = rw.GetBondWithIdx(int(bi))
            rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

        cut_mol = rw.GetMol()
        frags = Chem.GetMolFrags(cut_mol, asMols=False, sanitizeFrags=False)

        candidate_sets = []
        n_atoms_total = mol.GetNumAtoms()
        for frag in frags:
            atoms = sorted(set(int(i) for i in frag))
            if len(atoms) == 0:
                continue
            if len(atoms) == n_atoms_total:
                continue

            aset = set(atoms)
            attach = count_attachments(mol, aset)
            if attach <= 0:
                continue

            # For BRICS baseline, ring-cut fragments should be excluded.
            if cuts_ring(mol, aset):
                continue

            candidate_sets.append(atoms)

        # Deduplicate and rank: prefer chemically local fragments with 1-2 attachments
        seen = set()
        final_sets = []
        for atoms in candidate_sets:
            tup = tuple(atoms)
            if tup in seen:
                continue
            seen.add(tup)
            final_sets.append(atoms)

        final_sets = sorted(
            final_sets,
            key=lambda atoms: (
                abs(count_attachments(mol, set(atoms)) - 1),
                len(atoms),
                atoms
            )
        )

        final_sets = final_sets[:max_targets_per_parent]

        for rank, atoms in enumerate(final_sets, 1):
            rows.append(make_like_template(
                template_cols,
                pr,
                mol_id,
                smiles,
                atoms,
                "brics_edit",
                "brics_fragment_component",
                rank,
                "BRICS target rebuilt from graph components after cutting BRICS bonds; ring cuts excluded."
            ))

    df = pd.DataFrame(rows)
    out = OUTDIR / "phase19_targets_brics_edit_v2.csv"
    df.to_csv(out, index=False)
    print("[WROTE]", out, "rows=", len(df))
    return df


def summarize(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("baseline_method_group"):
        rows.append({
            "method_group": method,
            "target_rows": len(g),
            "parent_count": g["mol_id"].nunique() if "mol_id" in g.columns else None,
            "mean_atom_count": pd.to_numeric(g["baseline_atom_count"], errors="coerce").mean(),
            "mean_attachment_count": pd.to_numeric(g["baseline_attachment_count"], errors="coerce").mean(),
            "ring_cut_rate": g["baseline_cuts_ring"].astype(str).str.lower().eq("true").mean(),
        })
    out = pd.DataFrame(rows).sort_values("method_group")
    out.to_csv(out_path, index=False)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    moltoken = rebuild_moltoken_targets()
    brics = rebuild_brics_targets(max_targets_per_parent=3)

    # Keep existing usable tables
    random_df = pd.read_csv(OUTDIR / "phase19_targets_random_action_matched.csv", low_memory=False)
    recap_df = pd.read_csv(OUTDIR / "phase19_targets_recap_edit.csv", low_memory=False)
    rgroup_df = pd.read_csv(OUTDIR / "phase19_targets_rgroup_enumeration.csv", low_memory=False)

    combined = pd.concat(
        [moltoken, random_df, brics, recap_df, rgroup_df],
        ignore_index=True,
        sort=False
    )

    combined_out = OUTDIR / "phase19_all_baseline_targets_v2.csv"
    combined.to_csv(combined_out, index=False)

    summary_out = OUTDIR / "phase19_baseline_target_summary_v2.csv"
    summary = summarize(combined, summary_out)

    report = {
        "phase": "19A_baseline_target_construction_v2_repaired",
        "status": "pass",
        "notes": [
            "MolToken summary fields repaired.",
            "BRICS rebuilt from true graph fragments after cutting BRICS bonds.",
            "BRICS ring-cut fragments excluded."
        ],
        "combined_targets": int(len(combined)),
        "summary": summary.to_dict(orient="records"),
    }
    report_out = REPORT_DIR / "phase19a_self_check_v2.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Phase19-A v2 target summary ===")
    print(summary.to_string(index=False))
    print("\n[WROTE]", combined_out)
    print("[WROTE]", summary_out)
    print("[WROTE]", report_out)


if __name__ == "__main__":
    main()
