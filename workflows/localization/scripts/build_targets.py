#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 19-A: Build baseline editable-target tables for MolToken-Editor comparisons.

This script does NOT score molecules.
It only creates matched editable-region target tables for:
1. MolToken original targets
2. Random editable regions
3. BRICS fragments
4. RECAP fragments
5. Murcko side-chain regions

The output tables are designed to answer:
"Where to edit?" under different localization strategies.

Later, these target tables can be fed into the same Phase 10-D edit-generation
and scoring pipeline to ensure fair comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Recap


REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PHASE10 = (
    REPO_ROOT
    / "workflows"
    / "moltoken"
    / "tables"
)

RANDOM_SEED = 20260619
random.seed(RANDOM_SEED)


def parse_atom_indices(x) -> List[int]:
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [int(i) for i in x]
    s = str(x).strip()
    if not s:
        return []
    # allow "1,2,3", "[1, 2, 3]", "1;2;3"
    for ch in "[](){}":
        s = s.replace(ch, "")
    s = s.replace(";", ",").replace("|", ",")
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            pass
    return sorted(set(out))


def atom_indices_to_string(indices: List[int]) -> str:
    return ",".join(str(int(i)) for i in sorted(set(indices)))


def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    return {
        "mol_id": pick(["mol_id", "parent_id", "external_molecule_id"]),
        "smiles": pick(["canonical_smiles", "smiles", "parent_smiles", "SMILES"]),
        "atom_indices": pick(["atom_indices", "selected_atom_indices", "token_atom_indices"]),
        "attachment_count": pick(["attachment_count", "n_attachments", "num_attachments"]),
        "token_id": pick(["token_id", "moltoken_id", "candidate_id"]),
        "target_region": pick(["target_region", "region", "optimization_site"]),
        "library_id": pick(["parent_library_id", "library_id", "source_library_id"]),
    }


def count_attachments(mol: Chem.Mol, atom_set: Set[int]) -> int:
    """Number of bonds crossing selected atom set and outside atoms."""
    if not atom_set:
        return 0
    count = 0
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if (a in atom_set and c not in atom_set) or (c in atom_set and a not in atom_set):
            count += 1
    return count


def cuts_ring(mol: Chem.Mol, atom_set: Set[int]) -> bool:
    """Whether the selected region would cut ring bonds at boundary."""
    if not atom_set:
        return False
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if ((a in atom_set and c not in atom_set) or (c in atom_set and a not in atom_set)) and b.IsInRing():
            return True
    return False


def connected_components_within(mol: Chem.Mol, atom_indices: Set[int]) -> List[List[int]]:
    """Split selected atoms into connected components."""
    atom_indices = set(atom_indices)
    seen = set()
    comps = []
    for a in sorted(atom_indices):
        if a in seen:
            continue
        stack = [a]
        seen.add(a)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            atom = mol.GetAtomWithIdx(u)
            for nb in atom.GetNeighbors():
                v = nb.GetIdx()
                if v in atom_indices and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def make_target_row(
    template_cols: List[str],
    parent_row: pd.Series,
    mol_id: str,
    smiles: str,
    atom_indices: List[int],
    attachment_count: int,
    method_group: str,
    target_source: str,
    token_rank: int,
    extra: Optional[Dict] = None,
) -> Dict:
    """Create a row compatible with active MolToken table as much as possible."""
    extra = extra or {}
    row = {c: None for c in template_cols}

    # Preserve useful parent-level columns if present
    for c in template_cols:
        if c in parent_row.index:
            row[c] = parent_row[c]

    # Standard columns
    for c in ["mol_id", "parent_id", "external_molecule_id"]:
        if c in row:
            row[c] = mol_id

    for c in ["canonical_smiles", "smiles", "parent_smiles", "SMILES"]:
        if c in row:
            row[c] = smiles

    for c in ["atom_indices", "selected_atom_indices", "token_atom_indices"]:
        if c in row:
            row[c] = atom_indices_to_string(atom_indices)

    for c in ["attachment_count", "n_attachments", "num_attachments"]:
        if c in row:
            row[c] = attachment_count

    for c in ["token_id", "moltoken_id", "candidate_id"]:
        if c in row:
            row[c] = f"{target_source}_{mol_id}_{token_rank:03d}"

    # Add explicit columns even if original schema does not have them
    row["baseline_method_group"] = method_group
    row["baseline_target_source"] = target_source
    row["baseline_token_rank"] = token_rank
    row["baseline_atom_count"] = len(atom_indices)
    row["baseline_attachment_count"] = attachment_count
    row["baseline_cuts_ring"] = extra.get("cuts_ring", None)
    row["baseline_note"] = extra.get("note", "")

    return row


def get_parent_table(active_df: pd.DataFrame, quality_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Use active table as parent source; if quality has more parents, merge unique parent records."""
    active_cols = infer_columns(active_df)
    mol_col = active_cols["mol_id"]
    smi_col = active_cols["smiles"]
    if mol_col is None or smi_col is None:
        raise ValueError("Cannot infer mol_id/smiles columns from active MolToken table.")

    base = active_df.drop_duplicates(subset=[mol_col])[[c for c in active_df.columns]].copy()

    if quality_df is not None:
        qcols = infer_columns(quality_df)
        if qcols["mol_id"] and qcols["smiles"]:
            qbase = quality_df.drop_duplicates(subset=[qcols["mol_id"]]).copy()
            # Normalize names if possible
            if qcols["mol_id"] != mol_col:
                qbase[mol_col] = qbase[qcols["mol_id"]]
            if qcols["smiles"] != smi_col:
                qbase[smi_col] = qbase[qcols["smiles"]]
            known = set(base[mol_col].astype(str))
            qbase = qbase[~qbase[mol_col].astype(str).isin(known)]
            if len(qbase) > 0:
                missing_cols = [c for c in base.columns if c not in qbase.columns]
                for c in missing_cols:
                    qbase[c] = None
                base = pd.concat([base, qbase[base.columns]], ignore_index=True)

    return base


def build_moltoken_targets(active_df: pd.DataFrame, max_per_parent: int) -> pd.DataFrame:
    cols = infer_columns(active_df)
    mol_col = cols["mol_id"]
    active_df = active_df.copy()
    if "baseline_method_group" not in active_df.columns:
        active_df["baseline_method_group"] = "moltoken_editor"
    if "baseline_target_source" not in active_df.columns:
        active_df["baseline_target_source"] = "attribution_guided_moltoken"
    if "baseline_note" not in active_df.columns:
        active_df["baseline_note"] = "Original active MolToken targets."

    if max_per_parent > 0 and mol_col:
        active_df = active_df.groupby(mol_col, group_keys=False).head(max_per_parent).reset_index(drop=True)
    return active_df


def build_random_targets(
    active_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    max_per_parent: int,
    template_cols: List[str],
) -> pd.DataFrame:
    """Matched random editable region from contribution split quality table."""
    acols = infer_columns(active_df)
    qcols = infer_columns(quality_df)

    mol_col = acols["mol_id"]
    smi_col = acols["smiles"]
    atom_col = acols["atom_indices"]
    attach_col = acols["attachment_count"]

    q_mol_col = qcols["mol_id"]
    q_smi_col = qcols["smiles"]
    q_atom_col = qcols["atom_indices"]
    q_attach_col = qcols["attachment_count"]

    if not all([mol_col, smi_col, atom_col, q_mol_col, q_smi_col, q_atom_col]):
        raise ValueError("Cannot infer required columns for random baseline from active/quality tables.")

    active_sets_by_parent = {}
    for _, r in active_df.iterrows():
        mid = str(r[mol_col])
        aset = tuple(parse_atom_indices(r[atom_col]))
        active_sets_by_parent.setdefault(mid, set()).add(aset)

    rows = []
    for mid, g_active in active_df.groupby(mol_col):
        mid = str(mid)
        gq = quality_df[quality_df[q_mol_col].astype(str) == mid].copy()
        if len(gq) == 0:
            continue

        used = set()
        take_n = max_per_parent if max_per_parent > 0 else len(g_active)

        # desired attachment count distribution from MolToken targets
        desired_attachments = []
        if attach_col:
            for x in g_active[attach_col].tolist():
                try:
                    desired_attachments.append(int(float(x)))
                except Exception:
                    pass
        if not desired_attachments:
            desired_attachments = [None] * take_n

        # randomize order
        gq = gq.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        for rank in range(take_n):
            desired_attach = desired_attachments[rank % len(desired_attachments)]
            chosen = None
            for _, qr in gq.iterrows():
                atoms = parse_atom_indices(qr[q_atom_col])
                if not atoms:
                    continue
                tup = tuple(atoms)
                if tup in used:
                    continue
                if tup in active_sets_by_parent.get(mid, set()):
                    continue
                if q_attach_col and desired_attach is not None:
                    try:
                        qa = int(float(qr[q_attach_col]))
                        if qa != desired_attach:
                            continue
                    except Exception:
                        pass
                chosen = qr
                used.add(tup)
                break

            if chosen is None:
                continue

            smiles = str(chosen[q_smi_col])
            mol = mol_from_smiles(smiles)
            atoms = parse_atom_indices(chosen[q_atom_col])
            attach = None
            if q_attach_col:
                try:
                    attach = int(float(chosen[q_attach_col]))
                except Exception:
                    attach = None
            if attach is None and mol is not None:
                attach = count_attachments(mol, set(atoms))

            rows.append(make_target_row(
                template_cols,
                chosen,
                mid,
                smiles,
                atoms,
                int(attach or 0),
                "random_action_matched",
                "random_editable_region",
                rank + 1,
                {"cuts_ring": cuts_ring(mol, set(atoms)) if mol else None,
                 "note": "Random editable region matched by parent and attachment count where possible."}
            ))

    return pd.DataFrame(rows)


def build_brics_targets(parent_df: pd.DataFrame, template_cols: List[str], max_per_parent: int) -> pd.DataFrame:
    cols = infer_columns(parent_df)
    mol_col = cols["mol_id"]
    smi_col = cols["smiles"]

    rows = []
    for _, pr in parent_df.iterrows():
        mid = str(pr[mol_col])
        smi = str(pr[smi_col])
        mol = mol_from_smiles(smi)
        if mol is None:
            continue

        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds:
            continue

        cut_atoms = set()
        for item in brics_bonds:
            # item form: ((a,b), (label1,label2))
            try:
                a, b = item[0]
                cut_atoms.add(int(a))
                cut_atoms.add(int(b))
            except Exception:
                continue

        # Components induced by atoms adjacent to BRICS bonds
        comps = connected_components_within(mol, cut_atoms)
        # Also include one-bond neighborhoods for richer candidate regions
        candidate_sets = []
        for comp in comps:
            s = set(comp)
            for a in list(s):
                for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                    s.add(nb.GetIdx())
            candidate_sets.append(sorted(s))

        # Deduplicate and rank by size/attachment
        seen = set()
        final_sets = []
        for atoms in candidate_sets:
            tup = tuple(sorted(atoms))
            if tup in seen:
                continue
            seen.add(tup)
            if len(atoms) < 1:
                continue
            final_sets.append(atoms)

        final_sets = sorted(final_sets, key=lambda x: (len(x), x))
        if max_per_parent > 0:
            final_sets = final_sets[:max_per_parent]

        for rank, atoms in enumerate(final_sets, 1):
            aset = set(atoms)
            attach = count_attachments(mol, aset)
            rows.append(make_target_row(
                template_cols,
                pr,
                mid,
                smi,
                atoms,
                attach,
                "brics_edit",
                "brics_fragment_region",
                rank,
                {"cuts_ring": cuts_ring(mol, aset),
                 "note": "Editable region derived from BRICS bond neighborhoods."}
            ))

    return pd.DataFrame(rows)


def build_recap_targets(parent_df: pd.DataFrame, template_cols: List[str], max_per_parent: int) -> pd.DataFrame:
    cols = infer_columns(parent_df)
    mol_col = cols["mol_id"]
    smi_col = cols["smiles"]

    rows = []
    for _, pr in parent_df.iterrows():
        mid = str(pr[mol_col])
        smi = str(pr[smi_col])
        mol = mol_from_smiles(smi)
        if mol is None:
            continue

        try:
            tree = Recap.RecapDecompose(mol)
            leaves = tree.GetLeaves()
        except Exception:
            leaves = {}

        candidate_sets = []
        for frag_smi, node in leaves.items():
            frag = mol_from_smiles(frag_smi.replace("*", "[*]"))
            # RECAP fragments often contain dummy atoms; remove dummies for substructure match
            try:
                frag_clean = Chem.DeleteSubstructs(frag, Chem.MolFromSmarts("[#0]")) if frag else None
                if frag_clean is None or frag_clean.GetNumAtoms() == 0:
                    continue
                matches = mol.GetSubstructMatches(frag_clean, uniquify=True)
                for m in matches:
                    candidate_sets.append(sorted(m))
            except Exception:
                continue

        # Fallback: if RECAP gives no usable match, skip.
        seen = set()
        final_sets = []
        for atoms in candidate_sets:
            tup = tuple(sorted(atoms))
            if tup in seen:
                continue
            seen.add(tup)
            final_sets.append(atoms)

        final_sets = sorted(final_sets, key=lambda x: (len(x), x))
        if max_per_parent > 0:
            final_sets = final_sets[:max_per_parent]

        for rank, atoms in enumerate(final_sets, 1):
            aset = set(atoms)
            attach = count_attachments(mol, aset)
            rows.append(make_target_row(
                template_cols,
                pr,
                mid,
                smi,
                atoms,
                attach,
                "recap_edit",
                "recap_fragment_region",
                rank,
                {"cuts_ring": cuts_ring(mol, aset),
                 "note": "Editable region derived from RECAP decomposition leaves."}
            ))

    return pd.DataFrame(rows)


def build_murcko_sidechain_targets(parent_df: pd.DataFrame, template_cols: List[str], max_per_parent: int) -> pd.DataFrame:
    cols = infer_columns(parent_df)
    mol_col = cols["mol_id"]
    smi_col = cols["smiles"]

    rows = []
    for _, pr in parent_df.iterrows():
        mid = str(pr[mol_col])
        smi = str(pr[smi_col])
        mol = mol_from_smiles(smi)
        if mol is None:
            continue

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            matches = mol.GetSubstructMatches(scaffold, uniquify=True)
        except Exception:
            matches = []

        if not matches:
            continue

        scaffold_atoms = set(matches[0])
        side_atoms = set(range(mol.GetNumAtoms())) - scaffold_atoms
        if not side_atoms:
            continue

        comps = connected_components_within(mol, side_atoms)
        comps = sorted(comps, key=lambda x: (len(x), x))

        if max_per_parent > 0:
            comps = comps[:max_per_parent]

        for rank, atoms in enumerate(comps, 1):
            aset = set(atoms)
            attach = count_attachments(mol, aset)
            rows.append(make_target_row(
                template_cols,
                pr,
                mid,
                smi,
                atoms,
                attach,
                "rgroup_enumeration",
                "murcko_sidechain_region",
                rank,
                {"cuts_ring": cuts_ring(mol, aset),
                 "note": "Traditional baseline target: side-chain region outside Murcko scaffold."}
            ))

    return pd.DataFrame(rows)


def summarize_targets(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if df is None or len(df) == 0:
        summary = pd.DataFrame()
        summary.to_csv(out_path, index=False)
        return summary

    method_col = "baseline_method_group"
    mol_col = "mol_id" if "mol_id" in df.columns else None
    atom_col = "baseline_atom_count"
    attach_col = "baseline_attachment_count"
    cuts_col = "baseline_cuts_ring"

    rows = []
    for method, g in df.groupby(method_col):
        item = {
            "method_group": method,
            "target_rows": len(g),
            "parent_count": g[mol_col].nunique() if mol_col else None,
            "mean_atom_count": g[atom_col].mean() if atom_col in g.columns else None,
            "mean_attachment_count": g[attach_col].mean() if attach_col in g.columns else None,
            "ring_cut_rate": g[cuts_col].astype(str).str.lower().eq("true").mean() if cuts_col in g.columns else None,
        }
        rows.append(item)

    summary = pd.DataFrame(rows).sort_values("method_group")
    summary.to_csv(out_path, index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="workflows")
    parser.add_argument("--active", default=None)
    parser.add_argument("--quality", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--max-targets-per-parent", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.project_root)

    active_path = Path(args.active) if args.active else FROZEN_PHASE10 / "phase10c_active_moltoken_pool.csv"
    quality_path = Path(args.quality) if args.quality else FROZEN_PHASE10 / "phase10c_contribution_split_quality_table.csv"

    # Fallback to Chinese/original paths if needed
    if not active_path.exists():
        candidates = list(root.rglob("phase10c_active_moltoken_pool.csv"))
        if candidates:
            active_path = candidates[0]
    if not quality_path.exists():
        candidates = list(root.rglob("phase10c_contribution_split_quality_table.csv"))
        if candidates:
            quality_path = candidates[0]

    outdir = Path(args.outdir) if args.outdir else root / "localization/tables"
    outdir.mkdir(parents=True, exist_ok=True)

    if not active_path.exists():
        raise FileNotFoundError(f"active MolToken pool not found: {active_path}")
    if not quality_path.exists():
        raise FileNotFoundError(f"contribution split quality table not found: {quality_path}")

    active = pd.read_csv(active_path, low_memory=False)
    quality = pd.read_csv(quality_path, low_memory=False)

    template_cols = list(active.columns)
    parent_df = get_parent_table(active, quality)

    print("[INFO] active:", active_path, "rows=", len(active))
    print("[INFO] quality:", quality_path, "rows=", len(quality))
    print("[INFO] parent_df rows:", len(parent_df))

    moltoken_targets = build_moltoken_targets(active, args.max_targets_per_parent)
    random_targets = build_random_targets(active, quality, args.max_targets_per_parent, template_cols)
    brics_targets = build_brics_targets(parent_df, template_cols, args.max_targets_per_parent)
    recap_targets = build_recap_targets(parent_df, template_cols, args.max_targets_per_parent)
    murcko_targets = build_murcko_sidechain_targets(parent_df, template_cols, args.max_targets_per_parent)

    outputs = {
        "phase19_targets_moltoken_editor.csv": moltoken_targets,
        "phase19_targets_random_action_matched.csv": random_targets,
        "phase19_targets_brics_edit.csv": brics_targets,
        "phase19_targets_recap_edit.csv": recap_targets,
        "phase19_targets_rgroup_enumeration.csv": murcko_targets,
    }

    all_df = []
    for name, df in outputs.items():
        p = outdir / name
        df.to_csv(p, index=False)
        print(f"[WROTE] {p} rows={len(df)}")
        if len(df):
            all_df.append(df)

    if all_df:
        combined = pd.concat(all_df, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()

    combined_path = outdir / "phase19_all_baseline_targets.csv"
    combined.to_csv(combined_path, index=False)

    summary_path = outdir / "phase19_baseline_target_summary.csv"
    summary = summarize_targets(combined, summary_path)

    report = {
        "phase": "19A_baseline_target_construction",
        "status": "pass" if len(combined) > 0 else "failed",
        "seed": RANDOM_SEED,
        "active_path": str(active_path),
        "quality_path": str(quality_path),
        "outdir": str(outdir),
        "max_targets_per_parent": args.max_targets_per_parent,
        "total_targets": int(len(combined)),
        "summary": summary.to_dict(orient="records") if len(summary) else [],
    }
    report_path = outdir.parent / "reports/phase19a_self_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Phase19 target summary ===")
    print(summary.to_string(index=False))
    print("\n[WROTE]", combined_path)
    print("[WROTE]", summary_path)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
