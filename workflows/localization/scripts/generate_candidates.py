#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
PHASE10D_SCRIPT = ROOT / "moltoken/scripts/generate_edits.py"

PHASE19 = ROOT / "localization"
TABLES = PHASE19 / "tables"
REPORTS = PHASE19 / "reports"
PREDICTIONS = PHASE19 / "predictions"

METHOD_ORDER = [
    "moltoken_editor",
    "brics_edit",
    "recap_edit",
    "rgroup_enumeration",
    "random_action_matched",
]


def import_phase10d():
    if not PHASE10D_SCRIPT.exists():
        raise FileNotFoundError(f"Missing Phase10-D script: {PHASE10D_SCRIPT}")

    sys.path.insert(0, str(PHASE10D_SCRIPT.parent))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path.cwd()))

    spec = importlib.util.spec_from_file_location("phase10d_original", PHASE10D_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    # Redirect outputs to Phase19 to avoid overwriting Phase10D results.
    mod.TABLES = TABLES
    mod.REPORTS = REPORTS
    mod.PREDICTIONS = PREDICTIONS

    # Monkey-patch original parse_ints.
    # Original Phase10-D parse_ints fails on multi-atom strings such as
    # "16,17,18" or "16 17 18". Baseline targets require multi-atom regions.
    def robust_parse_ints(x):
        import pandas as _pd
        if x is None:
            return []
        try:
            if _pd.isna(x):
                return []
        except Exception:
            pass
        if isinstance(x, (list, tuple, set)):
            out = []
            for v in x:
                try:
                    out.append(int(float(v)))
                except Exception:
                    pass
            return sorted(set(out))
        st = str(x).strip()
        if not st:
            return []
        for ch in "[](){}":
            st = st.replace(ch, " ")
        for ch in ",;|":
            st = st.replace(ch, " ")
        out = []
        for part in st.split():
            try:
                out.append(int(float(part)))
            except Exception:
                pass
        return sorted(set(out))

    mod.parse_ints = robust_parse_ints

    # Monkey-patch target_direction so all Phase19 baselines share the same
    # direction rule: low LogD parents -> increase_logd; high LogD parents -> decrease_logd.
    def robust_target_direction(target):
        return infer_replace_direction(target)

    mod.target_direction = robust_target_direction
    return mod


def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def parse_atom_indices(x) -> list[int]:
    """
    Robust parser for atom index fields.

    Supports:
    - "1,2,3"
    - "1 2 3"
    - "1;2;3"
    - "[1, 2, 3]"
    - Python lists/tuples/sets
    """
    if x is None:
        return []

    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            try:
                out.append(int(float(v)))
            except Exception:
                pass
        return sorted(set(out))

    s = str(x).strip()
    if not s:
        return []

    for ch in "[](){}":
        s = s.replace(ch, " ")

    for ch in ",;|":
        s = s.replace(ch, " ")

    out = []
    for part in s.split():
        try:
            out.append(int(float(part)))
        except Exception:
            pass

    return sorted(set(out))


def atom_indices_to_string(indices: list[int]) -> str:
    return " ".join(str(int(i)) for i in sorted(set(indices)))


def bool_value(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y", "pass"}



def infer_replace_direction(row) -> str:
    """
    Infer optimization direction for Phase19 baseline targets.

    Priority:
    1. Existing replace_direction if valid.
    2. Original/parent target region: low -> increase_logd, high -> decrease_logd.
    3. Original prediction value: <1.0 -> increase_logd, >3.4 -> decrease_logd.
    4. Otherwise none.
    """
    def get_value(keys):
        for k in keys:
            try:
                if k in row:
                    v = row.get(k)
                    if v is not None and not pd.isna(v) and str(v).strip() != "":
                        return v
            except Exception:
                pass
        return None

    direction = get_value(["replace_direction", "direction", "optimization_direction"])
    if direction is not None:
        d = str(direction).strip().lower()
        if d in {"increase_logd", "increase", "up", "low_to_range"}:
            return "increase_logd"
        if d in {"decrease_logd", "decrease", "down", "high_to_range"}:
            return "decrease_logd"

    region = get_value([
        "main_original_target_region",
        "original_target_region",
        "parent_target_region",
        "target_region",
        "region",
    ])
    if region is not None:
        r = str(region).strip().lower()
        if "low" in r:
            return "increase_logd"
        if "high" in r:
            return "decrease_logd"

    pred = get_value([
        "main_original_pred_mean",
        "main_parent_pred_mean",
        "parent_main_pred",
        "original_main_pred",
        "main_original_pred",
    ])
    try:
        x = float(pred)
        if x < 1.0:
            return "increase_logd"
        if x > 3.4:
            return "decrease_logd"
    except Exception:
        pass

    return "none"



def count_attachments(mol: Chem.Mol, atom_set: set[int]) -> int:
    n = 0
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if (a in atom_set and c not in atom_set) or (c in atom_set and a not in atom_set):
            n += 1
    return n


def external_neighbor_atoms(mol: Chem.Mol, atom_set: set[int]) -> list[int]:
    anchors = set()
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        if a in atom_set and c not in atom_set:
            anchors.add(c)
        elif c in atom_set and a not in atom_set:
            anchors.add(a)
    return sorted(anchors)


def generic_delete_counterfactual(original_smiles: str, atoms: list[int]) -> tuple[Optional[str], str]:
    mol = mol_from_smiles(original_smiles)
    if mol is None:
        return None, "invalid_original"
    if not atoms:
        return None, "empty_atom_indices"

    n = mol.GetNumAtoms()
    if any(i < 0 or i >= n for i in atoms):
        return None, "atom_index_out_of_range"
    if len(set(atoms)) >= n:
        return None, "delete_whole_molecule"

    rw = Chem.RWMol(mol)
    for idx in sorted(set(atoms), reverse=True):
        try:
            rw.RemoveAtom(int(idx))
        except Exception:
            return None, "remove_atom_failed"

    edited = rw.GetMol()
    smi = canonical_smiles(edited)
    if smi is None:
        return None, "sanitize_or_canonicalize_failed"

    emol = mol_from_smiles(smi)
    if emol is None:
        return None, "invalid_edited"
    if len(Chem.GetMolFrags(emol)) != 1:
        return None, "edited_not_single_fragment"

    original_can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if smi == original_can:
        return None, "unchanged_smiles"

    return smi, "ok"


def prepare_targets(df: pd.DataFrame, target_cap_per_parent: int) -> pd.DataFrame:
    df = df.copy()

    for col in ["mol_id", "canonical_smiles", "atom_indices", "baseline_method_group"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Explicitly assign replace_direction for every baseline target.
    # This prevents original Phase10-D target_direction() from silently returning "none".
    df["replace_direction"] = df.apply(infer_replace_direction, axis=1)

    if "baseline_attachment_count" in df.columns:
        df["attachment_count"] = pd.to_numeric(df["baseline_attachment_count"], errors="coerce")
    else:
        df["attachment_count"] = pd.to_numeric(df.get("attachment_count", np.nan), errors="coerce")

    if "baseline_atom_count" in df.columns:
        df["n_atoms"] = pd.to_numeric(df["baseline_atom_count"], errors="coerce")
    else:
        df["n_atoms"] = df["atom_indices"].map(lambda x: len(parse_atom_indices(x)))

    if "token_id" not in df.columns:
        df["token_id"] = (
            df["baseline_target_source"].astype(str)
            + "_"
            + df["mol_id"].astype(str)
            + "_"
            + df.groupby(["baseline_method_group", "mol_id"]).cumcount().add(1).astype(str)
        )

    if "baseline_token_rank" in df.columns:
        df["active_moltoken_rank"] = pd.to_numeric(df["baseline_token_rank"], errors="coerce")
    elif "active_moltoken_rank" not in df.columns:
        df["active_moltoken_rank"] = df.groupby(["baseline_method_group", "mol_id"]).cumcount() + 1

    # Fair budget: same maximum target count per method-parent.
    if target_cap_per_parent > 0:
        df = df.sort_values(
            ["baseline_method_group", "mol_id", "active_moltoken_rank", "n_atoms"],
            ascending=[True, True, True, True],
        )
        df = (
            df.groupby(["baseline_method_group", "mol_id"], group_keys=False)
            .head(target_cap_per_parent)
            .reset_index(drop=True)
        )


    # Normalize atom_indices to whitespace-separated format for original Phase10-D parse_ints.
    # The original functions fail on comma-separated strings such as "16,17,18".
    df["atom_indices"] = df["atom_indices"].map(lambda x: atom_indices_to_string(parse_atom_indices(x)))

    ext_neighbors = []
    delete_smiles = []
    delete_ok = []
    delete_reason = []
    attachment_values = []

    for _, row in df.iterrows():
        smi = str(row["canonical_smiles"])
        mol = mol_from_smiles(smi)
        atoms = parse_atom_indices(row["atom_indices"])
        aset = set(atoms)

        if mol is not None and atoms:
            ext_neighbors.append(atom_indices_to_string(external_neighbor_atoms(mol, aset)))
            if pd.isna(row.get("attachment_count", np.nan)):
                attachment_values.append(count_attachments(mol, aset))
            else:
                attachment_values.append(int(float(row.get("attachment_count"))))
        else:
            ext_neighbors.append("")
            attachment_values.append(0)

        existing_cf = row.get("counterfactual_smiles", None)
        existing_usable = bool_value(row.get("counterfactual_usable_for_scoring", False))

        if isinstance(existing_cf, str) and existing_cf.strip() and existing_usable:
            delete_smiles.append(existing_cf)
            delete_ok.append(True)
            delete_reason.append("original_counterfactual")
        else:
            smi_del, reason = generic_delete_counterfactual(smi, atoms)
            delete_smiles.append(smi_del)
            delete_ok.append(reason == "ok")
            delete_reason.append(reason)

    df["external_neighbor_atoms"] = ext_neighbors
    df["counterfactual_smiles"] = delete_smiles
    df["counterfactual_usable_for_scoring"] = delete_ok
    df["phase19_delete_reason"] = delete_reason
    df["attachment_count"] = pd.Series(attachment_values).fillna(0).astype(int)
    df["n_boundary_atoms"] = df["attachment_count"]

    return df



def repair_attachment_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute atom_indices, attachment_count, n_boundary_atoms and external_neighbor_atoms
    from canonical_smiles + atom_indices for every Phase19 target.

    This fixes cases where attachment_count was accidentally written as 0,
    which prevents the original Phase10-D replace interface from generating candidates.
    """
    df = df.copy()

    normalized_atoms = []
    attachment_values = []
    external_neighbors = []

    for _, row in df.iterrows():
        smi = str(row.get("canonical_smiles", ""))
        atoms = parse_atom_indices(row.get("atom_indices", ""))
        mol = mol_from_smiles(smi)

        normalized_atoms.append(atom_indices_to_string(atoms))

        if mol is not None and atoms:
            aset = set(atoms)
            attachment_values.append(int(count_attachments(mol, aset)))
            external_neighbors.append(atom_indices_to_string(external_neighbor_atoms(mol, aset)))
        else:
            attachment_values.append(0)
            external_neighbors.append("")

    df["atom_indices"] = normalized_atoms
    df["attachment_count"] = attachment_values
    df["n_boundary_atoms"] = attachment_values
    df["external_neighbor_atoms"] = external_neighbors

    return df



def override_candidate_labels(cand: pd.DataFrame, method_group: str, action: str) -> pd.DataFrame:
    if cand is None or len(cand) == 0:
        return pd.DataFrame()

    cand = cand.copy()
    cand["method_group"] = method_group
    cand["method"] = f"{method_group}_{action}"
    cand["edit_action"] = action
    cand["baseline_method_group"] = method_group
    cand["phase19_baseline"] = True
    return cand


def generate_candidates_for_method(
    phase10d,
    targets: pd.DataFrame,
    donors: pd.DataFrame,
    linkers: pd.DataFrame,
    method_group: str,
    max_reagents_per_token: int,
    max_add_per_parent: int,
) -> pd.DataFrame:
    sub = targets[targets["baseline_method_group"] == method_group].copy()
    if sub.empty:
        return pd.DataFrame()

    parts = []

    try:
        delete_df = phase10d.build_delete_candidates(sub)
        parts.append(override_candidate_labels(delete_df, method_group, "delete"))
    except Exception as e:
        print(f"[WARN] delete failed for {method_group}: {e}")

    try:
        replace_df = phase10d.build_replace_candidates(
            sub,
            donors,
            linkers,
            max_reagents_per_token=max_reagents_per_token,
            method=f"{method_group}_replace",
        )
        parts.append(override_candidate_labels(replace_df, method_group, "replace"))
    except Exception as e:
        print(f"[WARN] replace failed for {method_group}: {e}")

    try:
        add_df = phase10d.build_add_candidates(
            sub,
            donors,
            max_add_per_parent=max_add_per_parent,
        )
        parts.append(override_candidate_labels(add_df, method_group, "add"))
    except Exception as e:
        print(f"[WARN] add failed for {method_group}: {e}")

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True, sort=False)


def summarize_candidates(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    summary = {
        "candidate_attempts": int(len(df)),
        "method_group_counts": df["method_group"].value_counts(dropna=False).to_dict(),
        "method_counts": df["method"].value_counts(dropna=False).to_dict(),
        "edit_action_counts": df["edit_action"].value_counts(dropna=False).to_dict(),
    }

    if "edit_valid" in df.columns:
        summary["edit_valid_counts"] = df["edit_valid"].astype(str).value_counts(dropna=False).to_dict()

    by_group = []
    for m, g in df.groupby("method_group"):
        row = {
            "method_group": m,
            "candidate_attempts": int(len(g)),
            "parent_count": int(g["mol_id"].nunique()) if "mol_id" in g.columns else 0,
            "delete_attempts": int((g["edit_action"] == "delete").sum()) if "edit_action" in g.columns else 0,
            "replace_attempts": int((g["edit_action"] == "replace").sum()) if "edit_action" in g.columns else 0,
            "add_attempts": int((g["edit_action"] == "add").sum()) if "edit_action" in g.columns else 0,
        }
        if "edit_valid" in g.columns:
            row["valid_edits"] = int(g["edit_valid"].map(bool_value).sum())
            row["valid_rate"] = float(g["edit_valid"].map(bool_value).mean())
        by_group.append(row)

    summary["by_method_group"] = by_group
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-table", default=str(TABLES / "phase19_common_clean_targets_v2.csv"))
    parser.add_argument("--comparison-name", default="common_budget2")
    parser.add_argument("--target-cap-per-parent", type=int, default=2)
    parser.add_argument("--max-reagents-per-token", type=int, default=8)
    parser.add_argument("--max-add-per-parent", type=int, default=10)
    parser.add_argument("--max-external-reagents-per-role", type=int, default=15)
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    phase10d = import_phase10d()

    target_path = Path(args.target_table)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target table: {target_path}")

    raw_targets = pd.read_csv(target_path, low_memory=False)
    targets = prepare_targets(raw_targets, target_cap_per_parent=args.target_cap_per_parent)
    targets = repair_attachment_fields(targets)

    prepared_path = TABLES / f"phase19b_{args.comparison_name}_prepared_targets.csv"
    targets.to_csv(prepared_path, index=False)

    print("[INFO] prepared targets:", len(targets))
    print(targets.groupby("baseline_method_group")["mol_id"].nunique())
    print(targets.groupby("baseline_method_group").size())

    donors = phase10d.sidechain_tools.build_donor_library(max_brics_per_role=45)
    linkers = phase10d.linker_tools.build_linker_library(max_brics_per_role=45)
    donors, linkers = phase10d.augment_reagents_from_phase10_libraries(
        donors,
        linkers,
        max_external_per_role=args.max_external_reagents_per_role,
    )

    donor_path = TABLES / f"phase19b_{args.comparison_name}_donor_library.csv"
    linker_path = TABLES / f"phase19b_{args.comparison_name}_linker_library.csv"
    donors.to_csv(donor_path, index=False)
    linkers.to_csv(linker_path, index=False)

    all_parts = []
    for method_group in METHOD_ORDER:
        cand = generate_candidates_for_method(
            phase10d,
            targets,
            donors,
            linkers,
            method_group=method_group,
            max_reagents_per_token=args.max_reagents_per_token,
            max_add_per_parent=args.max_add_per_parent,
        )
        print(f"[INFO] {method_group} candidates:", len(cand))
        if len(cand):
            all_parts.append(cand)

    candidates = pd.concat(all_parts, ignore_index=True, sort=False) if all_parts else pd.DataFrame()

    candidate_path = TABLES / f"phase19b_{args.comparison_name}_candidates_all.csv"
    candidates.to_csv(candidate_path, index=False)

    summary = summarize_candidates(candidates)
    report = {
        "phase": "19B_candidate_generation_only",
        "status": "pass" if not candidates.empty else "failed",
        "comparison_name": args.comparison_name,
        "target_table": str(target_path),
        "target_cap_per_parent": args.target_cap_per_parent,
        "max_reagents_per_token": args.max_reagents_per_token,
        "max_add_per_parent": args.max_add_per_parent,
        "max_external_reagents_per_role": args.max_external_reagents_per_role,
        "prepared_targets": int(len(targets)),
        "donor_count": int(len(donors)),
        "linker_count": int(len(linkers)),
        **summary,
        "outputs": {
            "prepared_targets": str(prepared_path),
            "donor_library": str(donor_path),
            "linker_library": str(linker_path),
            "candidates": str(candidate_path),
        },
    }

    report_path = REPORTS / f"phase19b_{args.comparison_name}_candidate_generation_self_check.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Candidate summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[WROTE]", prepared_path)
    print("[WROTE]", candidate_path)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
