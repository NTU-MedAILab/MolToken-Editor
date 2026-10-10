from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))

from moltoken_common import TargetWindow, ensure_dirs, morgan_features, rdkit_descriptor_frame


rdBase.DisableLog("rdApp.warning")

PYTHON = Path(sys.executable)
PHASE03 = ROOT / "moltoken"
PHASE04 = ROOT / "predictors"
PHASE05 = ROOT / "moltoken"
CHEMPROP_CLI = PHASE04 / "scripts" / "chemprop_stable_cli.py"
CONFIG_PATH = PHASE05 / "configs" / "quality_scoring.json"
TABLES = PHASE05 / "tables"
REPORTS = PHASE05 / "reports"
PREDICTIONS = PHASE05 / "tables" / "counterfactual_predictions"

SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
SPLIT_TYPES = ["random", "scaffold"]
TARGET = TargetWindow(low=1.0, high=3.4)


def parse_ints(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split(";") if part != ""]


def scalar_distance(value: float) -> float:
    return float(TARGET.distance([value])[0])


def region(value: float) -> str:
    return TARGET.region(float(value))


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if np.isnan(value):
        return low
    return float(max(low, min(high, value)))


def canonical_smiles(mol: Chem.Mol) -> str | None:
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def remove_atoms_counterfactual(smiles: str, token_atoms: list[int]) -> tuple[str | None, dict[str, object]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {"counterfactual_valid": False, "counterfactual_reason": "invalid_original"}
    atom_count = mol.GetNumAtoms()
    token_set = set(token_atoms)
    if not token_set or token_set >= set(range(atom_count)):
        return None, {"counterfactual_valid": False, "counterfactual_reason": "empty_or_whole_molecule"}

    external_neighbors = sorted(
        {
            nbr.GetIdx()
            for atom_idx in token_set
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if nbr.GetIdx() not in token_set
        }
    )
    attachment_count = len(external_neighbors)
    if attachment_count > 3:
        return None, {
            "counterfactual_valid": False,
            "counterfactual_reason": "too_many_attachment_points",
            "attachment_count": attachment_count,
        }

    remaining = [idx for idx in range(atom_count) if idx not in token_set]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining)}

    def build_without_token(add_bridge: bool) -> Chem.Mol | None:
        rw = Chem.RWMol(mol)
        for atom_idx in sorted(token_set, reverse=True):
            rw.RemoveAtom(atom_idx)
        if add_bridge and attachment_count == 2:
            a, b = [old_to_new[idx] for idx in external_neighbors if idx in old_to_new]
            if a != b and rw.GetBondBetweenAtoms(a, b) is None:
                rw.AddBond(a, b, Chem.BondType.SINGLE)
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return None

    method = "delete"
    out_mol = None
    if attachment_count == 2:
        out_mol = build_without_token(add_bridge=True)
        method = "delete_and_bridge" if out_mol is not None else "delete"
    if out_mol is None:
        out_mol = build_without_token(add_bridge=False)
    if out_mol is None:
        return None, {
            "counterfactual_valid": False,
            "counterfactual_reason": "sanitize_failed",
            "attachment_count": attachment_count,
        }

    cf_smiles = canonical_smiles(out_mol)
    if cf_smiles is None:
        return None, {
            "counterfactual_valid": False,
            "counterfactual_reason": "canonicalize_failed",
            "attachment_count": attachment_count,
        }
    n_frags = len(Chem.GetMolFrags(out_mol))
    heavy_atoms = int(out_mol.GetNumHeavyAtoms())
    usable = bool(n_frags == 1 and heavy_atoms >= 3)
    return cf_smiles, {
        "counterfactual_valid": True,
        "counterfactual_usable_for_scoring": usable,
        "counterfactual_reason": "ok",
        "counterfactual_method": method,
        "attachment_count": attachment_count,
        "counterfactual_fragments": int(n_frags),
        "counterfactual_single_fragment": bool(n_frags == 1),
        "counterfactual_heavy_atoms": heavy_atoms,
    }


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


def run_chemprop_counterfactual_predictions(
    input_csv: Path, force: bool = False, output_prefix: str = "counterfactual_main"
) -> None:
    ensure_dirs(PREDICTIONS)
    for split_type in SPLIT_TYPES:
        output = PREDICTIONS / f"{output_prefix}_{split_type}.csv"
        if output.exists() and not force:
            continue
        cmd = [
            str(PYTHON),
            str(CHEMPROP_CLI),
            "predict",
            "-i",
            str(Path(input_csv).resolve()),
            "-o",
            str(Path(output).resolve()),
            "-s",
            "smiles",
            "--model-paths",
            *[str(Path(path).resolve()) for path in model_paths(split_type)],
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
            cwd=Path(ROOT).resolve(),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        (REPORTS / f"{output_prefix}_{split_type}.log").write_text(
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
            raise RuntimeError(f"Chemprop prediction failed for {split_type}")


def read_main_counterfactual_predictions(
    valid_counterfactuals: pd.DataFrame, output_prefix: str = "counterfactual_main"
) -> pd.DataFrame:
    rows = valid_counterfactuals[["cf_row_id", "mol_id", "token_id"]].copy()
    pred_cols: list[pd.Series] = []
    for split_type in SPLIT_TYPES:
        pred_path = PREDICTIONS / f"{output_prefix}_{split_type}.csv"
        pred_df = pd.read_csv(pred_path)
        if "exp" not in pred_df.columns:
            raise ValueError(f"missing exp prediction column in {pred_path}")
        values = pred_df["exp"].to_numpy(dtype=float)
        if len(values) != len(rows):
            raise ValueError(
                "Prediction-row mismatch for "
                f"{pred_path}: expected {len(rows)}, "
                f"found {len(values)}. "
                "Regenerate this prediction set with "
                "--force-predict."
            )
        rows[f"main_counterfactual_pred_{split_type}"] = values
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
    if seed_std_cols:
        rows["main_counterfactual_seed_std"] = rows[seed_std_cols].mean(axis=1)
    else:
        rows["main_counterfactual_seed_std"] = rows["main_counterfactual_pred_std_across_splits"]
    return rows


def aggregate_main_original() -> pd.DataFrame:
    consensus = pd.read_csv(PHASE04 / "predictions" / "independent_100_predictor_consensus.csv")
    consensus = consensus[consensus["variant"] == "dmpnn_rdkit2d"].copy()
    grouped = (
        consensus.groupby("mol_id", as_index=False)
        .agg(
            main_original_pred_mean=("pred_mean", "mean"),
            main_original_pred_std=("pred_std", "mean"),
            main_original_pred_split_std=("pred_mean", "std"),
            exp=("exp", "first"),
            target_region_exp=("target_region", "first"),
            target_distance_exp=("target_distance_exp", "first"),
            nearest_train_tanimoto_main=("nearest_train_tanimoto", "max"),
        )
    )
    grouped["main_original_pred_split_std"] = grouped["main_original_pred_split_std"].fillna(0.0)
    grouped["risk_side_05"] = grouped["main_original_pred_mean"].map(region)
    grouped["main_original_target_distance"] = TARGET.distance(grouped["main_original_pred_mean"].to_numpy(dtype=float))
    return grouped


def aggregate_verifier_original() -> pd.DataFrame:
    verifier = pd.read_csv(PHASE04 / "predictions" / "independent_100_verifier_ensemble.csv")
    pivot = verifier.pivot_table(
        index="mol_id",
        columns="model",
        values="pred_mean",
        aggfunc="mean",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "xgboost": "xgboost_original_pred_mean",
            "random_forest": "rf_original_pred_mean",
        }
    )
    for col in ["xgboost_original_pred_mean", "rf_original_pred_mean"]:
        if col in pivot.columns:
            pivot[col.replace("_pred_mean", "_target_distance")] = TARGET.distance(pivot[col].to_numpy(dtype=float))
    return pivot


def build_features_for_smiles(smiles_values: pd.Series) -> np.ndarray:
    temp = pd.DataFrame({"canonical_smiles": smiles_values})
    fp = morgan_features(temp["canonical_smiles"], radius=2, n_bits=2048)
    desc = rdkit_descriptor_frame(temp, smiles_col="canonical_smiles").to_numpy(dtype=np.float32)
    return np.hstack([fp, desc]).astype(np.float32)


def predict_verifier_counterfactual(valid_counterfactuals: pd.DataFrame) -> pd.DataFrame:
    rows = valid_counterfactuals[["cf_row_id", "mol_id", "token_id"]].copy()
    X = build_features_for_smiles(valid_counterfactuals["counterfactual_smiles"])
    for split_type in SPLIT_TYPES:
        for model_key, out_prefix in [("xgboost", "xgboost"), ("random_forest", "rf")]:
            preds = []
            for seed in SEEDS:
                path = PHASE04 / "models" / "verifiers" / split_type / f"seed_{seed}" / f"{model_key}.joblib"
                model = joblib.load(path)
                preds.append(np.asarray(model.predict(X), dtype=float))
            arr = np.vstack(preds)
            rows[f"{out_prefix}_counterfactual_pred_{split_type}"] = arr.mean(axis=0)
            rows[f"{out_prefix}_counterfactual_seed_std_{split_type}"] = arr.std(axis=0)
    for out_prefix in ["xgboost", "rf"]:
        cols = [f"{out_prefix}_counterfactual_pred_{split_type}" for split_type in SPLIT_TYPES]
        rows[f"{out_prefix}_counterfactual_pred_mean"] = rows[cols].mean(axis=1)
        std_cols = [f"{out_prefix}_counterfactual_seed_std_{split_type}" for split_type in SPLIT_TYPES]
        rows[f"{out_prefix}_counterfactual_seed_std"] = rows[std_cols].mean(axis=1)
    return rows


def score_rows(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = config["score_weights"]
    penalties = config["penalty_weights"]

    main_cf_pred = df["main_counterfactual_pred_mean"].to_numpy(dtype=float)
    df["main_counterfactual_target_distance"] = TARGET.distance(main_cf_pred)
    df.loc[np.isnan(main_cf_pred), "main_counterfactual_target_distance"] = np.nan
    df["main_delta_distance"] = df["main_original_target_distance"] - df["main_counterfactual_target_distance"]

    for prefix in ["xgboost", "rf"]:
        original = f"{prefix}_original_pred_mean"
        counter = f"{prefix}_counterfactual_pred_mean"
        if original in df.columns and counter in df.columns:
            df[f"{prefix}_original_target_distance"] = TARGET.distance(df[original].to_numpy(dtype=float))
            counter_pred = df[counter].to_numpy(dtype=float)
            df[f"{prefix}_counterfactual_target_distance"] = TARGET.distance(counter_pred)
            df.loc[np.isnan(counter_pred), f"{prefix}_counterfactual_target_distance"] = np.nan
            df[f"{prefix}_delta_distance"] = df[f"{prefix}_original_target_distance"] - df[f"{prefix}_counterfactual_target_distance"]

    min_delta = 0.02
    support_cols = []
    for col in ["main_delta_distance", "xgboost_delta_distance", "rf_delta_distance"]:
        support_col = col.replace("_delta_distance", "_supports_harm")
        df[support_col] = df[col].fillna(-999) > min_delta
        support_cols.append(support_col)
    df["morgan_supports_harm"] = df["mean_combined_harm_score"].fillna(0.0) > 0.03
    support_cols.append("morgan_supports_harm")
    df["direction_agreement_score"] = df[support_cols].mean(axis=1).astype(float)

    df["contribution_strength"] = df["main_delta_distance"].fillna(0.0).clip(lower=0.0, upper=1.0)
    original_unc = df["main_original_pred_std"].fillna(0.5) + df["main_original_pred_split_std"].fillna(0.0)
    cf_unc = df["main_counterfactual_seed_std"].fillna(0.5) + df["main_counterfactual_pred_std_across_splits"].fillna(0.0)
    df["seed_stability_score"] = (1.0 - ((original_unc + cf_unc) / 1.0)).clip(lower=0.0, upper=1.0)

    boundary = df["n_boundary_atoms"].fillna(99).astype(float)
    df["boundary_clarity_score"] = np.where(
        boundary <= 2,
        1.0,
        np.where(boundary == 3, 0.65, np.maximum(0.0, 0.65 - 0.2 * (boundary - 3))),
    )
    n_atoms = df["n_atoms"].fillna(999).astype(float)
    df["specificity_score"] = np.where(
        (n_atoms >= 3) & (n_atoms <= 22),
        1.0,
        np.where((n_atoms >= 2) & (n_atoms <= 25), 0.7, np.maximum(0.15, 1.0 - np.abs(n_atoms - 16.0) / 30.0)),
    )
    cf_valid = df["counterfactual_usable_for_scoring"].fillna(False).astype(bool)
    cuts_ring = df["cuts_ring"].fillna(False).astype(bool)
    df["editability_score"] = np.where(
        (~cuts_ring) & cf_valid & (boundary <= 2),
        1.0,
        np.where((~cuts_ring) & cf_valid & (boundary == 3), 0.6, np.where((~cuts_ring) & (boundary <= 3), 0.35, 0.0)),
    )

    ambiguity = df["ambiguous_bit_fraction"].fillna(1.0).astype(float)
    df["ambiguity_penalty"] = (ambiguity / 0.5).clip(lower=0.0, upper=1.0)
    df["disagreement_penalty"] = 1.0 - df["direction_agreement_score"]
    df["counterfactual_penalty"] = np.where(cf_valid, 0.0, 1.0)
    df["size_penalty"] = np.where((n_atoms > 25) | (n_atoms < 2), 1.0, 0.0)
    df["boundary_penalty"] = ((boundary - 3.0) / 4.0).clip(lower=0.0, upper=1.0)

    score = np.zeros(len(df), dtype=float)
    for name, weight in weights.items():
        score += weight * df[name].fillna(0.0).to_numpy(dtype=float)
    for name, weight in penalties.items():
        score -= weight * df[name].fillna(0.0).to_numpy(dtype=float)
    df["split_quality_score"] = np.clip(score, 0.0, 1.0)

    thresholds = config["grade_thresholds"]
    df["split_quality_grade"] = np.where(
        df["split_quality_score"] >= thresholds["A"],
        "A",
        np.where(
            df["split_quality_score"] >= thresholds["B"],
            "B",
            np.where(df["split_quality_score"] >= thresholds["C"], "C", "D"),
        ),
    )

    df["predictor_support_level"] = np.where(
        df[["main_supports_harm", "xgboost_supports_harm", "rf_supports_harm"]].sum(axis=1) == 3,
        "strong",
        np.where(
            df[["main_supports_harm", "xgboost_supports_harm", "rf_supports_harm"]].sum(axis=1) >= 2,
            "medium",
            np.where(df["main_supports_harm"], "weak_main_only", "weak_or_none"),
        ),
    )
    return df


def assign_status(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    hard = config["hard_filters"]
    statuses = []
    reasons = []
    token_roles = []
    for row in df.to_dict(orient="records"):
        risk_side = row.get("risk_side_05")
        score = float(row.get("split_quality_score", 0.0))
        n_atoms = float(row.get("n_atoms", 999))
        boundary = float(row.get("n_boundary_atoms", 999))
        ambiguity = float(row.get("ambiguous_bit_fraction", 1.0))
        cf_valid = bool(row.get("counterfactual_usable_for_scoring", False))
        cuts_ring = bool(row.get("cuts_ring", False))
        support = row.get("predictor_support_level", "weak_or_none")

        if cuts_ring:
            statuses.append("drop")
            reasons.append("cuts_ring")
            token_roles.append("none")
        elif risk_side == "in_range":
            if score >= 0.45 and boundary <= hard["max_boundary_atoms_for_auto_target"]:
                statuses.append("retain_donor")
                reasons.append("main_predictor_in_target_window")
                token_roles.append("donor")
            else:
                statuses.append("manual_review")
                reasons.append("in_range_but_low_quality")
                token_roles.append("donor_candidate")
        elif risk_side not in {"high", "low"}:
            statuses.append("manual_review")
            reasons.append("uncertain_risk_side")
            token_roles.append("target_candidate")
        elif n_atoms > hard["max_token_atoms_for_auto_target"]:
            statuses.append("split_required")
            reasons.append("token_too_large")
            token_roles.append("target_candidate")
        elif boundary > hard["max_boundary_atoms_for_auto_target"]:
            statuses.append("manual_review")
            reasons.append("boundary_too_complex")
            token_roles.append("target_candidate")
        elif ambiguity > hard["max_ambiguous_bit_fraction_for_auto_target"]:
            statuses.append("manual_review")
            reasons.append("high_attribution_ambiguity")
            token_roles.append("target_candidate")
        elif not cf_valid:
            statuses.append("manual_review")
            reasons.append("invalid_counterfactual")
            token_roles.append("target_candidate")
        elif score >= 0.60 and support in {"strong", "medium", "weak_main_only"} and row.get("main_delta_distance", 0.0) > 0:
            statuses.append("retain_target")
            reasons.append("quality_score_and_main_support")
            token_roles.append("target")
        elif score >= 0.45:
            statuses.append("manual_review")
            reasons.append("moderate_quality")
            token_roles.append("target_candidate")
        else:
            statuses.append("drop")
            reasons.append("low_split_quality")
            token_roles.append("none")
    df["formal_token_status"] = statuses
    df["review_reason"] = reasons
    df["formal_token_role"] = token_roles
    return df


def build_counterfactual_table(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for idx, row in candidates.reset_index(drop=True).iterrows():
        atoms = parse_ints(row["atom_indices"])
        cf_smiles, meta = remove_atoms_counterfactual(row["canonical_smiles"], atoms)
        out = row.to_dict()
        out["cf_row_id"] = idx
        out["counterfactual_smiles"] = cf_smiles
        out.update(meta)
        rows.append(out)
    cf = pd.DataFrame(rows)
    valid = cf[(cf["counterfactual_valid"] == True) & cf["counterfactual_smiles"].notna()].copy()
    return cf, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--skip-chemprop-predict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(TABLES, REPORTS, PREDICTIONS)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    candidates = pd.read_csv(PHASE03 / "tables" / "pre_moltoken_regions.csv")
    candidates = candidates[candidates["is_candidate_moltoken"] == True].copy()
    candidates["candidate_row_id"] = np.arange(len(candidates))

    counterfactuals, valid_counterfactuals = build_counterfactual_table(candidates)
    counterfactuals.to_csv(TABLES / "counterfactual_moltoken_smiles.csv", index=False)
    chemprop_input = PREDICTIONS / "counterfactual_chemprop_input.csv"
    valid_counterfactuals = valid_counterfactuals[valid_counterfactuals["counterfactual_usable_for_scoring"] == True].copy()
    valid_counterfactuals[["counterfactual_smiles"]].rename(columns={"counterfactual_smiles": "smiles"}).to_csv(
        chemprop_input, index=False
    )

    if not args.skip_chemprop_predict and len(valid_counterfactuals) > 0:
        run_chemprop_counterfactual_predictions(chemprop_input, force=args.force_predict)

    main_original = aggregate_main_original()
    verifier_original = aggregate_verifier_original()
    main_cf = read_main_counterfactual_predictions(valid_counterfactuals) if len(valid_counterfactuals) else pd.DataFrame()
    verifier_cf = predict_verifier_counterfactual(valid_counterfactuals) if len(valid_counterfactuals) else pd.DataFrame()

    quality = counterfactuals.merge(main_original, on="mol_id", how="left")
    quality = quality.merge(verifier_original, on="mol_id", how="left")
    if not main_cf.empty:
        quality = quality.merge(main_cf.drop(columns=["mol_id", "token_id"]), on="cf_row_id", how="left")
    if not verifier_cf.empty:
        quality = quality.merge(verifier_cf.drop(columns=["mol_id", "token_id"]), on="cf_row_id", how="left")

    quality = score_rows(quality, config)
    quality = assign_status(quality, config)

    quality.to_csv(TABLES / "contribution_split_quality_table.csv", index=False)
    review_cols = [
        "mol_id",
        "token_id",
        "canonical_smiles",
        "atom_indices",
        "boundary_atoms",
        "external_neighbor_atoms",
        "risk_side",
        "risk_side_05",
        "n_atoms",
        "n_boundary_atoms",
        "ambiguous_bit_fraction",
        "token_confidence",
        "main_original_pred_mean",
        "main_counterfactual_pred_mean",
        "main_delta_distance",
        "direction_agreement_score",
        "split_quality_score",
        "split_quality_grade",
        "predictor_support_level",
        "formal_token_status",
        "formal_token_role",
        "review_reason",
    ]
    quality[review_cols].to_csv(TABLES / "formal_moltoken_review_sheet.csv", index=False)
    formal = quality[quality["formal_token_status"].isin(["retain_target", "retain_donor"])].copy()
    formal.to_csv(TABLES / "formal_moltoken_table.csv", index=False)

    summary = {
        "candidate_moltokens": int(len(candidates)),
        "counterfactual_valid": int((counterfactuals["counterfactual_valid"] == True).sum()),
        "quality_rows": int(len(quality)),
        "formal_rows": int(len(formal)),
        "status_counts": quality["formal_token_status"].value_counts().to_dict(),
        "role_counts": quality["formal_token_role"].value_counts().to_dict(),
        "grade_counts": quality["split_quality_grade"].value_counts().to_dict(),
        "risk_side_05_counts": quality["risk_side_05"].value_counts().to_dict(),
        "mean_split_quality_score": float(quality["split_quality_score"].mean()),
        "mean_main_delta_distance": float(quality["main_delta_distance"].mean()),
    }
    (REPORTS / "contribution_split_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
