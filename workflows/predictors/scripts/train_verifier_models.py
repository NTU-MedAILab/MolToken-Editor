from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))

from moltoken_common import TargetWindow, ensure_dirs, morgan_features, rdkit_descriptor_frame


PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE04 = ROOT / "predictors"
MODELS = PHASE04 / "models" / "verifiers"
PREDICTIONS = PHASE04 / "predictions"
REPORTS = PHASE04 / "reports"
FEATURES = PHASE04 / "features"

TARGET = TargetWindow(low=1.0, high=3.4)
SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
REGION_TO_ID = {"low": 0, "in_range": 1, "high": 2}


def load_frame(split_type: str, split: str) -> pd.DataFrame:
    if split_type == "random":
        return pd.read_csv(PHASE01 / "splits" / f"{split}.csv")
    return pd.read_csv(PHASE04 / "splits" / f"scaffold_{split}.csv")


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    fp = morgan_features(df["canonical_smiles"], radius=2, n_bits=2048)
    desc = rdkit_descriptor_frame(df, smiles_col="canonical_smiles").to_numpy(dtype=np.float32)
    names = [f"morgan_{i}" for i in range(fp.shape[1])] + [
        "MolWt",
        "MolLogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "HeavyAtomCount",
        "RingCount",
        "AromaticRingCount",
        "FractionCSP3",
        "QED",
    ]
    return np.hstack([fp, desc]).astype(np.float32), names


def target_distance(values: np.ndarray) -> np.ndarray:
    return TARGET.distance(values)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_region = np.array([REGION_TO_ID[TARGET.region(value)] for value in y_true])
    pred_region = np.array([REGION_TO_ID[TARGET.region(value)] for value in y_pred])
    sp = spearmanr(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(sp.statistic) if not np.isnan(sp.statistic) else float("nan"),
        "mean_true_target_distance": float(np.mean(target_distance(y_true))),
        "mean_pred_target_distance": float(np.mean(target_distance(y_pred))),
        "target_distance_mae": float(mean_absolute_error(target_distance(y_true), target_distance(y_pred))),
        "region_balanced_accuracy": float(balanced_accuracy_score(true_region, pred_region)),
        "region_macro_f1": float(f1_score(true_region, pred_region, average="macro")),
    }


def train_models(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, object]:
    return {
        "xgboost": XGBRegressor(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
        ).fit(X, y),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        ).fit(X, y),
    }


def main() -> None:
    ensure_dirs(MODELS, PREDICTIONS, REPORTS, FEATURES)
    all_metrics: list[dict[str, object]] = []
    all_prediction_frames: list[pd.DataFrame] = []

    independent = pd.read_csv(PHASE01 / "splits" / "independent_100.csv")
    X_independent, feature_names = build_features(independent)

    for split_type in ["random", "scaffold"]:
        frames = {split: load_frame(split_type, split) for split in ["train", "valid", "test"]}
        X_by_split: dict[str, np.ndarray] = {}
        for split, frame in frames.items():
            X, names = build_features(frame)
            X_by_split[split] = X
            if names != feature_names:
                raise ValueError("feature name mismatch")
        (FEATURES / f"{split_type}_feature_names.json").write_text(
            json.dumps(feature_names, indent=2), encoding="utf-8"
        )

        y_train = frames["train"]["exp"].to_numpy(dtype=float)
        for seed in SEEDS:
            models = train_models(X_by_split["train"], y_train, seed)
            for model_name, model in models.items():
                out_dir = MODELS / split_type / f"seed_{seed}"
                ensure_dirs(out_dir)
                joblib.dump(model, out_dir / f"{model_name}.joblib")

                for split, frame in frames.items():
                    y_true = frame["exp"].to_numpy(dtype=float)
                    y_pred = np.asarray(model.predict(X_by_split[split]), dtype=float)
                    row = metric_row(y_true, y_pred)
                    row.update(
                        {
                            "split_type": split_type,
                            "split": split,
                            "seed": seed,
                            "model": model_name,
                            "n": int(len(frame)),
                        }
                    )
                    all_metrics.append(row)

                ind_pred = np.asarray(model.predict(X_independent), dtype=float)
                pred_df = independent[
                    ["mol_id", "CMPD_CHEMBLID", "canonical_smiles", "exp", "target_region", "target_distance_exp"]
                ].copy()
                pred_df["split_type"] = split_type
                pred_df["seed"] = seed
                pred_df["model"] = model_name
                pred_df["pred"] = ind_pred
                pred_df["pred_target_distance"] = target_distance(ind_pred)
                all_prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(REPORTS / "verifier_model_metrics.csv", index=False)
    pred_df = pd.concat(all_prediction_frames, ignore_index=True)
    pred_df.to_csv(PREDICTIONS / "independent_100_verifier_seed_predictions.csv", index=False)

    ensemble = (
        pred_df.groupby(["mol_id", "split_type", "model"], as_index=False)
        .agg(
            pred_mean=("pred", "mean"),
            pred_std=("pred", "std"),
            pred_min=("pred", "min"),
            pred_max=("pred", "max"),
            pred_target_distance_mean=("pred_target_distance", "mean"),
            exp=("exp", "first"),
            target_region=("target_region", "first"),
            target_distance_exp=("target_distance_exp", "first"),
            canonical_smiles=("canonical_smiles", "first"),
        )
    )
    ensemble["pred_std"] = ensemble["pred_std"].fillna(0.0)
    ensemble.to_csv(PREDICTIONS / "independent_100_verifier_ensemble.csv", index=False)

    summary = {
        "seeds": SEEDS,
        "split_types": ["random", "scaffold"],
        "models": ["xgboost", "random_forest"],
        "metrics_rows": int(len(metrics_df)),
        "independent_prediction_rows": int(len(pred_df)),
        "ensemble_rows": int(len(ensemble)),
        "best_valid": (
            metrics_df[metrics_df["split"] == "valid"]
            .sort_values(["rmse", "mae"])
            .head(10)
            .to_dict(orient="records")
        ),
    }
    (REPORTS / "verifier_model_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
