from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))

from moltoken_common import TargetWindow, ensure_dirs, morgan_features, rdkit_descriptor_frame


PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE02 = ROOT / "predictors"
SPLITS = PHASE01 / "splits"
MODELS = PHASE02 / "models"
PREDICTIONS = PHASE02 / "predictions"
REPORTS = PHASE02 / "reports"
FEATURES = PHASE02 / "features"

SEED = 20260528
TARGET = TargetWindow(low=1.0, high=3.4)


def load_split(name: str) -> pd.DataFrame:
    path = SPLITS / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    fp = morgan_features(df["canonical_smiles"], radius=2, n_bits=2048)
    fp_names = [f"morgan_{i}" for i in range(fp.shape[1])]
    desc = rdkit_descriptor_frame(df, smiles_col="canonical_smiles")
    X = np.hstack([fp, desc.to_numpy(dtype=np.float32)])
    names = fp_names + list(desc.columns)
    return X, names, desc


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    sp = spearmanr(y_true, y_pred)
    spearman = float(sp.statistic) if not np.isnan(sp.statistic) else float("nan")
    true_dist = TARGET.distance(y_true)
    pred_dist = TARGET.distance(y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": spearman,
        "mean_true_target_distance": float(np.mean(true_dist)),
        "mean_pred_target_distance": float(np.mean(pred_dist)),
    }


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True)),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 17))),
            ]
        ).fit(X_train, y_train),
        "elasticnet": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        alphas=np.logspace(-4, 1, 20),
                        max_iter=20000,
                        cv=5,
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ).fit(X_train, y_train),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=1,
        ).fit(X_train, y_train),
        "xgboost": XGBRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
        ).fit(X_train, y_train),
    }


def main() -> None:
    ensure_dirs(MODELS, PREDICTIONS, REPORTS, FEATURES)

    split_names = ["train", "valid", "test", "independent_100"]
    frames = {name: load_split(name) for name in split_names}
    X_by_split: dict[str, np.ndarray] = {}
    desc_by_split: dict[str, pd.DataFrame] = {}
    feature_names: list[str] | None = None

    for name, frame in frames.items():
        X, names, desc = build_features(frame)
        X_by_split[name] = X
        desc_by_split[name] = desc
        if feature_names is None:
            feature_names = names
        np.savez_compressed(FEATURES / f"{name}_features.npz", X=X)
        desc.to_csv(FEATURES / f"{name}_rdkit_descriptors.csv", index=False)

    assert feature_names is not None
    (FEATURES / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    y_train = frames["train"]["exp"].to_numpy(dtype=float)
    models = train_models(X_by_split["train"], y_train)

    all_metrics: list[dict[str, object]] = []
    for model_name, model in models.items():
        joblib.dump(model, MODELS / f"{model_name}.joblib")
        for split_name, frame in frames.items():
            y_true = frame["exp"].to_numpy(dtype=float)
            y_pred = np.asarray(model.predict(X_by_split[split_name]), dtype=float)
            m = metrics(y_true, y_pred)
            m.update({"model": model_name, "split": split_name, "n": int(len(frame))})
            all_metrics.append(m)
            pred_df = frame[["mol_id", "CMPD_CHEMBLID", "canonical_smiles", "exp", "target_region"]].copy()
            pred_df["pred"] = y_pred
            pred_df["true_target_distance"] = TARGET.distance(pred_df["exp"])
            pred_df["pred_target_distance"] = TARGET.distance(pred_df["pred"])
            pred_df.to_csv(PREDICTIONS / f"{model_name}_{split_name}_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(REPORTS / "descriptor_model_metrics.csv", index=False)

    ranking = (
        metrics_df[metrics_df["split"] == "valid"]
        .sort_values(["rmse", "mae"], ascending=[True, True])
        [["model", "rmse", "mae", "r2", "spearman"]]
        .reset_index(drop=True)
    )
    best_model = str(ranking.iloc[0]["model"])

    summary = {
        "seed": SEED,
        "target_window": {"low": TARGET.low, "high": TARGET.high},
        "n_features": len(feature_names),
        "split_sizes": {name: int(len(frame)) for name, frame in frames.items()},
        "best_valid_model": best_model,
        "validation_ranking": ranking.to_dict(orient="records"),
    }
    (REPORTS / "descriptor_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# 02 预测模型训练报告",
        "",
        "## 模型角色",
        "",
        "- Ridge / ElasticNet：用于带符号贡献校验。",
        "- Random Forest / XGBoost：用于性质复核和候选筛选。",
        "- Chemprop / D-MPNN：作为正式主预测器扩展，本阶段另存运行脚本。",
        "",
        "## 数据规模",
        "",
        "| split | count |",
        "|---|---:|",
    ]
    for name in split_names:
        report_lines.append(f"| {name} | {len(frames[name])} |")
    report_lines.extend(
        [
            "",
            "## 传统模型指标",
            "",
            "| model | split | RMSE | MAE | R2 | Spearman |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics_df.sort_values(["model", "split"]).itertuples(index=False):
        report_lines.append(
            f"| {row.model} | {row.split} | {row.rmse:.4f} | {row.mae:.4f} | {row.r2:.4f} | {row.spearman:.4f} |"
        )
    report_lines.extend(
        [
            "",
            f"验证集 RMSE 最低模型：`{best_model}`。",
            "",
            "说明：本阶段输出的传统模型主要用于 03 阶段的贡献方向校验和候选复核；正式主预测器建议继续使用 Chemprop / D-MPNN。",
        ]
    )
    (REPORTS / "predictor_training_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
