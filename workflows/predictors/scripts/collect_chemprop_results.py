from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))

from moltoken_common import TargetWindow


PYTHON = Path(sys.executable)
PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE04 = ROOT / "predictors"
CHEMPROP_CLI = PHASE04 / "scripts" / "chemprop_stable_cli.py"
MODELS = PHASE04 / "models" / "chemprop_main"
PREDICTIONS = PHASE04 / "predictions"
REPORTS = PHASE04 / "reports"
LOGS = REPORTS / "chemprop_predict_logs"

TARGET = TargetWindow(low=1.0, high=3.4)
REGION_TO_ID = {"low": 0, "in_range": 1, "high": 2}
DEFAULT_SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
VARIANT_ARGS = {
    "dmpnn_rdkit2d": ["--molecule-featurizers", "rdkit_2d"],
}


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


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


def chemprop_input_path(split_type: str, dataset: str) -> Path:
    if dataset == "independent_100":
        return PHASE01 / "splits" / "chemprop_independent_100.csv"
    if split_type == "random":
        return PHASE01 / "splits" / f"chemprop_{dataset}.csv"
    return PHASE04 / "splits" / f"chemprop_scaffold_{dataset}.csv"


def metadata_path(split_type: str, dataset: str) -> Path:
    if dataset == "independent_100":
        return PHASE01 / "splits" / "independent_100.csv"
    if split_type == "random":
        return PHASE01 / "splits" / f"{dataset}.csv"
    return PHASE04 / "splits" / f"scaffold_{dataset}.csv"


def run_predict(
    *,
    variant: str,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    log_path: Path,
    batch_size: int,
) -> dict[str, object]:
    cmd = [
        str(PYTHON),
        str(CHEMPROP_CLI),
        "predict",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-s",
        "smiles",
        "--model-paths",
        str(model_path),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "0",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        *VARIANT_ARGS[variant],
    ]
    started = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    finished = datetime.now().isoformat(timespec="seconds")
    log_path.write_text(
        "\n".join(
            [
                f"started: {started}",
                f"finished: {finished}",
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
    return {"returncode": result.returncode, "log_path": str(log_path.relative_to(ROOT))}


def read_prediction(path: Path) -> np.ndarray:
    pred_df = pd.read_csv(path)
    if "exp" in pred_df.columns:
        return pred_df["exp"].to_numpy(dtype=float)
    candidate_cols = [col for col in pred_df.columns if col.startswith("exp")]
    if not candidate_cols:
        raise ValueError(f"cannot find prediction column in {path}")
    return pred_df[candidate_cols[0]].to_numpy(dtype=float)


def fingerprints(smiles_values: pd.Series) -> list:
    fps = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol is not None else None)
    return fps


def nearest_tanimoto(query_smiles: pd.Series, train_smiles: pd.Series) -> list[float]:
    train_fps = [fp for fp in fingerprints(train_smiles) if fp is not None]
    scores: list[float] = []
    for fp in fingerprints(query_smiles):
        if fp is None or not train_fps:
            scores.append(float("nan"))
            continue
        scores.append(float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-types", nargs="+", default=["random", "scaffold"], choices=["random", "scaffold"])
    parser.add_argument("--variants", nargs="+", default=["dmpnn_rdkit2d"], choices=list(VARIANT_ARGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--datasets", nargs="+", default=["train", "valid", "test", "independent_100"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def add_verifier_consensus(ensemble: pd.DataFrame) -> pd.DataFrame:
    verifier_path = PREDICTIONS / "independent_100_verifier_ensemble.csv"
    if not verifier_path.exists() or ensemble.empty:
        return ensemble
    verifier = pd.read_csv(verifier_path)
    keep = verifier[["mol_id", "split_type", "model", "pred_mean", "pred_std"]].copy()
    wide = keep.pivot_table(
        index=["mol_id", "split_type"],
        columns="model",
        values=["pred_mean", "pred_std"],
        aggfunc="first",
    )
    wide.columns = [f"{model}_{metric}" for metric, model in wide.columns]
    wide = wide.reset_index()
    merged = ensemble.merge(wide, on=["mol_id", "split_type"], how="left")
    for model in ["xgboost", "random_forest"]:
        col = f"{model}_pred_mean"
        if col in merged.columns:
            merged[f"abs_disagreement_{model}"] = (merged["pred_mean"] - merged[col]).abs()
    return merged


def main() -> None:
    args = parse_args()
    ensure_dirs(PREDICTIONS, REPORTS, LOGS)
    metrics: list[dict[str, object]] = []
    independent_rows: list[pd.DataFrame] = []
    predict_runs: list[dict[str, object]] = []

    for variant in args.variants:
        for split_type in args.split_types:
            train_meta = pd.read_csv(metadata_path(split_type, "train"))
            ad_reference = train_meta["canonical_smiles"] if "canonical_smiles" in train_meta.columns else train_meta["smiles"]
            for seed in args.seeds:
                model_path = MODELS / variant / split_type / f"seed_{seed}" / "model_0" / "best.pt"
                if not model_path.exists():
                    continue
                for dataset in args.datasets:
                    input_path = chemprop_input_path(split_type, dataset)
                    meta_path = metadata_path(split_type, dataset)
                    pred_path = PREDICTIONS / "chemprop_main" / variant / split_type / f"seed_{seed}" / f"{dataset}.csv"
                    log_path = LOGS / f"{variant}_{split_type}_seed_{seed}_{dataset}.log"
                    ensure_dirs(pred_path.parent)
                    if not (args.skip_existing and pred_path.exists()):
                        result = run_predict(
                            variant=variant,
                            model_path=model_path,
                            input_path=input_path,
                            output_path=pred_path,
                            log_path=log_path,
                            batch_size=args.batch_size,
                        )
                        result.update(
                            {
                                "variant": variant,
                                "split_type": split_type,
                                "seed": seed,
                                "dataset": dataset,
                            }
                        )
                        predict_runs.append(result)
                        if result["returncode"] != 0:
                            continue

                    if not pred_path.exists():
                        continue
                    meta = pd.read_csv(meta_path)
                    y_true = meta["exp"].to_numpy(dtype=float)
                    y_pred = read_prediction(pred_path)
                    if len(y_true) != len(y_pred):
                        raise ValueError(f"length mismatch for {pred_path}")
                    row = metric_row(y_true, y_pred)
                    row.update(
                        {
                            "variant": variant,
                            "split_type": split_type,
                            "seed": seed,
                            "split": dataset,
                            "n": int(len(meta)),
                            "model_path": str(model_path.relative_to(ROOT)),
                        }
                    )
                    metrics.append(row)

                    if dataset == "independent_100":
                        cols = [
                            "mol_id",
                            "CMPD_CHEMBLID",
                            "canonical_smiles",
                            "exp",
                            "target_region",
                            "target_distance_exp",
                        ]
                        frame = meta[cols].copy()
                        frame["variant"] = variant
                        frame["split_type"] = split_type
                        frame["seed"] = seed
                        frame["pred"] = y_pred
                        frame["pred_target_distance"] = target_distance(y_pred)
                        frame["nearest_train_tanimoto"] = nearest_tanimoto(
                            frame["canonical_smiles"], ad_reference
                        )
                        independent_rows.append(frame)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(REPORTS / "chemprop_main_metrics.csv", index=False)

    if independent_rows:
        independent_df = pd.concat(independent_rows, ignore_index=True)
    else:
        independent_df = pd.DataFrame()
    independent_df.to_csv(PREDICTIONS / "independent_100_chemprop_seed_predictions.csv", index=False)

    if not independent_df.empty:
        ensemble = (
            independent_df.groupby(["mol_id", "variant", "split_type"], as_index=False)
            .agg(
                pred_mean=("pred", "mean"),
                pred_std=("pred", "std"),
                pred_min=("pred", "min"),
                pred_max=("pred", "max"),
                pred_target_distance_mean=("pred_target_distance", "mean"),
                nearest_train_tanimoto=("nearest_train_tanimoto", "first"),
                exp=("exp", "first"),
                target_region=("target_region", "first"),
                target_distance_exp=("target_distance_exp", "first"),
                canonical_smiles=("canonical_smiles", "first"),
            )
        )
        ensemble["pred_std"] = ensemble["pred_std"].fillna(0.0)
        ensemble["pred_target_distance_mean"] = TARGET.distance(ensemble["pred_mean"].to_numpy(dtype=float))
        ensemble = add_verifier_consensus(ensemble)
    else:
        ensemble = pd.DataFrame()
    ensemble.to_csv(PREDICTIONS / "independent_100_chemprop_ensemble.csv", index=False)
    ensemble.to_csv(PREDICTIONS / "independent_100_predictor_consensus.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_rows": int(len(metrics_df)),
        "independent_prediction_rows": int(len(independent_df)),
        "ensemble_rows": int(len(ensemble)),
        "completed_predict_runs": int(sum(1 for row in predict_runs if row.get("returncode") == 0)),
        "failed_predict_runs": int(sum(1 for row in predict_runs if row.get("returncode") != 0)),
        "best_valid": (
            metrics_df[metrics_df["split"] == "valid"]
            .sort_values(["rmse", "mae"])
            .head(12)
            .to_dict(orient="records")
            if not metrics_df.empty
            else []
        ),
        "test_rows": (
            metrics_df[metrics_df["split"] == "test"]
            .sort_values(["variant", "split_type", "seed"])
            .to_dict(orient="records")
            if not metrics_df.empty
            else []
        ),
        "predict_runs": predict_runs,
    }
    (REPORTS / "chemprop_main_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "predict_runs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
