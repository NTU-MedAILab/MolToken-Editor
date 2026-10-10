from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))

from moltoken_common import TargetWindow, ensure_dirs, morgan_features, rdkit_descriptor_frame


rdBase.DisableLog("rdApp.warning")

PYTHON = Path(sys.executable)
PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE04 = ROOT / "predictors"
PHASE10 = ROOT / "moltoken"
CHEMPROP_CLI = PHASE04 / "scripts" / "chemprop_stable_cli.py"

CONFIG_PATH = PHASE10 / "configs" / "target_expansion.json"
CANDIDATE_PATH = PHASE10 / "tables" / "phase10_parent_screening_candidates.csv"
PREDICTION_INPUT_PATH = PHASE10 / "tables" / "phase10_parent_predictor_input.csv"
PREDICTION_DIR = PHASE10 / "tables" / "predictions" / "main_predictor"
LOG_DIR = PHASE10 / "reports" / "chemprop_predict_logs"

SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
SPLIT_TYPES = ["random", "scaffold"]
VARIANT_ARGS = {"dmpnn_rdkit2d": ["--molecule-featurizers", "rdkit_2d"]}
REGION_TO_SIDE = {"low": -1, "in_range": 0, "high": 1}


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def clean_number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number) or np.isinf(number):
        return default
    return number


def model_path(split_type: str, seed: int) -> Path:
    return (
        PHASE04
        / "models"
        / "chemprop_main"
        / "dmpnn_rdkit2d"
        / split_type
        / f"seed_{seed}"
        / "model_0"
        / "best.pt"
    )


def run_chemprop_predict(
    *,
    split_type: str,
    seed: int,
    input_path: Path,
    output_path: Path,
    batch_size: int,
    force: bool,
) -> dict:
    if output_path.exists() and not force:
        return {"returncode": 0, "status": "cached", "output_path": rel(output_path)}

    ensure_dirs(output_path.parent, LOG_DIR)
    log_path = LOG_DIR / f"phase10_parent_{split_type}_seed_{seed}.log"
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
        str(model_path(split_type, seed)),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "0",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        *VARIANT_ARGS["dmpnn_rdkit2d"],
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
    if result.returncode != 0:
        raise RuntimeError(f"chemprop prediction failed: {split_type} seed {seed}; see {log_path}")
    return {"returncode": 0, "status": "run", "output_path": rel(output_path), "log_path": rel(log_path)}


def read_prediction_values(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "exp" in df.columns:
        return df["exp"].to_numpy(dtype=float)
    exp_cols = [col for col in df.columns if col.startswith("exp")]
    if not exp_cols:
        raise ValueError(f"Cannot find prediction column in {path}")
    return df[exp_cols[0]].to_numpy(dtype=float)


def prepare_parent_library_map(config: dict) -> pd.DataFrame:
    source = config["source_libraries"]
    membership_path = ROOT / source["mini_library_membership"]
    mapping_long = pd.read_csv(membership_path)
    keep_cols = [
        col
        for col in ["library_id", "external_molecule_id", "molecule_chembl_id", "standardized_smiles", "inchikey"]
        if col in mapping_long.columns
    ]
    mapping_long = mapping_long[keep_cols].copy()
    mapping_long = mapping_long.dropna(subset=["external_molecule_id", "library_id"]).copy()
    mapping_long["library_id"] = mapping_long["library_id"].astype(str)
    mapping_long["external_molecule_id"] = mapping_long["external_molecule_id"].astype(str)

    grouped = (
        mapping_long.groupby("external_molecule_id", as_index=False)
        .agg(
            primary_library_id=("library_id", lambda values: sorted(set(values))[0]),
            library_memberships=("library_id", lambda values: ";".join(sorted(set(values)))),
            library_membership_count=("library_id", lambda values: len(set(values))),
        )
        .sort_values("external_molecule_id")
    )
    out_path = PHASE10 / "tables" / "phase10_parent_library_map.csv"
    grouped.to_csv(out_path, index=False, encoding="utf-8-sig")
    return grouped


def build_prediction_input(candidates: pd.DataFrame) -> None:
    input_df = pd.DataFrame({"smiles": candidates["phase10_parent_smiles"].astype(str)})
    input_df.to_csv(PREDICTION_INPUT_PATH, index=False, encoding="utf-8")


def aggregate_main_predictions(candidates: pd.DataFrame, target: TargetWindow, config: dict, args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict]]:
    build_prediction_input(candidates)
    logs: list[dict] = []
    seed_table = candidates[["external_molecule_id", "phase10_parent_smiles"]].copy()

    for split_type in SPLIT_TYPES:
        for seed in SEEDS:
            out_path = PREDICTION_DIR / split_type / f"seed_{seed}.csv"
            logs.append(
                run_chemprop_predict(
                    split_type=split_type,
                    seed=seed,
                    input_path=PREDICTION_INPUT_PATH,
                    output_path=out_path,
                    batch_size=args.batch_size,
                    force=args.force_predict,
                )
            )
            values = read_prediction_values(out_path)
            if len(values) != len(candidates):
                raise ValueError(f"Prediction length mismatch for {out_path}: {len(values)} != {len(candidates)}")
            seed_table[f"main_pred_{split_type}_seed_{seed}"] = values

    for split_type in SPLIT_TYPES:
        cols = [f"main_pred_{split_type}_seed_{seed}" for seed in SEEDS]
        seed_table[f"main_pred_{split_type}"] = seed_table[cols].mean(axis=1)
        seed_table[f"main_seed_std_{split_type}"] = seed_table[cols].std(axis=1, ddof=0)

    split_mean_cols = [f"main_pred_{split_type}" for split_type in SPLIT_TYPES]
    seed_std_cols = [f"main_seed_std_{split_type}" for split_type in SPLIT_TYPES]
    seed_table["main_pred_mean"] = seed_table[split_mean_cols].mean(axis=1)
    seed_table["main_split_std"] = seed_table[split_mean_cols].std(axis=1, ddof=0)
    seed_table["main_seed_std"] = seed_table[seed_std_cols].mean(axis=1)
    seed_table["main_uncertainty"] = np.sqrt(seed_table["main_split_std"] ** 2 + seed_table["main_seed_std"] ** 2)
    seed_table["main_uncertainty_sum"] = seed_table["main_split_std"] + seed_table["main_seed_std"]
    seed_table["main_target_region"] = seed_table["main_pred_mean"].apply(target.region)
    seed_table["main_target_distance"] = target.distance(seed_table["main_pred_mean"].to_numpy(dtype=float))

    seed_out = PHASE10 / "tables" / "phase10_parent_main_predictor_seed_predictions.csv"
    seed_table.to_csv(seed_out, index=False, encoding="utf-8-sig")

    consensus = candidates.merge(seed_table.drop(columns=["phase10_parent_smiles"]), on="external_molecule_id", how="left")
    return consensus, logs


def build_features(smiles_values: pd.Series) -> np.ndarray:
    temp = pd.DataFrame({"canonical_smiles": smiles_values.astype(str)})
    fp = morgan_features(temp["canonical_smiles"], radius=2, n_bits=2048)
    desc = rdkit_descriptor_frame(temp, smiles_col="canonical_smiles").to_numpy(dtype=np.float32)
    return np.hstack([fp, desc]).astype(np.float32)


def aggregate_auxiliary_predictions(candidates: pd.DataFrame, target: TargetWindow) -> pd.DataFrame:
    X = build_features(candidates["phase10_parent_smiles"])
    aux = candidates[["external_molecule_id", "phase10_parent_smiles"]].copy()
    model_specs = [("xgboost", "xgboost"), ("random_forest", "rf")]
    for model_key, prefix in model_specs:
        for split_type in SPLIT_TYPES:
            seed_preds = []
            for seed in SEEDS:
                path = PHASE04 / "models" / "verifiers" / split_type / f"seed_{seed}" / f"{model_key}.joblib"
                model = joblib.load(path)
                pred = np.asarray(model.predict(X), dtype=float)
                seed_preds.append(pred)
                aux[f"{prefix}_pred_{split_type}_seed_{seed}"] = pred
            arr = np.vstack(seed_preds)
            aux[f"{prefix}_pred_{split_type}"] = arr.mean(axis=0)
            aux[f"{prefix}_seed_std_{split_type}"] = arr.std(axis=0)
        pred_cols = [f"{prefix}_pred_{split_type}" for split_type in SPLIT_TYPES]
        std_cols = [f"{prefix}_seed_std_{split_type}" for split_type in SPLIT_TYPES]
        aux[f"{prefix}_pred_mean"] = aux[pred_cols].mean(axis=1)
        aux[f"{prefix}_split_std"] = aux[pred_cols].std(axis=1, ddof=0)
        aux[f"{prefix}_seed_std"] = aux[std_cols].mean(axis=1)
        aux[f"{prefix}_uncertainty"] = np.sqrt(aux[f"{prefix}_split_std"] ** 2 + aux[f"{prefix}_seed_std"] ** 2)
        aux[f"{prefix}_target_region"] = aux[f"{prefix}_pred_mean"].apply(target.region)
        aux[f"{prefix}_target_distance"] = target.distance(aux[f"{prefix}_pred_mean"].to_numpy(dtype=float))

    out_path = PHASE10 / "tables" / "phase10_auxiliary_verifier_predictions.csv"
    aux.to_csv(out_path, index=False, encoding="utf-8-sig")
    return aux


def fingerprints(smiles_values: pd.Series) -> list:
    fps = []
    for smiles in smiles_values.astype(str):
        mol = Chem.MolFromSmiles(smiles)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol is not None else None)
    return fps


def nearest_training_similarity(candidates: pd.DataFrame) -> pd.Series:
    train_frames = []
    for path in [PHASE01 / "splits" / "train.csv", PHASE04 / "splits" / "scaffold_train.csv"]:
        df = pd.read_csv(path)
        smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        train_frames.append(df[[smiles_col]].rename(columns={smiles_col: "smiles"}))
    train_smiles = pd.concat(train_frames, ignore_index=True)["smiles"].dropna().drop_duplicates()
    train_fps = [fp for fp in fingerprints(train_smiles) if fp is not None]
    values: list[float] = []
    for fp in fingerprints(candidates["phase10_parent_smiles"]):
        if fp is None or not train_fps:
            values.append(float("nan"))
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        values.append(float(max(sims)) if sims else float("nan"))
    return pd.Series(values, index=candidates.index)


def annotate_screening(consensus: pd.DataFrame, aux: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["predictor_screening"]
    merged = consensus.merge(
        aux.drop(columns=["phase10_parent_smiles"]),
        on="external_molecule_id",
        how="left",
    )
    merged["nearest_train_tanimoto"] = nearest_training_similarity(merged)
    merged["main_side"] = merged["main_target_region"].map(REGION_TO_SIDE)
    merged["xgboost_side"] = merged["xgboost_target_region"].map(REGION_TO_SIDE)
    merged["rf_side"] = merged["rf_target_region"].map(REGION_TO_SIDE)

    same_xgb = (merged["main_side"] != 0) & (merged["xgboost_side"] == merged["main_side"])
    same_rf = (merged["main_side"] != 0) & (merged["rf_side"] == merged["main_side"])
    opp_xgb = (merged["main_side"] != 0) & (merged["xgboost_side"] == -merged["main_side"])
    opp_rf = (merged["main_side"] != 0) & (merged["rf_side"] == -merged["main_side"])
    merged["auxiliary_same_side_count"] = same_xgb.astype(int) + same_rf.astype(int)
    merged["auxiliary_opposite_side_count"] = opp_xgb.astype(int) + opp_rf.astype(int)

    merged["mapping_ok"] = merged["primary_library_id"].notna()
    merged["outside_target_ok"] = merged["main_target_region"].isin(["low", "high"])
    merged["active_distance_ok"] = merged["main_target_distance"] >= cfg["active_distance_min"]
    merged["tier_a_distance_ok"] = merged["main_target_distance"] >= cfg["tier_a_active_distance_min"]
    merged["uncertainty_ok"] = merged["main_uncertainty"] <= cfg["prediction_uncertainty_max"]
    merged["nearest_min_ok"] = merged["nearest_train_tanimoto"] >= cfg["nearest_train_tanimoto_min"]
    merged["nearest_preferred_ok"] = merged["nearest_train_tanimoto"] >= cfg["preferred_nearest_train_tanimoto_min"]
    merged["exact_overlap_ok"] = merged["nearest_train_tanimoto"] <= cfg["exact_train_overlap_tanimoto_max"]
    merged["auxiliary_support_ok"] = merged["auxiliary_same_side_count"] >= 1
    merged["auxiliary_no_opposite_ok"] = merged["auxiliary_opposite_side_count"] == 0

    active_mask = (
        merged["mapping_ok"]
        & merged["outside_target_ok"]
        & merged["active_distance_ok"]
        & merged["uncertainty_ok"]
        & merged["nearest_min_ok"]
        & merged["exact_overlap_ok"]
    )
    tier_a_mask = (
        active_mask
        & merged["tier_a_distance_ok"]
        & merged["nearest_preferred_ok"]
        & merged["auxiliary_support_ok"]
        & merged["auxiliary_no_opposite_ok"]
    )
    tier_b_mask = active_mask & ~tier_a_mask & merged["auxiliary_no_opposite_ok"]
    merged["active_parent_flag"] = active_mask
    merged["active_parent_tier"] = np.where(tier_a_mask, "A", np.where(tier_b_mask, "B", "none"))

    statuses = []
    for _, row in merged.iterrows():
        if row["active_parent_tier"] == "A":
            statuses.append("active_tier_a")
        elif row["active_parent_tier"] == "B":
            statuses.append("active_tier_b")
        elif not row["mapping_ok"]:
            statuses.append("inactive_without_library_mapping")
        elif not row["outside_target_ok"]:
            statuses.append("inactive_in_target_window")
        elif not row["active_distance_ok"]:
            statuses.append("inactive_small_target_distance")
        elif not row["uncertainty_ok"]:
            statuses.append("inactive_high_uncertainty")
        elif not row["nearest_min_ok"]:
            statuses.append("inactive_far_from_training_domain")
        elif not row["exact_overlap_ok"]:
            statuses.append("inactive_training_overlap")
        elif not row["auxiliary_no_opposite_ok"]:
            statuses.append("inactive_auxiliary_opposition")
        else:
            statuses.append("inactive_other")
    merged["screening_status"] = statuses

    score = (
        merged["main_target_distance"].astype(float)
        + 0.22 * merged["auxiliary_same_side_count"].astype(float)
        + 0.12 * merged["nearest_train_tanimoto"].astype(float).fillna(0.0)
        + 0.08 * merged["qed"].astype(float).fillna(0.0)
        - 0.65 * merged["main_uncertainty"].astype(float).fillna(1.0)
        - 0.18 * merged["auxiliary_opposite_side_count"].astype(float)
    )
    score += np.where(merged["active_parent_tier"] == "A", 0.35, 0.0)
    merged["selection_score"] = score
    return merged


def scaffold_key(row: pd.Series) -> str:
    scaffold = str(row.get("murcko_scaffold", "") or "").strip()
    if scaffold and scaffold.lower() != "nan":
        return scaffold
    return f"acyclic_{row['external_molecule_id']}"


def select_active_pool(screened: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config["predictor_screening"]
    gates = config["acceptance_gates"]
    preferred = int(cfg["preferred_active_parent_count"])
    side_quota = int(np.ceil(preferred * cfg["low_side_fraction_min"]))
    max_scaffold = int(cfg["max_molecules_per_murcko_scaffold"])
    max_library = int(cfg["max_molecules_per_mini_library"])

    eligible = screened[screened["active_parent_tier"].isin(["A", "B"])].copy()
    eligible["tier_rank"] = eligible["active_parent_tier"].map({"A": 0, "B": 1}).fillna(2)
    eligible["scaffold_selection_key"] = eligible.apply(scaffold_key, axis=1)
    eligible = eligible.sort_values(
        ["tier_rank", "selection_score", "main_target_distance", "main_uncertainty"],
        ascending=[True, False, False, True],
    )

    selected_indices: list[int] = []
    scaffold_counts: dict[str, int] = {}
    library_counts: dict[str, int] = {}

    def can_add(row: pd.Series) -> bool:
        if row.name in selected_indices:
            return False
        scaffold = row["scaffold_selection_key"]
        library = row.get("primary_library_id", "")
        return scaffold_counts.get(scaffold, 0) < max_scaffold and library_counts.get(library, 0) < max_library

    def add_row(row: pd.Series) -> None:
        selected_indices.append(row.name)
        scaffold = row["scaffold_selection_key"]
        library = row.get("primary_library_id", "")
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
        library_counts[library] = library_counts.get(library, 0) + 1

    for side in ["low", "high"]:
        side_df = eligible[eligible["main_target_region"] == side]
        for _, row in side_df.iterrows():
            if sum(screened.loc[idx, "main_target_region"] == side for idx in selected_indices) >= side_quota:
                break
            if can_add(row):
                add_row(row)

    tier_a_min = int(gates["screened_tier_a_parent_count_min"])
    for _, row in eligible[eligible["active_parent_tier"] == "A"].iterrows():
        if sum(screened.loc[idx, "active_parent_tier"] == "A" for idx in selected_indices) >= tier_a_min:
            break
        if len(selected_indices) >= preferred:
            break
        if can_add(row):
            add_row(row)

    for _, row in eligible.iterrows():
        if len(selected_indices) >= preferred:
            break
        if can_add(row):
            add_row(row)

    selected = screened.loc[selected_indices].copy()
    if selected.empty:
        selected["selected_rank"] = []
        return selected
    selected = selected.sort_values(
        ["active_parent_tier", "selection_score", "main_target_distance"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    selected["selected_rank"] = np.arange(1, len(selected) + 1)
    return selected


def gate_status(name: str, actual: float, threshold: float, direction: str = ">=") -> dict:
    if direction == ">=":
        passed = actual >= threshold
    else:
        passed = actual <= threshold
    return {
        "gate": name,
        "actual": float(actual),
        "threshold": float(threshold),
        "direction": direction,
        "status": "pass" if passed else "needs_fix",
    }


def write_reports(screened: pd.DataFrame, selected: pd.DataFrame, logs: list[dict], config: dict) -> None:
    gates = config["acceptance_gates"]
    cfg = config["predictor_screening"]
    selected_count = len(selected)
    selected_low = int((selected["main_target_region"] == "low").sum()) if selected_count else 0
    selected_high = int((selected["main_target_region"] == "high").sum()) if selected_count else 0
    selected_tier_a = int((selected["active_parent_tier"] == "A").sum()) if selected_count else 0
    low_fraction = selected_low / selected_count if selected_count else 0.0
    high_fraction = selected_high / selected_count if selected_count else 0.0
    nearest_mean = clean_number(selected["nearest_train_tanimoto"].mean()) if selected_count else 0.0
    uncertainty_mean = clean_number(selected["main_uncertainty"].mean()) if selected_count else 999.0

    gate_rows = [
        gate_status("parent_candidate_count", len(screened), gates["parent_candidate_count_min"]),
        gate_status("selected_active_parent_count", selected_count, gates["screened_active_parent_count_min"]),
        gate_status("selected_tier_a_parent_count", selected_tier_a, gates["screened_tier_a_parent_count_min"]),
        gate_status("selected_low_fraction", low_fraction, gates["screened_active_low_fraction_min"]),
        gate_status("selected_high_fraction", high_fraction, gates["screened_active_high_fraction_min"]),
        gate_status("selected_nearest_train_tanimoto_mean", nearest_mean, gates["screened_active_nearest_train_tanimoto_mean_min"]),
        gate_status(
            "selected_prediction_uncertainty_mean",
            uncertainty_mean,
            gates["screened_active_prediction_uncertainty_mean_max"],
            direction="<=",
        ),
    ]
    verdict = "pass" if all(row["status"] == "pass" for row in gate_rows) else "needs_fix"

    summary = {
        "phase": "10B_parent_predictor_screening",
        "verdict": verdict,
        "candidate_count": int(len(screened)),
        "mapped_candidate_count": int(screened["primary_library_id"].notna().sum()),
        "active_eligible_count": int(screened["active_parent_flag"].sum()),
        "tier_a_count": int((screened["active_parent_tier"] == "A").sum()),
        "tier_b_count": int((screened["active_parent_tier"] == "B").sum()),
        "selected_count": int(selected_count),
        "selected_low_count": int(selected_low),
        "selected_high_count": int(selected_high),
        "selected_tier_a_count": int(selected_tier_a),
        "selected_nearest_train_tanimoto_mean": nearest_mean,
        "selected_prediction_uncertainty_mean": uncertainty_mean,
        "main_predictor_logs": logs,
        "gates": gate_rows,
        "screening_thresholds": cfg,
    }

    summary_path = PHASE10 / "reports" / "phase10b_parent_screening_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    status_counts = screened["screening_status"].value_counts().to_dict()
    lines = [
        "# 10-B 父分子主预测器筛选报告",
        "",
        f"结论：{verdict}",
        "",
        "## 核心结果",
        "",
        f"- 候选父分子：{len(screened)}",
        f"- 有小库映射的候选父分子：{int(screened['primary_library_id'].notna().sum())}",
        f"- 主预测器判定为主动父分子的数量：{int(screened['active_parent_flag'].sum())}",
        f"- Tier A 数量：{int((screened['active_parent_tier'] == 'A').sum())}",
        f"- Tier B 数量：{int((screened['active_parent_tier'] == 'B').sum())}",
        f"- 冻结父分子数量：{selected_count}",
        f"- 冻结父分子低侧/高侧：{selected_low}/{selected_high}",
        f"- 冻结 Tier A 数量：{selected_tier_a}",
        f"- 冻结父分子平均训练相似度：{nearest_mean:.3f}",
        f"- 冻结父分子平均预测不确定性：{uncertainty_mean:.3f}",
        "",
        "## 筛选状态分布",
        "",
    ]
    for key, value in status_counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 门控检查", ""])
    for row in gate_rows:
        lines.append(
            f"- {row['gate']}: {row['actual']:.3f} {row['direction']} {row['threshold']:.3f} -> {row['status']}"
        )
    lines.extend(
        [
            "",
            "## 输出表",
            "",
            f"- `{rel(PHASE10 / 'tables' / 'phase10_parent_predictor_consensus.csv')}`",
            f"- `{rel(PHASE10 / 'tables' / 'phase10_parent_main_predictor_seed_predictions.csv')}`",
            f"- `{rel(PHASE10 / 'tables' / 'phase10_auxiliary_verifier_predictions.csv')}`",
            f"- `{rel(PHASE10 / 'tables' / 'phase10_active_parent_pool.csv')}`",
            "",
            "## 后续动作",
            "",
            "- 若结论为 pass，下一步进入 10-C：对冻结父分子做 Mol Token 贡献切分和可编辑位点筛选。",
            "- 若结论为 needs_fix，优先检查低/高两侧是否失衡、Tier A 是否不足、预测不确定性是否偏高。",
            "",
        ]
    )
    report_path = PHASE10 / "reports" / "phase10b_parent_screening_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    self_check_lines = [
        "# 10-B 自检记录",
        "",
        f"总体结论：{verdict}",
        "",
        "## 检查项",
        "",
        f"- 输入候选表存在且可读取：pass",
        f"- 父分子到小库映射覆盖率：{summary['mapped_candidate_count']}/{summary['candidate_count']}",
        f"- 主预测器逐 seed 输出数量：{len(logs)}",
        f"- 冻结清单数量：{selected_count}",
        f"- 每个 Murcko scaffold 最多数量阈值：{cfg['max_molecules_per_murcko_scaffold']}",
        f"- 每个小库最多数量阈值：{cfg['max_molecules_per_mini_library']}",
        "",
        "## 门控",
        "",
    ]
    for row in gate_rows:
        self_check_lines.append(
            f"- {row['gate']}: {row['status']} ({row['actual']:.3f} {row['direction']} {row['threshold']:.3f})"
        )
    (PHASE10 / "reports" / "phase10b_parent_screening_self_check.md").write_text(
        "\n".join(self_check_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N candidates for a quick check.")
    args = parser.parse_args()

    ensure_dirs(PHASE10 / "tables", PHASE10 / "reports", PREDICTION_DIR, LOG_DIR)
    config = load_config()
    target_cfg = config["predictor_screening"]
    target = TargetWindow(low=target_cfg["target_window_low"], high=target_cfg["target_window_high"])

    candidates = pd.read_csv(CANDIDATE_PATH)
    if args.limit:
        candidates = candidates.head(args.limit).copy()
    candidates = candidates.dropna(subset=["external_molecule_id", "phase10_parent_smiles"]).copy()
    candidates["external_molecule_id"] = candidates["external_molecule_id"].astype(str)

    library_map = prepare_parent_library_map(config)
    candidates = candidates.merge(library_map, on="external_molecule_id", how="left")

    consensus, logs = aggregate_main_predictions(candidates, target, config, args)
    aux = aggregate_auxiliary_predictions(candidates, target)
    screened = annotate_screening(consensus, aux, config)
    selected = select_active_pool(screened, config)

    screened_path = PHASE10 / "tables" / "phase10_parent_predictor_consensus.csv"
    selected_path = PHASE10 / "tables" / "phase10_active_parent_pool.csv"
    screened.to_csv(screened_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    write_reports(screened, selected, logs, config)


if __name__ == "__main__":
    main()
