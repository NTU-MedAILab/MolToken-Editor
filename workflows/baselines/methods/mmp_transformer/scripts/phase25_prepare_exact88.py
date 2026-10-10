#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
import pandas as pd

P25 = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = P25.parents[1]
src = (
    COMPARISON_ROOT
    / "inputs"
    / "common_exact88"
    / "phase26_locked_exact88.csv"
)
out_dir = P25 / "inputs/exact88"
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "phase25_locked_exact88.csv"
manifest_path = out_dir / "phase25_exact88_manifest.json"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def detect(columns, candidates):
    return next((c for c in candidates if c in columns), None)

if not src.exists():
    raise FileNotFoundError(f"Missing frozen common Exact88 table: {src}")

df = pd.read_csv(src)
id_col = detect(df.columns, ["mol_id","parent_id","locked_parent_id","external_id","id"])
smi_col = detect(df.columns, ["smiles","parent_smiles","canonical_smiles","source_smiles"])
dir_col = detect(df.columns, ["optimization_direction","direction","locked_direction"])
if None in [id_col, smi_col, dir_col]:
    raise ValueError(f"Required columns not detected. Available: {list(df.columns)}")

out = pd.DataFrame({
    "mol_id": df[id_col].astype(str).str.strip(),
    "smiles": df[smi_col].astype(str).str.strip(),
    "optimization_direction": df[dir_col].astype(str).str.strip().str.lower(),
})
out["optimization_direction"] = out["optimization_direction"].replace({
    "increase": "increase_logd",
    "decrease": "decrease_logd",
})

errors = []
counts = out["optimization_direction"].value_counts().to_dict()
if len(out) != 88: errors.append(f"Expected 88 rows, found {len(out)}")
if out["mol_id"].nunique() != 88: errors.append("mol_id is not unique")
if counts.get("increase_logd", 0) != 66: errors.append(f"increase_logd={counts.get('increase_logd',0)}, expected 66")
if counts.get("decrease_logd", 0) != 22: errors.append(f"decrease_logd={counts.get('decrease_logd',0)}, expected 22")
if out["smiles"].eq("").any(): errors.append("Blank SMILES detected")
if errors:
    print("\n".join("[ERROR] " + e for e in errors))
    sys.exit(1)

out = out.sort_values("mol_id").reset_index(drop=True)
out.to_csv(out_csv, index=False)
manifest = {
    "phase": "phase25_mmp_transformer_exact88_lock",
    "status": "completed",
    "source_path": str(src.resolve()),
    "source_sha256": sha256(src),
    "output_path": str(out_csv.resolve()),
    "output_sha256": sha256(out_csv),
    "row_count": len(out),
    "unique_parent_count": out["mol_id"].nunique(),
    "direction_counts": counts,
    "target_window": [1.0, 3.4],
    "top_k": [1,3,5,10],
    "planned_sampling_seeds": 3,
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("===== Phase 25 Exact-88 lock completed =====")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
