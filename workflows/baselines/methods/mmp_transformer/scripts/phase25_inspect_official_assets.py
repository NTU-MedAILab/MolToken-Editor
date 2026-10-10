#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, pickle, sys, traceback
import pandas as pd

P25 = Path(__file__).resolve().parents[1]
repo = P25 / "external/deep-molecular-optimization"
model_root = P25 / "models/official_pretrained/extracted"
out_dir = P25 / "tables/preflight"
out_dir.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

if not repo.exists():
    raise FileNotFoundError(f"Repository missing: {repo}")

checkpoints = sorted(model_root.rglob("model_*.pt"))
vocabs = sorted(model_root.rglob("vocab.pkl")) or sorted(repo.rglob("vocab.pkl"))
if not checkpoints:
    raise FileNotFoundError(f"No model_*.pt found under {model_root}")

report = {
    "phase": "phase25_official_asset_preflight",
    "status": "manual_review",
    "checkpoints": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in checkpoints],
    "vocab_files": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in vocabs],
    "official_test_files": [],
    "condition_tokens": {},
}

samples = []
for p in sorted(repo.rglob("test*.csv")):
    try:
        df = pd.read_csv(p)
        report["official_test_files"].append({
            "path": str(p.resolve()), "rows": len(df),
            "columns": list(df.columns), "sha256": sha256(p),
        })
        if len(df):
            s = df.head(2).copy()
            s.insert(0, "_source_file", p.name)
            samples.append(s)
    except Exception as exc:
        report["official_test_files"].append({"path": str(p.resolve()), "read_error": repr(exc)})

if samples:
    pd.concat(samples, ignore_index=True, sort=False).to_csv(
        out_dir / "official_test_schema_samples.csv", index=False
    )

tokens = []
if vocabs:
    sys.path.insert(0, str(repo))
    try:
        with open(vocabs[0], "rb") as f:
            vocab = pickle.load(f)
        objects = []
        for attr in ["tokens","token2index","index2token","_tokens","_token2index","_index2token","vocab","stoi","itos"]:
            if hasattr(vocab, attr):
                objects.append(getattr(vocab, attr))
        if hasattr(vocab, "__dict__"):
            objects.extend(vocab.__dict__.values())
        seen = set()
        for obj in objects:
            values = list(obj.keys()) + list(obj.values()) if isinstance(obj, dict) else list(obj) if isinstance(obj, (list,tuple,set)) else []
            for value in values:
                if isinstance(value, str) and value not in seen:
                    seen.add(value)
                    tokens.append(value)
        tokens.sort()
    except Exception as exc:
        report["vocab_load_error"] = {
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

for group, patterns in {
    "logd": ["logd","log_d"],
    "solubility": ["solubility"],
    "clint": ["clint","clearance"],
}.items():
    report["condition_tokens"][group] = [
        t for t in tokens if any(pat in t.lower() for pat in patterns)
    ]

neutral_markers = ["no_change","unchanged","low->low","high->high","0->0","1->1","00","11"]
neutral_sol = [t for t in report["condition_tokens"]["solubility"] if any(m in t.lower().replace(" ","") for m in neutral_markers)]
neutral_cl = [t for t in report["condition_tokens"]["clint"] if any(m in t.lower().replace(" ","") for m in neutral_markers)]
report["single_logd_pretrained_gate"] = {
    "neutral_solubility_token_candidates": neutral_sol,
    "neutral_clint_token_candidates": neutral_cl,
    "automatic_gate_pass": bool(neutral_sol and neutral_cl),
    "note": "Token semantics must still be confirmed against official test rows.",
}
report["status"] = "candidate_pass" if neutral_sol and neutral_cl else "manual_review"

lines = []
for group in ["logd","solubility","clint"]:
    lines += [f"===== {group.upper()} condition tokens ====="]
    lines += report["condition_tokens"][group] or ["<none detected>"]
    lines += [""]

(out_dir / "condition_token_candidates.txt").write_text("\n".join(lines), encoding="utf-8")
(out_dir / "official_asset_preflight.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

print("===== Phase 25 official asset preflight =====")
print("Checkpoints:", len(checkpoints))
print("Vocab files:", len(vocabs))
print("Official test files:", len(report["official_test_files"]))
print("Status:", report["status"])
print("Automatic single-logD gate:", report["single_logd_pretrained_gate"]["automatic_gate_pass"])
print("Report:", out_dir / "official_asset_preflight.json")
