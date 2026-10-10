#!/usr/bin/env python3
"""Build a reproducible directory of the ten locked Chemprop-v2 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SPLITS = ("random", "scaffold")
EXPECTED_SEEDS = (20260528, 20260529, 20260530, 20260531, 20260601)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(root: Path, split: str, seed: int) -> Path:
    return root / split / f"seed_{seed}" / "model_0" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink.")
    args = parser.parse_args()

    root = args.model_root.resolve()
    expected = [
        (split, seed, checkpoint(root, split, seed))
        for split in EXPECTED_SPLITS
        for seed in EXPECTED_SEEDS
    ]
    missing = [str(path) for _, _, path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The locked 10-model Chemprop-v2 ensemble is incomplete:\n"
            + "\n".join(missing)
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for split, seed, source in expected:
        target = args.out_dir / f"main_logd_{split}_seed_{seed}.pt"
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.copy:
            shutil.copy2(source, target)
            mode = "copy"
        else:
            os.symlink(source, target)
            mode = "symlink"
        records.append(
            {
                "split": split,
                "seed": seed,
                "source": str(source),
                "source_sha256": sha256(source),
                "target": str(target.resolve()),
                "mode": mode,
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ensemble_size": len(records),
        "chemprop_generation": "v2",
        "molecule_featurizer": "RDKit2D",
        "molecule_feature_dimension": 217,
        "models": records,
    }
    manifest_path = args.out_dir / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
