from __future__ import annotations

# Phase20 fix: force MKL/OpenMP compatibility before importing numpy/torch/sklearn/xgboost.
import os
os.environ.pop('MKL_SERVICE_FORCE_INTEL', None)
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
try:
    import numpy as _phase20_numpy_mkl_guard
except Exception:
    pass

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / ".tmp" / "chemprop"
MPL = ROOT / ".mplconfig"
LOG_DIR = ROOT / ".tmp" / "chemprop_logs"


def configure_environment() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    MPL.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = str(TMP)
    os.environ["TEMP"] = str(TMP)
    os.environ["TMPDIR"] = str(TMP)
    os.environ["MPLCONFIGDIR"] = str(MPL)
    os.environ["CHEMPROP_LOG_DIR"] = str(LOG_DIR)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["NO_COLOR"] = "1"
    tempfile.tempdir = str(TMP)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def patch_chemprop_runtime() -> None:
    from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint
    import chemprop.cli.predict as predict_cli
    import chemprop.cli.train as train_cli

    class KeepAllModelCheckpoint(LightningModelCheckpoint):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("save_top_k", -1)
            super().__init__(*args, **kwargs)

    train_cli.ModelCheckpoint = KeepAllModelCheckpoint

    train_original_trainer = train_cli.pl.Trainer
    predict_original_trainer = predict_cli.pl.Trainer

    def quiet_train_trainer(*args, **kwargs):
        kwargs["enable_progress_bar"] = False
        return train_original_trainer(*args, **kwargs)

    def quiet_predict_trainer(*args, **kwargs):
        kwargs["enable_progress_bar"] = False
        return predict_original_trainer(*args, **kwargs)

    train_cli.pl.Trainer = quiet_train_trainer
    predict_cli.pl.Trainer = quiet_predict_trainer


if __name__ == "__main__":
    configure_environment()
    patch_chemprop_runtime()
    from chemprop.cli.main import main as chemprop_main

    chemprop_main()
