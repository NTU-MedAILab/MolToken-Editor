#!/usr/bin/env python3
"""Run REINVENT4 with an explicit seed and write an auditable run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--device", default=None, help="Optional CLI device override.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    plugin_dir = args.plugin_dir.resolve()
    if not plugin_dir.is_dir():
        raise FileNotFoundError(plugin_dir)
    executable = shutil.which("reinvent")
    if executable is None and not args.dry_run:
        raise FileNotFoundError("reinvent executable is not on PATH")
    executable = executable or "reinvent"

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-s",
        str(args.seed),
        "-l",
        str(args.log.resolve()),
    ]
    if args.device:
        command.extend(["-d", args.device])
    command.append(str(config))

    started = datetime.now(timezone.utc).isoformat()
    returncode = None
    if not args.dry_run:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(plugin_dir) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        result = subprocess.run(command, check=False, env=environment)
        returncode = result.returncode
    finished = datetime.now(timezone.utc).isoformat()

    manifest = {
        "status": "dry_run" if args.dry_run else ("completed" if returncode == 0 else "failed"),
        "started_at_utc": started,
        "finished_at_utc": finished,
        "command": command,
        "random_seed": args.seed,
        "config": str(config),
        "config_sha256": sha256(config),
        "plugin_dir": str(plugin_dir),
        "log": str(args.log.resolve()),
        "returncode": returncode,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if returncode not in (None, 0):
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
