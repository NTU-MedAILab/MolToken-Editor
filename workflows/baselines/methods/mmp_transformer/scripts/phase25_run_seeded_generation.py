#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official MMP-Transformer generator "
            "with a fixed random seed."
        )
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--test-file-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--save-directory", required=True)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument(
        "--decode-type",
        default="multinomial",
        choices=["multinomial", "greedy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo = Path(args.repo).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    checkpoint = model_path / f"model_{args.epoch}.pt"
    vocab_path = data_path / "vocab.pkl"
    test_path = data_path / f"{args.test_file_name}.csv"

    required_paths = [
        repo,
        checkpoint,
        vocab_path,
        test_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ["PYTHONHASHSEED"] = str(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("===== Phase 25 seeded generation =====")
    print("Seed:", args.seed)
    print("Repository:", repo)
    print("Input CSV:", test_path)
    print("Vocabulary:", vocab_path)
    print("Checkpoint:", checkpoint)
    print("Batch size:", args.batch_size)
    print("Number of samples:", args.num_samples)
    print("Decode type:", args.decode_type)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    sys.argv = [
        "generate.py",
        "--model-choice",
        "transformer",
        "--data-path",
        str(data_path),
        "--test-file-name",
        args.test_file_name,
        "--model-path",
        str(model_path),
        "--save-directory",
        args.save_directory,
        "--epoch",
        str(args.epoch),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
        "--decode-type",
        args.decode_type,
    ]

    import generate

    generate.run_main()

    print("PHASE25_GENERATION_FINISHED")


if __name__ == "__main__":
    main()
