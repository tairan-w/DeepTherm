
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deeptherm.training import predict_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a DeepTherm checkpoint ensemble")
    parser.add_argument("--data", required=True, help="Input CSV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir")
    group.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    if args.smiles_column not in frame:
        raise ValueError(f"Missing SMILES column {args.smiles_column!r}")
    checkpoints = args.checkpoints
    if checkpoints is None:
        checkpoints = sorted(str(path) for path in Path(args.checkpoint_dir).rglob("model_*.pt"))
    predictions, task_names, weights = predict_ensemble(
        checkpoints,
        frame[args.smiles_column].astype(str).tolist(),
        args.batch_size,
        args.device,
    )
    for index, task_name in enumerate(task_names):
        frame[f"pred_{task_name}"] = predictions[:, index]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote={output} models={len(checkpoints)} weights={weights.tolist()}")


if __name__ == "__main__":
    main()
