

from __future__ import annotations

import argparse
from pathlib import Path

from deeptherm.data import read_csv_dataset
from deeptherm.training import TrainingConfig, train_ensemble


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DeepTherm architecture")
    parser.add_argument("stage", choices=("pretrain", "finetune"))
    parser.add_argument("--data", required=True, help="CSV containing SMILES and target columns")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--targets", nargs="+", help="Target columns (auto-detect numeric columns if omitted)")
    parser.add_argument("--pretrained", help="QM9 DeepTherm checkpoint used to initialize the encoder")
    parser.add_argument("--descriptor", choices=("none", "morgan", "rdkit"), default=None)
    parser.add_argument("--split", choices=("random", "scaffold", "complexity", "functional-group"),
                        default="random")
    parser.add_argument("--split-fractions", type=float, nargs=3, default=(0.81, 0.09, 0.10),
                        metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument("--test-smarts", help="SMARTS held out by --split functional-group")
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=300)
    parser.add_argument("--ffn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "pretrain" and args.pretrained:
        raise ValueError("--pretrained is only valid during fine-tuning")
    if args.ensemble_size is not None and args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be positive")
    frame, task_names = read_csv_dataset(args.data, args.smiles_column, args.targets)
    descriptor = args.descriptor or ("none" if args.stage == "pretrain" else "morgan")
    ensemble_size = args.ensemble_size or (1 if args.stage == "pretrain" else 10)
    model_options = {
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "num_attention_heads": args.attention_heads,
        "ffn_hidden_size": args.ffn_hidden_size,
        "ffn_num_layers": args.ffn_layers,
        "dropout": args.dropout,
    }
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
    )
    results = train_ensemble(
        frame=frame,
        smiles_column=args.smiles_column,
        task_names=task_names,
        save_dir=Path(args.save_dir),
        descriptor_kind=descriptor,
        model_options=model_options,
        training_config=training_config,
        split_strategy=args.split,
        split_fractions=tuple(args.split_fractions),
        seed=args.seed,
        ensemble_size=ensemble_size,
        test_smarts=args.test_smarts,
        pretrained_checkpoint=args.pretrained,
    )
    for result in results:
        print(
            f"saved={result['checkpoint']} validation_mae={result['validation']['mae']:.6f} "
            f"test_mae={result['test']['mae']:.6f}"
        )


if __name__ == "__main__":
    main()
