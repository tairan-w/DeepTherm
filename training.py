
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from deeptherm.data import Standardizer, ThermochemistryDataset, collate_graphs, split_indices
from deeptherm.features import ATOM_FDIM, BOND_FDIM, descriptor_size, fixed_descriptors
from deeptherm.models import DeepThermModel, ModelConfig


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15
    num_workers: int = 0
    gradient_clip: float = 5.0
    device: str = "auto"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        raise ValueError("A training batch has no finite labels")
    return torch.square(predictions[mask] - targets[mask]).mean()


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    per_task = []
    for task in range(targets.shape[1]):
        mask = np.isfinite(targets[:, task]) & np.isfinite(predictions[:, task])
        if not np.any(mask):
            per_task.append({"mae": None, "rmse": None, "r2": None, "count": 0})
            continue
        actual, predicted = targets[mask, task], predictions[mask, task]
        error = predicted - actual
        denominator = np.square(actual - actual.mean()).sum()
        per_task.append(
            {
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "r2": None if denominator == 0 else float(1.0 - np.square(error).sum() / denominator),
                "count": int(mask.sum()),
            }
        )
    valid_maes = [item["mae"] for item in per_task if item["mae"] is not None]
    return {"mae": float(np.mean(valid_maes)), "per_task": per_task}


def _loader(dataset: ThermochemistryDataset, config: TrainingConfig, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_graphs,
        generator=generator,
    )


def predict_loader(
    model: DeepThermModel,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            if batch.targets is not None:
                raw_targets = batch.targets.cpu().numpy()
                raw_targets[~batch.target_mask.cpu().numpy()] = np.nan
                targets.append(raw_targets)
    predicted = np.concatenate(predictions)
    actual = np.concatenate(targets) if targets else None
    if target_scaler is not None:
        predicted = target_scaler.inverse_transform(predicted)
        if actual is not None:
            actual = target_scaler.inverse_transform(actual)
    return predicted, actual


def save_checkpoint(
    path: Path,
    model: DeepThermModel,
    model_config: ModelConfig,
    target_scaler: Standardizer,
    descriptor_scaler: Standardizer | None,
    descriptor_kind: str,
    task_names: Sequence[str],
    validation_metrics: dict,
    training_metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "target_scaler": target_scaler.to_dict(),
            "descriptor_scaler": None if descriptor_scaler is None else descriptor_scaler.to_dict(),
            "descriptor_kind": descriptor_kind,
            "task_names": list(task_names),
            "validation_metrics": validation_metrics,
            "training_metadata": training_metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[DeepThermModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    model = DeepThermModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload


def _fit_one(
    frame: pd.DataFrame,
    smiles_column: str,
    task_names: Sequence[str],
    descriptor_kind: str,
    model_options: dict,
    training_config: TrainingConfig,
    split_strategy: str,
    split_fractions: tuple[float, float, float],
    seed: int,
    test_smarts: str | None,
    pretrained_checkpoint: str | Path | None,
    checkpoint_path: Path,
) -> dict:
    seed_everything(seed)
    smiles = frame[smiles_column].astype(str).tolist()
    target_values = frame[list(task_names)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    train_indices, valid_indices, test_indices = split_indices(
        smiles, split_strategy, split_fractions, seed, test_smarts
    )
    if any(len(values) == 0 for values in (train_indices, valid_indices, test_indices)):
        raise ValueError("A data split is empty; adjust fractions or supply more data")

    train_targets = target_values[train_indices]
    target_scaler = Standardizer.fit(train_targets, np.isfinite(train_targets))
    descriptor_scaler = None
    if descriptor_size(descriptor_kind):
        raw_descriptors = np.stack([fixed_descriptors(value, descriptor_kind) for value in smiles])
        descriptor_scaler = Standardizer.fit(raw_descriptors[train_indices])

    def dataset(indices: np.ndarray) -> ThermochemistryDataset:
        return ThermochemistryDataset(
            [smiles[index] for index in indices],
            target_values[indices],
            descriptor_kind,
            descriptor_scaler,
            target_scaler,
        )

    train_data, valid_data, test_data = dataset(train_indices), dataset(valid_indices), dataset(test_indices)
    model_config = ModelConfig(
        atom_dim=ATOM_FDIM,
        bond_dim=BOND_FDIM,
        descriptor_dim=descriptor_size(descriptor_kind),
        num_tasks=len(task_names),
        **model_options,
    )
    model = DeepThermModel(model_config)
    transfer_metadata = None
    if pretrained_checkpoint:
        pretrained_payload = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=True)
        if "model_state" not in pretrained_payload:
            raise ValueError("Pretrained checkpoint is not a DeepTherm checkpoint")
        model.load_pretrained_encoder(pretrained_payload["model_state"])
        transfer_metadata = {
            "checkpoint": str(pretrained_checkpoint),
            "source_tasks": pretrained_payload.get("task_names", []),
        }

    device = resolve_device(training_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loader = _loader(train_data, training_config, True, seed)
    valid_loader = _loader(valid_data, training_config, False, seed)
    test_loader = _loader(test_data, training_config, False, seed)

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(batch), batch.targets, batch.target_mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        valid_predictions, valid_targets = predict_loader(model, valid_loader, device, target_scaler)
        valid_metrics = regression_metrics(valid_targets, valid_predictions)
        validation_loss = valid_metrics["mae"]
        history.append({"epoch": epoch, "train_mse_scaled": float(np.mean(batch_losses)),
                        "validation_mae": validation_loss})
        print(
            f"seed={seed} epoch={epoch:03d} train_mse={np.mean(batch_losses):.6f} "
            f"validation_mae={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                model_config,
                target_scaler,
                descriptor_scaler,
                descriptor_kind,
                task_names,
                valid_metrics,
                {
                    "seed": seed,
                    "best_epoch": epoch,
                    "split_strategy": split_strategy,
                    "split_fractions": split_fractions,
                    "split_indices": {
                        "train": train_indices.tolist(),
                        "validation": valid_indices.tolist(),
                        "test": test_indices.tolist(),
                    },
                    "training_config": asdict(training_config),
                    "transfer": transfer_metadata,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                break

    best_model, payload = load_checkpoint(checkpoint_path, device)
    test_predictions, test_targets = predict_loader(best_model, test_loader, device, target_scaler)
    test_metrics = regression_metrics(test_targets, test_predictions)
    result = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": payload["validation_metrics"],
        "test": test_metrics,
        "history": history,
    }
    result_path = checkpoint_path.with_suffix(".metrics.json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_ensemble(
    frame: pd.DataFrame,
    smiles_column: str,
    task_names: Sequence[str],
    save_dir: str | Path,
    descriptor_kind: str,
    model_options: dict,
    training_config: TrainingConfig,
    split_strategy: str = "random",
    split_fractions: tuple[float, float, float] = (0.81, 0.09, 0.10),
    seed: int = 0,
    ensemble_size: int = 1,
    test_smarts: str | None = None,
    pretrained_checkpoint: str | Path | None = None,
) -> list[dict]:
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for model_index in range(ensemble_size):
        model_seed = seed + model_index
        results.append(
            _fit_one(
                frame,
                smiles_column,
                task_names,
                descriptor_kind,
                model_options,
                training_config,
                split_strategy,
                split_fractions,
                model_seed,
                test_smarts,
                pretrained_checkpoint,
                output / f"model_{model_index:02d}.pt",
            )
        )
    (output / "ensemble_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def predict_ensemble(
    checkpoint_paths: Sequence[str | Path],
    smiles: Sequence[str],
    batch_size: int = 128,
    device_name: str = "auto",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not checkpoint_paths:
        raise ValueError("No checkpoints were supplied")
    device = resolve_device(device_name)
    all_predictions, validation_maes = [], []
    task_names = None
    for path in checkpoint_paths:
        model, payload = load_checkpoint(path, device)
        if task_names is None:
            task_names = payload["task_names"]
        elif task_names != payload["task_names"]:
            raise ValueError("Ensemble checkpoints have different target columns")
        descriptor_scaler = (
            None if payload["descriptor_scaler"] is None
            else Standardizer.from_dict(payload["descriptor_scaler"])
        )
        target_scaler = Standardizer.from_dict(payload["target_scaler"])
        dataset = ThermochemistryDataset(
            smiles, None, payload["descriptor_kind"], descriptor_scaler, None
        )
        config = TrainingConfig(batch_size=batch_size, device=device_name)
        loader = _loader(dataset, config, False, 0)
        predictions, _ = predict_loader(model, loader, device, target_scaler)
        all_predictions.append(predictions)
        validation_maes.append(float(payload["validation_metrics"]["mae"]))

    errors = np.asarray(validation_maes, dtype=np.float64)
    weights = 1.0 / np.maximum(errors, 1e-12)
    weights /= weights.sum()
    ensemble = np.tensordot(weights, np.stack(all_predictions), axes=(0, 0))
    return ensemble, list(task_names), weights
