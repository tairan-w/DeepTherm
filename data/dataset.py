"""CSV datasets, graph batching, scaling, and paper evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset

from deeptherm.features import fixed_descriptors, smiles_to_graph, validate_smiles
from deeptherm.models.model import GraphBatch


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray | None = None) -> "Standardizer":
        masked = np.where(mask, values, np.nan) if mask is not None else values
        mean = np.nanmean(masked, axis=0)
        scale = np.nanstd(masked, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> "Standardizer":
        return cls(np.asarray(value["mean"], dtype=np.float32),
                   np.asarray(value["scale"], dtype=np.float32))


class ThermochemistryDataset(Dataset):
    def __init__(
        self,
        smiles: Sequence[str],
        targets: np.ndarray | None,
        descriptor_kind: str,
        descriptor_scaler: Standardizer | None = None,
        target_scaler: Standardizer | None = None,
    ):
        self.smiles = [str(value) for value in smiles]
        validate_smiles(self.smiles)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        if self.targets is not None and len(self.smiles) != len(self.targets):
            raise ValueError("SMILES and target row counts differ")
        self.descriptor_kind = descriptor_kind
        self.descriptors = np.stack(
            [fixed_descriptors(value, descriptor_kind) for value in self.smiles]
        ) if descriptor_kind != "none" else np.empty((len(self.smiles), 0), dtype=np.float32)
        if descriptor_scaler is not None and self.descriptors.shape[1]:
            self.descriptors = descriptor_scaler.transform(self.descriptors)
        self.target_scaler = target_scaler

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> dict:
        targets = None if self.targets is None else self.targets[index]
        if targets is not None and self.target_scaler is not None:
            targets = self.target_scaler.transform(targets)
        return {
            "smiles": self.smiles[index],
            "graph": smiles_to_graph(self.smiles[index]),
            "descriptors": self.descriptors[index],
            "targets": targets,
        }


def collate_graphs(samples: Sequence[dict]) -> GraphBatch:
    atoms, bonds, sources, destinations, reverses, scopes = [], [], [], [], [], []
    descriptors, targets = [], []
    atom_offset = bond_offset = 0
    for sample in samples:
        graph = sample["graph"]
        atoms.append(graph.atom_features)
        bonds.append(graph.bond_features)
        sources.append(graph.bond_src + atom_offset)
        destinations.append(graph.bond_dst + atom_offset)
        reverses.append(graph.bond_reverse + bond_offset)
        scopes.append((atom_offset, graph.atom_features.shape[0]))
        atom_offset += graph.atom_features.shape[0]
        bond_offset += graph.bond_features.shape[0]
        descriptors.append(sample["descriptors"])
        if sample["targets"] is not None:
            targets.append(sample["targets"])

    target_array = np.stack(targets) if targets else None
    return GraphBatch(
        atom_features=torch.as_tensor(np.concatenate(atoms), dtype=torch.float32),
        bond_features=torch.as_tensor(np.concatenate(bonds), dtype=torch.float32),
        bond_src=torch.as_tensor(np.concatenate(sources), dtype=torch.long),
        bond_dst=torch.as_tensor(np.concatenate(destinations), dtype=torch.long),
        bond_reverse=torch.as_tensor(np.concatenate(reverses), dtype=torch.long),
        atom_scope=scopes,
        descriptors=torch.as_tensor(np.stack(descriptors), dtype=torch.float32),
        targets=None if target_array is None else torch.as_tensor(
            np.nan_to_num(target_array, nan=0.0), dtype=torch.float32
        ),
        target_mask=None if target_array is None else torch.as_tensor(
            np.isfinite(target_array), dtype=torch.bool
        ),
    )


def read_csv_dataset(
    path: str | Path, smiles_column: str = "smiles", target_columns: Sequence[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    if smiles_column not in frame:
        raise ValueError(f"Missing SMILES column {smiles_column!r}")
    if target_columns:
        columns = list(target_columns)
    else:
        columns = [column for column in frame.columns if column != smiles_column]
        columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    if not columns:
        raise ValueError("No numeric target columns were found; pass --targets explicitly")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")
    return frame, columns


def split_indices(
    smiles: Sequence[str],
    strategy: str,
    fractions: tuple[float, float, float],
    seed: int,
    test_smarts: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(smiles) < 3:
        raise ValueError("At least three molecules are required for train/validation/test splitting")
    if not np.isclose(sum(fractions), 1.0) or any(value <= 0 for value in fractions):
        raise ValueError("Split fractions must be positive and sum to one")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(smiles))

    def partition(ordered_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_train = min(len(ordered_indices) - 2, max(1, int(len(ordered_indices) * fractions[0])))
        n_valid = min(
            len(ordered_indices) - n_train - 1,
            max(1, int(len(ordered_indices) * fractions[1])),
        )
        return (
            ordered_indices[:n_train],
            ordered_indices[n_train:n_train + n_valid],
            ordered_indices[n_train + n_valid:],
        )

    if strategy == "random":
        rng.shuffle(indices)
        return partition(indices)

    if strategy == "complexity":
        indices = np.asarray(sorted(indices, key=lambda i: Chem.MolFromSmiles(smiles[i]).GetNumHeavyAtoms()))
        return partition(indices)

    if strategy == "functional-group":
        if not test_smarts:
            raise ValueError("--test-smarts is required for a functional-group split")
        query = Chem.MolFromSmarts(test_smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS: {test_smarts!r}")
        test = np.asarray([i for i, value in enumerate(smiles)
                           if Chem.MolFromSmiles(value).HasSubstructMatch(query)])
        remainder = np.asarray([i for i in indices if i not in set(test.tolist())])
        if len(test) == 0 or len(remainder) < 2:
            raise ValueError("Functional-group split produced an empty test set or insufficient training data")
        rng.shuffle(remainder)
        valid_size = max(1, round(len(remainder) * fractions[1] / (fractions[0] + fractions[1])))
        return remainder[valid_size:], remainder[:valid_size], test

    if strategy == "scaffold":
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(smiles):
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=value, includeChirality=True)
            # Acyclic molecules otherwise all collapse into one giant empty-scaffold group.
            key = scaffold or f"acyclic:{Chem.MolToSmiles(Chem.MolFromSmiles(value))}"
            groups.setdefault(key, []).append(index)
        ordered = sorted(groups.values(), key=lambda group: (-len(group), group[0]))
        buckets: list[list[int]] = [[], [], []]
        limits = np.asarray(fractions) * len(smiles)
        for group in ordered:
            ratios = [len(buckets[i]) / max(limits[i], 1.0) for i in range(3)]
            buckets[int(np.argmin(ratios))].extend(group)
        for bucket in buckets:
            rng.shuffle(bucket)
        if any(not bucket for bucket in buckets):
            raise ValueError("Scaffold split produced an empty partition; use more molecular diversity")
        return tuple(np.asarray(bucket, dtype=np.int64) for bucket in buckets)  # type: ignore[return-value]

    raise ValueError(f"Unknown split strategy: {strategy}")
