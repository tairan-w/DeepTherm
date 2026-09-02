"""RDKit graph and fixed-descriptor featurization for DeepTherm."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator


ATOM_NUMBERS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRAL_TAGS = list(range(4))
HYDROGEN_COUNTS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
RADICAL_ELECTRONS = [0, 1, 2]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = list(range(6))


def _one_hot_with_unknown(value, choices: list) -> list[float]:
    encoded = [0.0] * (len(choices) + 1)
    try:
        encoded[choices.index(value)] = 1.0
    except ValueError:
        encoded[-1] = 1.0
    return encoded


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Paper-aligned categorical atom attributes plus mass and radical state."""
    values = (
        _one_hot_with_unknown(atom.GetAtomicNum(), ATOM_NUMBERS)
        + _one_hot_with_unknown(atom.GetTotalDegree(), DEGREES)
        + _one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot_with_unknown(int(atom.GetChiralTag()), CHIRAL_TAGS)
        + _one_hot_with_unknown(atom.GetTotalNumHs(), HYDROGEN_COUNTS)
        + _one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), atom.GetMass() * 0.01]
        + _one_hot_with_unknown(atom.GetNumRadicalElectrons(), RADICAL_ELECTRONS)
    )
    return np.asarray(values, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    values = (
        _one_hot_with_unknown(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot_with_unknown(int(bond.GetStereo()), STEREO_TYPES)
    )
    return np.asarray(values, dtype=np.float32)


ATOM_FDIM = len(ATOM_NUMBERS) + 1 + len(DEGREES) + 1 + len(FORMAL_CHARGES) + 1 \
    + len(CHIRAL_TAGS) + 1 + len(HYDROGEN_COUNTS) + 1 + len(HYBRIDIZATIONS) + 1 \
    + 2 + len(RADICAL_ELECTRONS) + 1
BOND_FDIM = len(BOND_TYPES) + 1 + 2 + len(STEREO_TYPES) + 1


@dataclass(frozen=True)
class MoleculeGraph:
    atom_features: np.ndarray
    bond_features: np.ndarray
    bond_src: np.ndarray
    bond_dst: np.ndarray
    bond_reverse: np.ndarray


@lru_cache(maxsize=100_000)
def smiles_to_graph(smiles: str) -> MoleculeGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")

    atoms = np.stack([atom_features(atom) for atom in mol.GetAtoms()])
    directed_features: list[np.ndarray] = []
    sources: list[int] = []
    destinations: list[int] = []
    reverses: list[int] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        forward_index = len(directed_features)
        reverse_index = forward_index + 1
        directed_features.extend([feature, feature])
        sources.extend([begin, end])
        destinations.extend([end, begin])
        reverses.extend([reverse_index, forward_index])

    bonds = (
        np.stack(directed_features)
        if directed_features
        else np.empty((0, BOND_FDIM), dtype=np.float32)
    )
    return MoleculeGraph(
        atom_features=atoms,
        bond_features=bonds,
        bond_src=np.asarray(sources, dtype=np.int64),
        bond_dst=np.asarray(destinations, dtype=np.int64),
        bond_reverse=np.asarray(reverses, dtype=np.int64),
    )


@lru_cache(maxsize=1)
def _morgan_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


@lru_cache(maxsize=1)
def rdkit_descriptor_names() -> tuple[str, ...]:
    return tuple(name for name, _ in Descriptors.descList)


def descriptor_size(kind: str) -> int:
    if kind == "none":
        return 0
    if kind == "morgan":
        return 1024
    if kind == "rdkit":
        return len(rdkit_descriptor_names())
    raise ValueError(f"Unknown descriptor kind: {kind}")


@lru_cache(maxsize=100_000)
def fixed_descriptors(smiles: str, kind: str = "morgan") -> np.ndarray:
    """Computes the main-paper ECFP option or the full RDKit 2D descriptor set."""
    if kind == "none":
        return np.empty((0,), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if kind == "morgan":
        fingerprint = _morgan_generator().GetFingerprint(mol)
        output = np.zeros((1024,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, output)
        return output
    if kind == "rdkit":
        values = []
        for _, function in Descriptors.descList:
            try:
                value = float(function(mol))
            except Exception:
                value = 0.0
            values.append(value if np.isfinite(value) else 0.0)
        return np.asarray(values, dtype=np.float32)
    raise ValueError(f"Unknown descriptor kind: {kind}")


def validate_smiles(smiles_values: Iterable[str]) -> None:
    failures = [value for value in smiles_values if Chem.MolFromSmiles(str(value)) is None]
    if failures:
        preview = ", ".join(repr(value) for value in failures[:5])
        raise ValueError(f"Found {len(failures)} invalid SMILES; first values: {preview}")
