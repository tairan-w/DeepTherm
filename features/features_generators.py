from .featurization import fixed_descriptors


def morgan_1024(smiles: str):
    return fixed_descriptors(smiles, "morgan")


def rdkit_2d(smiles: str):
    return fixed_descriptors(smiles, "rdkit")


__all__ = ["morgan_1024", "rdkit_2d"]
