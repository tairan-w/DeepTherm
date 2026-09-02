import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.Descriptors import MolWt, NumHeteroatoms, NumRotatableBonds
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumRings, CalcNumHBA
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField


def get_molecular_descriptors(mol):
    """ Computes molecular descriptors for a given molecule. """
    if mol is None:
        return None


    mol = Chem.AddHs(mol)

    # Generate 3D conformer for UFF calculations
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        uff_energy = UFFGetMoleculeForceField(mol).CalcEnergy()
    except:
        uff_energy = None


    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        max_partial_charge = max(
            [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms() if atom.HasProp('_GasteigerCharge')],
            default=0)
        min_partial_charge = min(
            [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms() if atom.HasProp('_GasteigerCharge')],
            default=0)
    except:
        max_partial_charge, min_partial_charge = 0, 0

    # Compute molecular descriptors
    descriptors = {
        "Molecular Formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": MolWt(mol),
        "Number of heterocycles": NumHeteroatoms(mol),
        "Number of oxygen atom": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
        "O% Proportion of oxygen atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8) / mol.GetNumAtoms(),
        "Number of carbon atom": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6),
        "C% Proportion of carbon atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6) / mol.GetNumAtoms(),
        "Number of hydrogen atom": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1),
        "H% Proportion of hydrogen atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1) / mol.GetNumAtoms(),
        "Number of rings": CalcNumRings(mol),
        "Number of rotatable bonds": NumRotatableBonds(mol),
        "Oxygen balance": 100 * (2 * sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8) - sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)) / MolWt(mol),
        "MaxPartialCharge": max_partial_charge,
        "MinPartialCharge": min_partial_charge,
        "Molecular volume": CalcExactMolWt(mol),
        "Number of aromatic carbocycle": sum(1 for ring in Chem.GetSymmSSSR(mol) if any(mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring)),
        "Number of aromatic heterocycles": sum(1 for ring in Chem.GetSymmSSSR(mol) if any(mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring)),
        "Number of methyl groups": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and sum(1 for b in atom.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.SINGLE) == 1),
        "Number of hydrogen bond acceptors": CalcNumHBA(mol),
        "Total energy": uff_energy,
        "TPSA": CalcTPSA(mol)
    }

    return descriptors


input_file = r"dataset"
df = pd.read_csv(input_file, delimiter=",", encoding="utf-8")
df.columns = df.columns.str.strip()
print("Column Names:", df.columns)


if "SMILES" not in df.columns:
    raise KeyError("Error: 'SMILES' column not found. Check delimiter and column names.")


molecule_data = []
for index, row in df.iterrows():
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        descriptors = get_molecular_descriptors(mol)

        if descriptors:
            molecule_entry = {
                "SMILES": smiles,
                "Hf(298K)": row["Hf(298K)"],
                "S(298K)": row["S(298K)"],
                "C300": row["C300"],
                "C400": row["C400"],
                "C500": row["C500"],
                "C600": row["C600"],
                "C800": row["C800"],
                "C1000": row["C1000"],
                "C1500": row["C1500"],
            }
            molecule_entry.update(descriptors)
            molecule_data.append(molecule_entry)


output_df = pd.DataFrame(molecule_data)
output_df.to_csv("thermo_molecular_features.csv", index=False)

print("output_dataset")
