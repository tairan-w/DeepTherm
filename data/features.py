import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import BondType


def get_atomic_features(mol):
    atomic_features = {
        "Atom type": [],
        "Formal charge": [],
        "Chirality": [],
        "Hydrogen atoms": [],
        "Hybridization": [],
        "Aromaticity": [],
        "Atomic mass": [],
        "Bonds": [],
        "Electronegativity": [],
        "Ionization energy": [],
        "Partial charge": [],
        "Donor/acceptor": [],
        "Saturation degree": [],
        "Atomic volume": []
    }

    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)

    for atom in mol.GetAtoms():
        atomic_features["Atom type"].append(atom.GetAtomicNum())
        atomic_features["Formal charge"].append(atom.GetFormalCharge())
        atomic_features["Chirality"].append(atom.GetChiralTag())
        atomic_features["Hydrogen atoms"].append(atom.GetTotalNumHs())
        atomic_features["Hybridization"].append(atom.GetHybridization())
        atomic_features["Aromaticity"].append(int(atom.GetIsAromatic()))
        atomic_features["Atomic mass"].append(atom.GetMass() / 100)
        atomic_features["Bonds"].append(len(atom.GetBonds()))
        atomic_features["Electronegativity"].append(Descriptors.MolMR(mol))
        atomic_features["Ionization energy"].append(Descriptors.MolLogP(mol))
        atomic_features["Partial charge"].append(crippen_contribs[atom.GetIdx()][0])
        atomic_features["Donor/acceptor"].append(1 if atom.GetAtomicNum() in [7, 8] else 0)
        atomic_features["Saturation degree"].append(
            sum(1 for b in atom.GetBonds() if b.GetBondType() == BondType.SINGLE))
        atomic_features["Atomic volume"].append(atom.GetAtomicNum() / 100)

    return atomic_features



def get_bond_features(mol):
    bond_features = {
        "Bond type": [],
        "Conjugated": [],
        "In ring": [],
        "Stereo": [],
        "Rotatable bond": [],
        "Ring size": [],
        "Hydrogen bonding": [],
        "Local environment": []
    }

    for bond in mol.GetBonds():
        bond_features["Bond type"].append(str(bond.GetBondType()))
        bond_features["Conjugated"].append(int(bond.GetIsConjugated()))
        bond_features["In ring"].append(int(bond.IsInRing()))
        bond_features["Stereo"].append(str(bond.GetStereo()))
        bond_features["Rotatable bond"].append(int(bond.GetBondType() == BondType.SINGLE and not bond.IsInRing()))
        bond_features["Ring size"].append(rdMolDescriptors.CalcNumRings(mol) if bond.IsInRing() else 0)
        bond_features["Hydrogen bonding"].append(
            int(bond.GetBeginAtom().GetAtomicNum() in [7, 8] or bond.GetEndAtom().GetAtomicNum() in [7, 8]))
        bond_features["Local environment"].append(bond.GetBeginAtom().GetAtomicNum() + bond.GetEndAtom().GetAtomicNum())

    return bond_features


# Read input file
input_file = r"dataset"
df = pd.read_csv(input_file, delimiter=",")
df.columns = df.columns.str.strip()
print("Column Names:", df.columns)

# Verify if "SMILES" column exists
if "SMILES" not in df.columns:
    print("Error: 'SMILES' column not found. Check delimiter and column names.")
    exit()

# Process each molecule
molecule_data = []
for index, row in df.iterrows():
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:  # Check if the molecule was successfully parsed
        atomic_features = get_atomic_features(mol)
        bond_features = get_bond_features(mol)

        # Flatten atomic and bond features into a consistent column format
        atomic_feature_dict = {f"Atom_{key}_{i}": val for key, values in atomic_features.items() for i, val in
                               enumerate(values)}
        bond_feature_dict = {f"Bond_{key}_{i}": val for key, values in bond_features.items() for i, val in
                             enumerate(values)}

        # Combine all features into a single dictionary
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
            "C1500": row["C1500"]
        }
        molecule_entry.update(atomic_feature_dict)
        molecule_entry.update(bond_feature_dict)

        molecule_data.append(molecule_entry)

output_df = pd.DataFrame(molecule_data)
output_df.to_csv("thermo_features_output.csv", index=False)

print("output_dataset")
