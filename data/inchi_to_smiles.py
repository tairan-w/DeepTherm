import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi


def inchi_to_smiles(inchi_str):
    mol = inchi.MolFromInchi(inchi_str)
    if mol:
        return Chem.MolToSmiles(mol)
    else:
        return None


def main():
    input_file = 'local_dataset'
    output_file = 'output_dataset'

    df = pd.read_csv(input_file)

    if df.shape[1] < 2:
        print("Input file must have at least two columns")
        return

    df['SMILES'] = df.iloc[:, 1].apply(inchi_to_smiles)

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()