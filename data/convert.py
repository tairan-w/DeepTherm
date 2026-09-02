import pandas as pd
from rdkit import Chem

def read_smiles(file_path):
    df = pd.read_csv(file_path)
    smiles_list = df['SMILES'].tolist()
    return smiles_list

def standardize_smiles(smiles_list):
    standardized_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            standardized_smiles.append(canonical_smiles)
    return standardized_smiles


def check_duplicates(smiles_list):
    smiles_set = set(smiles_list)
    duplicates = [smiles for smiles in smiles_set if smiles_list.count(smiles) > 1]
    return duplicates


def write_smiles(file_path, smiles_list):
    df = pd.DataFrame(smiles_list, columns=['SMILES'])
    df.to_csv(file_path, index=False)


def main(input_file_path, output_file_path):
    smiles_list = read_smiles(input_file_path)
    standardized_smiles = standardize_smiles(smiles_list)
    write_smiles(output_file_path, standardized_smiles)
    duplicates = check_duplicates(standardized_smiles)
    if duplicates:
        print(f'Found duplicates: {duplicates}')
    else:
        print('No duplicates found.')

input_file_path = "local_dataset"
output_file_path = 'output_dataset'
main(input_file_path, output_file_path)