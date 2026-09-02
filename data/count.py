import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops, rdMolDescriptors

smiles_df = pd.read_csv('local_dataset', header=None, names=['SMILES'])

stable_molecules_count = 0
radicals_count = 0
functional_group_counts = {}

functional_groups = {
    'Alcohol': '[OH]',
    'Carboxylic Acid': 'C(=O)O',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Ketone': '[CX3](=O)[#6]',
    'Ether': 'C-O-C',
    'Ester': '[CX3](=O)[OX2H1]',
    'Alkene': 'C=C',
    'Aromatic': 'c1ccccc1',  # 苯环
}

for index, row in smiles_df.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    if Descriptors.NumRadicalElectrons(mol) > 0:
        radicals_count += 1
    else:
        stable_molecules_count += 1


    for fg_name, fg_smarts in functional_groups.items():
        fg_pattern = Chem.MolFromSmarts(fg_smarts)
        if mol.HasSubstructMatch(fg_pattern):
            if fg_name in functional_group_counts:
                functional_group_counts[fg_name] += 1
            else:
                functional_group_counts[fg_name] = 1


print(f'Stable molecules: {stable_molecules_count}')
print(f'Radicals: {radicals_count}')
print('Functional group counts:')
for fg_name, count in functional_group_counts.items():
    print(f'{fg_name}: {count}')