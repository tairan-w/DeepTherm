import pandas as pd
from rdkit import Chem

functional_groups = {
    'Vinyl carbon': '[CX3]=[CX2]',
    'Allenic group': '[C]=[C]=[C]',
    'Acetylenic carbon': '[CX2]#C',
    'Aromatic ring': 'c1ccccc1',
    'Non-aromatic cyclic group': '[R2]',
    'Polycyclic aromatic rings': 'c1ccc2ccccc2c1',

    'Methyl Radical': '[CH3*]',
    'Alkenyl Radical': '[C]=[C*]',
    'Alkynyl Radical': '[C]#[C*]',
    'Aryl Radical': '[c*]',

    'Hydroxyl group': '[OX2H]',
    'Ether group': '[CX4][OX2][CX4]',
    'Peroxide group': '[OX2][OX2]',
    'Epoxide group': '[C]1[O][C]1',
    'Carbonyl group': '[CX3]=O',
    'Ester group': '[CX3](=O)[OX2][CX4]',
    'Anhydride group': '[CX3](=O)[OX2][CX3](=O)',
    'Phenol group': '[cH][OX2H]',
    'Carboxyl group': '[CX3](=O)[OX2H]',
    'Peroxycarboxylic acid group': '[CX3](=O)[OX2][OX2H]',
    'Enol group': '[CX3](=C[OX2H])',
    'Aldehyde group': '[CX3H1](=O)',
    'Ketone group': '[CX3](=O)[#6]',

    'Hydroxyl radical': '[OH*]',
    'Peroxy radical': '[O][O*]',
    'Alkoxy radical': '[O*][CX4]',
    'Carbonyl radical': '[C](=O)*'
}

def detect_functional_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fg_detected = {key: 0 for key in functional_groups.keys()}
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            fg_detected[name] = 1
    return fg_detected

example_smiles = "CC(=O)Oc1ccccc1C(=O)O"
print(detect_functional_groups(example_smiles))


file_path = 'dataset'
df = pd.read_csv(file_path)


total_fg_counts = {key: 0 for key in functional_groups.keys()}


df['Functional Groups'] = df['SMILES'].apply(detect_functional_groups)


for _, row in df.iterrows():
    fg_count = row['Functional Groups']
    if fg_count:
        for fg, presence in fg_count.items():
            total_fg_counts[fg] += presence

w
print("官能团的总数量统计：")
for fg, count in total_fg_counts.items():
    print(f"{fg}: {count}")
