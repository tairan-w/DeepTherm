
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Function to determine if a molecule is close-shelled or an open-shelled radical
def classify_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "invalid"  # Mark invalid SMILES
        return "close-shelled" if Descriptors.NumRadicalElectrons(mol) == 0 else "open-shelled"
    except Exception as e:
        return "invalid"

data_file = "dataset"
data = pd.read_csv(data_file)

data["SMILES"] = data["SMILES"].astype(str).fillna("")


data = data[data["SMILES"] != ""]


data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(len(data_shuffled) * 0.9)
training_data = data_shuffled[:train_size]
test_data = data_shuffled[train_size:]


training_data["Classification"] = training_data["SMILES"].apply(classify_molecule)
test_data["Classification"] = test_data["SMILES"].apply(classify_molecule)


training_data = training_data[training_data["Classification"] != "invalid"]
test_data = test_data[test_data["Classification"] != "invalid"]


training_close_shelled = training_data[training_data["Classification"] == "close-shelled"]
training_open_shelled = training_data[training_data["Classification"] == "open-shelled"]


test_close_shelled = test_data[test_data["Classification"] == "close-shelled"]
test_open_shelled = test_data[test_data["Classification"] == "open-shelled"]


training_data.to_csv("training_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
training_close_shelled.to_csv("training_close_shelled.csv", index=False)
training_open_shelled.to_csv("training_open_shelled.csv", index=False)
test_close_shelled.to_csv("test_close_shelled.csv", index=False)
test_open_shelled.to_csv("test_open_shelled.csv", index=False)
