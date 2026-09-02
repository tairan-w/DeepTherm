
import pandas as pd


file_path = "dataset"
df = pd.read_csv(file_path, delimiter=",")


print("Column Names:", df.columns)


df.columns = df.columns.str.strip()

# Check if 'SMILES' exists in the DataFrame after stripping spaces
if "SMILES" not in df.columns:
    print("Possible column name issue. Available columns:", df.columns)
    exit()

# Iterate through rows safely
for index, row in df.iterrows():
    smiles = row["SMILES"]
    print(smiles)  # Process as needed