import pandas as pd
import matplotlib.pyplot as plt


input_file = "dataset"
df = pd.read_csv(input_file, delimiter=",", encoding="utf-8")
df.columns = df.columns.str.strip()
print("Column Names:", df.columns)

df_numeric = df.drop(columns=["SMILES"])

# Compute correlation matrix
correlation_matrix = df_numeric.corr(method='pearson')

correlation_matrix.to_csv("correlation_matrix.csv")
print("Correlation matrix saved to correlation_matrix.csv")
