import pandas as pd
from rdkit import Chem


df = pd.read_csv("datasets/coprinet.csv")

mols = []


for i in range(len(df)):
    mols.append(Chem.MolFromSmiles(df["SMILES"][i]))

print(mols[0])