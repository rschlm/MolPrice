
"""
This file is used to clone the Mcule database, preprocess it and save a corrected version of it.
"""


import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import os

def get_db(link) -> None:
    """get the database and unzip it"""
    os.system(f"curl {link} -o mcule_purchasable_in_stock_prices.csv.gz")
    os.system("gzip -d mcule_purchasable_in_stock_prices.csv.gz")

def check_smiles(smiles) -> bool:
    """Check if the SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def correct_df() -> None:
    """Correct the dataset and save it in a new file"""
    df = pd.read_csv("mcule_purchasable_in_stock_prices.csv", delimiter=",", low_memory=False)
    # Remove invalid SMILES compounds
    print("Removing invalid SMILES compounds")
    df['valid_smiles'] = df['SMILES'].apply(check_smiles)
    valid_df = df[df['valid_smiles'] == True]
    # remove compunds with price > 600 USD
    print("Removing compounds with price > 600 USD")
    valid_df = valid_df[valid_df["price 1 (USD)"] < 600]
    valid_df.to_csv('datasets/mcule_purchasable_in_stock_prices_valid_smiles.csv', index=False)
    print("File saved in /datasets folder")

if __name__ == "__main__":
    print("Please enter the link of the database ")
    link = input("> ")
    get_db(link)
    correct_df()
