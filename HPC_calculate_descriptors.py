"""
Calculate Morgan fp and Mordred descriptors, make ad dataframe and save it to a csv file.
"""


from mordred import Calculator, descriptors
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
import pandas as pd
import numpy as np


def _filter_invalid_descriptors(value):
    """
    Filter out the invalid descriptors.
    """
    isvalid = value.startswith("invalid value") or value.startswith("missing")
    return isvalid


def calculate_Mordred_descriptors(df) -> pd.DataFrame:
    """
    Calculate the Mordred descriptors for each molecule in the dataframe.
    """
    calc = Calculator(descriptors, ignore_3D=False)
    df_X = calc.pandas(df["Mol"])

    df_X = df_X.astype(float).fillna(0)

    return df_X


def calculate_Morgan_fp(df) -> pd.DataFrame:
    """
    Calculate the Morgan fingerprint for each molecule in the dataframe.
    """
    df["morgan_fp"] = df["Mol"].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048))
    return df


def generate_molecules(df) -> pd.DataFrame:
    """
    Generate molecules from the SMILES strings.
    """
    df["Mol"] = df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    return df


if __name__ == "__main__":
    df = pd.read_csv(
        "datasets/mcule_purchasable_in_stock_prices_valid_smiles.csv")
    print("Dataset loaded")

    # randomize the dataset
    df = df.sample(frac=1, random_state=0)

    df = df[:1000]

    new_df = generate_molecules(df)
    print("Molecules generated")

    df_features = calculate_Mordred_descriptors(new_df)
    df_features["price 1 (USD)"] = df["price 1 (USD)"]
    # df_features = calculate_Morgan_fp(new_df)
    print("Descriptors calculated")
    df_features.to_csv("datasets/mordred_descriptors.csv", index=False)
