"""
Program for training the model in HPC.
"""


# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit.Chem import Draw
import wandb


def load_data():
    """
    Load the data and preprocess it.
    """
    df = pd.read_csv("datasets/mcule_purchasable_in_stock_prices_230324_RKoqmy_valid_smiles.csv")
    truncated_df = df[df["price 1 (USD)"] < 600] # Remove molecules with price > 600 USD


def split_data():
    """
    Split the data into training, validation and test sets.
    """
    pass


def run_simple_mlr():
    """
    Run a simple multiple linear regression on the data.
    """
    pass

