"""
Streamlit app for the AI4Chemistry project.
Made by Loïc Bassement, Rémi Schlama, Gabor Dienes, Leander Choudhury.
Spring 2023.
"""


import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from model_nn import MPNN, MCULE_DATA
import pytorch_lightning as pl
import numpy as np
import random
from deepchem.splits import RandomSplitter

import streamlit as st
from streamlit_ketcher import st_ketcher

st.title("MolPrice")
st.write("MolPrice is a tool to predict the price of a molecule based on the structure.")

n_mols = st.sidebar.number_input('Number of molecules from catalogue', min_value=1, max_value=500, value=10, step=1,
                                 help='Number of molecules to select from Mcule database')

random_seed = st.sidebar.number_input('Random seed', min_value=1, max_value=100, value=33, step=1,
                                      help='Random seed to select molecules from Mcule database')


tab1, tab2 = st.tabs(['Input', 'Output'])

with tab1:
    st.write('### Draw your molecule of interest')
    st.write('Draw the molecule you want to predict the price from the Mcule database and click **Apply**')
    molecule = st_ketcher(value='', key='molecule')

    if st.button('Apply'):
        # Process the input molecule

        smiles = str(molecule)
        graph = smiles2graph(smiles)
        x = torch.tensor(graph['node_feat'], dtype=torch.long)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Load the model and make predictions
        model = MPNN(hidden_dim=80,
                     out_dim=1,
                     std=std,
                     train_data=train_dataset,
                     valid_data=valid_dataset,
                     test_data=test_dataset,
                     lr=0.001,
                     batch_size=64)
        model.load_state_dict(torch.load('gnn_model.pt'))
        model.eval()
        with torch.no_grad():
            output = model(data)

        # Display the output
        predicted_price = output.item()
        st.write(f'Predicted Price: {predicted_price}')

with tab2:
    # Add the desired output components
    pass
