"""
Streamlit app for the AI4Chemistry project.
Made by Loïc Bassement, Rémi Schlama, Gabor Dienes, Leander Choudhury.
Spring 2023.
"""

import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from model_nn import MPNN, MCULE_DATA
import streamlit as st
from streamlit_ketcher import st_ketcher

st.title("MolPrice")
st.write("MolPrice is a tool to predict the price of a molecule based on the structure.")

# Load the pretrained model
model = MPNN(hidden_dim=80, out_dim=1, std=42, train_data=None,
             valid_data=None, test_data=None, lr=0.001, batch_size=64)
model.load_state_dict(torch.load('gnn_model.pt'))
model.eval()


tab1, tab2 = st.tabs(['Input', 'About'])

with tab1:
    st.write('### Draw your molecule of interest')
    st.write('Draw the molecule you want to predict the price from the Mcule database and click **Apply**')
    molecule = st_ketcher(value='', key='molecule')

    if st.button('Apply'):

        smiles = str(molecule)
        graph = smiles2graph(smiles)
        x = torch.tensor(graph['node_feat'], dtype=torch.long)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        with torch.no_grad():
            output = model(data)

        predicted_price = output.item()
        st.write(f'Predicted Price: {predicted_price:.4f}')
