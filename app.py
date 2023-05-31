"""
Streamlit app for the AI4Chemistry project.
Made by Loïc Bassement, Rémi Schlama, Gabor Dienes, Leander Choudhury.
Spring 2023.
"""

import streamlit as st
import torch
from ogb.utils import smiles2graph
from model_nn import MPNN
from torch_geometric.data import Data
from streamlit_chemistry import st_chemistry

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
    molecule = st_chemistry(value='', key='molecule')

    if st.button('Apply'):
        # Process the input molecule
        smiles = molecule['smiles']
        graph = smiles2graph(smiles)
        x = torch.tensor(graph['node_feat'], dtype=torch.long)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Load the model and make predictions
        model = MPNN()
        model.load_state_dict(torch.load('gnn_model.pth'))
        model.eval()
        with torch.no_grad():
            output = model(data)

        # Display the output
        predicted_price = output.item()
        st.write(f'Predicted Price: {predicted_price}')

with tab2:
    # Add the desired output components
    pass
