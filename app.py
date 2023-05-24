"""
Streamlit app for the AI4Chemistry project.
Made by Loïc Bassement, Rémi Schlama, Gabor Dienes, Leander Choudhury.
Spring 2023.
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# from model import *
# from utils import download_model_uspto480k, download_mcule_molecules, take_random_subset_mols, check_rxn
from streamlit_ketcher import st_ketcher
from rdkit.Chem.QED import qed

st.title("MolPrice")
st.write("MolPrice is a tool to predict the price of a molecule based on the structure.")

n_mols = st.sidebar.number_input('Number of molecules from catalogue', min_value=1, 
                        max_value=500, value=10, step=1, 
                        help='Number of molecules to select from Mcule database')

random_seed = st.sidebar.number_input('Random seed', min_value=1, 
                        max_value=100, value=33, step=1,
                        help='Random seed to select molecules from Mcule database')



tab1, tab2 = st.tabs(['Input', 'Output'])

with tab1:
    st.write('''### Draw your molecule of interest''')
    st.write('''Draw the molecule you want to predict the price from the Mcule database and click **Apply**''')
    molecule = st_ketcher(value='', key='molecule')

    if st.button('Apply'):
        st.snow()