# import libraries
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepchem.splits import RandomSplitter

import torch.nn.functional as F
from torch.nn import GRU
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, MLP, global_add_pool
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
)

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils import smiles2graph

# setting random seeds for reproductibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

class MCULE_DATA(InMemoryDataset):
    # path to the data
    path_to_data = '/datasets/mcule_purchasable_in_stock_prices_230324_RKoqmy_valid_smiles.csv'

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['mcule_purchasable_in_stock_prices_230324_RKoqmy_valid_smiles.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # load raw data from a csv file
        df = pd.read_csv(self.raw_paths[0])
        smiles = df['SMILES'].values.tolist()
        target = df['price 1 (USD)'].values.tolist()

        # Convert SMILES into graph data
        print('Converting SMILES strings into graphs...')
        data_list = []
        for i, smi in enumerate(tqdm(smiles)):

            # get graph data from SMILES
            graph = smiles2graph(smi)

            # convert to tensor and pyg data
            x = torch.tensor(graph['node_feat'], dtype=torch.long)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        # save data
        torch.save(self.collate(data_list), self.processed_paths[0])



# create dataset
dataset = MCULE_DATA('./datasets/').shuffle()

# Normalize target to mean = 0 and std = 1.
mean = dataset.data.y.mean()
std = dataset.data.y.std()
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean.item(), std.item()

# split data
splitter = RandomSplitter()
train_idx, valid_idx, test_idx = splitter.split(dataset,frac_train=0.7, frac_valid=0.1, frac_test=0.2)
train_dataset = dataset[list(train_idx)]
valid_dataset = dataset[list(valid_idx)]
test_dataset = dataset[list(test_idx)]