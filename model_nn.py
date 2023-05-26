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

batch_size=64

class MPNN(pl.LightningModule):
    def __init__(self, hidden_dim, out_dim,
                train_data, valid_data, test_data,
                std, batch_size=32, lr=1e-3):
        super().__init__()
        self.std = std  # std of data's target
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.lr = lr
        # Initial layers
        self.atom_emb = AtomEncoder(emb_dim=hidden_dim)
        self.bond_emb = BondEncoder(emb_dim=hidden_dim)
        # Message passing layers
        nn = MLP([hidden_dim, hidden_dim*2, hidden_dim*hidden_dim])
        self.conv = NNConv(hidden_dim, hidden_dim, nn, aggr='mean')
        self.gru = GRU(hidden_dim, hidden_dim)
        # Readout layers
        self.mlp = MLP([hidden_dim, int(hidden_dim/2), out_dim])

    def forward(self, data, mode="train"):

        # Initialization
        x = self.atom_emb(data.x)
        h = x.unsqueeze(0)
        edge_attr = self.bond_emb(data.edge_attr)
        
        # Message passing
        for i in range(3):
            m = F.relu(self.conv(x, data.edge_index, edge_attr))  # send message and aggregation
            x, h = self.gru(m.unsqueeze(0), h)  # node update
            x = x.squeeze(0)

        # Readout
        x = global_add_pool(x, data.batch)
        x = self.mlp(x)

        return x.view(-1)
        
    def training_step(self, batch, batch_idx):
        # Here we define the train loop.
        out = self.forward(batch, mode="train")
        loss = F.mse_loss(out, batch.y)
        #print(batch.y.shape)
        self.log("Train loss", loss, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Define validation step. At the end of every epoch, this will be executed
        out = self.forward(batch, mode="valid")
        loss = F.mse_loss(out * self.std, batch.y * self.std)  # report MSE
        #print(f'validation{batch.y.shape}')
        self.log("Valid MSE", loss, batch_size=self.batch_size)
        
    def test_step(self, batch, batch_idx):
        # What to do in test
        out = self.forward(batch, mode="test")
        loss = F.mse_loss(out * self.std, batch.y * self.std)  # report MSE
        self.log("Test MSE", loss, batch_size=self.batch_size)

    def configure_optimizers(self):
        # Here we configure the optimization algorithm.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

class MCULE_DATA(InMemoryDataset):
    # path to the data
    path_to_data = '/datasets/mcule_purchasable_in_stock_prices_valid_smiles.csv'

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['mcule_purchasable_in_stock_prices_valid_smiles.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # load raw data from a csv file
        df = pd.read_csv(self.raw_paths[0])
        smiles = df['SMILES'][0:10000].values.tolist()
        target = df['price 1 (USD)'][0:10000].values.tolist()

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

"""For the creation of file to work, you need to:
- create a folder called 'raw' in the dataset folder containing your raw data: here the file cleaned of bad smiles
- it will generate a folder called processed with the data
- put random seed 0 to be consistent in the splitting
- Argument of the MCULE_DATA class is the folder where you can find the raw folder
"""

dataset = MCULE_DATA('./datasets/').shuffle()
# split data
splitter = RandomSplitter()
train_idx, valid_idx, test_idx = splitter.split(dataset,frac_train=0.7, frac_valid=0.1, frac_test=0.2)
train_dataset = dataset[list(train_idx)]
valid_dataset = dataset[list(valid_idx)]
test_dataset = dataset[list(test_idx)]


mean = dataset.data.y.mean()
std = dataset.data.y.std()

#training the model

gnn_model = MPNN(
    hidden_dim=80,
    out_dim=1,
    std=std,
    train_data=train_dataset,
    valid_data=valid_dataset,
    test_data=test_dataset,
    lr=0.001,
    batch_size=64
)

trainer = pl.Trainer(
    max_epochs = 200,
)

trainer.fit(
    model=gnn_model,
)

results = trainer.test(ckpt_path="best")
test_mse = results[0]["Test MSE"]
test_rmse = test_mse ** 0.5
print(f"\nMPNN model performance: RMSE on test set = {test_rmse:.4f}.\n")

