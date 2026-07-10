'''
Pytorch Geometric Dataset
'''

import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData

from smiles_to_geometric_data import Chiral_MFP_Graph

class ChiralGNN_Dataset_MorganFP(Dataset):
    def __init__(self, df, radius=2, fpSize=2048):
        super().__init__()
        self.df = df
        self.processing = Chiral_MFP_Graph(radius=radius, fpSize=fpSize)

    def len(self) -> int:
        return len(self.df)

    def get(self, idx : int):
        smiles = self.df.loc[idx, 'SMILES']
        rotation = self.df.loc[idx, 'Rotation']
        if rotation == '+':
            rotation = 1
        else:
            rotation = 0
        edge_tuples, node_info, bond_types = self.processing.smiles_to_MFP_graph(smiles=smiles)

        idx_data = Data(x=node_info,
                        edge_index=edge_tuples.t().contiguous(),
                        edge_attr=bond_types,
                        y=torch.tensor([rotation]))
        return idx_data