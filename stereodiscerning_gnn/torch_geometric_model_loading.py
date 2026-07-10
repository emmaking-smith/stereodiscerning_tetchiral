'''
Setting up various torch geometric models.

UPDATE 17 SEP: Works on dummy main() code.
'''

import numpy as np
import pandas as pd
import torch
import torch_geometric

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, GIN, AttentiveFP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Geometric_Models(torch.nn.Module):
    def __init__(self,
                 input_layer_size : int,
                 hidden_layer_size : int,
                 output_layer_size : int,
                 num_message_passes : int,
                 model_name : str):
        super().__init__()

        self.model_name = model_name

        graph_dict = {
            'GCN' : GCN,
            'GAT' : GAT,
            'SAGE' : GraphSAGE,
            'GIN' : GIN,
            'Attentive' : AttentiveFP
        }

        if model_name == 'Attentive':
            self.gnn = graph_dict[model_name](in_channels=input_layer_size,
                                              hidden_channels=hidden_layer_size,
                                              out_channels=output_layer_size,
                                              num_layers=num_message_passes,
                                              edge_dim=1,
                                              num_timesteps=3)
        else:
            self.gnn = graph_dict[model_name](in_channels=input_layer_size,
                                              hidden_channels=hidden_layer_size,
                                              out_channels=output_layer_size,
                                              num_layers=num_message_passes)

    def forward(self, batch : torch_geometric.data.batch.Batch) -> torch.tensor:
        graph_embeddings = self.gnn(x=batch['x'],
                                    edge_index=batch['edge_index'],
                                    edge_attr=batch['edge_attr'],
                                    batch=batch['batch'])
        if self.model_name != 'Attentive':
            graph_embeddings = global_mean_pool(graph_embeddings, batch['batch'])
        output = torch.nn.Sigmoid()(graph_embeddings).reshape([-1])
        return output

    def calculate_loss(self, predicted : torch.tensor,
                       true : torch.tensor) -> torch.tensor:
        return torch.nn.BCELoss()(predicted, true)

def train_one_epoch(model : torch.nn.Module,
                    dataloader : torch_geometric.loader.DataLoader,
                    optimizer : torch.optim) -> list:
    '''
    Training the generic model (Geometric Models) one epoch.
    '''
    model.train()
    losses = []
    for batch in dataloader:
        batch = batch.to(device)
        true_values = batch['y'].to(device).float()
        optimizer.zero_grad()
        preds = model(batch)
        loss = model.calculate_loss(preds, true_values)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
    return losses

def validate_test_one_epoch(model : torch.nn.Module,
                            dataloader : torch_geometric.loader.DataLoader,
                            ) -> tuple[list, list]:
    '''
    Validating or testing on a single epoch.
    '''
    model.eval()
    losses = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            true_values = batch['y'].to(device).float()
            preds = model(batch)
            loss = model.calculate_loss(preds, true_values)
            losses.append(loss.cpu().detach().numpy())
            predictions.append(preds.cpu().detach().numpy())
    return losses, predictions