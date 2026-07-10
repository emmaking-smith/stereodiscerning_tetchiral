'''
Saving out the differences in enantiomers in
GIN vs SAGE to understand why one is so much better
than the other.
'''

import pandas as pd
import argparse
import os
import torch
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric_model_loading import Geometric_Models
from geometric_dataset import ChiralGNN_Dataset_MorganFP

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dict',
                        type=str,
                        default='../data/SAGE_model_state_dict')
    return parser.parse_args()

class Latent_Embeddings:
    def __init__(self, model_name : str):
        '''
        Saving out the latent embeddings
        after each CONV layer in the model.
        '''
        self.model_name = model_name
        self.batch_name = None # set this before each forward pass
        self.hooks = []
    def make_hook(self, conv_idx : int):
        '''
        Function to register hooks.
        '''
        def hook(module, input, output):
            filename = f"Latent_Embeddings/{self.model_name}_layer{conv_idx}_{self.batch_name}.pt"
            torch.save(output.detach().cpu(), filename)
            print(f"Saved: {filename}")
        return hook
    def register(self, model):
        '''
        Register hooks for each conv layer.
        '''
        if self.model_name != 'ATTEN':
            model.gnn.convs[1].aggr_module.register_forward_hook(self.make_hook(1))
            model.gnn.convs[2].aggr_module.register_forward_hook(self.make_hook(2))
        else:
            model.gnn.atom_convs[1].aggr_module.register_forward_hook(self.make_hook(1))
            model.gnn.mol_conv.aggr_module.register_forward_hook(self.make_hook(2))


def main():
    args = init_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setting up the dataframe.
    df = pd.DataFrame(
        {
            'SMILES' : [
                'O[C@](C)([H])N', # stereocenter idx = 1
                'O[C@@](C)([H])N', # stereocenter idx = 1
            ],

            'Rotation' : [
                1,
                0,
            ]
        }
    )

    molecule_names = [
        'S-1-aminoethan-1-ol',
        'R-1-aminoethan-1-ol'
        ]

    # Set up the dataset and loader.
    dataset = ChiralGNN_Dataset_MorganFP(df=df, radius=2, fpSize=2048)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Set up model.
    input_layer_size = dataset[0]['x'].size()[-1]

    SAGE = Geometric_Models(input_layer_size=input_layer_size,
                             hidden_layer_size=128,
                             output_layer_size=1,
                             num_message_passes=3,
                             model_name='SAGE')

    Path('Latent_Embeddings/').mkdir(exist_ok=True, parents=True)
    SAGE.load_state_dict(torch.load(args.state_dict, map_location=device))

    # Saving out the embeddings of SAGE.
    SAGE_Embeddings = Latent_Embeddings(model_name='SAGE')
    SAGE_Embeddings.register(SAGE)
    for idx, batch in enumerate(loader):
        SAGE_Embeddings.batch_name = molecule_names[idx]
        preds = SAGE(batch)

if __name__ == '__main__':
    main()