'''
Predicts the optical rotation + or -
of SMILES strings. Takes in a text file
where each line has a new compound on it.
'''

import argparse
import os

import pandas as pd
import torch

from pathlib import Path
from rdkit import Chem

from torch_geometric.loader import DataLoader
from torch_geometric_model_loading import Geometric_Models
from torch_geometric.data import Dataset, Data
from smiles_to_geometric_data import Chiral_MFP_Graph

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        '-f',
                        type=str,
                        help='Path to text (.txt) file of SMILES strings.')
    parser.add_argument('--save_name',
                        '-s',
                        type=str,
                        default='./predictions.csv',
                        help='Name of generated save file.')
    return parser.parse_args()

class Test_SMILES_Dataset(Dataset):
    def __init__(self, smiles_list : list[str], radius=2, fpSize=2048):
        super().__init__()
        self.smiles_list = smiles_list
        self.processing = Chiral_MFP_Graph(radius=radius, fpSize=fpSize)

    def len(self) -> int:
        return len(self.smiles_list)

    def get(self, idx : int):
        smiles = self.smiles_list[idx]
        edge_tuples, node_info, bond_types = self.processing.smiles_to_MFP_graph(smiles=smiles)

        idx_data = Data(x=node_info,
                        edge_index=edge_tuples.t().contiguous(),
                        edge_attr=bond_types
                        )

        return idx_data

def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def main():
    # Setup.
    args = init_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = os.path.split(args.save_name)[0]
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Loading the SMILES.
    smiles = []
    with open(args.file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            smiles.append(line.replace('\n', ''))
    canonical_smiles = [canonicalize_smiles(x) for x in smiles]

    # Create the dataloader.
    loader = DataLoader(Test_SMILES_Dataset(smiles_list=canonical_smiles),
                        batch_size=1,
                        shuffle=False)

    # Initialize the model.
    model = Geometric_Models(input_layer_size=2048,
                             hidden_layer_size=128,
                             output_layer_size=1,
                             num_message_passes=3,
                             model_name='Attentive')
    model_weights_path = 'data/best_model_state_dict'
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)

    # Run the predictions.
    model_outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            model_outputs.append(preds.cpu().detach().numpy())

    predicted_rotation = ['+' if x >= 0.5 else '-' for x in model_outputs]

    # Save out the predictions.
    df = pd.DataFrame()
    df['SMILES'] = smiles
    df['canonical_smiles'] = canonical_smiles
    df['model_output'] = model_outputs
    df['predicted_rotation'] = predicted_rotation
    df.to_csv(args.save_name)

if __name__ == '__main__':
    main()