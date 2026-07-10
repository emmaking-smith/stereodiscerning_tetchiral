'''
Testing ChiENN (https://github.com/gmum/ChiENN/tree/master) with our data.

Please follow their instructions for installation and conda environment.
'''

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import torch
import math
from typing import Iterator
from tqdm import tqdm
from itertools import chain

from dataclasses import dataclass
from torch_geometric.graphgym.optim import SchedulerConfig, OptimizerConfig
import torch_geometric.graphgym.register as register
from torch.nn import Parameter
from torch.optim import AdamW, Optimizer
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric import seed_everything

from chienn import collate_with_circle_index
from chienn.data.featurize import smiles_to_data_with_circle_index
from chienn.model.chienn_model import ChiENNModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed',
                        type=int)
    parser.add_argument('--fold',
                        type=int,
                        help='Which fold of the cross validation should be left out in testing?')
    parser.add_argument('--cv',
                        type=int,
                        help='Number of folds in cross validation.',
                        default=5)
    parser.add_argument('--save-dir',
                        type=str)
    parser.add_argument('--epochs',
                        type=int,
                        default=100)
    return parser.parse_args()

def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer, # adamW
                           base_lr=cfg.optim.base_lr, # 0.0001
                           weight_decay=cfg.optim.weight_decay, # 1e-5
                           momentum=cfg.optim.momentum) # 0.9
@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
    num_warmup_epochs: int = 10
    train_mode: str = 'custom'
    eval_period: int = 1

def new_scheduler_config(cfg): # Note: no reduce factor, min_lr, or schedule_patience so attributes removed
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode, eval_period=cfg.train.eval_period)

@register.register_optimizer('adamW')
def adamW_optimizer(params: Iterator[Parameter], base_lr: float,
                   weight_decay: float) -> AdamW:
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

@register.register_scheduler('cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler

class EKS_Dataset(Dataset):
    '''
    Dataset for preprocessing the dataframes (df).
    '''
    def __init__(self, df):
        super(EKS_Dataset, self).__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Following directions in
        https://github.com/gmum/ChiENN/tree/master
        '''
        rotation = self.df.loc[idx, 'Rotation']
        if rotation == '+':
            rotation = 1.
        else:
            rotation = 0.
        return torch.tensor([idx]), torch.tensor([rotation])

    def EKS_collate_fn(self, batch):
        idxs = []
        rotation_labels = []

        for item in batch:
            idxs.append(int(item[0]))
            rotation_labels.append(item[1])

        data_list = [self.df.loc[i, 'ChiENN_Data'] for i in idxs]

        # Add the label data.
        for i, data in enumerate(data_list):
            if data is not None:
                data.y = rotation_labels[i]

        # Remove None entries.
        data_list = self.remove_Nones_in_list(data_list)

        new_batch = collate_with_circle_index(data_list, k_neighbors=3)
        return new_batch

    def remove_Nones_in_list(self, data_list : list) -> list:
        return list(filter(lambda x: x is not None, data_list))

def main():
    args = init_args()
    df = pd.read_pickle('../data/processed_data_with_xyz.pickle')
    chienn_data = []

    for i in tqdm(range(len(df))):
        try:
            chienn_data.append(smiles_to_data_with_circle_index(df.loc[i, 'SMILES']))
        except:
            chienn_data.append(None)

    df['ChiENN_Data'] = chienn_data

    cfg_files = [
        'experiments/configs/models/common.yaml',
        'experiments/configs/models/ChiENN/ChiENN.yaml',
        'experiments/configs/datasets/bace.yaml'
    ]  # bace yaml files.

    set_cfg(cfg)
    for file in cfg_files:
        cfg.set_new_allowed(True)
        cfg.merge_from_file(file)

    print('cfg')
    print(cfg)
    print()

    print('Device:', device)
    print()

    np.random.seed(args.random_seed)
    idxs = np.array(df.index)
    np.random.shuffle(idxs)
    idxs = np.array_split(idxs, args.cv)

    save_dir = str(os.path.join(args.save_dir, str(args.random_seed), 'fold_' + str(args.fold)))

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Splitting into training, validation, and testing.
    test_idxs = idxs[args.fold]
    train_test_idxs = idxs.copy()
    del train_test_idxs[args.fold]

    # Every other idx not in the test fold becomes part of the
    # training or validation fold.
    train_test_idxs = np.concatenate(train_test_idxs).reshape([-1])
    train_idxs = train_test_idxs[0:len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1))]
    val_idxs = train_test_idxs[len(train_test_idxs) - int(np.floor(len(train_test_idxs) * 0.1)) : ]

    assert set(train_idxs).intersection(test_idxs) == set()
    assert set(train_idxs).intersection(val_idxs) == set()
    assert set(val_idxs).intersection(test_idxs) == set()

    train_df = df.loc[train_idxs].reset_index(drop=True)
    val_df = df.loc[val_idxs].reset_index(drop=True)
    test_df = df.loc[test_idxs].reset_index(drop=True)

    del df

    # Seed everything.
    seed_everything(cfg.seed) # seed = 0

    # Initializing the datasets.
    train_dataset = EKS_Dataset(df=train_df)
    val_dataset = EKS_Dataset(df=val_df)
    test_dataset = EKS_Dataset(df=test_df)

    # Creating the GraphDataModule needed for training.
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.EKS_collate_fn)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfg.train.batch_size,
                            shuffle=True,
                            collate_fn=val_dataset.EKS_collate_fn)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=cfg.train.batch_size,
                             shuffle=False,
                             collate_fn=test_dataset.EKS_collate_fn)

    # Define model and optimizers.
    model = ChiENNModel(k_neighbors=3)
    model.to(device)

    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

    # Training. Max epochs = 100.
    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_losses = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            out = model(data).reshape([-1])
            result = torch.nn.BCELoss()(out, data.y)
            result.backward()

            optimizer.step()
            scheduler.step()
            train_losses.append(result.detach().cpu().numpy())

        # Validation check.
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                data = batch.to(device)
                out = model(data).reshape([-1])
                result = torch.nn.BCELoss()(out, data.y)
                val_losses.append(result.cpu().detach().numpy())

    # Testing and model weights saving.
    model.eval()
    test_preds = []
    ys = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch.to(device)
            out = model(data).reshape([-1])
            test_preds.append(out.cpu().detach().numpy())
            ys.append(data.y.cpu().detach().numpy())

    test_predictions = list(chain.from_iterable(test_preds))
    ground_truth = list(chain.from_iterable(ys))

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict'))

    # To save on space, just save out the true.npy and pred.npy
    np.save(os.path.join(save_dir, 'true.npy'), ground_truth)
    np.save(os.path.join(save_dir, 'pred.npy'), test_predictions)

if __name__ == '__main__':
    main()





