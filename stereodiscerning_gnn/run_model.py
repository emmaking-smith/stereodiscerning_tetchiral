'''
Training the GNN models on the set of features desired.

5-Fold Cross Validation
'''

import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging
import torch

from torch_geometric.loader import DataLoader

from torch_geometric_model_loading import Geometric_Models, train_one_epoch, validate_test_one_epoch
from geometric_dataset import ChiralGNN_Dataset_MorganFP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default='data/processed_data_with_xyz.pickle')
    parser.add_argument('--model-name',
                        type=str,
                        choices = ['GCN', 'GAT', 'SAGE', 'GIN', 'Attentive'],
                        help='Choose one of the available options: GCN, GAT, SAGE, GIN, Attentive.')
    parser.add_argument('--random-seed',
                        type=int)
    parser.add_argument('--epochs',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024)
    parser.add_argument('--hidden_layer_size',
                        type=int,
                        default=128)
    parser.add_argument('--fold',
                        type=int,
                        help='Which fold of the cross validation should be left out in testing?')
    parser.add_argument('--cv',
                        type=int,
                        help='Number of folds in cross validation.',
                        default=5)
    parser.add_argument('--save-dir',
                        type=str)
    return parser.parse_args()

def logger_setup(fold : int, save_dir : str) -> logging.Logger:
    '''
    Returns a specific logger for each fold.
    '''
    log_file = os.path.join(save_dir, 'epoch_loss.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger(f'fold_{fold}')
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()  # Clear existing handlers

    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def main():
    args = init_args()

    df = pd.read_pickle(args.data)
    save_dir = str(os.path.join(args.save_dir, args.model_name, str(args.random_seed), 'fold_' + str(args.fold)))

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # Get the logger ready.
    logger = logger_setup(args.fold, save_dir)

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

    # Set up dataset & datalaoder.

    train_dataset = ChiralGNN_Dataset_MorganFP(df=train_df, radius=2, fpSize=2048)
    val_dataset = ChiralGNN_Dataset_MorganFP(df=val_df, radius=2, fpSize=2048)
    test_dataset = ChiralGNN_Dataset_MorganFP(df=test_df, radius=2, fpSize=2048)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False)

    # Set up model & optimizer.
    input_layer_size = train_dataset[0]['x'].size()[-1]
    model = Geometric_Models(input_layer_size=input_layer_size,
                               hidden_layer_size=args.hidden_layer_size,
                               output_layer_size=1,
                               num_message_passes=3,
                               model_name=args.model_name)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr)
    model.to(device)

    # Train - Val Loop.
    for epoch in range(args.epochs):
        train_losses = train_one_epoch(model, train_dataloader, optimizer)
        logger.debug('Epoch %d | Mean Train Loss : %.3f', epoch, np.mean(train_losses))
        val_losses, _ = validate_test_one_epoch(model, val_dataloader)
        logger.debug('Epoch %d | Mean Val Loss : %.3f', epoch, np.mean(val_losses))

    # Testing.
    test_losses, test_preds = validate_test_one_epoch(model, test_dataloader)
    test_df['pred'] = test_preds
    logger.debug('*** Fold %d *** Mean Test Loss : %.3f', args.fold, np.mean(test_losses))

    # Save out model and preds.
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_state_dict'))

    # To save on space, just save out the true.npy and pred.npy
    ground_truth = [1 if x == '+' else 0 for x in test_df['Rotation']]
    predictions = [float(x[0]) for x in test_df['pred']]
    np.save(os.path.join(save_dir, 'true.npy'), ground_truth)
    np.save(os.path.join(save_dir, 'pred.npy'), predictions)

if __name__ == '__main__':
    main()
