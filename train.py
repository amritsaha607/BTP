import yaml
import os
import glob
import argparse
import wandb
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import BasicModel
from data.dataset import AreaDataset
from data.collate import collate
from utils.decorators import timer


# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
args = parser.parse_args()


version = args.version
cfg_path = os.path.join('configs/{}.yml'.format(version.replace('_', '/')))
configs = yaml.safe_load(open(cfg_path))

random_seed = int(configs['random_seed'])
batch_size = int(configs['batch_size'])
n_epoch = int(configs['n_epoch'])
train_root = configs['train_root']
val_root = configs['val_root']
learning_rate = float(configs['lr'])
weight_decay = float(configs['weight_decay'])
adam_eps = float(configs['adam_eps'])
adam_amsgrad = bool(configs['adam_amsgrad'])


# Set seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
try:
    torch.cuda.manual_seed(random_seed)
except:
    pass


# Dataset
dataset = AreaDataset(
    root=train_root,
    formats=['.csv'],
)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)


# Model, loss, optimizer
model = BasicModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    eps=adam_eps,
    # amsgrad=adam_amsgrad
)


# Training Function

@timer
def train(epoch, loader, optimizer, metrics=[]):

    """
        epoch : Epoch no
        loader : Training dataloader
        optimizer : Optimizer used
        metrics : metrics to log
    """

    n = len(loader)
    tot_loss, loss_count = 0.0, 0

    model.train()
    for batch_idx, (x, y) in enumerate(loader):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1
        # tot_loss += loss.item()
        # loss_count += 1

        n_arrow = 50*(batch_idx+1)//n
        progress = "Epoch {} [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
            epoch, "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
        )
        print(progress, end='\r')

    print(loss_count)
    logg = {
        'training_loss': tot_loss/loss_count,
    }
    return logg



def run():
    for epoch in range(1, n_epoch+1):
        train(epoch, dataloader, optimizer)


if __name__=='__main__':
    run()