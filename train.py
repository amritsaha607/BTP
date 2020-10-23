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
parser.add_argument('--save', type=int, default=100, help='Version of experiment')
args = parser.parse_args()


version = args.version
save = args.save
cfg_path = os.path.join('configs/{}.yml'.format(version.replace('_', '/')))
configs = yaml.safe_load(open(cfg_path))

random_seed = int(configs['random_seed'])
batch_size = int(configs['batch_size'])
n_epoch = int(configs['n_epoch'])
train_root = configs['train_root']
val_root = configs['val_root']
optimizer_ = configs['optimizer']
learning_rate = float(configs['lr'])
weight_decay = float(configs['weight_decay'])
adam_eps = float(configs['adam_eps'])
adam_amsgrad = bool(configs['adam_amsgrad'])
CHECKPOINT_DIR = configs['CHECKPOINT_DIR']
ckpt_dir = os.path.join('checkpoints', version.replace('_', '/'))

# Checkpoint Directory
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Set seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Dataset
train_set = AreaDataset(
    root=train_root,
    formats=['.csv'],
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)
val_set = AreaDataset(
    root=val_root,
    formats=['.csv'],
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)

# Model, loss
model = BasicModel()
criterion = nn.MSELoss()
if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

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

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        n_arrow = 50*(batch_idx+1)//n
        progress = "Epoch {} (Training) [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
            epoch, "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
        )
        print(progress, end='\r')

    print()
    logg = {
        'training_loss': tot_loss/loss_count,
    }
    return logg


# Validation Function
@timer
def validate(epoch, loader, metrics=[]):

    """
        epoch : Epoch no
        loader : Validation dataloader
        metrics : metrics to log
    """

    n = len(loader)
    tot_loss, loss_count = 0.0, 0

    model.eval()
    for batch_idx, (x, y) in enumerate(loader):

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        n_arrow = 50*(batch_idx+1)//n
        progress = "Epoch {} (Validation) [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
            epoch, "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
        )
        print(progress, end='\r')

    print()
    logg = {
        'val_loss': tot_loss/loss_count,
    }
    return logg



def run():

    # Optimizer
    if optimizer_=='adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=adam_eps,
            amsgrad=adam_amsgrad
        )

    # Scheduler (optional)
    scheduler = None
    if 'scheduler' in configs:
        sch_factor = configs['scheduler']
        lr_lambda = lambda epoch: sch_factor**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize wandb
    run_name = "train_{}".format(version)
    wandb.init(name=run_name, project="DL Nanophotonics", dir='/content/wandb/')
    wandb.watch(model, log='all')

    config = wandb.config

    config.version = version
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.train_root = train_root
    config.val_root = val_root
    config.optimizer = optimizer_
    config.lr = learning_rate
    config.weight_decay = weight_decay
    config.adam_eps = adam_eps
    config.amsgrad = adam_amsgrad
    config.CHECKPOINT_DIR = CHECKPOINT_DIR
    config.scheduler = sch_factor if scheduler is not None else None
    config.cuda = torch.cuda.is_available()
    config.log_interval = 1
    

    # Train & Validate over multiple epochs
    for epoch in range(1, n_epoch+1):
        logg = {}
        
        logg_train = train(epoch, train_loader, optimizer, metrics=[])
        logg_val = validate(epoch, val_loader, metrics=[])

        logg.update(logg_train)
        logg.update(logg_val)

        if scheduler and epoch%10==0:
            scheduler.step()
            print("\nepoch {}, lr : {}\n".format(epoch, [param_group['lr'] for param_group in optimizer.param_groups]))

        wandb.log(logg, step=epoch)

        if save and epoch%save==0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'epoch_{}.pth'.format(epoch)))


if __name__=='__main__':
    run()