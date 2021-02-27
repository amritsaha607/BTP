from collections import defaultdict
import yaml
import os
import glob
import argparse
import wandb
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import BasicModel, E1Model
from models.utils.loss import SeperateLoss
from data.dataset import AreaDataset
from data.collate import collate
from utils.utils import getLossWeights, getLabel, isMode
from utils.decorators import timer
from utils.operations import dictAdd, dictMultiply


# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
parser.add_argument('--cont', type=int, default=None, help='To continue training from a specific epoch, \
        put the last epoch you have trained last time \
        e.g. if you have trained till epoch 50 and want to continue from 51, put 50 here, not 51')
parser.add_argument('--wid', type=str, default=None, help='For continuing runs, provide the id of wandb run')
parser.add_argument(
    '--BEST_VAL_LOSS', 
    type=float, default=None, 
    help="For continuing runs, provide the best validation loss that you've got till now",
)
parser.add_argument('--model', type=int, default=0, help='Model ID')
parser.add_argument('--verbose', type=int, default=1, help='To show training progress or not')
parser.add_argument('--data_factors', type=str, default='f0', 
    help='To spply factor in dataset')
parser.add_argument(
    '--mode', type=str, default='default', 
    help="Mode selects which parameter to predict\
        default - predict all\
        r - predict r"
)
parser.add_argument('--save', type=int, default=100, help='Version of experiment')
args = parser.parse_args()


version = args.version
cont = args.cont
wid = args.wid
verbose = args.verbose
model_ID = args.model
data_factors = args.data_factors
mode = args.mode
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
# CHECKPOINT_DIR = configs['CHECKPOINT_DIR']
# ckpt_dir = os.path.join('checkpoints', version.replace('_', '/'))
ckpt_dir = os.path.join('checkpoints', mode, version.split('_')[0], str(model_ID), version.split('_')[1])
LOSS_WEIGHT_DIR = configs['LOSS_WEIGHT_DIR']
input_key = configs['input_key'] if 'input_key' in configs else 'A_tot'

WANDB_PROJECT_NAME = 'DL Nanophotonics'
WANDB_PROJECT_DIR = '/content/wandb/'

# Checkpoint Directory
if save and (not os.path.exists(ckpt_dir)):
    os.makedirs(ckpt_dir)

# Set seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Dataset
DATA_FACTOR_ROOT = 'configs/data_factors'
data_factors = yaml.safe_load(open(os.path.join(DATA_FACTOR_ROOT, '{}.yml'.format(data_factors))))
data_factors = {key: float(val) for key, val in data_factors.items()}
collate = collate(mode)
train_set = AreaDataset(
    root=train_root,
    formats=['.csv'],
    factors=data_factors,
    input_key=input_key,
    mode=mode,
    shuffle=True,
    batch_size=batch_size,
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
    factors=data_factors,
    input_key=input_key,
    mode=mode,
    shuffle=True,
    batch_size=batch_size,
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)


# Samples
f = glob.glob(os.path.join(train_root, '*', '*.csv'))[0]
n_samples = pd.read_csv(f).values.shape[0]

# Model, loss
# if mode=='default':
#     model_out_dim = 6+2*n_samples
# elif mode=='r':
#     model_out_dim = 2
# elif mode=="eps_sm":
#     model_out_dim = 4
# elif mode=='eps':
#     model_out_dim = 4+2*n_samples
# else:
#     raise AssertionError("Unknown mode [{}] found!".format(mode))
model_out_dim = 2

if isMode(mode, 'e1'):
    E1_CLASSES = [
        'al2o3',
        'sio2',
    ]
    model = E1Model(
        classes = E1_CLASSES,
        model_id = model_ID,
        input_dim = n_samples,
        out_dim = model_out_dim,
    )
    E1_BEST_LOSSES = defaultdict(float)
    for e1_cls in E1_CLASSES:
        E1_BEST_LOSSES[e1_cls] = float('inf')
else:
    model = BasicModel(
        input_dim = n_samples,
        out_dim = model_out_dim,
        model_id = model_ID,
    )
criterion = SeperateLoss()
loss_mode, loss_split_mode = 'mse', 'split_re'
loss_weights = None
if 'loss_weight' in configs:
    loss_mode = 'weighted'
    loss_split_mode += '_weighted'
    loss_weight_cfg = os.path.join(LOSS_WEIGHT_DIR, '{}.yml'.format(configs['loss_weight']))
    loss_weight_cfg = yaml.safe_load(open(loss_weight_cfg))
    loss_weight_cfg = {key: float(val) for key, val in loss_weight_cfg.items()}
    loss_weights = getLossWeights(
        weights_dict=loss_weight_cfg,
        n=n_samples
    )

if cont is not None:
    cont = int(cont)
    latest_ckpt = os.path.join(ckpt_dir, 'latest_{}.pth'.format(cont))
    model.load_state_dict(torch.load(latest_ckpt, map_location=torch.device('cpu')))

if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

# Training Function
@timer
def train(epoch, loader, optimizer, metrics=[], 
            verbose=1, topups=['loss_split_re']):

    """
        epoch : Epoch no
        loader : Training dataloader
        optimizer : Optimizer used
        metrics : metrics to log
    """

    if (mode=='r' or mode=="eps_sm" or mode=='eps') and ('loss_split_re' in topups):
        raise ValueError("You can't add {} topup in {} mode"
            .format("loss_split_re", mode))

    if isMode(mode, 'e1'):
        e1_losses = defaultdict(float)
        e1_loss_counts = defaultdict(float)

    n = len(loader)
    tot_loss, loss_count = 0.0, 0
    if 'loss_split_re' in topups:
        tot_loss_split = None

    model.train()
    for batch_idx, (x, y) in enumerate(loader):

        # y = getLabel(y, mode=mode)

        # For e1 mode, break x into parts
        if isMode(mode, 'e1'):
            x, x_e1 = x

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        if isMode(mode, 'e1'):
            y_pred = model(x, x_e1)
        else:
            y_pred = model(x)

        loss = criterion(y_pred, y, mode=loss_mode, run='train', weights=loss_weights)
        loss.backward()
        optimizer.step()

        if 'loss_split_re' in topups:
            loss_split = criterion(
                y_pred, y, 
                mode=loss_split_mode, run='train', 
                weights=loss_weights
            )
            tot_loss_split = dictAdd([tot_loss_split, loss_split]) if tot_loss_split else loss_split

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

            if isMode(mode, 'e1'):
                e1_losses[f"training_loss_{x_e1}"] += loss.item()
                e1_loss_counts[f"training_loss_{x_e1}"] += 1

        if verbose:
            n_arrow = 50*(batch_idx+1)//n
            progress = "Training - [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
                "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
            )
            print(progress, end='\r')

    print()
    logg = {
        'training_loss': tot_loss/loss_count,
    }

    # Classwise loss of different materials with different e1
    if isMode(mode, 'e1'):
        for key in e1_losses:
            e1_losses[key] /= e1_loss_counts[key]
        logg.update(e1_losses)

    if 'loss_split_re' in topups:
        for key in tot_loss_split:
            tot_loss_split[key] /= loss_count
        logg.update(tot_loss_split)

    return logg


# Validation Function
@timer
def validate(epoch, loader, metrics=[], 
            verbose=1, topups=['loss_split_re']):

    """
        epoch : Epoch no
        loader : Validation dataloader
        metrics : metrics to log
    """

    if isMode(mode, 'e1'):
        e1_losses = defaultdict(float)
        e1_loss_counts = defaultdict(float)

    n = len(loader)
    tot_loss, loss_count = 0.0, 0
    if 'loss_split_re' in topups:
        tot_loss_split = None

    model.eval()
    for batch_idx, (x, y) in enumerate(loader):

        # y = getLabel(y, mode=mode)

        # For e1 mode, break x into parts
        if isMode(mode, 'e1'):
            x, x_e1 = x

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        if isMode(mode, 'e1'):
            y_pred = model(x, x_e1)
        else:
            y_pred = model(x)

        loss = criterion(y_pred, y, mode=loss_mode, run='val', weights=loss_weights)

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

            if isMode(mode, 'e1'):
                e1_losses[f"val_loss_{x_e1}"] += loss.item()
                e1_loss_counts[f"val_loss_{x_e1}"] += 1

        if 'loss_split_re' in topups:
            loss_split = criterion(
                y_pred, y, 
                mode=loss_split_mode, run='val',
                weights=loss_weights
            )
            tot_loss_split = dictAdd([tot_loss_split, loss_split]) if tot_loss_split else loss_split

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        if verbose:
            n_arrow = 50*(batch_idx+1)//n
            progress = "Validation - [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
                "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
            )
            print(progress, end='\r')

    print()
    logg = {
        'val_loss': tot_loss/loss_count,
    }

    # Classwise loss of different materials with different e1
    if isMode(mode, 'e1'):
        for key in e1_losses:
            e1_losses[key] /= e1_loss_counts[key]
        logg.update(e1_losses)

    if 'loss_split_re' in topups:
        for key in tot_loss_split:
            tot_loss_split[key] /= loss_count
        logg.update(tot_loss_split)

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
    run_name = "train_{}_{}".format(version, mode)
    if args.cont is not None:
        wandb.init(id=args.wid, name=run_name, 
            project=WANDB_PROJECT_NAME, dir=WANDB_PROJECT_DIR, resume=True)
    else:
        wandb.init(name=run_name, 
            project=WANDB_PROJECT_NAME, dir=WANDB_PROJECT_DIR)

    wandb.watch(model, log='all')

    config = wandb.config

    if not cont:
        config.version = version
        config.model_ID = model_ID
        config.batch_size = batch_size
        config.n_epoch = n_epoch
        config.train_root = train_root
        config.val_root = val_root
        config.data_factors = args.data_factors
        config.optimizer = optimizer_
        config.lr = learning_rate
        config.weight_decay = weight_decay
        config.adam_eps = adam_eps
        config.amsgrad = adam_amsgrad
        config.CHECKPOINT_DIR = ckpt_dir
        config.scheduler = sch_factor if scheduler is not None else None
        config.cuda = torch.cuda.is_available()
        config.log_interval = 1
    
    BEST_LOSS = float('inf')

    if cont:
        BEST_LOSS = float(args.BEST_VAL_LOSS) if args.BEST_VAL_LOSS is not None else float('inf')
        if scheduler:
            print("Setting up scheduler to continuing state...\n")
            for epoch in range(1, cont+1):
                if epoch%10==0:
                    scheduler.step()

    topups = []
    # topups = ['loss_split_re']

    # Train & Validate over multiple epochs
    start_epoch = cont+1 if cont is not None else 1
    for epoch in range(start_epoch, n_epoch+1):

        print("Epoch {}".format(epoch))

        logg = {}

        logg_train = train(
            epoch,
            train_loader,
            optimizer,
            metrics=[],
            topups=topups,
            verbose=verbose
        )
        logg_val = validate(
            epoch,
            val_loader,
            metrics=[],
            topups=topups,
            verbose=verbose
        )

        logg.update(logg_train)
        logg.update(logg_val)

        if scheduler and epoch%10==0:
            scheduler.step()
            print("\nepoch {}, lr : {}\n"
                    .format(epoch, [param_group['lr'] for param_group in optimizer.param_groups]))

        wandb.log(logg, step=epoch)

        if save:
            if logg['val_loss'] < BEST_LOSS:
                BEST_LOSS = logg['val_loss']
                os.system('rm {}'.format(os.path.join(ckpt_dir, 'best_*.pth')))
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_{}.pth'.format(epoch)))
            # if epoch==n_epoch:
            os.system('rm {}'.format(os.path.join(ckpt_dir, 'latest_*.pth')))
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'latest_{}.pth'.format(epoch)))

            if isMode(mode, 'e1'):
                for e1_cls in E1_CLASSES:
                    if logg[f"val_loss_{e1_cls}"] < E1_BEST_LOSSES[e1_cls]:
                        E1_BEST_LOSSES[e1_cls] = logg[f"val_loss_{e1_cls}"]
                        os.system('rm {}'.format(os.path.join(ckpt_dir, f'e1_best_{e1_cls}_*')))
                        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'e1_best_{e1_cls}_{epoch}.pth'))


if __name__=='__main__':
    run()
