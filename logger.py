import wandb
import argparse
import os
import pickle
from tqdm import tqdm


# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--theme', type=str, default="train", help="train/val/predict")
parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
parser.add_argument('--model', type=int, default=0, help='Model ID')
parser.add_argument(
    '--mode', type=str, default='default', 
    help="Mode selects which parameter to predict\
        default - predict all\
        r - predict r\
        r_e1 - predict r with e1 class\
        r_e1_e2 - predict r with e1 & e2 classes"
)
parser.add_argument('--domain', type=int, default=0, 
    help="Pipeline domain\
        0 -> model predicts (r1, r2)\
        1 -> model predicts (r1/r2, r2-r1)")
args = parser.parse_args()


# Extract parameters from argument parser
theme = args.theme
version = args.version
model_ID = args.model
mode = args.mode
domain = args.domain


# Initiate global variables
WANDB_PROJECT_NAME = 'DL Nanophotonics'
WANDB_PROJECT_DIR = '/content/wandb/'


# Get File Location Root
pickle_root = os.path.join('cache', theme, f'dom_{domain}', mode, version.split('_')[0])


# Training
if theme.lower() == 'train':
    pickle_name = os.path.join(pickle_root, str(model_ID), "{}.pkl".format(version.split('_')[1]))
    [config, loggs] = pickle.load(open(pickle_name, 'rb'))

    print(config)
    run_name = "train_{}_{}_dom{}".format(version, mode, domain)

    wandb.init(
        name=run_name,
        config=config,
        project=WANDB_PROJECT_NAME, 
        dir=WANDB_PROJECT_DIR,
    )
    for step_no, logg in tqdm(enumerate(loggs)):
        wandb.log(logg, step=step_no+1)

