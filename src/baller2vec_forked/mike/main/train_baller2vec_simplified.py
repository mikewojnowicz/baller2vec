import os 
import sys
import yaml

import torch

from baller2vec_forked.mike.train_model import (
    init_datasets,
    init_model,
    train_model
)



###
# MAIN
###

### CONFIGS
DATA_DIR="/Users/mwojno01/Repos/baller2vec_forked/data/"
GAMES_DIR=f"{DATA_DIR}/across_game_splits/games"
EXPERIMENTS_DIR=f"{DATA_DIR}/across_game_splits/experiments"
INFO_DIR = f"{DATA_DIR}/across_game_splits/info"

JOB = "tiny" # TODO: In Alcorn code, JOB=$(date +%Y%m%d%H%M%S)
JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

SEED = 2010
N_EPOCHS = 3

### CODE 
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)


try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
except IndexError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

### Initialize datasets.
(
    train_dataset,
    train_loader,
    valid_dataset,
    valid_loader,
    test_dataset,
    test_loader,
) = init_datasets(opts, INFO_DIR, GAMES_DIR, SEED)

### Initialize model.

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specify the first GPU device
    print("Using CUDA")
else:
    device = torch.device("cpu")  # Use CPU if CUDA is not available
    print("Using CPU")

model = init_model(opts, train_dataset).to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params}")

### Train model.
train_model(
    opts,
    model,
    train_loader,
    valid_loader,
    test_loader, 
    device,
    N_EPOCHS,  
    JOB_DIR,
)


