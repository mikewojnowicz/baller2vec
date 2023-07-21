import numpy as np
import pickle
import random
import sys
import time
from typing import Tuple, List, Dict  
import yaml

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from baller2vec_forked.settings import *
from baller2vec_forked.baller2vec import Baller2Vec, Baller2VecSeq2Seq
from baller2vec_forked.baller2vec_dataset import Baller2VecDataset



###
# MAIN
###

### CONFIGS
SEED = 2010
N_EPOCHS = 650
EXPERIMENTS_DIR = "/Users/mwojno01/Repos/baller2vec_forked/data/single_game_splits/experiments"
JOB = "tiny" # TODO: In Alcorn code, JOB=$(date +%Y%m%d%H%M%S)


### CODE 
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
except IndexError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
task = opts["train"]["task"]
patience = opts["train"]["patience"]

### Initialize datasets.
(
    train_dataset,
    train_loader,
    valid_dataset,
    valid_loader,
    test_dataset,
    test_loader,
) = init_datasets(opts)

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
    N_EPOCHS,  
    train_loader,
    valid_loader,
    test_loader,
)


