import numpy as np
import pickle
import random
import time
from typing import Tuple, List, Dict  

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from baller2vec_forked.settings import *
from baller2vec_forked.baller2vec import Baller2Vec, Baller2VecSeq2Seq
from baller2vec_forked.baller2vec_dataset import Baller2VecDataset


### 
# Init datasets
###

def get_train_valid_test_gameids(opts, seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Reads in the train/valid/test game ids from disk, if they exist.
    Otherwise, creates them using `train_valid_prop` and `train_prop` from the .yaml file, and writes to disk.
    
    Returns lists of game ids for each case, e.g. ['0021500295'] for the first element in the tuple (train_gameids).
    """

    try:
        with open("train_gameids.txt") as f:
            train_gameids = f.read().split()

        with open("valid_gameids.txt") as f:
            valid_gameids = f.read().split()

        with open("test_gameids.txt") as f:
            test_gameids = f.read().split()

    except FileNotFoundError:
        print("No {train/valid/test}_gameids.txt files found. Generating new ones.")

        gameids = list(set([np_f.split("_")[0] for np_f in os.listdir(GAMES_DIR)]))
        gameids.sort()
        np.random.seed(seed)
        np.random.shuffle(gameids)
        n_train_valid = int(opts["train"]["train_valid_prop"] * len(gameids))
        n_train = int(opts["train"]["train_prop"] * n_train_valid)
        train_valid_gameids = gameids[:n_train_valid]

        train_gameids = train_valid_gameids[:n_train]
        valid_gameids = train_valid_gameids[n_train:]
        test_gameids = gameids[n_train_valid:]
        train_valid_test_gameids = {
            "train": train_gameids,
            "valid": valid_gameids,
            "test": test_gameids,
        }
        for (train_valid_test, gameids) in train_valid_test_gameids.items():
            with open(f"{train_valid_test}_gameids.txt", "w") as f:
                for gameid in gameids:
                    f.write(f"{gameid}\n")

    np.random.seed(seed)

    return (train_gameids, valid_gameids, test_gameids)


def init_datasets(opts : Dict, info_dir : str, games_dir :str , seed : int) -> Tuple:
    """
    Arguments:
        opts: From the configs file.  E.g. gives the size of the MLP in each layer.

    Returns:
        The object returned is a Tuple, whose names are given below.

            train_dataset,
            train_loader,
            valid_dataset,
            valid_loader,
            test_dataset,
            test_loader

        Each *_loader is an Iterable (e.g., we can do `for tensors in valid_loader`) of dictionaries
            whose keys are given below:
        
            dict_keys(['player_idxs', 'player_xs', 'player_ys', 'player_hoop_sides', 'ball_xs', 'ball_ys', 
            'ball_zs', 'game_contexts', 'events', 'player_trajs', 'ball_trajs', 'score_changes', 'ball_locs'])

            and whose values, on a quick check, are tensors giving information across a minibatch of timesteps.        

    """
    baller2vec_info = pickle.load(open(f"{info_dir}/baller2vec_info.pydict", "rb"))
    n_player_ids = len(baller2vec_info["player_idx2props"])
    filtered_player_idxs = set()
    for (player_idx, player_props) in baller2vec_info["player_idx2props"].items():
        if "playing_time" not in player_props:
            continue

        if player_props["playing_time"] < opts["train"]["min_playing_time"]:
            filtered_player_idxs.add(player_idx)

    (train_gameids, valid_gameids, test_gameids) = get_train_valid_test_gameids(opts, seed)

    dataset_config = opts["dataset"]
    dataset_config["gameids"] = train_gameids
    dataset_config["N"] = opts["train"]["train_samples_per_epoch"]
    dataset_config["starts"] = []
    dataset_config["mode"] = "train"
    dataset_config["n_player_ids"] = n_player_ids
    dataset_config["filtered_player_idxs"] = filtered_player_idxs
    train_dataset = Baller2VecDataset(**dataset_config)
    # TODO: What does this DataLoader do?
    # TODO: Alcorn's original file had worker_init_fn=worker_init_fn as an argument;
    # This seemed to introduce some worker-specific random seed.  Figure out what the
    # implications are of ignoring this.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    N = opts["train"]["valid_samples"]
    samps_per_gameid = int(np.ceil(N / len(valid_gameids)))
    starts = []
    for gameid in valid_gameids:
        y = np.load(f"{games_dir}/{gameid}_y.npy")
        max_start = len(y) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["gameids"] = np.repeat(valid_gameids, samps_per_gameid)
    dataset_config["N"] = len(dataset_config["gameids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "valid"
    valid_dataset = Baller2VecDataset(**dataset_config)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    samps_per_gameid = int(np.ceil(N / len(test_gameids)))
    starts = []
    for gameid in test_gameids:
        y = np.load(f"{games_dir}/{gameid}_y.npy")
        # The `chunk_size` (105) is the default Hz (25) multiplied by the yaml-specified
        # `secs` for the training dataset (4.2)
        max_start = len(y) - train_dataset.chunk_size
        gaps = max_start // samps_per_gameid
        starts.append(gaps * np.arange(samps_per_gameid))

    dataset_config["gameids"] = np.repeat(test_gameids, samps_per_gameid)
    dataset_config["N"] = len(dataset_config["gameids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "test"
    test_dataset = Baller2VecDataset(**dataset_config)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    if opts["train"]["task"] != "event":
        opts["n_seq_labels"] = train_dataset.n_score_changes + 1
    else:
        opts["n_seq_labels"] = len(baller2vec_info["event2event_idx"])

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )

### 
# Init Model
###

def init_model(opts : Dict, train_dataset : Baller2VecDataset) -> Baller2Vec:
    """
    Arguments:
        opts: From the configs file.  E.g. gives the size of the MLP in each layer.
    """ 
    model_config = opts["model"]
    # Add one for the generic player.
    model_config["n_player_ids"] = train_dataset.n_player_ids + 1
    model_config["seq_len"] = train_dataset.chunk_size // train_dataset.hz - 1
    model_config["n_player_labels"] = train_dataset.player_traj_n ** 2 
    # TODO: `n_player_labels` is number of bins for representing player loc in (x,y) plane?
    # Or is it the dimensionality for the player embedding?
    if opts["train"]["task"] == "seq2seq":
        model = Baller2VecSeq2Seq(**model_config)
    else:
        model_config["n_seq_labels"] = opts["n_seq_labels"]  
        # `n_seq_labels` is interpreted as `num_score_changes` for task=="player_traj"
        model_config["n_players"] = train_dataset.n_players
        if opts["train"]["task"] == "ball_loc":
            model_config["n_ball_labels"] = (
                train_dataset.n_ball_loc_y_bins * train_dataset.n_ball_loc_x_bins
            )
        else:
            model_config["n_ball_labels"] = train_dataset.ball_traj_n ** 3 # Number of bins for representing ball loc in (x,y,z) plane?

        model = Baller2Vec(**model_config)

    return model

### 
# Train
###

def get_preds_labels(
    tensors: Dict[str, torch.Tensor],
    model : Baller2Vec, 
    task : str, 
    device : torch.device, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        tensors: a dict which has has keys 
            dict_keys(['player_idxs', 'player_xs', 'player_ys', 'player_hoop_sides', 
            'ball_xs', 'ball_ys', 'ball_zs', 'game_contexts', 
            'events', 'player_trajs', 'ball_trajs', 'score_changes', 'ball_locs'])

            For example, the `tensors["player_trajs"]` tensor is a (T_slice, J) tensor containing 
            the bin labels for the J players over the course of T_slice.  The size of T_slice is
            determined in the Baller2VecDataset class by the configurable number of seconds for a minibatch
            (Alcorn used 4.2) and the configurable sampling rate (Alcorn used hz=5).  
            The default Hz is 25 Hz, since the raw data is recorded at 25 Hz, so some of the data is skipped.

            e.g.
                tensor([[72, 47, 51, 46, 59, 59, 60, 70, 59, 51],
                        [72, 47, 51, 36, 49, 59, 71, 70, 59, 40],
                        [72, 47, 40, 36, 60, 59, 60, 70, 59, 40],
                        [...]

        task: e.g. "player_traj".  Specified in configs.  Determines the training objective 
        device: has type torch.device, e.g.
            device(type='cpu')
    Results:
        preds: A tensor of shape (T_slice x J, bin_size_per_court_dimension**2) and with dtype=reals.
            Some real-valued model output (perhaps the log of an unnormalized prob?) presumably reflecting
            the potential that each entity j at each time t was in each of the court bins in terms of differences
            in (x,y) location. 
            
                e.g. tensor([[-0.9522, -0.3529, -0.7651,  ..., -0.7389, -0.4591, -0.9369],
                            [-0.9058, -0.4411, -0.8383,  ..., -0.7552, -0.5721, -0.7922],
                            [-0.8677, -0.4146, -0.8075,  ..., -0.6958, -0.5371, -0.8496],
                            [...]
        labels:  A tensor of shape (T_slice x J, )  and with dtype=ints.  Presumably reflecting the true
            difference in (x,y) location. 
                e.g. tensor([51, 59, 59, 59, 47, 51, 70, 60, 46, 72,  .... ]
    """
    if ("player" in task) or ("ball" in task):
        player_trajs = tensors["player_trajs"].flatten()
        n_player_trajs = len(player_trajs)
        if task == "player_traj":
            # I think that for each timestep, the label (player_trajs) gives the binned loction of the player on the court.
            # For example,
            #   In [15]: tensors["player_trajs"]
            #   Out[15]: 
            #    tensor([[48, 60, 60, 59, 60, 59, 60, 60, 60, 61],
            #            [48, 60, 60, 59, 60, 59, 60, 60, 59, 60],
            #            [59, 60, 60, 59, 60, 59, 60, 60, 70, 71],
            labels = player_trajs.to(device)
            preds = model(tensors)["player"][:n_player_trajs]

        else:
            if task == "ball_loc":
                labels = tensors["ball_locs"].flatten().to(device)
            else:
                labels = tensors["ball_trajs"].flatten().to(device)

            preds = model(tensors)["ball"][n_player_trajs:][: len(labels)]
    elif (task == "event") or (task == "score"):
        if task == "event":
            labels = tensors["events"].flatten().to(device)
        else:
            labels = tensors["score_changes"].flatten().to(device)

        preds = model(tensors)["seq_label"][-model.seq_len :]
    elif task == "seq2seq":
        # Randomly choose which team to encode.
        start_stops = {"enc": (0, 5), "dec": (5, 10)}
        if random.random() < 0.5:
            start_stops = {"enc": (5, 10), "dec": (0, 5)}

        (start, stop) = start_stops["dec"]
        labels = tensors["player_trajs"][:, start:stop].flatten().to(device)
        preds = model(tensors, start_stops)[: len(labels)]

    return (preds, labels)



def train_model(
    opts,
    model: Baller2Vec,
    train_loader,
    valid_loader,
    test_loader,
    device : torch.device,
    n_epochs: int, 
    job_dir: str, 
    verbose: bool = True, 
):
    """
    n_epochs was hardcoded to 650 in the Alcorn code.
    """
    task = opts["train"]["task"]
    patience = opts["train"]["patience"]

    # Initialize optimizer.
    if ((task == "event") or (task == "score")) and (opts["train"]["prev_model"]):
        old_job = opts["train"]["prev_model"]
        old_job_dir = f"{EXPERIMENTS_DIR}/{old_job}"
        model.load_state_dict(torch.load(f"{old_job_dir}/best_params.pth"))
        train_params = [params for params in model.event_classifier.parameters()]
    else:
        train_params = [params for params in model.parameters()]

    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Continue training on a prematurely terminated model.
    try:
        model.load_state_dict(torch.load(f"{job_dir}/best_params.pth"))

        try:
            optimizer.load_state_dict(torch.load(f"{job_dir}/optimizer.pth"))
        except ValueError:
            print("Old optimizer doesn't match.")

    except FileNotFoundError:
        pass

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    no_improvement = 0
    for epoch in range(n_epochs):
        print(f"\nepoch: {epoch}", flush=True)

        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            n_valid = 0
            for valid_tensors in valid_loader:
                #breakpoint()
                # Skip bad sequences.
                if len(valid_tensors["player_idxs"]) < model.seq_len:
                    continue

                (preds, labels) = get_preds_labels(valid_tensors, model, task, device)
                loss = criterion(preds, labels)
                total_valid_loss += loss.item()
                n_valid += 1

            probs = torch.softmax(preds, dim=1)
            (probs, preds) = probs.max(1)
            if verbose:
                print(probs.view(model.seq_len, model.n_players), flush=True)
                print(preds.view(model.seq_len, model.n_players), flush=True)
                print(labels.view(model.seq_len, model.n_players), flush=True)

            total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            torch.save(optimizer.state_dict(), f"{job_dir}/optimizer.pth")
            torch.save(model.state_dict(), f"{job_dir}/best_params.pth")

            test_loss_best_valid = 0.0
            with torch.no_grad():
                n_test = 0
                for test_tensors in test_loader:
                    # Skip bad sequences.
                    if len(test_tensors["player_idxs"]) < model.seq_len:
                        continue

                    (preds, labels) = get_preds_labels(test_tensors, model, task, device)
                    loss = criterion(preds, labels)
                    test_loss_best_valid += loss.item()
                    n_test += 1

            test_loss_best_valid /= n_test

        elif no_improvement < patience:
            no_improvement += 1
            if no_improvement == patience:
                if ((task == "event") or (task == "score")) and (
                    opts["train"]["prev_model"]
                ):
                    print("Now training full model.")
                    train_params = [params for params in model.parameters()]
                    optimizer = optim.Adam(
                        train_params, lr=opts["train"]["learning_rate"]
                    )
                else:
                    print("Reducing learning rate.")
                    for g in optimizer.param_groups:
                        g["lr"] *= 0.1

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        print(f"total_valid_loss: {total_valid_loss}")
        print(f"best_valid_loss: {best_valid_loss}")
        print(f"test_loss_best_valid: {test_loss_best_valid}")

        model.train()
        total_train_loss = 0.0
        n_train = 0
        start_time = time.time()
        # TODO: Figure out how to get insight into, and control,
        # how many training tensors are passed through.
        for (train_idx, train_tensors) in enumerate(train_loader):
            if train_idx % 1000 == 0:
                print(f"Train idx={train_idx}", end="\r")

            # Skip bad sequences.
            if len(train_tensors["player_idxs"]) < model.seq_len:
                continue

            optimizer.zero_grad()
            (preds, labels) = get_preds_labels(train_tensors, model, task, device)
            loss = criterion(preds, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        print(f"epoch_time: {epoch_time:.2f}", flush=True)

