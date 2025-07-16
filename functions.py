### This is a file for storing all of the functions that are being used in my project
# Imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import Dataset
from copy import deepcopy
import torchmetrics
from torchinfo import summary


import numpy as np
import math
import random

import os
import pickle
import pathlib
from pathlib import Path
import io 
from typing import Union

import time

import LDAQ
import nidaqmx
from nidaqmx.system import System


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.stats import qmc
from scipy.signal import welch

def list_functions():
    """ 
    This function contains a list of all functions in this file.
    Use it when you forget the names of the functions 
    """
    functions = ["ImpactPredictor - ML model",
                 "ImpactDataset - Make dataset",
                 "make_dataloaders - Make dataloaders form the dataset",
                 "train_model - Function to train the model",
                 "save_model_weights - Saves the model weights",
                 "load_model_weights - Loads model weights",
                 "gather_data - Function for gathering data via LDAQ (open NI-MAX)",
                 "plot_data - Plots data form the dictionary of gather_data()",
                 "plot_loss_curves - Plots the loss curves of the traing proces",
                 "plot_accuracy_curves - Plots the test accuracy curves",
                 "open_pkl_dict - Opens a specified .pkl file and loads it as a dict",
                 "live_inference - Performes live inference directly after gathering data",
                 "save_model_results - Saves the dictionary of the model results to .pkl",
                 "plot_inference - Plots predictions of the model vs. the labels",
                 "print_model_summary - Print the parameters of the model",
                 "prediction_heatmap - Makes a heatmap based on the location of predictions",
                 "generate_latin_hypercube_points - Generate random sample points with LHC distribution (x,y,F)",
                 "run_random_sampling - Runs the sampling with gather_data and samples across the generated random points",
                 "interpolate_nan_in_directory - goes into a datadir and lineary interpolates to get rid of nan values",
                 "cluster_location_labels - remaps the location labels to the neares grid point",
                 "model_factory - creates a fresh instance of the CNN model",
                 "dataset_factory - creates a fresh dataset",
                 "save_lowpass_filtered_data - saves the filtered data dicts",
                 "apply_lowpass - plots the filtered data vs. the source data",
                 "plot_fft - plots the fft of the desired signal",
                 "filter_array - filters a np array and returns the filtered signal",
                 "estimate_first_mode - estimates the first own frequency",
                 "FFT_array - perform fast forier transform on array"]
    for i in functions:
        print(i)


### PYTORCH MODEL PART - CONTAINS ALL FUNCTIONS THAT USE PYTORCH AND ARE CONNECTED TO THE MODEL

# Model definition

class ResBlock(nn.Module):
    """
    A simple residual block: Conv -> BN -> ReLU -> Conv -> BN, with skip connection.
    Downsamples via MaxPool after addition.
    """
    def __init__(self, channels, kernel_size, pool_size, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.bn2   = nn.BatchNorm1d(channels)
        self.pool  = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + res
        out = F.relu(out)
        out = self.dropout(out)
        out = self.pool(out)
        return out

class ImpactPredictor(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        sequence_length: int,
        cnn_filters: list,
        kernel_size: int,
        pool_size: int,
        force_reg: bool,
        loc_class: bool,
        grid_resolution: tuple,
        latent_dim: int,
        dropout_rate: float,
        head_hidden_dim: int,
        head_hidden_layers: int
    ):
        """
        Multi-task model to predict {impact} location (classification or regression)
        and force (classification or regression).

        Args:
            num_sensors (int): Number of input sensor channels.
            sequence_length (int): Time-series length.
            cnn_filters (list): Output channels for each Conv1D layer.
            kernel_size (int): Kernel size for Conv1D.
            pool_size (int): MaxPool1D kernel size.
            force_reg (bool): If True, use regression head for force.
            loc_class (bool): If True, use classification head for location.
            grid_resolution (tuple): (nx, ny) bins for loc-class.
            latent_dim (int): Dimensionality of shared FC layer.
            dropout_rate (float): Dropout probability.
            head_hidden_dim (int): Hidden dim for each head layer.
            head_hidden_layers (int): Number of hidden layers in each head.
        """
        super().__init__()
        self.force_reg = force_reg
        self.loc_class = loc_class

        # --- CNN encoder with Residual Blocks ---
        layers = []
        in_ch = num_sensors
        seq_len = sequence_length
        for out_ch in cnn_filters:
            # 1x1 conv if channel dims change
            if in_ch != out_ch:
                layers.append(nn.Conv1d(in_ch, out_ch, 1))
            layers.append(ResBlock(out_ch, kernel_size, pool_size, dropout_rate))
            seq_len //= pool_size
            in_ch = out_ch
        self.cnn_encoder = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        # --- Shared FC ---
        self.fc_shared = nn.Sequential(
            nn.Linear(in_ch, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # --- Location head: classification ---
        num_bins = grid_resolution[0] * grid_resolution[1]
        loc_cls_layers = []
        in_dim = latent_dim
        for _ in range(head_hidden_layers):
            loc_cls_layers += [
                nn.Linear(in_dim, head_hidden_dim),
                nn.BatchNorm1d(head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = head_hidden_dim
        loc_cls_layers.append(nn.Linear(in_dim, num_bins))
        self.loc_class_head = nn.Sequential(*loc_cls_layers)

        # --- Location head: regression ---
        loc_reg_layers = []
        in_dim = latent_dim
        for _ in range(head_hidden_layers):
            loc_reg_layers += [
                nn.Linear(in_dim, head_hidden_dim),
                nn.BatchNorm1d(head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = head_hidden_dim
        loc_reg_layers.append(nn.Linear(in_dim, 2))
        self.location_head = nn.Sequential(*loc_reg_layers)

        # --- Force head: classification ---
        force_cls_layers = []
        in_dim = latent_dim
        for _ in range(head_hidden_layers):
            force_cls_layers += [
                nn.Linear(in_dim, head_hidden_dim),
                nn.BatchNorm1d(head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = head_hidden_dim
        force_cls_layers.append(nn.Linear(in_dim, 3))
        self.force_head = nn.Sequential(*force_cls_layers)

        # --- Force head: regression ---
        force_reg_layers = []
        in_dim = latent_dim
        for _ in range(head_hidden_layers):
            force_reg_layers += [
                nn.Linear(in_dim, head_hidden_dim),
                nn.BatchNorm1d(head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5)
            ]
            in_dim = head_hidden_dim
        force_reg_layers.append(nn.Linear(in_dim, 1))
        self.force_reg_head = nn.Sequential(*force_reg_layers)

    def forward(self, x: torch.Tensor):
        # x: (N, T, C) -> (N, C, T)
        x = x.permute(0, 2, 1)
        x = self.cnn_encoder(x)
        x = self.adaptive_pool(x).squeeze(-1)
        latent = self.fc_shared(x)

        loc_out = self.loc_class_head(latent) if self.loc_class else self.location_head(latent)
        force_out = self.force_reg_head(latent) if self.force_reg else self.force_head(latent)

        return loc_out, force_out




# Dataset calss definition
class ImpactDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        loc_class: bool,
        trim_data: bool,
        grid_size: tuple = None,
        force_regression: bool = False,
        classification_boundaries: tuple = None,
        transform=None,
        skip_data: int = None
    ):
        """
        Args:
            data_dir (str): Folder of .pkl measurement dicts.
            loc_class (bool): True→location as grid‐cell class; False→(x,y) regression.
            trim_data (bool): True→keep only first 0.12 s of each trace (51200 Hz → 6144 samples).
            grid_size (tuple, optional): (nx, ny) for location‐classification bins.
            force_regression (bool): True→force regression; False→force classification via `classification_boundaries`.
            classification_boundaries (tuple, optional): (b1,b2) thresholds for 3‐way force classes.
            transform (callable, optional): e.g. normalization on each sample tensor.
            skip_data (int, optional): If n>1, keep only every n-th sample (down-sampling).
        """
        super().__init__()
        self.transform = transform
        self.loc_class = loc_class
        self.force_regression = force_regression
        self.grid_size = grid_size
        self.trim_data = trim_data
        # only use skip_data if it's >1
        self.skip_data = skip_data if (skip_data is not None and skip_data > 1) else None

        if self.trim_data:
            # 0.12 s × 51200 Hz
            self.trim_len = int(0.12 * 51200)

        if not force_regression:
            if (classification_boundaries is None) or (len(classification_boundaries) != 2):
                raise ValueError("Provide 2 boundaries for force classification.")
        if self.loc_class and (grid_size is None or len(grid_size) != 2):
            raise ValueError("Provide grid_size=(nx,ny) for location classification.")

        self.classification_boundaries = classification_boundaries

        # load all samples
        self.samples = []
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise ValueError(f"{data_dir} is not a directory")

        for pkl in sorted(data_path.glob("*.pkl")):
            with open(pkl, 'rb') as f:
                dd = pickle.load(f)
            loc = dd.get("label_loc")
            max_force = dd.get("hammer_max_force")
            traces = dd.get("data", [])
            if loc is None or max_force is None or not traces:
                continue

            # force label
            if self.force_regression:
                force_label = float(max_force)
            else:
                b1, b2 = self.classification_boundaries
                force_label = 0 if max_force < b1 else 1 if max_force < b2 else 2

            # location label
            if self.loc_class:
                x, y = loc
                nx, ny = self.grid_size
                xi = int(np.clip(x * nx, 0, nx-1))
                yi = int(np.clip(y * ny, 0, ny-1))
                loc_label = yi * nx + xi
            else:
                loc_label = tuple(loc)

            for arr in traces:
                self.samples.append((arr, loc_label, force_label))

        if not self.samples:
            raise RuntimeError(f"No samples found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, loc_label, force_label = self.samples[idx]
        # 1) optional trim
        if self.trim_data:
            arr = arr[:self.trim_len]
        # 2) optional down-sample
        if self.skip_data:
            arr = arr[::self.skip_data]

        x = torch.from_numpy(arr.astype(np.float32))
        if self.transform:
            x = self.transform(x)

        # labels
        if self.loc_class:
            loc_tensor = torch.tensor(loc_label, dtype=torch.long)
        else:
            loc_tensor = torch.tensor(loc_label, dtype=torch.float32)

        if self.force_regression:
            force_tensor = torch.tensor(force_label, dtype=torch.float32)
        else:
            force_tensor = torch.tensor(force_label, dtype=torch.long)

        return x, loc_tensor, force_tensor



## function that clusteres data based on proximity
def cluster_location_labels(data_dir:str,
                            grid_resoulution: (tuple),
                            overwrite: bool = False,
                            output_dir: str = None):
    """
    Remap continous location labels saved in .pkl measurment dicitionaries to the nearest grid-center on a normalized [0, 1]^2 board, given a grid size of 'grid_resolution'.

     his function loads each .pkl in `data_dir`, computes the nearest of the equally
    spaced grid points ((i+0.5)/nx, (j+0.5)/ny) to the original `label_loc`, and
    replaces `label_loc` with that grid‐center tuple. The modified dict is then saved
    (either overwriting the original or to `output_dir`).

    Args:
        data_dir (str): Path to directory containing .pkl measurement files.
        grid_resolution (tuple): (nx, ny) number of bins along x and y axes.
        overwrite (bool): If True, overwrite the original .pkl files; otherwise write
            cleaned files into `output_dir`.
        output_dir (str): Directory to write remapped files if `overwrite=False`.

    Raises:
        ValueError: If `data_dir` is not a valid directory or if `output_dir` is needed but
            not provided.
    """

    # Prepare paths
    src = Path(data_dir)
    if not src.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory")
    if not overwrite:
        if output_dir is None:
            raise ValueError("output_dir must be provided if 'overwrite = False")
        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)
    
    # unpack grid
    nx, ny = grid_resoulution
    # generate all gird centers on [0,1]
    cell_centers = []
    for iy in range(ny):
        for ix in range(nx):
            cx = (ix + 0.5) / nx # center of the bins
            cy = (iy + 0.5) / ny
            cell_centers.append((cx, cy))
    centers_arr = np.array(cell_centers) # shape [nx*ny, 2]

    # process each file
    for pkl_file in sorted(src.glob("*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        orig_loc = data.get("label_loc")
        if orig_loc is None:
            # skip if no label loc
            continue
        loc_arr = np.array(orig_loc, dtype=float)
        # compute distance to each center
        diffs = centers_arr - loc_arr
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        new_loc = tuple(centers_arr[idx].tolist())

        # overwrite the label_loc
        data["label_loc"] = new_loc

        # Build new filename: prefix with rounded coords
        rounded = f"{cx:.1f}_{cy:.1f}"
        orig_name = pkl_file.name
        new_name = f"{rounded}_{orig_name}"
        if overwrite:
            save_path = pkl_file.parent / new_name
        else:
            save_path = dst / new_name
        
        # save the dictionary
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if overwrite and save_path != pkl_file:
            pkl_file.unlink()
        print(f"Remapped {orig_name} -> grid center {new_name}")

    
# Dataloader defintion
def make_dataloaders(dataset: Dataset, train_frac: float, batch_size: int):
    """"
    First returns the train_dataloader then test_datalaoder
    Args:
        dataset (torch.utils.data.Dataset): Dateset that will be used for training/testing the model
        train_frac (None): The fraction of the data that will be used for the training dataset.
        batch_size (int): Batch size of the dataloaders
    Returns:
        train_dataLoader(torch.utils.data.DataLoader): The dataloader used for training the model
        test_dataLoader(torch.utils.data.DataLoader): The dataloader used for testing the model
    """
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, test_size],
        generator=torch.Generator().manual_seed(42) # manual seed for repeatability
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader

# Training function definition
def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion_loc: torch.nn.Module,
    criterion_force: torch.nn.Module,
    loss_weight_loc: tuple,
    loss_weight_force: tuple,
    early_stop_patience: int,
    device: torch.device = None
):
    """
    Trains and evaluates a multi-task model using a OneCycleLR schedule.

    Args:
        model (nn.Module): The multi-task model.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation data loader.
        num_epochs (int): Total epochs to train.
        optimizer (Optimizer): Optimizer (e.g., Adam) with initial lr set.
        criterion_loc (nn.Module): Loss for location.
        criterion_force (nn.Module): Loss for force.
        loss_weight_loc (tuple): Weight for location loss. You put the start and the end value of the location loss weight -> changes through the training linearly.
        loss_weight_force (tuple): Weight for force loss. You put the start and the end value of the location loss weight -> changes through the training linearly.
        early_stop_patience (int): Epochs with no improvement before stopping.
        device (torch.device, optional): Device to train on; auto-detects if None.

    Returns:
        dict: Training history containing losses and accuracies.
    """
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Setup changing location loss weights
    loc_weights = np.linspace(loss_weight_loc[0], loss_weight_loc[1], num_epochs, dtype=np.float32)

    # Setup changing force loss weights
    force_weights = np.linspace(loss_weight_force[0], loss_weight_force[1], num_epochs, dtype=np.float32)


    # Detect classification vs regression
    loc_class = getattr(model, 'loc_class', False)
    force_regression = getattr(model, 'force_reg', False)

    # OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    max_lr = optimizer.param_groups[0]['lr']
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=num_epochs * steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    # Early stopping params
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # History
    history = {'train_loss': [], 'test_loss': []}
    if not force_regression:
        history.update({'train_force_acc': [], 'test_force_acc': []})
    if loc_class:
        history.update({'train_loc_acc': [], 'test_loc_acc': []})

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        running_loss = correct_force = total_force = 0
        correct_loc = total_loc = 0

        for x, loc_t, force_t in train_loader:
            x, loc_t, force_t = x.to(device), loc_t.to(device), force_t.to(device)
            optimizer.zero_grad()
            loc_preds, force_preds = model(x)
            # Location loss
            if loc_class:
                loc_t = loc_t.long().squeeze()
                loss_loc = criterion_loc(loc_preds, loc_t)
                preds_loc = loc_preds.argmax(dim=1)
                correct_loc += (preds_loc == loc_t).sum().item()
                total_loc += loc_t.size(0)
            else:
                loss_loc = criterion_loc(loc_preds, loc_t)
            # Force loss
            if force_regression:
                force_t = force_t.float().unsqueeze(1)
                loss_force = criterion_force(force_preds, force_t)
            else:
                force_t = force_t.long().squeeze()
                loss_force = criterion_force(force_preds, force_t)
                preds_force = force_preds.argmax(dim=1)
                correct_force += (preds_force == force_t).sum().item()
                total_force += force_t.size(0)
            # Combined loss
            loss = loc_weights[epoch-1] * loss_loc + force_weights[epoch-1] * loss_force # weights of the loss function change linearly thorugh the training
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / steps_per_epoch
        history['train_loss'].append(avg_train_loss)
        if not force_regression and total_force:
            history['train_force_acc'].append(correct_force / total_force)
        if loc_class and total_loc:
            history['train_loc_acc'].append(correct_loc / total_loc)

        # --- Validation ---
        model.eval()
        val_loss = 0
        correct_force = total_force = correct_loc = total_loc = 0
        with torch.inference_mode():
            for x, loc_t, force_t in test_loader:
                x, loc_t, force_t = x.to(device), loc_t.to(device), force_t.to(device)
                loc_preds, force_preds = model(x)
                # Location
                if loc_class:
                    loc_t = loc_t.long().squeeze()
                    l_loc = criterion_loc(loc_preds, loc_t)
                    preds_loc = loc_preds.argmax(dim=1)
                    correct_loc += (preds_loc == loc_t).sum().item()
                    total_loc += loc_t.size(0)
                else:
                    l_loc = criterion_loc(loc_preds, loc_t)
                # Force
                if force_regression:
                    force_t = force_t.float().unsqueeze(1)
                    l_force = criterion_force(force_preds, force_t)
                else:
                    force_t = force_t.long().squeeze()
                    l_force = criterion_force(force_preds, force_t)
                    preds_force = force_preds.argmax(dim=1)
                    correct_force += (preds_force == force_t).sum().item()
                    total_force += force_t.size(0)
                val_loss += (loc_weights[epoch-1] * l_loc + force_weights[epoch-1] * l_force).item() # weights of the loss function change linearly thorugh the training

        avg_val_loss = val_loss / len(test_loader)
        history['test_loss'].append(avg_val_loss)
        if not force_regression and total_force:
            history['test_force_acc'].append(correct_force / total_force)
        if loc_class and total_loc:
            history['test_loc_acc'].append(correct_loc / total_loc)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Logging
        msg = f"Epoch {epoch}/{num_epochs} | Train L {avg_train_loss:.4f} | Val L {avg_val_loss:.4f}"
        if not force_regression:
            msg += f" | Train F-Acc {history['train_force_acc'][-1]:.3f} | Val F-Acc {history['test_force_acc'][-1]:.3f}"
        if loc_class:
            msg += f" | Train L-Acc {history['train_loc_acc'][-1]:.3f} | Val L-Acc {history['test_loc_acc'][-1]:.3f}"
        print(msg)

    return history

# helper funcitons for hyperparameter search
def model_factory(num_sensors: int,
                  sequence_length: int,
                  cnn_filters: list,
                  kernel_size: int,
                  pool_size: int,
                  force_reg: bool,
                  loc_class: bool,
                  grid_resolution: tuple,
                  latent_dim: int,
                  dropout_rate: float,
                  head_hidden_dim: int,
                  head_hidden_layers: int):
    """
    Returns a fresh instance of the CNN model. Used for testing the hyperparameters.
    """
    

    model = ImpactPredictor(num_sensors=num_sensors,
                            sequence_length=sequence_length,
                            cnn_filters=cnn_filters,
                            kernel_size=kernel_size,
                            pool_size=pool_size,
                            force_reg=force_reg,
                            loc_class=loc_class,
                            grid_resolution=grid_resolution,
                            latent_dim=latent_dim,
                            dropout_rate=dropout_rate,
                            head_hidden_dim=head_hidden_dim,
                            head_hidden_layers=head_hidden_layers)
    return model

def dataset_factory(dataset: Dataset,
                    batch_size: int,
                    train_frac: float):
    
    """
    Makes a fresh dataset when provided with the dataset batch size and the train_frac
    """
    
    train_loader, test_loader = make_dataloaders(dataset=dataset, train_frac=train_frac, batch_size=batch_size)
    return train_loader, test_loader
    

# hyperparameter search 
def hyperparameter_search(
    model_factory,           # callable that returns a fresh ImpactPredictor
    dataset_factory,         # callable that returns (train_loader, test_loader)
    lr_range: tuple,                # tuple (lr_min, lr_max)
    wd_range: tuple,                # tuple (wd_min, wd_max)
    loc_w_range: tuple,             # tuple (loc_w_start, loc_w_end)
    n_trials: int = 20,
    epochs_per_trial: int = 20):
    """
    Random search over (lr, weight_decay) with OneCycleLR and a dynamic location-weight.

    Args:
        model_factory:    no-arg callable returning a fresh ImpactPredictor
        dataset_factory:  callable(batch_size) -> (train_loader, test_loader)
        lr_range:         (min_lr, max_lr)
        wd_range:         (min_wd, max_wd)
        loc_w_range:      (start_weight, end_weight) to ramp over epochs
        n_trials:         number of random configurations
        epochs_per_trial: epochs to train each trial
        batch_size:       batch size for loaders

    Returns:
        List of (val_loss, lr, wd) sorted ascending by val_loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for trial in range(1, n_trials + 1):
        # Sample hyperparameters on log scale
        lr = 10 ** random.uniform(math.log10(lr_range[0]), math.log10(lr_range[1]))
        wd = 10 ** random.uniform(math.log10(wd_range[0]), math.log10(wd_range[1]))

        # Prepare model and data
        model = model_factory().to(device)
        train_loader, test_loader = dataset_factory()

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        steps_per_epoch = len(train_loader)
        total_steps = epochs_per_trial * steps_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )

        # Loss functions
        criterion_loc = nn.MSELoss()
        criterion_force = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = 5

        # Training loop
        for epoch in range(1, epochs_per_trial + 1):
            model.train()
            train_loss = 0.0
            # dynamic location weight ramp
            loc_w = loc_w_range[0] + (loc_w_range[1] - loc_w_range[0]) * ((epoch - 1) / (epochs_per_trial - 1))

            for x, loc_t, force_t in train_loader:
                x, loc_t, force_t = x.to(device), loc_t.to(device), force_t.to(device)
                optimizer.zero_grad()
                loc_preds, force_preds = model(x)
                # location loss
                loss_loc = criterion_loc(loc_preds, loc_t if loc_preds.ndim==2 else loc_t)
                # force loss
                loss_force = criterion_force(force_preds, force_t if force_preds.ndim==2 else force_t)
                loss = loc_w * loss_loc + loss_force
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, loc_t, force_t in test_loader:
                    x, loc_t, force_t = x.to(device), loc_t.to(device), force_t.to(device)
                    loc_preds, force_preds = model(x)
                    loss_loc = criterion_loc(loc_preds, loc_t)
                    loss_force = criterion_force(force_preds, force_t)
                    val_loss += (loc_w * loss_loc + loss_force).item()
            avg_val = val_loss / len(test_loader)
            # early stop
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop:
                    break

        print(f"Trial {trial}/{n_trials} lr={lr:.2e}, wd={wd:.2e} → val_loss={best_val_loss:.4f}")
        results.append((best_val_loss, lr, wd))

    results.sort(key=lambda x: x[0])
    print("\nTop configs:")
    for loss, lr, wd in results[:5]:
        print(f"  loss={loss:.4f}, lr={lr:.2e}, wd={wd:.2e}")
    return results



def save_model_weights(model, dir_name, weights_name):
    """
    Saves PyTorch model weights to a .pkl file using torch.save.
    Creates the directory if it doesn't exist.

    Parameters:
        model (torch.nn.Module): The model to save.
        dir_name (str): Directory where the weights should be saved.
        weights_name (str): Name of the weights file (without extension).
    """
    os.makedirs(dir_name, exist_ok=True)
    filepath = os.path.join(dir_name, weights_name + '.pkl')
    torch.save(model.state_dict(), filepath)
    print(f"Saved dict to: {filepath}")



def save_model_results(dir_name: str,
                       file_name: str,
                       model_results_dict: dict):
    """
    Saves a dictionary of the model results.
    Args:
        dir_name (str): name of the directory (it will create one if it doesnt exist yet)
        file_name (str): name of the saved .pkl file
        model_results_dict (dict): dictionary of the model results
    """
    dir_path = Path(dir_name)
    dir_path.mkdir(parents=True, exist_ok=True)

    # filename and path
    file_name = file_name + ".pkl"
    file_path = dir_path / file_name

    # write pickle file
    with open(file_path, "wb") as file:
        pickle.dump(model_results_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dict to: {file_path}")

    
# Helper class to unpickle CUDA files
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_model_weights(
    model: torch.nn.Module,
    filepath: str,
    map_location: Union[str, torch.device] = None):
    """
    Loads weights from a .pkl/.pt file into `model`, remapping to CPU if requested.

    Tries torch.load() first, then falls back to pickle.load() with CPU_Unpickler if needed.

    Params:
        model        : your nn.Module
        filepath     : path to the saved state dict
        map_location : e.g. "cpu", torch.device("cpu"), or torch.device("cuda")
    """
    target_dev = torch.device(map_location) if isinstance(map_location, str) else map_location

    try:
        state_dict = torch.load(filepath, map_location=target_dev)
    except Exception as e:
        print(f"[torch.load failed] Falling back to pickle loading: {e}")
        with open(filepath, 'rb') as f:
            if target_dev == torch.device('cpu'):
                unpickler = CPU_Unpickler(f)
                state_dict = unpickler.load()
            else:
                state_dict = pickle.load(f)

    model.load_state_dict(state_dict, strict=False)


def plot_loss_curves(model_data: dict):
    """
    Returns a graph of the training process showing the loss curves.
    
    Args:
        model_data (dict): dictionary of training data from the train_model() funciton

    Returns:
        A graph of the loss data
    """
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(model_data["train_loss"], "b--", marker=".", label="Train Loss")
    ax.plot(model_data["test_loss"], "r--", marker=".", label="Test Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()

def plot_accuracy_curves(history: dict):
    """
    Plot training & validation accuracy for location and force (if available).

    Args:
        history (dict): returned by train_model(), e.g. keys include
            -  'test_loc_acc'   (only if loc_class=True)
            -  'test_force_acc' (only if force_reg=False)
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    plt.figure(figsize=(7,5))

    has_loc =  'test_loc_acc' in history
    has_force = 'test_force_acc' in history

    if has_loc:
        plt.plot(epochs,
                 history['test_loc_acc'],
                 'b-.', marker='o', label='Test loc. acc')

    if has_force:
        plt.plot(epochs,
                 history['test_force_acc'],
                 'r--', marker='x', label='Test force acc')

    if not (has_loc or has_force):
        raise ValueError("No accuracy keys found in history to plot.")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()

def plot_inference(data_dict: dict,
                   model: torch.nn.Module,
                   weight_path: str,
                   board_size_xy: tuple,
                   class_boundaries: tuple,
                   grid_size: tuple,
                   skip_data: int = None,
                   trim_duration: float = None):
    """
    Perform inference on a sample dict and plot location & force predictions,
    with optional down-sampling/filtering.

    Args:
        data_dict (dict): Contains keys:
            - "data": list/array of sensor samples (T×C)
            - "label_loc": tuple (x,y) true coordinates
            - "hammer_max_force": float true force
            - "label_force": true force class (if available)
        model (nn.Module): ImpactPredictor with loc_class & force_reg flags.
        weight_path (str): Path to saved model weights.
        board_size_xy (tuple): Physical board size (X_max, Y_max).
        class_boundaries (tuple): Boundaries for force classes.
        grid_size (tuple): (nx, ny) number of bins in x and y.
        skip_data (int, optional): If >1, down-sample input by taking every nth sample.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load weights
    load_model_weights(model, weight_path, map_location=device)
    model.to(device).eval()

    # pick random sample
    idx = random.randint(0, len(data_dict["data"]) - 1)
    x = torch.from_numpy(data_dict["data"][idx].astype(np.float32)).to(device)
    # print(f"Initial tensor: {x.shape}")

    # trim duration
    if trim_duration is not None:
        x = x[:int(trim_duration*51200)]
    # print(f"Tensor after trimming: {x.shape}")

    # down sample
    if skip_data is not None and skip_data > 1:
        x = x[::skip_data]
    print(f"Tensor after downsampling: {x.shape}")
    x = x.unsqueeze(dim=0)
    # print(f"tensor that goes in the model: {x.shape}")

    # inference
    with torch.inference_mode():
        loc_out, force_out = model(x)
    loc_out = loc_out.squeeze().cpu().numpy()  # for class: logits length nx*ny; for reg: [2]

    # prepare force text
    if model.force_reg:
        force_pred = force_out.item()
        force_text = f"Force pred: {force_pred:.2f} N\nForce true: {data_dict['hammer_max_force']:.2f} N"
    else:
        class_pred = force_out.argmax(dim=1).item()
        # convert true force to class
        def f2c(val, bounds):
            for i, b in enumerate(bounds):
                if val < b:
                    return i
            return len(bounds)
        class_true = f2c(data_dict["hammer_max_force"], class_boundaries)
        force_text = f"Force class pred: {class_pred}\nForce class true: {class_true}"

    board_x, board_y = board_size_xy
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title("Inference: Location & Force")
    ax.set_xlim(0-0.2, board_x+0.2)
    ax.set_ylim(0-0.2, board_y+0.2)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    if model.loc_class:
        # classification: color bins
        nx, ny = grid_size
        cell_w = board_x / nx
        cell_h = board_y / ny

        # predicted & true bin indices
        pred_idx = int(np.argmax(loc_out))
        x_true, y_true = data_dict['label_loc']
        x_bin_true = min(int((x_true / board_x) * nx), nx - 1)
        y_bin_true = min(int((y_true / board_y) * ny), ny - 1)
        true_idx = y_bin_true * nx + x_bin_true

        # draw grid
        for ix in range(nx):
            for iy in range(ny):
                ax.add_patch(Rectangle((ix * cell_w, iy * cell_h),
                                    cell_w, cell_h,
                                    fill=False,
                                    edgecolor='gray',
                                    linewidth=0.5))

        # smaller, centered pred‐bin rectangle
        rect_w = cell_w * 0.6      # 60% of cell width
        rect_h = cell_h * 0.6      # 60% of cell height
        px = (pred_idx % nx) * cell_w + (cell_w - rect_w) / 2
        py = (pred_idx // nx) * cell_h + (cell_h - rect_h) / 2
        ax.add_patch(Rectangle((px, py),
                            rect_w, rect_h,
                            facecolor='lightcoral',
                            alpha=1.0,
                            label='Pred bin'))

        # full‐cell true‐bin rectangle
        tx = x_bin_true * cell_w
        ty = y_bin_true * cell_h
        ax.add_patch(Rectangle((tx, ty),
                            cell_w, cell_h,
                            facecolor='lightgreen',
                            alpha=0.5,
                            label='True bin'))

        ax.legend(loc='upper right', facecolor='white')

    else:
        # regression: scatter true & pred points
        loc_pred_xy = tuple(loc_out.tolist())
        ax.scatter(*data_dict['label_loc'], c='lightgreen', marker='x', s=120, label='True location')
        ax.scatter(*loc_pred_xy, c='lightcoral', marker='o', s=120, label='Pred location')
        ax.legend(loc='upper right', facecolor='white')

    # display force info
    if not model.loc_class:
        print(f"The model predicted the impact at coord: {loc_pred_xy}")
    ax.text(-0.2, 1.2, force_text,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', alpha=0.75))
    plt.tight_layout()
    plt.show()

def print_model_summary(model: nn.Module, input_size: tuple):
    print(summary(model=model,
                  input_size=input_size))



def prediction_heatmap(
    model: torch.nn.Module,
    weight_path: str,
    data_dir: str,
    force_class_boundaries: tuple,
    board_size_xy: tuple,
    grid_res: tuple,
    skip_data: int = None,
    trim_duration: float = None,
    mode: str = "mean_error",      # "mean_error", "hit_rate", "mean_hit_error", "force_accuracy", "location_accuracy", "force_error"
    tol: float = 2.0,
    device: torch.device = None,
    verbose: bool = True
):
    """
    Compute and plot a spatial heat-map summarizing model performance.

    Args:
        model (nn.Module): model to test (must have preloaded weights).
        weight_path (str): path to model weights.
        data_dir (str): directory with .pkl measurement files.
        force_class_boundaries (tuple): boundaries for force classes. Tuple of length 2. 
        board_size_xy (tuple): (width, height) of the board in same units as labels.
        grid_res (tuple): (nx, ny) grid resolution.
        skip_data (int): down‑sampling factor.
        trim_duration (float): truncate each trial to this duration (sec).
        mode (str): one of:
            - "mean_error"        : mean Euclidean error of all samples (location only)
            - "hit_rate"          : fraction of samples with error < tol (location only)
            - "mean_hit_error"    : mean error of only the "hits" (error < tol, location only)
            - "force_accuracy"    : force classification accuracy (or regression within tolerance)
            - "location_accuracy" : location classification/regression accuracy
            - "force_error"       : mean absolute error for force regression
        tol (float): tolerance threshold (same units as board_size_xy).
        device (torch.device): inference device. Auto‑detect if None.
        verbose (bool): print progress per file.

    Returns:
        None (displays a matplotlib heatmap).
    """
    # set up device & load weights
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model_weights(model=model, filepath=weight_path, map_location=device)
    model.to(device).eval()

    board_w, board_h = board_size_xy
    nx, ny       = grid_res
    b1, b2 = force_class_boundaries

    # where to bin along X,Y
    x_edges = np.linspace(0, board_w, nx + 1)
    y_edges = np.linspace(0, board_h, ny + 1)

    # collect errors & hits per true‑cell
    err_lists = [[[] for _ in range(nx)] for _ in range(ny)]
    hit_lists = [[[] for _ in range(nx)] for _ in range(ny)]
    force_correct_lists = [[[] for _ in range(nx)] for _ in range(ny)]
    force_error_lists = [[[] for _ in range(nx)] for _ in range(ny)]
    loc_correct_lists = [[[] for _ in range(nx)] for _ in range(ny)]

    pkl_paths = sorted(Path(data_dir).glob("*.pkl"))
    if not pkl_paths:
        raise RuntimeError(f"No .pkl files in {data_dir}")

    # precompute bin centers (for classification→distance errors)
    cell_w = board_w / nx
    cell_h = board_h / ny
    centers = [
        ((ix + 0.5) * cell_w, (iy + 0.5) * cell_h)
        for iy in range(ny) for ix in range(nx)
    ]

    for idx, pkl_path in enumerate(pkl_paths, 1):
        if verbose:
            print(f"({idx}/{len(pkl_paths)}) {pkl_path.name}")
        d = open_pkl_dict(pkl_path)
        true_loc = np.array(d['label_loc'], dtype=float)
        
        # Get force label if available
        true_force = d.get('label_force', None)
        # Also get hammer max force if available
        hammer_max_force = d.get('hammer_max_force', None)

        # find true‑cell indices
        ix = np.clip(np.digitize(true_loc[0], x_edges) - 1, 0, nx-1)
        iy = np.clip(np.digitize(true_loc[1], y_edges) - 1, 0, ny-1)
        true_bin = iy * nx + ix

        for sensor_np in d['data']:
            x_t = torch.from_numpy(sensor_np.astype(np.float32)).to(device)
            if trim_duration is not None:
                x_t = x_t[: int(trim_duration * 51200)]
            if skip_data is not None and skip_data > 1:
                x_t = x_t[::skip_data]
            x_t = x_t.unsqueeze(0)  # shape (1, T, C)

            with torch.inference_mode():
                model_out = model(x_t)
                
                # Handle different model output formats
                if isinstance(model_out, tuple):
                    if len(model_out) == 2:
                        loc_out, force_out = model_out
                    else:
                        loc_out = model_out[0]
                        force_out = model_out[1] if len(model_out) > 1 else None
                else:
                    loc_out = model_out
                    force_out = None

            loc_out = loc_out.squeeze().cpu().numpy()
            if force_out is not None:
                force_out = force_out.squeeze().cpu().numpy()

            # compute location error + hit
            if model.loc_class:
                # classification → compare bins
                pred_bin = int(np.argmax(loc_out))
                px, py   = centers[pred_bin]
                error    = np.linalg.norm([px, py] - true_loc)
                hit      = (pred_bin == true_bin)
                loc_correct = (pred_bin == true_bin)
            else:
                # regression → direct (x,y)
                pred_xy = loc_out
                error   = np.linalg.norm(pred_xy - true_loc)
                hit     = (error < tol)
                loc_correct = hit

            err_lists[iy][ix].append(error)
            hit_lists[iy][ix].append(float(hit))
            loc_correct_lists[iy][ix].append(float(loc_correct))

            # compute force accuracy if available
            if force_out is not None and true_force is not None:
                if model.force_reg:
                    # Force regression - compare predicted vs true force
                    pred_force = float(force_out)
                    force_error = abs(pred_force - true_force)
                    force_error_lists[iy][ix].append(force_error)
                    
                    # Consider it "correct" if within some tolerance (e.g., 10% of true force)
                    force_tolerance = 0.1 * abs(true_force) if true_force != 0 else 1.0
                    force_correct = (force_error < force_tolerance)
                else:
                    # Force classification - convert continuous force to discrete classes
                    true_force_class = 0 if hammer_max_force < b1 else 1 if hammer_max_force < b2 else 2

                    
                    pred_force_class = int(np.argmax(force_out))
                    force_correct = (pred_force_class == true_force_class)
                
                force_correct_lists[iy][ix].append(float(force_correct))

    # build the heatmap array
    heat = np.full((ny, nx), np.nan, dtype=float)

    for iy in range(ny):
        for ix in range(nx):
            errs = np.array(err_lists[iy][ix])
            hits = np.array(hit_lists[iy][ix])
            loc_corrects = np.array(loc_correct_lists[iy][ix])
            force_corrects = np.array(force_correct_lists[iy][ix])
            force_errors = np.array(force_error_lists[iy][ix])
            
            if errs.size == 0:
                continue

            if mode == 'mean_error':
                heat[iy, ix] = errs.mean()

            elif mode == 'hit_rate':
                heat[iy, ix] = hits.mean()

            elif mode == 'mean_hit_error':
                if model.loc_class:
                    # classification: mis-classification rate
                    heat[iy, ix] = 1.0 - hits.mean()
                else:
                    # regression: mean error of only the "hits"
                    if hits.sum() > 0:
                        heat[iy, ix] = errs[hits.astype(bool)].mean()
                    else:
                        heat[iy, ix] = np.nan

            elif mode == 'force_accuracy':
                if force_corrects.size > 0:
                    heat[iy, ix] = force_corrects.mean()
                else:
                    heat[iy, ix] = np.nan

            elif mode == 'location_accuracy':
                heat[iy, ix] = loc_corrects.mean()

            elif mode == 'force_error':
                if force_errors.size > 0:
                    heat[iy, ix] = force_errors.mean()
                else:
                    heat[iy, ix] = np.nan

            else:
                raise ValueError("mode must be 'mean_error', 'hit_rate', 'mean_hit_error', 'force_accuracy', 'location_accuracy', or 'force_error'")

    # choose colormap & label
    if mode == 'hit_rate':
        cmap  = 'viridis'
        label = f'Hit rate (< {tol})'
    elif mode == 'mean_error':
        cmap  = 'magma_r'
        label = 'Mean error'
    elif mode == 'mean_hit_error':
        cmap  = 'magma_r'
        label = 'Mean error of hits' if not model.loc_class else 'Mis‑classification rate'
    elif mode == 'force_accuracy':
        cmap  = 'viridis'
        label = 'Force accuracy' if not model.force_reg else 'Force accuracy (within tolerance)'
    elif mode == 'location_accuracy':
        cmap  = 'viridis'
        label = 'Location accuracy'
    elif mode == 'force_error':
        cmap  = 'magma_r'
        label = 'Mean force error'

    # plot
    plt.figure(figsize=(6,5))
    im = plt.imshow(
        heat,
        origin='lower',
        extent=[0, board_w, 0, board_h],
        cmap=cmap,
        aspect='equal'
    )
    plt.colorbar(im, label=label)
    plt.title(f"Spatial {label} (grid {nx}×{ny})")
    plt.xlabel('X'); plt.ylabel('Y')
    plt.tight_layout()
    plt.show()


### DAQ FUNCTIONS 

# Data gathering function definition
def gather_data(location: tuple,
                sample_rate: int,
                trigger_force: int,
                measurment_duration: float,
                num_presamples: int,
                hammer_trigger: bool,
                resistance_trigger: int,
                R_shunt: int,
                num_samples: int,
                DIR_NAME: str,
                save_dict:bool):
    """
    Args:
        location (tuple): The (x,y) coordinates of the impact
        sample_rate (int): Sample rate of measuments in Hz
        trigger_force (int): The force that will be used to trigger the impact
        measurment_duration (float): The duration of the measurment in seconds
        num_presamples (int): Number of presamples for the measurment
        hammer_trigger (bool): Is the trigger going to be a hammer. If false then the trigger will be via sensors.
        resistance_trigger (int): The change of resistance that will be used to trigger the measurment
        R_shunt (int): Shunt resistance in Ohm
        num_samples (int): the number of individual measurments that you want to make
        DIR_NAME (str): Name of the directory in which the measurments are sotred (they are sotred as .pkl)
        save_dict (bool): Saves the measurments to specified directory
    Returns:
        Dictionary of measurments:
        {
        "data" : list the gathered data from the sensors (AC change of resistance),
        "label_loc" : tuple of the location,
        "label_force" : int of the trigger force,
        "measured_force" : list of the measured force,
        "hammer_max_force" : float of the max impact force 
        }

    """
    # Set up DAQ 
    # Make input tasks - need to make in NI max -> differential measurments

    input_sensor_shunt_task = LDAQ.national_instruments.NITask("input_sensor_shunt", sample_rate=sample_rate)
    # Sensor channels
    for ch in range(4):

        input_sensor_shunt_task.add_channel(
            channel_name=f"sensor_{ch}",
            device_ind=1,
            channel_ind=ch,
            sensitivity=1,
            sensitivity_units="V",
            units="V",

        )
    # Shunt channels
    for ch in range(4):

        input_sensor_shunt_task.add_channel(
            channel_name=f"shunt_{ch}",
            device_ind=2,
            channel_ind=ch,
            sensitivity=1,
            sensitivity_units="V",
            units="V"
        )
    
    # Hammer channel
    input_sensor_shunt_task.add_channel(
        channel_name="hammer",
        device_ind=3,
        channel_ind=0,
        sensitivity=2.273, #V/N check the label -> preracun
        sensitivity_units="mV/N",
        units="N"
    )
    # ACQ objects
    acq_sensor_shunt = LDAQ.national_instruments.NIAcquisition(input_sensor_shunt_task, acquisition_name="NI_sensor_shunt")

    # See if object is ok
    print([acq_sensor_shunt])

    # Visualization of the measurment

    viz = LDAQ.Visualization() 
    viz.add_lines((0,0), source="NI_sensor_shunt", channels=[0])

    viz.add_lines((0,1), source="NI_sensor_shunt", channels=[1])
    viz.add_lines((0,2), source="NI_sensor_shunt", channels=[2])
    viz.add_lines((0,3), source="NI_sensor_shunt", channels=[3])
    viz.add_lines((1,0), source="NI_sensor_shunt", channels=[4])
    viz.add_lines((1,1), source="NI_sensor_shunt", channels=[5])
    viz.add_lines((1,2), source="NI_sensor_shunt", channels=[6])
    viz.add_lines((1,3), source="NI_sensor_shunt", channels=[7])
    viz.add_lines((2,0), source="NI_sensor_shunt", channels=[8])
    viz.config_subplot((0,0), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((0,1), xlim=(-5,5), ylim=(-5,5))    

    viz.config_subplot((0,2), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((0,3), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((1,0), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((1,1), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((1,2), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((1,3), xlim=(-5,5), ylim=(-5,5))    
    viz.config_subplot((2,0), xlim=(-1, 50), ylim=(-5,5))    
    # LDAQ.Core object
    ldaq = LDAQ.Core(acquisitions=[acq_sensor_shunt])
    # Trigger gre na kladivo -> Newtni
    if hammer_trigger:
        ldaq.set_trigger(
            source="NI_sensor_shunt",
            channel=8,
            level=trigger_force, # N
            duration=measurment_duration,
            presamples=num_presamples
        )
    else:
        # for i in range(4):
        # the trigger is set to voltage not the resistance level
        ldaq.set_trigger(
            source="NI_sensor_shunt",
            channel=2,
            level=resistance_trigger,
            trigger_type="up",
            duration=measurment_duration,
            presamples=num_presamples
        )

    # Make num_samples measurments and append them to a dictionary
    results_dict = {
        "data": [],
        "label_loc": location,
        "label_force": trigger_force,
        "measured_force": [],
        "hammer_max_force" : None
    }

    for i in range(num_samples):
        # Run the measurment
        ldaq.run()
        
        # Get the measurment data
        measurment = ldaq.get_measurement_dict()

        # Samo ac komponento - torej od meritve R(t) odstejem avg(R(t))
        SEN_1_DATA_DC_AC =  measurment["NI_sensor_shunt"]["data"][:,0] / measurment["NI_sensor_shunt"]["data"][:,4] * R_shunt
        SEN_2_DATA_DC_AC =  measurment["NI_sensor_shunt"]["data"][:,1] / measurment["NI_sensor_shunt"]["data"][:,5] * R_shunt
        SEN_3_DATA_DC_AC =  measurment["NI_sensor_shunt"]["data"][:,2] / measurment["NI_sensor_shunt"]["data"][:,6] * R_shunt
        SEN_4_DATA_DC_AC =  measurment["NI_sensor_shunt"]["data"][:,3] / measurment["NI_sensor_shunt"]["data"][:,7] * R_shunt
        
        
        HAMMER_FORCE = measurment["NI_sensor_shunt"]["data"][:,8] # potek sile kladiva
        HAMMER_FORCE_MAX = np.max(HAMMER_FORCE) # maximum force -> bomo vzeli, da je to sila udarca

        # Izracin DC komponenta
        SEN_1_DATA_DC = np.mean(SEN_1_DATA_DC_AC)
        SEN_2_DATA_DC = np.mean(SEN_2_DATA_DC_AC)
        SEN_3_DATA_DC = np.mean(SEN_3_DATA_DC_AC)
        SEN_4_DATA_DC = np.mean(SEN_4_DATA_DC_AC)
        
        
        SEN_1_DATA_AC = SEN_1_DATA_DC_AC - SEN_1_DATA_DC
        SEN_2_DATA_AC = SEN_2_DATA_DC_AC - SEN_2_DATA_DC
        SEN_3_DATA_AC = SEN_3_DATA_DC_AC - SEN_3_DATA_DC
        SEN_4_DATA_AC = SEN_4_DATA_DC_AC - SEN_4_DATA_DC


        # Get the data into right shape [num_samples, num_sensors]
        sample_tensor = np.vstack((SEN_1_DATA_AC, SEN_2_DATA_AC, SEN_3_DATA_AC, SEN_4_DATA_AC)).T
        print(f"The measurment tensor shape: {sample_tensor.shape}")

        # Append the data to the dictionary
        results_dict["data"].append(sample_tensor)
        results_dict["measured_force"].append(HAMMER_FORCE)
        results_dict["hammer_max_force"] = HAMMER_FORCE_MAX


    if save_dict:
        # make dir
        dir_path = Path(DIR_NAME)
        dir_path.mkdir(parents=True, exist_ok=True)

        # filename and path
        file_name = f"{location[0]}_{location[1]}_{trigger_force}_{num_samples}.pkl"
        file_path = dir_path / file_name

        # write pickle file
        with open(file_path, "wb") as file:
            pickle.dump(results_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved dict to: {file_path}")

    return results_dict



# plot_data definition
def plot_data(results_dict, sample_rate=51200.0):
    """
    Plot sensor data, hammer force, and FFT of sensor data for the first measurement.

    Args:
        results_dict (dict): Dictionary returned by gather_data, containing:
            - "data": list of arrays [N, 4] (N=time samples, 4 sensors)
            - "measured_force": list of hammer force time series [N]
        sample_rate (float): Sampling rate in Hz (default: 100).
    """
    # Use only the first measurement
    data = results_dict["data"][0]         # shape: [N, 4]
    hammer_force = results_dict["measured_force"][0]  # shape: [N]

    N = data.shape[0]
    t = np.arange(N) / sample_rate         # time vector

    # 1) Plot each sensor's time series in its own subplot
    fig1, axes1 = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i in range(4):
        axes1[i].plot(t, data[:, i], label=f"Sensor {i+1}")
        axes1[i].set_ylabel("AC Resistance")
        axes1[i].legend(loc="upper right")
        axes1[i].grid(True)
    axes1[-1].set_xlabel("Time (s)")
    fig1.suptitle("Sensor Time Series")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 2) Plot hammer force time series in its own figure
    if len(hammer_force) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(t, hammer_force, color="k", label="Hammer Force")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Force (N)")
        ax2.set_title("Hammer Force Time Series")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()

    plt.show()




### Other functions 

# open an exaple measurment from a saved file
def open_pkl_dict(file_path: str):
    """
    Load a dictionary from a .pkl file.

    Parameters:
        file_path (str): Path to the .pkl file.

    Returns:
        dict: The loaded dictionary.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data

# plot the prediction of the model on an overaly of the board

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
# Ensure gather_data and load_model_weights are imported

def live_inference(
    model: torch.nn.Module,
    weight_path: str,
    board_size_xy: tuple,
    location: tuple,
    class_boundaries: tuple,
    grid_res: tuple,
    sample_rate: int,
    trigger_force: int,
    measurement_duration: float,
    num_presamples: int,
    hammer_trigger: bool,
    resistance_trigger: int,
    DIR_NAME: str,
    R_shunt: float,
    filter_parameters: list,
    filter_fn: callable = None,
    skip_data: int = None,
    device: torch.device = None
) -> dict:
    """
    Acquire one impact from the DAQ, run the pretrained model on it, and visualize
    prediction vs. ground-truth for both location and force (regression or classification).

    Args:
        model (nn.Module): Instantiated ImpactPredictor matching training architecture.
        weight_path (str): Path to saved state_dict weights.
        board_size_xy (tuple): (width, height) of board in same units as labels.
        location (tuple): Ground-truth (x, y) coordinate for the strike.
        class_boundaries (tuple): Boundaries for force classification.
        grid_res (tuple): (nx, ny) bins for location classification.
        sample_rate (int): DAQ sampling rate in Hz.
        trigger_force (int): Trigger threshold (N) for hammer trigger.
        measurement_duration (float): Acquisition window length (s).
        num_presamples (int): Number of pre-trigger samples.
        hammer_trigger (bool): Whether to trigger on hammer channel.
        resistance_trigger (int): Threshold for resistance trigger if not hammer.
        DIR_NAME (str): Directory name for temporary files (gather_data temp).
        R_shunt (float): Shunt resistance used in AC-resistance calc.
        filter_parameters (list): parameters for the filter function (in order).
        filter_fn (callable): filter function (expects input shape [T, C] or [N, T, C]).
        skip_data (int): If >1, down-sample by this factor along time axis.
        device (torch.device, optional): Device for inference; auto-detect if None.

    Returns:
        dict: {
            'pred_loc': np.ndarray shape (2,) or int class index,
            'pred_force': float (regression) or int (classification),
            'true_loc': np.ndarray shape (2,),
            'true_force': float,
            'sensor_data': np.ndarray shape (T', C) of the preprocessed data.
        }
    """
    # --- 1. Setup ---
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    load_model_weights(model, weight_path, map_location=device)

    # --- 2. Acquire Data ---
    print("Acquiring data...")
    results = gather_data(
        location=location, sample_rate=sample_rate, trigger_force=trigger_force,
        measurment_duration=measurement_duration, num_presamples=num_presamples,
        hammer_trigger=hammer_trigger, resistance_trigger=resistance_trigger,
        R_shunt=R_shunt, num_samples=1, DIR_NAME=DIR_NAME, save_dict=False
    )

    # Extract the raw sensor trace [T, C] and ground truth
    raw_sensor_np = results['data'][0]
    processed_sensor_np = raw_sensor_np.copy()  # Start with a copy to modify
    
    true_loc = np.array(location, dtype=float)
    true_force = results['hammer_max_force']
    print(f"Data acquired. True force: {true_force:.2f} N. Initial shape: {raw_sensor_np.shape}")

    # --- 3. Preprocess Data ---
    # a) Apply filtering if requested
    if filter_fn is not None:
        print(f"Applying filter with parameters: {filter_parameters}")
        # The corrected filter_fn now handles 2D input correctly
        processed_sensor_np = filter_fn(processed_sensor_np, *filter_parameters)
        print(f"Shape after filtering: {processed_sensor_np.shape}")

    # b) Down-sample in time if requested
    if skip_data is not None and skip_data > 1:
        print(f"Down-sampling by factor of {skip_data}...")
        processed_sensor_np = processed_sensor_np[::skip_data].copy()
        print(f"Shape after down-sampling: {processed_sensor_np.shape}")

    # --- 4. Inference ---
    # Prepare tensor for model: shape (1, T', C)
    x = torch.from_numpy(processed_sensor_np.astype(np.float32)).unsqueeze(0).to(device)
    print(f"Final tensor shape for model: {x.shape}")

    with torch.inference_mode():
        loc_out, force_out = model(x)
    loc_out = loc_out.squeeze().cpu().numpy()

    # --- 5. Decode and Visualize ---
    # Decode force prediction
    if model.force_reg:
        force_pred = float(force_out.item())
        force_str  = f"Force: {force_pred:.2f} N (pred)\nvs. {true_force:.2f} N (true)"
    else:
        # Helper to map continuous force to discrete class
        def force_to_class(val, bounds):
            for i, b in enumerate(bounds):
                if val < b: return i
            return len(bounds)
        cls_pred   = int(force_out.argmax(dim=1).item())
        cls_true   = force_to_class(true_force, class_boundaries)
        force_pred = cls_pred
        force_str  = f"Force: class {cls_pred} (pred)\nvs. class {cls_true} (true)"

    # Plot location and force
    board_w, board_h = board_size_xy
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title('Live Inference – Location & Force')
    ax.set_xlim(-0.1 * board_w, 1.1 * board_w)
    ax.set_ylim(-0.1 * board_h, 1.1 * board_h)
    ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')

    # Location plotting: classification vs. regression
    if getattr(model, 'loc_class', False):
        loc_pred_val = int(np.argmax(loc_out))
        nx, ny  = grid_res
        cell_w, cell_h = board_w / nx, board_h / ny
        tx, ty   = true_loc
        tx_bin, ty_bin = min(int((tx/board_w)*nx), nx-1), min(int((ty/board_h)*ny), ny-1)
        
        # Draw grid
        for ix in range(nx):
            for iy in range(ny):
                ax.add_patch(Rectangle((ix*cell_w, iy*cell_h), cell_w, cell_h, fill=False, edgecolor='gray', linewidth=0.5))
        # Highlight predicted bin
        ax.add_patch(Rectangle(((loc_pred_val % nx)*cell_w, (loc_pred_val//nx)*cell_h), cell_w, cell_h, facecolor='lightcoral', alpha=0.6, label='Predicted Bin'))
        # Highlight true bin
        ax.add_patch(Rectangle((tx_bin*cell_w, ty_bin*cell_h), cell_w, cell_h, facecolor='lightgreen', alpha=0.6, label='True Bin'))
        ax.legend(loc='upper right', facecolor='white')
    else:
        loc_pred_val = loc_out
        ax.scatter(*true_loc,    c='green', marker='X', s=150, label='True Location', zorder=10, edgecolors='k')
        ax.scatter(*loc_pred_val, c='red',   marker='o', s=150, label='Predicted Location', zorder=10, edgecolors='k', alpha=0.8)
        ax.legend(loc='upper right', facecolor='white')

    # Annotate force result
    ax.text(0.05, 0.95, force_str, transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
    plt.tight_layout()
    plt.show()

    return {
        'pred_loc':   loc_pred_val,
        'pred_force': force_pred,
        'true_loc':   true_loc,
        'true_force': true_force,
        'sensor_data': processed_sensor_np
    }

#### MONTE CARLO PART OF THE CODE - logic on how to train the model -> probabily a seperate montecarlo training function
def generate_latin_hypercube_points(n_samples: int,
                                    x_range: tuple,
                                    y_range: tuple,
                                    seed: int = None,
                                    resolution: float = 0.1):
    """
    Generates `n_samples` points (x, y, F) via a Latin Hypercube design,
    scaled to the specified ranges.

    Args:
        n_samples (int): Number of points to sample.
        x_range (tuple): Lower/upper bound for the x-coordinate (in my case (0,1))
        y_range (tuple): Lower/upper bound for the y-coordinate.
        seed (int): Random seed for reproducibility (optional)
        resolution (float): Quantization step for all three axes.

    Returns:
        List of tuples [(x1, y1, F1), ..., (xn, yn, Fn)].
    """
    # Create a 3D LHC in [0, 1]^3
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n_samples) # shape (n_samples, 3)

    # Scale each dim to its bounds
    l_bounds = [x_range[0], y_range[0]]
    u_bounds = [x_range[1], y_range[1]]
    scaled = qmc.scale(sample=unit_samples, l_bounds=l_bounds, u_bounds=u_bounds) # still the same shape

    def quantize(val, vmin, vmax, step):
        # turn into an integer grid index, clip, then back to value
        idx = round((val - vmin) / step)
        max_idx = int(round((vmax - vmin) / step))
        idx = min(max(idx, 0), max_idx)
        return round(vmin + idx * step, ndigits=1)

    # quantize each coord
    pts = []
    for x,y in scaled:
        xq = quantize(x, x_range[0], x_range[1], resolution)
        yq = quantize(y, y_range[0], y_range[1], resolution)
        pts.append((xq,yq))

    return pts



def run_random_sampling(sample_points: list,
                        sample_rate: int,
                        measurment_duration: float,
                        num_samples: int,
                        num_presamples: int,
                        hammer_trigger: bool,
                        trigger_force: int,
                        resistance_trigger: float,
                        R_shunt: float,
                        save_dir_name: str,
                        save_dict: bool):
    """
    For each (x, y, F) in `sample_points`:
      1. Print "→ Next: hit at (x, y) with F N"
      2. Call your gather_data(...) with save_dict=True into save_dir
      3. That saves one .pkl per point using the location & force in its filename

    Returns
    -------
    None
        All data are written to disk under `save_dir/*.pkl`.
    """
    Path(save_dir_name).mkdir(parents=True, exist_ok=True)

    for idx, (x,y) in enumerate(sample_points, start=1):
        print(f"[{idx:3d}/{len(sample_points)}] Next: hit at x = {x:.1f} | y = {y:.1f}")

        # call gather_data_function

        d = gather_data(location=(x,y),
                        sample_rate=sample_rate,
                        trigger_force=trigger_force,
                        measurment_duration=measurment_duration,
                        num_presamples=num_presamples,
                        hammer_trigger=hammer_trigger,
                        resistance_trigger=resistance_trigger,
                        R_shunt=R_shunt,
                        num_samples=num_samples,
                        DIR_NAME=save_dir_name,
                        save_dict=save_dict)
        
    print("Done gathering random samples")


    import pickle
from pathlib import Path
import numpy as np

def _interp_nan_array(arr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate over NaNs in a 1D array using numpy.interp.
    Edges are filled with the nearest valid value.
    """
    # arr: 1D float array
    n = arr.shape[0]
    # indices
    x = np.arange(n)
    # mask of valid (non-nan) points
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        # all NaN, return zeros
        return np.zeros_like(arr)
    if valid.sum() == n:
        return arr.copy()
    # xp: indices of valid
    xp = x[valid]
    fp = arr[valid]
    # interpolate; left/right filled with fp[0]/fp[-1]
    return np.interp(x, xp, fp)


def interpolate_nan_in_directory(data_dir: str,
                                 overwrite: bool = True,
                                 output_dir: str = None) -> None:
    """
    For every .pkl in `data_dir`, load the results_dict, interpolate NaNs
    in each sensor time series, and save back (either overwrite or to `output_dir`).

    Args:
        data_dir (str): Path to directory with .pkl files.
        overwrite (bool): If True, overwrite original files; else write to `output_dir`.
        output_dir (str): Path to save cleaned files if overwrite=False.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory")

    if not overwrite:
        if output_dir is None:
            raise ValueError("output_dir must be provided if overwrite=False")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    for pkl_file in sorted(data_path.glob("*.pkl")):
        # Load dictionary
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)

        cleaned = []
        # Process each sample array in results['data']
        for arr in results.get('data', []):
            # arr: 2D array shape [T, channels]
            arr = np.asarray(arr, dtype=float)
            # interpolate each column independently
            for c in range(arr.shape[1]):
                arr[:, c] = _interp_nan_array(arr[:, c])
            cleaned.append(arr)
        # Replace data
        results['data'] = cleaned

        # Save
        if overwrite:
            save_path = pkl_file
        else:
            save_path = out_path / pkl_file.name
        with open(save_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Cleaned NaNs and saved: {save_path}")




### FILTERING DATA

# fft of data with given parameters
def plot_fft(
    path: str,
    sample_rate: float = 51200.0,
    max_freq: float = 500.0
):
    """
    Load one or all .pkl data dictionaries and overlay their average FFT spectra.

    Args:
        path (str): Path to a single `.pkl` file or a directory of `.pkl` files.
        sample_rate (float): Sampling rate in Hz.
        max_freq (float): Maximum frequency (Hz) to display.

    Behavior:
        - If `path` is a file, computes its average spectrum.
        - If `path` is a directory, computes and plots each file’s average spectrum
          all on the same axes, so you can compare resonant peaks across impacts.

    FFT details:
      * We use `np.fft.rfft` to compute the one-sided spectrum of each time-series.
      * Amplitude is normalized by dividing by the number of time samples.
      * We average first over sensors, then over multiple samples in the same file.
    """
    p = Path(path)
    files = []
    if p.is_dir():
        files = sorted(p.glob("*.pkl"))
    elif p.is_file():
        files = [p]
    else:
        raise ValueError(f"{path} is not a file or directory")
    if not files:
        raise RuntimeError("No .pkl files found")

    plt.figure(figsize=(10, 6))
    for f in files:
        with open(f, 'rb') as fin:
            dd = pickle.load(fin)
        data_list = dd.get('data', [])
        if not data_list:
            continue
        # assume first sample defines T and C
        T, C = np.asarray(data_list[0]).shape
        freqs = np.fft.rfftfreq(T, d=1/sample_rate)
        idx_max = np.searchsorted(freqs, max_freq, 'right')

        # accumulate amplitude across sensors & samples
        amp_sum = np.zeros(idx_max)
        count = 0
        for arr in data_list:
            sig = np.asarray(arr)
            N = sig.shape[0]
            fft_vals = np.fft.rfft(sig, axis=0)  # shape [freq_bins, C]
            amp = np.abs(fft_vals) / N           # normalize
            # average across sensors to get 1D
            amp_mean = amp.mean(axis=1)          # shape [freq_bins]
            amp_sum += amp_mean[:idx_max]
            count += 1
        avg_amp = amp_sum / count

        plt.plot(freqs[:idx_max], avg_amp, label=f.name)

    plt.xlim(0, max_freq)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (norm)")
    plt.title("Overlayed Average FFT Spectra")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Plot the filter data:
def apply_lowpass(
    filepath: str,
    cutoff: float = 100.0,
    order: int = 4,
    sample_rate: float = 51200.0,
    trim_duration: float = None):
    """
    Load a .pkl data dictionary, apply a Butterworth low-pass filter to each sensor channel,
    plot original vs. filtered time series, and return the filtered signal.

    Returns:
        t (np.ndarray): time vector for the signal
        filtered (np.ndarray): filtered signal array shape [T, C]
    """
    # Load the pickle file
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f)
    signal = np.array(data_dict['data'][0])  # shape [T, C]
    
    # Trim to desired duration
    if trim_duration is not None:
        max_samples = int(trim_duration * sample_rate)
        signal = signal[:max_samples]
    
    T, num_sensors = signal.shape
    t = np.arange(T) / sample_rate

    # Design Butterworth filter
    nyq = 0.5 * sample_rate
    wn = cutoff / nyq
    b, a = butter(order, wn, btype='low', analog=False)

    # Apply filter to each channel
    filtered = filtfilt(b, a, signal, axis=0)

    # Plot
    fig, axes = plt.subplots(num_sensors, 1, figsize=(10, 3*num_sensors), sharex=True)
    if num_sensors == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, signal[:, i], label='Original', alpha=0.5)
        ax.plot(t, filtered[:, i], label='Filtered')
        ax.set_ylabel(f"Sensor {i}")
        ax.legend(loc='upper right')
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    title = f"Original vs. {cutoff} Hz Low-pass Filtered"
    if trim_duration is not None:
        title += f" (first {trim_duration:.2f}s)"
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





# apply filter to data and save the files
def save_lowpass_filtered_data(
    data_dir: str,
    cutoff: float = 100.0,
    order: int = 4,
    sample_rate: float = 51200.0,
    trim_duration: float = None,
    output_dir: str = None
):
    """
    Apply a Butterworth low‑pass filter to every .pkl in `data_dir` and optionally save the cleaned dicts.

    Args:
        data_dir (str): Directory containing `.pkl` measurement dicts.
        cutoff (float): Low‑pass cutoff frequency in Hz (default 100.0).
        order (int): Butterworth filter order (default 4).
        sample_rate (float): Sampling rate in Hz (default 51200.0).
        trim_duration (float, optional): If set, trim each time series to the first 
                                         `trim_duration` seconds before filtering.
        output_dir (str, optional): If provided, save filtered `.pkl` files into this directory 
                                    (creates it if needed). If None, does not write files.

    For each `.pkl`:
      1. Load the dict, expecting key `"data"` → list of [T, C] arrays.
      2. For each array: trim to `trim_duration`, design Butterworth low‑pass, apply zero‑phase filter.
      3. Replace the dict’s `"data"` list with the filtered arrays.
      4. If `output_dir` is set, write the modified dict under the same filename into that directory.
    """
    src = Path(data_dir)
    if not src.is_dir():
        raise ValueError(f"{data_dir} is not a valid directory")
    dst = None
    if output_dir:
        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)

    # design filter once
    nyq = 0.5 * sample_rate
    wn  = cutoff / nyq
    b, a = butter(order, wn, btype='low', analog=False)

    for pkl_path in sorted(src.glob("*.pkl")):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        raw_list = data.get('data', [])
        cleaned_list = []
        for arr in raw_list:
            arr = np.asarray(arr, dtype=float)
            # trim if requested
            if trim_duration is not None:
                max_samples = int(trim_duration * sample_rate)
                arr = arr[:max_samples]
            # filter each channel
            filtered = filtfilt(b, a, arr, axis=0)
            cleaned_list.append(filtered)
        data['data'] = cleaned_list

        if dst:
            save_path = dst / pkl_path.name
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved filtered: {save_path}")
        else:
            print(f"Processed (not saved): {pkl_path.name}")

    # Optional: return list of processed filenames
    return [p.name for p in src.glob("*.pkl")]

def filter_array(array: np.array,
                 cutoff: float,
                 trim_duration: float = None,
                 order: int = 4,
                 sample_rate: int = 51200):
    """
    Takes an array and applies a Butterworth low-pass filter to an array. 
    
    Args: 
        - array (np.array): A numpy array that you want to filter
        - cutoff (float): The cutoff frequency of the filter
        - trim_duration (float): Trim the duration of the signal
        - order (int): order of the filter (how sharp the cutoff is)
        - sample_rate (int): Sample rate of acquisition
    Returns:
        - filtered array of the data    
    """
    if trim_duration is not None:
        max_samples = int(trim_duration * sample_rate)
        signal = array[:max_samples]
    
    samples, T, num_sensors = signal.shape
    t = np.arange(T) / sample_rate

    # Butterworth filter
    nyq = 0.5 * sample_rate
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="low", analog=False)

    # Apply filter to each channel
    filtered = filtfilt(b, a, signal, axis=0)

    return filtered

### find first own freq


def estimate_first_mode(
    path: str,
    sample_rate: float = 51200.0,
    max_freq: float = 5000.0,
    low_freq_cutoff: float = 100.0,
    peak_prominence: float = 0.1
) -> float:
    """
    Estimate the first mechanical resonance frequency from impact data.

    Args:
        path (str): Path to a single .pkl or directory of .pkl files (unfiltered).
        sample_rate (float): Original acquisition rate (Hz).
        max_freq (float): Upper search limit for peaks (Hz).
        low_freq_cutoff (float): Ignore any spectral content below this (Hz).
        peak_prominence (float): Fraction of max amplitude for peak threshold.

    Returns:
        float: Estimated first resonance frequency (Hz).
    """
    p = Path(path)
    files = sorted(p.glob("*.pkl")) if p.is_dir() else [p]

    freqs = None
    amp_sum = None
    count = 0

    for fpath in files:
        with open(fpath, "rb") as fin:
            dd = pickle.load(fin)
        data_list = dd.get("data", [])
        if not data_list:
            continue

        # assume first sample to get length
        T, C = np.asarray(data_list[0]).shape
        this_freqs = np.fft.rfftfreq(T, d=1/sample_rate)
        idx_max = np.searchsorted(this_freqs, max_freq, 'right')

        if freqs is None:
            freqs = this_freqs[:idx_max]
            amp_sum = np.zeros_like(freqs)

        for arr in data_list:
            sig = np.asarray(arr, dtype=float)    # shape (T, C)
            # remove DC offset / drift
            sig = detrend(sig, axis=0, type='constant')
            # FFT and amplitude
            fft_vals = np.fft.rfft(sig, axis=0)
            amp = np.abs(fft_vals) / T            # normalize by length
            amp_sum += amp.mean(axis=1)[:idx_max]
            count += 1

    if count == 0:
        raise RuntimeError("No valid data found in path.")

    avg_amp = amp_sum / count

    # Mask out low frequencies below low_freq_cutoff
    valid = (freqs >= low_freq_cutoff) & (freqs <= max_freq)
    f_valid = freqs[valid]
    amp_valid = avg_amp[valid]

    # Find peaks
    height = amp_valid.max() * peak_prominence
    peaks, _ = find_peaks(amp_valid, height=height)
    if not len(peaks):
        raise RuntimeError(
            "No peaks found—try lowering peak_prominence "
            "or increasing max_freq."
        )

    first_peak = f_valid[peaks[0]]

    # Optional annotated plot
    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs, avg_amp, label="Avg spectrum")
    plt.semilogy(f_valid[peaks], amp_valid[peaks], 'ro', label="Peaks")
    plt.axvline(first_peak, color='gray', ls='--',
                label=f"1st mode ≈ {first_peak:.0f} Hz")
    plt.xlim(0, max_freq)
    plt.ylim(bottom=amp_valid[peaks].min()*0.1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (norm)")
    plt.title("Estimated 1st Resonance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return first_peak

def filter_array_LI(array: np.ndarray,
                 cutoff: float,
                 trim_duration: float = None,
                 order: int = 4,
                 sample_rate: int = 51200):
    """
    Takes an array and applies a Butterworth low-pass filter to an array.
    Handles both 2D (T, C) and 3D (N, T, C) inputs robustly.

    Args:
        - array (np.ndarray): A numpy array that you want to filter.
        - cutoff (float): The cutoff frequency of the filter.
        - trim_duration (float): Trim the duration of the signal.
        - order (int): order of the filter (how sharp the cutoff is).
        - sample_rate (int): Sample rate of acquisition.
    Returns:
        - filtered array of the data.
    """
    signal = np.asarray(array, dtype=float)

    if trim_duration is not None:
        max_samples = int(trim_duration * sample_rate)
        # Handle both 2D and 3D cases for trimming
        if signal.ndim == 2:  # Shape (T, C)
            signal = signal[:max_samples, :]
        elif signal.ndim == 3:  # Shape (N, T, C)
            signal = signal[:, :max_samples, :]

    # Determine the time axis based on dimension
    time_axis = 0 if signal.ndim == 2 else 1

    # Safety check: filtfilt requires signal length > padlen (approx 3*order)
    if signal.shape[time_axis] <= 3 * order:
        print(f"Warning: Signal length ({signal.shape[time_axis]}) is too short "
              f"for filtering with order {order}. Skipping filter.")
        return signal

    # Butterworth filter design
    nyq = 0.5 * sample_rate
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="low", analog=False)

    # Apply filter to each channel along the correct time axis
    filtered = filtfilt(b, a, signal, axis=time_axis)

    return filtered


