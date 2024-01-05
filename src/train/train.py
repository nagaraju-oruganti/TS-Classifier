import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp              # mixed precision
from torch import autocast
from torchsummary import summary
from datetime import datetime

from sklearn.metrics import mean_squared_error

## Local imports
from models.lstm import HSLSTMForDirection
from utils import utils
from data import dataset
from data.dataset import MakeDataloaders
from train.trainer import trainer

import warnings
warnings.filterwarnings('ignore')

import gc

def train_model(config):
    print();print('-'*50);print(); print(f'Training fold {config.fold}')
    print('Horizon definition: ', config.horizon_def)
    device = config.device
    
    config.dest_path = os.path.join(config.models_dir, config.model_name)
    os.makedirs(config.dest_path, exist_ok=True)
    
    ## class weights
    class_weights = dataset.estimate_class_weights(config, fold = config.fold)
    print(f'class weights: {class_weights}')
    
    ## Train with batches
    config.n_batches = utils.estimate_n_batches(config = config)
    print(f'Training with {config.n_batches} batches')
    
    # define model
    if config.model_kind == 'hslstm':
        model = HSLSTMForDirection(config = config, class_weights = class_weights)
    else:
        model = None
    model.to(device)
    
    # optmizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor=0.5, patience=4)
    
    # Trainer
    results = trainer(config, model, optimizer, scheduler)
    
    ### SAVE RESULTS
    with open(os.path.join(config.dest_path, f'results{config.fold}.pkl'), 'wb') as f:
        pickle.dump(results, f)
        
    ## Remove parent folder of prep data
    utils.remove(root = config.prep_data_path)

    return results