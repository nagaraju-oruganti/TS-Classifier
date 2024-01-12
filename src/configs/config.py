### Configuration file for supervised learning problems

import os
import torch
import numpy as np

# Local modules
from utils.random_seed import seed_everything

INPUT_FREQ = [3, 5, 10, 15, 30, 60, 120, 150, 180, 240]
HORIZONS = list(range(2, 25, 3))
PCT_THRESHOLDS = [1, 2, 3, 4]

class Config:
    
    def apply_seed(self, seed):
        seed_everything(seed=seed)
        
    # DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # DATA CONFIGURATION
    data_dir = 'data'
    models_dir ='models'
    raw_data_path = '../../Datasets/data/new_binance/BTC_futures_1m.csv'
    model_name = 'delete'                     # creates path to save the trained models
    prep_data_path = ''
    
    # DATA SPLITS
    # Train, valid, and test splits
    splits = {
        'train': ('01-01-2020', '12-31-2022'),
        'test' : ('01-01-2023', '12-31-2023')
    }
    
    folds = {
        1: {
            'train': ('01-01-2020', '06-30-2021'),
            'valid': ('07-01-2021', '12-31-2021')
        },
        2: {
            'train': ('01-01-2020', '12-31-2021'),
            'valid': ('01-01-2022', '06-30-2022')
        },
        3: {
            'train': ('01-01-2020', '06-30-2022'),
            'valid': ('07-01-2022', '12-31-2022')
        },
    }
    
    ## PROBLEM DEFINITION
    ### (1) x% change in horizon periods (h) with input data from time interval (t)
    horizon_def = {
        'in_freq'       : 30,           # minutes
        'pct_thresh'    : 2/100.,
        'lookahead'     : 60 * 4,       # minutes
    }
    project_on_intrarange = False
    ## TICKERS
    tickers = ['1000SHIB', 'ADA', 'ATOM', 'AVAX', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETC', 'ETH',
               'FIL', 'ICP', 'LDO', 'LINK', 'LTC', 'MATIC', 'SOL', 'TRX', 'UNI', 'XLM', 'XMR', 'XRP']
    
    ## Model definition
    model_kind = 'hslstm'
    output_size = 4
    model_params = dict(
        hslstm={
            'num_layers': 4,
            'dropout_prob': 0,
        }
    )
    
    ## Dataset parameters
    max_len = 64
    step_size = 1
    features = ['High', 'Low', 'Close', 'pct_chg_close', 'pct_chg_volume']
    target = 'label'
    dt_column = 'Datetime'
    
    ## Train parameters
    train_batch_size = 16
    valid_batch_size = 32
    sample_run = False
    learning_rate = 5e-5
    lr_multiplier = 0.5
    lr_patience = 4
    num_epochs = 1000
    iters_to_accumulate = 4
    
    ## Train convergence parameters
    save_epoch_wait = 0
    early_stop_count = 10
    save_checkpoint = True
    
    ## Intermin output
    show_label_distributions = True
    
    ## MISCELLANEOUS
    LABEL_MAPPER = {
        'neither'   : 0,
        'long'      : 1,
        'short'     : 2,
        'either'    : 3}
    REVERSE_LABEL_MAPPER = {v:k for k, v in LABEL_MAPPER.items()}
    
    ## Train with batches
    n_batches = {'train': 4, 'valid' : 2, 'test' : 2}
    train_with_external_drive = False