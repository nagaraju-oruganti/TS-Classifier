########################################################################
## CREATE DATA FOR TRAINING AND MODELING
########################################################################

# IMPORT LIBRARIES
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# Surpress warnings
import warnings
warnings.filterwarnings('ignore')

# Garbage collector to free-up memory
import gc

### Import local classes
from utils.class_weights import estimator
from data.labeler import Labeler

################################################################################################
# PyTorch Dataset
################################################################################################
class MyDataset(Dataset):
    
    def __init__(self, config, df):
        self.config = config
        self.dt_column = config.dt_column
        self.df = df
        self.data = self.prepare_data()
        
    def prepare_data(self):
        
        # Parse problem specific configuration parameters
        features = self.config.features
        target  = self.config.target
        
        # Timeseries parameters
        max_len = self.config.max_len
        step_size = self.config.step_size
        
        data = []
        for tic in self.config.tickers:
            sub = self.df[self.df['ticker'] == tic]
            sub[self.dt_column] = pd.to_datetime(sub[self.dt_column])
            sub.sort_values(by = self.dt_column, ascending = False, inplace = True)
            for i in range(len(sub) - max_len, -1, -step_size):
                
                start, end = i, i+max_len       # boundery indices
                
                patch = sub[features][start:end]
                if len(patch) == max_len:
                    data.append(dict(
                        ticker      = tic,
                        start       = str(sub[self.dt_column].values[start]),
                        end         = str(sub[self.dt_column].values[end - 1]),
                        inputs      = patch.values,
                        target      = sub[target].values[end-1],
                    ))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
        
        ticker = sample['ticker'], 
        start  = sample['start']
        end    = sample['end']
        inputs = torch.tensor(sample['inputs'], dtype = torch.float32).permute(1, 0)
        target = torch.tensor(sample['target'], dtype = torch.long)
        
        return (ticker, start, end, inputs, target)
            
################################################################################################
# CREATE DATALOADERS FOR MODEL TRAINING
################################################################################################
class MakeDataloaders:
    
    def __init__(self, config, fold = 1):
        self.config = config
        self.fold = fold
        self.dt_column = config.dt_column
        self.max_len = self.config.max_len
        
    def read_data(self):
        ### Get normalized dataset from preprocessing
        prep = Labeler(config = self.config)
        df = prep.make()
        df = df[df['ticker'].isin(self.config.tickers)]
        return df
        
    def gather_datasets(self, df):
        
        ### Make train, validation datasets
        df[self.dt_column] = pd.to_datetime(df[self.dt_column])
        folds = self.config.folds[self.fold]
        
        # Train dataset
        (start, end) = folds['train']
        start = pd.to_datetime(start) + timedelta(hours = -self.max_len)
        self.train = df[(df[self.dt_column] >= start) & (df[self.dt_column] < end)]
        
        # validation dataset
        (start, end) = folds['valid']
        start = pd.to_datetime(start) + timedelta(hours = -self.max_len)
        self.valid = df[(df[self.dt_column] >= start) & (df[self.dt_column] < end)]
        
    def make(self):
        
        # Read data
        df = self.read_data()
        class_weights = estimator(df = df)
        
        # folds
        self.gather_datasets(df)
        
        print(self.train.shape, self.valid.shape)
        
        # Dataloaders
        train_loader = DataLoader(
            dataset     = MyDataset(config = self.config, 
                                    df = self.train if not self.config.sample_run else self.train.head(5000)),
            batch_size  = self.config.train_batch_size,
            shuffle     = True,
            drop_last   = False
        )
        
        valid_loader = DataLoader(
            dataset     = MyDataset(config = self.config, 
                                    df = self.valid if not self.config.sample_run else self.valid.head(5000)),
            batch_size  = self.config.valid_batch_size,
            shuffle     = True,
            drop_last   = False
        )
        
        print(f'Samples in train: {len(train_loader.dataset)}')
        print(f'Samples in valid: {len(valid_loader.dataset)}')
        
        return train_loader, valid_loader, None, class_weights
        
if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    preprocess = PreProcess(config=config)
    preprocess.run()
    _ = MakeDataloaders(config = config, fold = 1).make()