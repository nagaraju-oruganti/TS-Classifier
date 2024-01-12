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
from utils.utils import divide_date_range
from data.labeler import Labeler

################################################################################################
# PyTorch Dataset
################################################################################################
class MyDataset(Dataset):
    
    def __init__(self, config, df, data = None, original_start_date = None):
        self.config = config
        self.dt_column = config.dt_column
        self.df = df
        self.original_start_date = pd.to_datetime('01/01/1970') if original_start_date is None else original_start_date
        self.data = self.prepare_data() if data is None else data
        
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
            sub.sort_values(by = self.dt_column, ascending = True, inplace = True)
            for i in range(len(sub) - max_len, -1, -step_size):
                
                start, end = i, i+max_len       # boundery indices
                if sub[self.dt_column].values[end-1] >= self.original_start_date:
                    patch = sub[features][start:end]
                    if len(patch) == max_len:
                        data.append(dict(
                            ticker      = tic,
                            start       = str(sub[self.dt_column].values[start]),
                            end         = str(sub[self.dt_column].values[end - 1]),
                            inputs      = patch.values,
                            target      = sub[target].values[end - 1],
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
        self.init_data()
        
    def init_data(self):
        df = self.read_data()               # Read data
        class_weights = estimator(df = df)
        self.gather_datasets(df)            # folds
        
    def read_data(self):
        ### Get normalized dataset from preprocessing
        prep = Labeler(config = self.config)
        df = prep.make(verbose = False)
        df = df[df['ticker'].isin(self.config.tickers)]
        return df
        
    def gather_datasets(self, df):
        
        ### Make train, validation datasets
        df[self.dt_column] = pd.to_datetime(df[self.dt_column])
        print(df[self.dt_column].min(), df[self.dt_column].max())
        folds = self.config.folds[self.fold]
        
        # Train dataset
        (start, end) = folds['train']
        start   = pd.to_datetime(start) + timedelta(hours = -self.max_len)
        end     = pd.to_datetime(end)
        self.train = df[(df[self.dt_column] >= start) & (df[self.dt_column] < end)]
        self.train_date_ranges = self.date_boundaries(start, end, kind = 'train')
        
        # validation dataset
        (start, end) = folds['valid']
        start   = pd.to_datetime(start) + timedelta(hours = -self.max_len)
        end     = pd.to_datetime(end)
        self.valid = df[(df[self.dt_column] >= start) & (df[self.dt_column] < end)]
        self.valid_date_ranges = self.date_boundaries(start, end, kind = 'valid')

        # test dataset
        (start, end) = self.config.splits['test']
        start   = pd.to_datetime(start) + timedelta(hours = -self.max_len)
        end     = pd.to_datetime(end)
        self.test = df[(df[self.dt_column] >= start) & (df[self.dt_column] <= end)]
        self.test_date_ranges = self.date_boundaries(start, end, kind = 'test')
        
        ## Data mappers
        self.data_mapper = {'train': self.train, 'valid': self.valid, 'test': self.test}
        self.date_range_mapper = {'train': self.train_date_ranges, 'valid': self.valid_date_ranges, 'test': self.test_date_ranges}
        
    def date_boundaries(self, start, end, kind = 'train'):
       return divide_date_range(start, end, num_batches = self.config.n_batches[kind], leap_back = -self.max_len)
    
    def make_data(self, kind, batch_idx, verbose = False):
        prep_data_path = os.path.join(self.config.prep_data_path, self.config.model_name, kind)
        os.makedirs(prep_data_path, exist_ok = True)
        data_path = os.path.join(prep_data_path, f'batch_{batch_idx}.pkl')
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                batch_data = pickle.load(f)
            dataset = MyDataset(config = self.config, df = pd.DataFrame(), data = batch_data)
        else:
            df = self.data_mapper[kind].copy()
            original_start_date, start_date, end_date = self.date_range_mapper[kind][batch_idx]
            start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
            df = df[(df[self.dt_column] >= start_date) & (df[self.dt_column] <= end_date)]
            dataset = MyDataset(config = self.config, df = df, data = None, original_start_date = original_start_date)
            with open(data_path, 'wb') as f:
                pickle.dump(dataset.data, f)
            
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.config.train_batch_size if kind == 'train' else self.config.valid_batch_size,
            shuffle = kind == 'train',
            drop_last = False
        )
        if verbose:
            print(f'Samples in {kind}: {len(dataloader.dataset)} | {df.shape}')
        return dataloader
    
    def make_train(self, batch_idx, verbose = False):
        return self.make_data(kind = 'train', batch_idx = batch_idx, verbose = verbose)
    
    def make_valid(self, batch_idx, verbose = False):
        return self.make_data(kind = 'valid', batch_idx = batch_idx, verbose = verbose)
    
    def make_test(self, batch_idx, verbose = False):
        return self.make_data(kind = 'test', batch_idx = batch_idx, verbose = verbose)
        
    def make(self, batch_idx, verbose = True):
        
        train_df = self.train.copy()
        valid_df = self.valid.copy()
        
        start_date, end_date = self.train_date_ranges[batch_idx]
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        train_df = train_df[(train_df[self.dt_column] >= start_date) & (train_df[self.dt_column] <= end_date)]
        
        start_date, end_date = self.valid_date_ranges[batch_idx]
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        valid_df = valid_df[(valid_df[self.dt_column] >= start_date) & (valid_df[self.dt_column] <= end_date)]

        print(train_df.shape, valid_df.shape)

        train_dataset = MyDataset(config = self.config, df = train_df)
        valid_dataset = MyDataset(config = self.config, df = valid_df)
        
        # Dataloaders
        train_loader = DataLoader(
            dataset     = train_dataset,
            batch_size  = self.config.train_batch_size,
            shuffle     = True,
            drop_last   = False
        )
        
        valid_loader = DataLoader(
            dataset     = valid_dataset,
            batch_size  = self.config.valid_batch_size,
            shuffle     = True,
            drop_last   = False,
            pin_memory  = True
        )
        
        if verbose:
            print(f'Samples in train: {len(train_loader.dataset)} | {train_df.shape}')
            print(f'Samples in valid: {len(valid_loader.dataset)} | {valid_df.shape}')
        
        return train_loader, valid_loader, None
    
### Estimate class weights
def estimate_class_weights(config, fold):
    prep = Labeler(config = config)
    df = prep.make()
    df = df[df['ticker'].isin(config.tickers)]
    class_weights = estimator(df = df)
    return class_weights
        
if __name__ == '__main__':
    from helper_config import Config
    config = Config()
    preprocess = PreProcess(config=config)
    preprocess.run()
    _ = MakeDataloaders(config = config, fold = 1).make()