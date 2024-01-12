import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc

## local libraries
from data.dataset import MakeDataloaders, estimate_class_weights
from models.lstm import HSLSTMForDirection
from utils import utils

class Predictor():
    def __init__(self, config, n_batches):
        self.config = config
        self.device = config.device
        self.n_batches = n_batches
        self.prep()
        
    def prep(self):
        # dataset
        self.data_maker = MakeDataloaders(config = self.config, fold = self.config.fold)
        
        # model
        class_weights = estimate_class_weights(self.config, fold = self.config.fold)
        self.model = HSLSTMForDirection(config = self.config, class_weights = class_weights)
        
        # Load saved model
        self.repo_path = f'{self.config.models_dir}/{self.config.model_name}'
        self.model.load_state_dict(torch.load(f'{self.repo_path}/model3.pth')['model_state_dict'])
        self.model.to(self.device)
        
    def make_dataset(self, batch_idx):
        return self.data_maker.make_test(batch_idx = batch_idx, verbose = False)
    
    def batch_predictions(self, dataloader):
        results = []
        self.model.eval()
        with torch.no_grad():
            for (ticker, start, end, inputs, targets) in dataloader:
                logits = self.model(inputs.to(self.config.device), y = None)
                probs = F.softmax(logits, dim = 1)
                preds = torch.argmax(probs, dim = 1)
                
                preds = preds.to('cpu').numpy().tolist()
                probs = probs.to('cpu').numpy().tolist()
                targets = targets.to('cpu').numpy().tolist()

                # save
                for i, _ in enumerate(targets):
                    
                    tic = ticker[0][i]
                    s = start[i]
                    e = end[i]
                    
                    item = [tic, s, e, targets[i]]
                    item.extend(probs[i])
                    item.append(preds[i])
                    
                    results.append(item)
        
        return results
            
    def predict(self):
        results = []
        for idx in range(self.n_batches):
            dataloader = self.make_dataset(batch_idx = idx)
            batch_results = self.batch_predictions(dataloader = dataloader)
            results.extend(batch_results)
            del dataloader
            _ = gc.collect()

        # remove data from external drive
        utils.remove(root = self.config.prep_data_path)
        
        # save
        cols = ['ticker', 'start', 'end', 'target', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'pred_label']
        df = pd.DataFrame(results, columns = cols)
        df.to_csv(f'{self.repo_path}/test_results.csv', index = False)
        print('Prediction is complete')