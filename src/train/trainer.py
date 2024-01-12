import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp              # mixed precision
from torch import autocast
from torchsummary import summary
from datetime import datetime

from sklearn.metrics import mean_squared_error

# local modules
from evaluate import evaluators
from data import dataset

import warnings
warnings.filterwarnings('ignore')

import gc

#TODO: Make this function into a class

#### Trainer
def trainer(config, model, optimizer, scheduler):
    
    ## DATALOADERS
    dataloaders = dataset.MakeDataloaders(config = config, fold = config.fold)
    
    def update_que():
        que.set_postfix({
            'batch_loss'        : f'{batch_loss_list[-1]:4f}',
            'epoch_loss'        : f'{np.mean(batch_loss_list):4f}',
            'learning_rate'     : optimizer.param_groups[0]["lr"],
            })
    
    def save_checkpoint(model, epoch, eval_results, df_eval_results, best = False):
        if best:
            save_path = os.path.join(config.dest_path, f'model{config.fold}.pth')
            if config.save_checkpoint:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }
                torch.save(checkpoint, save_path)
            
            # save evaluation results
            with open(os.path.join(config.dest_path, f'eval_results{config.fold}.pkl'), 'wb') as f:
                pickle.dump(eval_results, f)
                
            # in dataframe format
            df_eval_results.to_csv(os.path.join(config.dest_path, f'df_eval_results{config.fold}.csv'), index = False)
                
            print(f'>>> [{datetime.now()}] - Checkpoint and predictions saved')
        
    def dis(x): return f'{x:.6f}'
        
    def run_evaluation_sequence(ref_score, counter):
        
        def print_result():
            print('')
            text =  f'>>> [{datetime.now()} | {epoch + 1}/{NUM_EPOCHS} | Early stopping counter {counter}] \n'
            text += f'    loss          - train: {dis(train_loss)}     valid: {dis(valid_loss)} \n'
            text += f'    score         - train: {dis(train_score)}     valid: {dis(valid_score)} \n'
            text += f'    learning rate        : {optimizer.param_groups[0]["lr"]:.5e}'
            print(text + '\n')
        
        # Evaluation
        dict_output = evaluators.evaluate_with_batches(model, dataloaders, config.n_batches, device, verbose = False)
        train_score, train_loss = dict_output['train_score'], dict_output['train_loss']
        valid_score, valid_loss = dict_output['valid_score'], dict_output['valid_loss']
        eval_results = dict_output['eval_results']
        df_eval_results = dict_output['df_eval_results']
        
        # append results
        lr =  optimizer.param_groups[0]["lr"]
        results.append((epoch, train_loss, valid_loss, train_score, valid_score, lr))
        
        # Learning rate scheduler
        eval_metric = valid_score
        scheduler.step(eval_metric)           # apply scheduler on validation accuracy
        
        ### Save checkpoint
        if ((epoch + 1) > config.save_epoch_wait):
            save_checkpoint(model, epoch, eval_results, df_eval_results, best = eval_metric > ref_score)
        
        # Tracking early stop
        counter = 0 if eval_metric >= ref_score else counter + 1
        ref_score = max(ref_score, eval_metric)
        done = counter >= config.early_stop_count
        
        # show results
        print_result()
        
        # Save results
        with open(os.path.join(config.dest_path, f'results{config.fold}.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        return ref_score, counter, done 
    
    ### MIXED PRECISION
    scaler = amp.GradScaler()
    
    results = []
    device = config.device
    precision = torch.bfloat16 if str(device) == 'cpu' else torch.float16
    NUM_EPOCHS = config.num_epochs
    n_batches = config.n_batches['train']
    iters_to_accumlate = config.iters_to_accumulate
    
    # dummy value for placeholders
    eval_results = []
    ref_score, counter = 1e-3, 0
    train_loss, valid_loss, train_f1, valid_f1 = 0, 0, 0, 0
    
    ## Evaluation baseline before training
    print('Baseline:')
    epoch, bidx = -1, 0
    ref_score, counter, done = run_evaluation_sequence(ref_score, counter)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        batch_loss_list = []
        for batch_idx in range(n_batches):
            train_loader = dataloaders.make_train(batch_idx = batch_idx)
            que = tqdm(enumerate(train_loader), total = len(train_loader), desc = f'Epoch {epoch + 1} | batch {batch_idx + 1} / {n_batches}')
            for i, ((_,_,_, inputs, targets)) in que:
                
                ###### TRAINING SECQUENCE            
                #with autocast(device_type = str(device), dtype = precision):
                with autocast(enabled=True, device_type = str(device), dtype=precision) as _autocast, \
                    torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
                    _, loss = model(inputs.to(device), targets.to(device))            # Forward pass
                    loss = loss / iters_to_accumlate
                
                # - Accmulates scaled gradients  
                scaler.scale(loss).backward()           # scale loss
                
                if (i + 1) % iters_to_accumlate == 0:
                    scaler.step(optimizer)                  # step
                    scaler.update()
                    optimizer.zero_grad()
                #######
                
                batch_loss_list.append((loss.item() * iters_to_accumlate))
                
                # Update que status
                update_que()
                
            # clean memory
            del train_loader
            _ = gc.collect()
            
        ### Run evaluation sequence
        ref_score, counter, done = run_evaluation_sequence(ref_score, counter)
        if done:
            return results
            
    return results

def save_config(config, path):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)