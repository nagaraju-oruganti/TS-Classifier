import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
import gc

def evaluate_with_batches(model, dataloader, n_batches, device, verbose):
    train_y_trues, train_y_preds = [], []
    train_loss_list, valid_loss_list = [], []
    for batch_idx in range(n_batches['train']):
        # dataloaders for batch
        train_loader = dataloader.make_train(batch_idx, verbose = verbose)
        # Train dataset
        _, train_loss, y_trues, y_preds, _ = evaluator(model, train_loader, device = device)
        train_loss_list.append(train_loss)
        train_y_trues.extend(y_trues)
        train_y_preds.extend(y_preds)
        
        del train_loader, y_trues, y_preds
        _ = gc.collect()
        
    
    valid_y_trues, valid_y_preds = [], []
    valid_results = []
    for batch_idx in range(n_batches['valid']):
        # valid dataset
        valid_loader = dataloader.make_valid(batch_idx, verbose = verbose)
        _, valid_loss, y_trues, y_preds, results = evaluator(model, valid_loader, device = device)
        valid_loss_list.append(valid_loss)
        valid_y_trues.extend(y_trues)
        valid_y_preds.extend(y_preds)
        valid_results.extend(results)
        
        del valid_loader, y_trues, y_preds, results
        _ = gc.collect()
        
    # scoring
    dict_output = {
        'train_score'   : f1_score(y_true = train_y_trues, y_pred = train_y_preds, average = 'weighted'),
        'valid_score'   : f1_score(y_true = valid_y_trues, y_pred = valid_y_preds, average = 'weighted'),
        'train_loss'    : np.mean(train_loss_list),
        'valid_loss'    : np.mean(valid_loss_list),
        'eval_results'  : valid_results
    }
    
    return dict_output
        

def evaluator(model, dataloader, device = 'cpu'):
    results = []
    y_trues, y_preds = [], []
    batch_loss_list = []
    model.eval()
    with torch.no_grad():
        for (ticker, start, end, inputs, targets) in dataloader:
            logits, loss = model(inputs.to(device), targets.to(device))
            batch_loss_list.append(loss.item())
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
            
            y_trues.extend(targets)
            y_preds.extend(preds)
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    # scoring
    score = f1_score(y_true = y_trues, y_pred = y_preds, average = 'weighted')
    eval_loss = np.mean(batch_loss_list)
    return score, eval_loss, y_trues, y_preds, results