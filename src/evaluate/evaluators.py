import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

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
            
            preds = preds.numpy().tolist()
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
    return score, eval_loss, results