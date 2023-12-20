import torch

def estimator(df):
    
    # https://naadispeaks.blog/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
    
    class_weights = []
    dist = dict(df['label'].value_counts())
    
    total_samples = len(df)
    for c in [0, 1, 2, 3]:
        class_weights.append(
            1 - (dist[c] / total_samples)
        )
    
    return class_weights
