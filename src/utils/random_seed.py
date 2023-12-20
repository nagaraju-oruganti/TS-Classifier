
'''Module providing a function to random seed for numpy, pytorch and os'''

import random
import os
import numpy as np
import torch


def seed_everything(seed: int):
    '''
        function to apply random seed
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f'set seed to {seed}')
