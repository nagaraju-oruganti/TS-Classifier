from datetime import datetime, timedelta

def divide_date_range(start_date, end_date, num_batches, leap_back):
    '''
        Leap back is the parameter that decides overlap durations for continuous input generation
    '''
    
    # duration
    total_duration = end_date - start_date
    batch_duration = total_duration / num_batches

    # boundaries
    batch_boundaries = []
    for i in range(num_batches):
        batch_start = start_date + i * batch_duration
        original_start_date = batch_start
        batch_start = batch_start + timedelta(hours = -leap_back)
        batch_end = start_date + (i + 1) * batch_duration
        batch_boundaries.append((original_start_date, batch_start, batch_end))     
        #print((i, original_start_date, batch_start, batch_end))
        
    return batch_boundaries

def estimate_n_batches(config):
    
    if config.train_with_external_drive:
        return {'train': 1, 'valid': 1, 'test': 1}
    
    in_freq = config.horizon_def['in_freq']
    # assuming 4 batches for 4-hours (240m) intervals
    if in_freq >= 30:
        return {'train' : 4, 'valid' : 1, 'test' : 4}
    elif in_freq >= 15:
        return {'train' : 8, 'valid' : 2, 'test' : 6}
    elif in_freq >= 10:
        return {'train' : 12, 'valid' : 4, 'test' : 6}
    elif in_freq >= 5:
        return {'train' : 32, 'valid' : 16, 'test' : 32}
    return {'train' : 48, 'valid' : 24, 'test' : 48}

import shutil,os
def remove(root): shutil.rmtree(root, ignore_errors=True)