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
        batch_start = batch_start + timedelta(hours = -leap_back)
        batch_end = start_date + (i + 1) * batch_duration
        batch_boundaries.append((batch_start, batch_end))

    return batch_boundaries

def estimate_n_batches(config):
    in_freq = config.horizon_def['in_freq']
    # assuming 4 batches for 4-hours (240m) intervals
    if in_freq >= 60:
        return {'train' : 4, 'valid' : 1}
    elif in_freq >= 15:
        return {'train' : 8, 'valid' : 2}
    return {'train' : 24, 'valid' : 6}

import shutil,os
def remove(root): shutil.rmtree(root, ignore_errors=True)