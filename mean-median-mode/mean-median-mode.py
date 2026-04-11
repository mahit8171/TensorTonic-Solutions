import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Mean
    mean = float(np.mean(x))
    
    # Median
    median = float(np.median(x))
    
    # Mode (smallest value in case of tie)
    freq = Counter(x)
    max_freq = max(freq.values())
    mode = min([k for k, v in freq.items() if v == max_freq])
    
    return mean, median, mode