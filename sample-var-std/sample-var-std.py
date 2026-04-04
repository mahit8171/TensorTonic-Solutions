import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    if n < 2:
        raise ValueError("At least 2 data points required")

    mean = np.mean(x)
    
    # Bessel's correction (ddof=1)
    var = np.sum((x - mean) ** 2) / (n - 1)
    
    std = np.sqrt(var)
    
    return var, std