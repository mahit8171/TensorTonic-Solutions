import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Convert to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Compute L2 distance
    return float(np.sqrt(np.sum((x - y) ** 2)))