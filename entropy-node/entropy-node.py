import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # count occurrences of each class
    values, counts = np.unique(y, return_counts=True)
    
    # convert counts to probabilities
    probs = counts / counts.sum()
    
    # compute entropy (ignore log(0))
    entropy = -np.sum(probs * np.log2(probs ))
    
    return entropy