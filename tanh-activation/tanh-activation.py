import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.array(x)
    y = x * -1

    return ((np.exp(x) - np.exp(y))/(np.exp(x) + np.exp(y)))