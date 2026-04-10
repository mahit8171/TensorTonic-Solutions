import numpy as np

def softmax(x):
    """
    Compute softmax for 1D or 2D numpy array.
    
    Parameters:
    x : np.ndarray
    
    Returns:
    np.ndarray (same shape as x)
    """
    # Convert to numpy array (safe)
    x = np.array(x)
    
    if x.ndim == 1:
        # Subtract max for numerical stability
        x_stable = x - np.max(x)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x)
    
    elif x.ndim == 2:
        # Apply row-wise
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    else:
        raise ValueError("Input must be 1D or 2D array")