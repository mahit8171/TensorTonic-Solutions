import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Performs forward pass of Batch Normalization.

    Parameters:
    x     : Input data (N,D) or (N,C,H,W)
    gamma : Scale parameter (D,) or (C,)
    beta  : Shift parameter (D,) or (C,)
    eps   : Small constant for numerical stability

    Returns:
    y     : Normalized, scaled, shifted output
    """

    # 🔹 Convert inputs to NumPy arrays (fix for list input issue)
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)

    
    #  Case 1: 2D Input (N, D)
   
    if x.ndim == 2:
        # Compute mean for each feature (column-wise)
        mean = np.mean(x, axis=0, keepdims=True)

        # Compute variance for each feature
        var = np.var(x, axis=0, keepdims=True)

        # Normalize input
        # (x - mean) / sqrt(var + eps)
        x_hat = (x - mean) / np.sqrt(var + eps)

        # Scale and shift
        # gamma and beta automatically broadcast over rows
        y = gamma * x_hat + beta

   
    #  Case 2: 4D Input (N, C, H, W)
    
    elif x.ndim == 4:
        # Compute mean per channel over (N, H, W)
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)

        # Compute variance per channel
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        # Normalize
        x_hat = (x - mean) / np.sqrt(var + eps)

        # Reshape gamma and beta for broadcasting
        # (C,) → (1, C, 1, 1)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        # Scale and shift
        y = gamma * x_hat + beta

    else:
        # Invalid input shape
        raise ValueError("Input must be 2D or 4D")

    return y