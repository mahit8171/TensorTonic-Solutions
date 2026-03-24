import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.

    Parameters:
    p (list or np.ndarray): Predicted probabilities, shape (N,)
    y (list or np.ndarray): True binary labels (0 or 1), shape (N,)
    gamma (float): Focusing parameter (gamma >= 0)

    Returns:
    float: Mean focal loss over all samples
    """

    # Convert inputs to NumPy arrays (important for vectorized operations)
    p = np.array(p)
    y = np.array(y)

    # Small value to prevent log(0), which is undefined
    eps = 1e-12

    # Clip probabilities to range [eps, 1-eps]
    # This ensures numerical stability
    p = np.clip(p, eps, 1 - eps)

    # Compute focal loss for each sample using formula:
    # FL = - (1 - p)^gamma * y * log(p)
    #      - (p^gamma) * (1 - y) * log(1 - p)
    loss = - (1 - p) ** gamma * y * np.log(p) \
           - (p ** gamma) * (1 - y) * np.log(1 - p)

    # Return mean loss across all samples
    return np.mean(loss)