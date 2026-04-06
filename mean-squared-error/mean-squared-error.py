import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Compute Mean Squared Error (MSE) between predictions and targets.

    Args:
        y_pred: array-like of predicted values
        y_true: array-like of true values

    Returns:
        float: MSE value, or None if shapes do not match
    """

    # Convert inputs to NumPy arrays for vectorized operations
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)

    # Return None if shapes are not equal
    if y_pred.shape != y_true.shape:
        return None

    # Compute average of squared differences
    return float(np.mean((y_pred - y_true) ** 2))