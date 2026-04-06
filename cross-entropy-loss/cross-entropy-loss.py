import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    Args:
        y_true: array-like of shape (N,) with class indices
        y_pred: array-like of shape (N, K) with predicted probabilities

    Returns:
        float: average cross-entropy loss, or None if shape mismatch
    """

    # Convert inputs to NumPy arrays
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=float)

    # Check if number of samples match
    if y_pred.shape[0] != y_true.shape[0]:
        return None

    # Select probabilities corresponding to correct class for each sample
    correct_class_probs = y_pred[np.arange(len(y_true)), y_true]

    # Compute negative log likelihood and take mean
    loss = -np.mean(np.log(correct_class_probs))

    return float(loss)