import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.

    Args:
        p (array-like): Predicted probabilities
        y (array-like): Ground truth (0 or 1)
        eps (float): Small value for numerical stability

    Returns:
        float: Dice loss
    """

    # Convert to numpy arrays (float)
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Flatten to handle both 1D and 2D
    p = p.flatten()
    y = y.flatten()

    # Intersection
    intersection = np.sum(p * y)

    # Sum of predictions and ground truth
    sum_p = np.sum(p)
    sum_y = np.sum(y)

    # Dice coefficient
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)

    # Dice loss
    return 1 - dice