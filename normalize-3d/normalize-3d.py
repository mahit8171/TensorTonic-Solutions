import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.

    Args:
        v: list or np.ndarray of shape (3,) or (N, 3)

    Returns:
        np.ndarray of same shape as input, dtype float
    """

    # Convert input to NumPy array of floats
    # - Handles list/tuple/array input
    # - Ensures decimal division (important for normalization)
    # - Avoids unnecessary copy if already ndarray
    v = np.asarray(v, dtype=float)

    # Check if input is a single vector (shape = (3,))
    is_single = (v.ndim == 1)

    # If single vector → reshape to (1,3) so we can treat everything as batch
    if is_single:
        v = v.reshape(1, 3)

    # Compute L2 norm (length) of each vector
    # axis=1 → compute across columns (x, y, z)
    # keepdims=True → keeps shape (N,1) instead of (N,)
    # This helps in broadcasting during division
    norms = np.linalg.norm(v, axis=1, keepdims=True)

    # Small threshold to avoid division by zero
    eps = 1e-10

    # Create boolean mask:
    # True → valid vector (norm > 0)
    # False → zero vector (norm ≈ 0)
    mask = norms > eps   # shape: (N,1)

    # Create output array filled with zeros
    # Same shape as input
    # Zero vectors will remain unchanged
    out = np.zeros_like(v)

    # Convert mask from (N,1) → (N,) for row indexing
    valid = mask[:, 0]

    # Normalize only valid (non-zero) vectors
    # v[valid] → select only rows where norm > 0
    # norms[valid] → corresponding norms (shape: (k,1))
    # Broadcasting: (k,3) / (k,1) → (k,3)
    out[valid] = v[valid] / norms[valid]

    # If original input was single vector → return shape (3,)
    # Else return batch shape (N,3)
    return out[0] if is_single else out