import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v)

    # Case 1: Single vector (shape: (3,))
    if v.ndim == 1:
        return float(np.sqrt(np.sum(v**2)))

    # Case 2: Batch of vectors (shape: (N, 3))
    elif v.ndim == 2:
        return np.sqrt(np.sum(v**2, axis=1))

    else:
        raise ValueError("Input must be shape (3,) or (N, 3)")