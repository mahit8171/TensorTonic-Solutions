import numpy as np

def clip_gradients(g, max_norm):
    # Ensure numpy array
    g = np.array(g, dtype=float)
    
    # Edge case: no clipping if max_norm <= 0
    if max_norm <= 0:
        return g
    
    # Compute L2 norm
    norm = np.linalg.norm(g)
    
    # Clip only if needed
    if norm > max_norm:
        g = g * (max_norm / norm)
    
    return g