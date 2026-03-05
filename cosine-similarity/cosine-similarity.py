import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    

    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a , b)
    euclidean_a = np.linalg.norm(a)
    euclidean_b = np.linalg.norm(b)
    euclidean_norm = euclidean_a*euclidean_b
    
    if(euclidean_norm == 0):
        return 0.00

    return dot/ euclidean_norm

    
    