import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    G = np.array(G, dtype=float)

    
    G = G + g**2

    
    w = w - (lr * g) / np.sqrt(G + eps)

    return w.tolist(), G.tolist()