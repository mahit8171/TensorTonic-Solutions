import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    
    # Convert lists to numpy arrays
    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    # Step 1: update first moment
    m_t = beta1 * m + (1 - beta1) * grad

    # Step 2: update second moment
    v_t = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: Nesterov adjusted momentum
    nesterov = beta1 * m_t + (1 - beta1) * grad

    # Step 4: update parameters
    w_t = w - lr * (nesterov / (np.sqrt(v_t) + eps))

    return w_t.tolist(), m_t.tolist(), v_t.tolist()