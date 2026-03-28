import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Perform a single forward step of a vanilla RNN with tanh activation.

    Args:
        x_t (np.ndarray): Input at time step t, shape (D,)
        h_prev (np.ndarray): Previous hidden state, shape (H,)
        Wx (np.ndarray): Input-to-hidden weights, shape (D, H)
        Wh (np.ndarray): Hidden-to-hidden weights, shape (H, H)
        b (np.ndarray): Bias vector, shape (H,)

    Returns:
        h_t (np.ndarray): Next hidden state, shape (H,)
    """

    # Step 1: Compute pre-activation (linear combination)
    pre_act = x_t @ Wx + h_prev @ Wh + b

    # Step 2: Apply tanh activation
    h_t = np.tanh(pre_act)

    return h_t