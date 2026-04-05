def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    
    Parameters:
        X (list of list of float): Feature matrix (n x d)
        y (list of float): Target vector (n,)
        lam (float): Regularization parameter (lambda)
    
    Returns:
        list of float: Weight vector (d,)
    """
    import numpy as np

    # Convert inputs to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Number of features (d)
    d = X.shape[1]

    # Identity matrix (d x d)
    I = np.eye(d)

    # Compute ridge regression formula
    w = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

    return w.tolist()
    
    