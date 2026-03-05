import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(steps):
        linear_model = np.dot(X , weights) + bias

        y_predicted = _sigmoid(linear_model)

         # 2. Compute Gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # 3. Update Parameters
        weights -= lr * dw
        bias -= lr * db

        if epoch % 100 == 0:
            # Binary Cross Entropy Loss
            loss = -np.mean(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9))
            print(f"Epoch {epoch}: Loss {loss:.4f}")
    

    return weights, bias
        

    