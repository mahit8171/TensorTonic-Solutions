import numpy as np 

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True positives (correct predictions)
    TP = np.sum(y_true == y_pred)

    # count how many true

    # False positives and false negatives
    FP = np.sum(y_true != y_pred)

    # count how many false 
    
    FN = FP   # for single-label multi-class

    denominator = 2*TP + FP + FN
    if denominator == 0:
        return 0.0

    return (2*TP) / denominator