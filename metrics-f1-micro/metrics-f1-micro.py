def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    if not y_true:
        return 0.0

    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1

    return correct / len(y_true)