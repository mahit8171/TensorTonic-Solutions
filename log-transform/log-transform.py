import math

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    return [float(math.log1p(v)) for v in values]