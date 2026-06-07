def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    rows = len(data)
    cols = len(data[0])

    # Initialize result matrix
    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        # Get all values in the current column
        column = [data[i][j] for i in range(rows)]

        min_val = min(column)
        max_val = max(column)
        range_val = max_val - min_val

        # If all values are the same, keep column as 0.0
        if range_val == 0:
            continue

        # Scale each value in the column
        for i in range(rows):
            result[i][j] = float((data[i][j] - min_val) / range_val)

    return result