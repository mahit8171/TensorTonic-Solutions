def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here 
    n = len(values)
    k = len(weights)
    
    result = []
    weight_sum = sum(weights)

    for i in range(n - k + 1):
        weighted_sum = 0
        
        for j in range(k):
            weighted_sum += weights[j] * values[i + j]
        
        result.append(weighted_sum / weight_sum)

    return result

    