def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    H, W = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])

    # Step 1: Create padded image
    padded_H = H + 2 * padding
    padded_W = W + 2 * padding

    padded = [[0] * padded_W for _ in range(padded_H)]

    for i in range(H):
        for j in range(W):
            padded[i + padding][j + padding] = image[i][j]

    # Step 2: Compute output dimensions
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1

    output = [[0] * W_out for _ in range(H_out)]

    # Step 3: Perform convolution
    for i in range(H_out):
        for j in range(W_out):
            val = 0
            for m in range(kh):
                for n in range(kw):
                    val += padded[i * stride + m][j * stride + n] * kernel[m][n]
            output[i][j] = val

    return output