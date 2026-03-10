def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    A = set(set_a)
    B = set(set_b)
    if len(A) == 0 and len(B) == 0 :
        return 0.0

    intersrction = A & B
    union = A | B

    return len(intersrction)/len(union)