def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    
    Args:
        sentences (list[list[str]]): List of tokenized sentences
    
    Example:
        [["i", "love", "ml"], ["i", "love", "coding"]]
        -> {"i": 2, "love": 2, "ml": 1, "coding": 1}
    """
    
    freq = {}  # Initialize empty dictionary
    
    # Iterate through each sentence
    for sentence in sentences:
        # Iterate through each word in the sentence
        for word in sentence:
            # Update count
            freq[word] = freq.get(word, 0) + 1
    
    return freq