def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Top-k recommended items
    top_k = recommended[:k]
    
    # Convert relevant list to set for fast lookup
    relevant_set = set(relevant)
    
    # Count how many recommended items are relevant
    hits = 0
    for item in top_k:
        if item in relevant_set:
            hits += 1
    
    # Compute precision and recall
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return [precision, recall]