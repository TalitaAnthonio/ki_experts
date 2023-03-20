def compute_precision_at_k(golden_predictions, predictions, k): 
    """Evaluate the performance.
    On average, how many instances does it predict correctly"""

    correct_predictions = 0 
    for prediction in predictions: 
        if prediction in golden_predictions: 
            correct_predictions +=1 

    precision_at_k = correct_predictions/(min(len(golden_predictions), k))
    return precision_at_k