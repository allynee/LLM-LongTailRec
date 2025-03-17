"""
This file functions and examples to calculate Diversity, Coverage, NDCG@10.
These metrics can be imported and used for model evaluation across various tasks.
Other common metrics are available in scikit-learn.

Metrics can be imported as follows from within /baselines/script/:

import metrics
if __name__ == '__main__':
    relevance = [3, 2, 3, 0, 1, 2, 3, 2, 3, 0]  # True relevance scores
    ranking = [2, 0, 6, 8, 5, 1, 4, 3, 7, 9]    # Predicted ranking (as indices)
    
    # Calculate NDCG at different k values
    ndcg_5 = metrics.ndcg_at_k(relevance, ranking, k=5)
    print(ndcg_5)
"""
from typing import List, Dict, Union, Tuple, Optional, Callable, Any, Set
import numpy as np
from collections import Counter
from sklearn.metrics import (
    ndcg_score
)

# Ranking and Recommendation Metrics
def ndcg_at_k(y_true: List[int], y_pred: List[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank k (NDCG@k).
    
    NDCG@k = DCG@k / IDCG@k
    
    Where:
    DCG@k = Σ(2^relevance_i - 1) / log2(i + 1), for i from 1 to k
    IDCG@k = DCG@k for the ideal ranking
    
    Parameters:
    -----------
    y_true : List[int]
        Ground truth relevance scores (higher is more relevant)
    y_pred : List[int]
        Predicted ranking of items
    k : int, optional
        Rank position to calculate NDCG for
        
    Returns:
    --------
    float
        NDCG@k score between 0.0 and 1.0
    """
    try:
        # Convert to numpy arrays for sklearn's ndcg_score
        # Reshape for sklearn's expected format [n_samples, n_labels]
        true_relevance = np.asarray([y_true])
        
        # Create a scores matrix where the predicted ranks get higher scores
        # (reverse of rank position to make higher ranks have higher scores)
        scores = np.zeros((1, len(y_true)))
        for i, idx in enumerate(y_pred[:k]):
            # Give score based on rank position (higher rank = higher score)
            scores[0, idx] = len(y_pred) - i
        
        return ndcg_score(true_relevance, scores, k=k)
    except (ValueError, ImportError):
        # Fall back to original implementation if scikit-learn ndcg_score isn't available
        # or has formatting issues
        def dcg_at_k(r: List[int], k: int) -> float:
            """Calculate DCG@k."""
            r = np.array(r)[:k]
            return np.sum((2 ** r - 1) / np.log2(np.arange(1, len(r) + 1) + 1))
        
        # Get relevance scores for predicted ranking
        actual_relevance = [y_true[i] for i in y_pred[:k]]
        
        # Calculate DCG@k
        dcg = dcg_at_k(actual_relevance, k)
        
        # Calculate IDCG@k (DCG@k with perfect ranking)
        ideal_relevance = sorted(y_true, reverse=True)[:k]
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0  # Avoid division by zero
            
        return dcg / idcg


def diversity(recommendations: List[List[int]], item_features: Dict[int, List[Any]]) -> float:
    """
    Calculate diversity of recommendations.
    
    Diversity measures how different the recommended items are from each other.
    Higher diversity means more varied recommendations.
    
    Parameters:
    -----------
    recommendations : List[List[int]]
        List of recommendation lists (item IDs) for each user
    item_features : Dict[int, List[Any]]
        Dictionary mapping item IDs to their feature vectors
        
    Returns:
    --------
    float
        Diversity score between 0.0 and 1.0
    """
    # This is a specialized metric not available in scikit-learn
    # Calculate average pairwise distance between items in recommendations
    
    diversity_scores = []
    
    for user_recs in recommendations:
        if len(user_recs) <= 1:
            continue
            
        # Calculate all pairwise distances
        distances = []
        for i in range(len(user_recs)):
            for j in range(i + 1, len(user_recs)):
                item_i = user_recs[i]
                item_j = user_recs[j]
                
                # Only consider items that exist in our feature dictionary
                if item_i in item_features and item_j in item_features:
                    # Calculate Jaccard distance between feature vectors
                    features_i = set(item_features[item_i])
                    features_j = set(item_features[item_j])
                    
                    if len(features_i.union(features_j)) == 0:
                        continue
                        
                    jaccard_distance = 1 - len(features_i.intersection(features_j)) / len(features_i.union(features_j))
                    distances.append(jaccard_distance)
        
        if distances:
            diversity_scores.append(np.mean(distances))
    
    # Calculate average diversity across all users
    if not diversity_scores:
        return 0.0
        
    return np.mean(diversity_scores)


def coverage(recommendations: List[List[int]], catalog_size: int) -> float:
    """
    Calculate catalog coverage of recommendations.
    
    Coverage measures the percentage of items in the catalog that are recommended at least once.
    
    Parameters:
    -----------
    recommendations : List[List[int]]
        List of recommendation lists (item IDs) for each user
    catalog_size : int
        Total number of items in the catalog
        
    Returns:
    --------
    float
        Coverage score between 0.0 and 1.0
    """
    # This is a specialized recommendation metric not available in scikit-learn
    # Flatten all recommendations and count unique items
    all_recommended_items = set()
    for user_recs in recommendations:
        all_recommended_items.update(user_recs)
    
    # Calculate coverage
    coverage_score = len(all_recommended_items) / catalog_size if catalog_size > 0 else 0.0
    
    return coverage_score


# Testing code
if __name__ == "__main__":
    print("===== Ranking and Recommendation Metrics Demo =====")
    
    # Sample data for NDCG@k
    relevance = [3, 2, 3, 0, 1, 2, 3, 2, 3, 0]  # True relevance scores
    ranking = [2, 0, 6, 8, 5, 1, 4, 3, 7, 9]    # Predicted ranking (as indices)
    
    # Calculate NDCG at different k values
    ndcg_5 = ndcg_at_k(relevance, ranking, k=5)
    ndcg_10 = ndcg_at_k(relevance, ranking, k=10)
    
    # Print results
    print(f"NDCG@5: {ndcg_5:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")

    # Sample data - recommendations for users
    recommendations = [
        [101, 102, 103, 104],  # User 1's recommendations
        [102, 105, 106, 107],  # User 2's recommendations
        [103, 108, 109, 110],  # User 3's recommendations
        [101, 105, 108, 111]   # User 4's recommendations
    ]
    
    # Item features for diversity calculation
    item_features = {
        101: ["action", "thriller", "2000s"],
        102: ["comedy", "romance", "2010s"],
        103: ["action", "sci-fi", "2010s"],
        104: ["drama", "crime", "1990s"],
        105: ["comedy", "family", "2000s"],
        106: ["action", "comedy", "2010s"],
        107: ["drama", "romance", "2000s"],
        108: ["sci-fi", "thriller", "2020s"],
        109: ["action", "adventure", "2020s"],
        110: ["horror", "thriller", "2010s"],
        111: ["documentary", "history", "2020s"]
    }
    
    # Calculate diversity
    div_score = diversity(recommendations, item_features)
    
    # Calculate coverage
    total_catalog_size = 15  # Assuming total catalog has 15 items (101-115)
    cov_score = coverage(recommendations, total_catalog_size)
    
    # Print results
    print(f"\nDiversity Score: {div_score:.4f}")
    print(f"Coverage Score: {cov_score:.4f}")