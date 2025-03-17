"""
This file contains basic metrics used in addition to Diversity, Coverage, NDCG@10.
These metrics can be imported and used for model evaluation across various tasks.
"""
from typing import List, Dict, Union, Tuple, Optional, Callable, Any, Set
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

# Classification Metrics
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    float
        Accuracy score between 0.0 and 1.0
    """
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Union[float, np.ndarray]:
    """
    Calculate precision score.
    
    Precision = TP / (TP + FP)
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    average : str, optional
        Averaging method for multi-class classification:
        - 'binary': Only report results for the positive class
        - 'micro': Calculate metrics globally by counting total TP, FP, etc.
        - 'macro': Calculate metrics for each class and take unweighted mean
        - 'weighted': Calculate metrics for each class and take weighted mean by support
        - None: Return a value for each class
        
    Returns:
    --------
    Union[float, np.ndarray]
        Precision score(s)
    """
    if average == 'binary':
        cm = confusion_matrix(y_true, y_pred)
        TP = cm[1, 1]
        FP = cm[0, 1]
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # For multi-class, we would implement the other averaging methods
    # This is a simplified implementation
    return NotImplemented


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Union[float, np.ndarray]:
    """
    Calculate recall score.
    
    Recall = TP / (TP + FN)
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    average : str, optional
        Averaging method for multi-class classification
        
    Returns:
    --------
    Union[float, np.ndarray]
        Recall score(s)
    """
    if average == 'binary':
        cm = confusion_matrix(y_true, y_pred)
        TP = cm[1, 1]
        FN = cm[1, 0]
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # For multi-class, we would implement the other averaging methods
    return NotImplemented


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Union[float, np.ndarray]:
    """
    Calculate F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    average : str, optional
        Averaging method for multi-class classification
        
    Returns:
    --------
    Union[float, np.ndarray]
        F1 score(s) between 0.0 and 1.0
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels
    y_score : np.ndarray
        Predicted probabilities or scores
        
    Returns:
    --------
    float
        AUC-ROC score between 0.0 and 1.0
    """
    # This is a simplified implementation
    # In practice, you'd want to use sklearn.metrics.roc_auc_score
    
    # Sort instances by predicted score
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true[sorted_indices]
    
    # Calculate true positive rate and false positive rate
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random guess
    
    # Calculate area
    area = 0.0
    false_pos_count = 0
    
    for label in sorted_y_true:
        if label == 1:
            area += false_pos_count / n_neg
        else:
            false_pos_count += 1
    
    return area / n_pos


# Regression Metrics
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    MSE = (1/n) * Σ(y_true - y_pred)²
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        Mean squared error (non-negative)
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        Mean absolute error (non-negative)
    """
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (Coefficient of Determination).
    
    R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        R² score, can be negative if model is worse than mean prediction
    """
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    if ss_total == 0:
        return 0.0  # Undefined, return 0
    
    return 1 - (ss_residual / ss_total)


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
    # This is a simplified implementation
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
    # Flatten all recommendations and count unique items
    all_recommended_items = set()
    for user_recs in recommendations:
        all_recommended_items.update(user_recs)
    
    # Calculate coverage
    coverage_score = len(all_recommended_items) / catalog_size if catalog_size > 0 else 0.0
    
    return coverage_score


# Testing code
def demonstrate_ranking_metrics():
    """Demonstrate the usage of ranking metrics."""
    print("\n===== Ranking Metrics =====")
    
    # Sample data for NDCG@k
    relevance = [3, 2, 3, 0, 1, 2, 3, 2, 3, 0]  # True relevance scores
    ranking = [2, 0, 6, 8, 5, 1, 4, 3, 7, 9]    # Predicted ranking (as indices)
    
    # Calculate NDCG at different k values
    ndcg_5 = ndcg_at_k(relevance, ranking, k=5)
    ndcg_10 = ndcg_at_k(relevance, ranking, k=10)
    
    # Print results
    print(f"NDCG@5: {ndcg_5:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")


def demonstrate_recommendation_metrics():
    """Demonstrate the usage of recommendation system metrics."""
    print("\n===== Recommendation Metrics =====")
    
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
    print(f"Diversity Score: {div_score:.4f}")
    print(f"Coverage Score: {cov_score:.4f}")
    
    # Detailed explanation of diversity and coverage
    print("\nDiversity Explanation:")
    print("- Measures how different the recommended items are from each other")
    print("- Higher score (closer to 1.0) means more diverse recommendations")
    print("- Calculated using the Jaccard distance between item features")
    
    print("\nCoverage Explanation:")
    print("- Measures what percentage of the catalog is being recommended")
    print("- Higher score (closer to 1.0) means more items from catalog are recommended")
    print(f"- In this example, {int(cov_score * total_catalog_size)} out of {total_catalog_size} items are recommended")

    
if __name__ == "__main__":
    demonstrate_ranking_metrics()
    demonstrate_recommendation_metrics()
