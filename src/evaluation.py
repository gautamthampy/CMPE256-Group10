"""
Evaluation metrics for recommendation systems.
"""
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict


def dcg_at_k(relevance: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    Args:
        relevance: Binary relevance list (1 if relevant, 0 otherwise)
        k: Cutoff position
        
    Returns:
        DCG@K score
    """
    relevance = np.array(relevance[:k])
    if relevance.size == 0:
        return 0.0
    
    # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        recommended: List of recommended item IDs (in order)
        relevant: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        NDCG@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    # Get relevance list for recommendations
    recommended = recommended[:k]
    relevance = [1 if item in relevant else 0 for item in recommended]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance, k)
    
    # Calculate ideal DCG (if we ranked all relevant items first)
    ideal_relevance = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Precision at K.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        Precision@K score
    """
    if k == 0:
        return 0.0
    
    recommended_k = set(recommended[:k])
    return len(recommended_k & relevant) / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Recall at K.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        Recall@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = set(recommended[:k])
    return len(recommended_k & relevant) / len(relevant)


def average_precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Average Precision at K.
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        AP@K score
    """
    if len(relevant) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(len(relevant), k)


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Calculate Hit Rate at K (binary: did we recommend at least one relevant item?).
    
    Args:
        recommended: List of recommended item IDs
        relevant: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        1.0 if at least one relevant item in top-k, 0.0 otherwise
    """
    recommended_k = set(recommended[:k])
    return 1.0 if len(recommended_k & relevant) > 0 else 0.0


def catalog_coverage(all_recommendations: List[List[int]], n_items: int) -> float:
    """
    Calculate catalog coverage: percentage of items recommended at least once.
    
    Args:
        all_recommendations: List of recommendation lists for all users
        n_items: Total number of items in catalog
        
    Returns:
        Coverage percentage (0-100)
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    
    return 100 * len(recommended_items) / n_items


class Evaluator:
    """Evaluator for recommendation systems."""
    
    def __init__(self, k: int = 20):
        """
        Initialize evaluator.
        
        Args:
            k: Cutoff position for metrics
        """
        self.k = k
    
    def evaluate_user(self, recommended: List[int], relevant: Set[int]) -> Dict[str, float]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            
        Returns:
            Dictionary of metric scores
        """
        return {
            'ndcg': ndcg_at_k(recommended, relevant, self.k),
            'precision': precision_at_k(recommended, relevant, self.k),
            'recall': recall_at_k(recommended, relevant, self.k),
            'map': average_precision_at_k(recommended, relevant, self.k),
            'hit_rate': hit_rate_at_k(recommended, relevant, self.k),
        }
    
    def evaluate_all(self, recommendations: Dict[int, List[int]], 
                    test_data: Dict[int, List[int]],
                    n_items: int = None) -> Dict[str, float]:
        """
        Evaluate recommendations for all users.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended items
            test_data: Dictionary mapping user_id to list of relevant items
            n_items: Total number of items (for coverage calculation)
            
        Returns:
            Dictionary of average metric scores
        """
        all_scores = defaultdict(list)
        all_recs = []
        
        for user_id, relevant_items in test_data.items():
            if user_id not in recommendations:
                continue
            
            relevant = set(relevant_items)
            if len(relevant) == 0:
                continue
            
            recommended = recommendations[user_id]
            all_recs.append(recommended)
            
            # Evaluate for this user
            scores = self.evaluate_user(recommended, relevant)
            for metric, value in scores.items():
                all_scores[metric].append(value)
        
        # Calculate averages
        avg_scores = {
            metric: np.mean(values) for metric, values in all_scores.items()
        }
        
        # Add coverage if n_items provided
        if n_items is not None:
            avg_scores['coverage'] = catalog_coverage(all_recs, n_items)
        
        return avg_scores
    
    def print_results(self, results: Dict[str, float], model_name: str = "Model"):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Dictionary of metric scores
            model_name: Name of the model
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS: {model_name}")
        print(f"{'='*60}")
        
        print(f"\nRanking Metrics (@ {self.k}):")
        print(f"  NDCG@{self.k}:      {results.get('ndcg', 0):.4f}")
        print(f"  Precision@{self.k}: {results.get('precision', 0):.4f}")
        print(f"  Recall@{self.k}:    {results.get('recall', 0):.4f}")
        print(f"  MAP@{self.k}:       {results.get('map', 0):.4f}")
        print(f"  Hit Rate@{self.k}:  {results.get('hit_rate', 0):.4f}")
        
        if 'coverage' in results:
            print(f"\nDiversity:")
            print(f"  Coverage:    {results['coverage']:.2f}%")
        
        print(f"{'='*60}\n")


def compare_models(results_dict: Dict[str, Dict[str, float]], k: int = 20):
    """
    Compare multiple models side by side.
    
    Args:
        results_dict: Dictionary mapping model name to results dictionary
        k: Cutoff position
    """
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON @ {k}")
    print(f"{'='*80}\n")
    
    metrics = ['ndcg', 'precision', 'recall', 'map', 'hit_rate', 'coverage']
    
    # Header
    print(f"{'Model':<25} {'NDCG':>8} {'Precision':>10} {'Recall':>8} {'MAP':>8} {'Hit Rate':>10} {'Coverage':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    
    # Results for each model
    for model_name, results in results_dict.items():
        print(f"{model_name:<25} ", end="")
        for metric in metrics:
            if metric in results:
                if metric == 'coverage':
                    print(f"{results[metric]:>9.2f}% ", end="")
                else:
                    print(f"{results[metric]:>8.4f} ", end="")
            else:
                print(f"{'N/A':>8} ", end="")
        print()
    
    print(f"{'='*80}\n")

