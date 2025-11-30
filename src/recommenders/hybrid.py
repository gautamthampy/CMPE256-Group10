"""
Hybrid recommender combining multiple algorithms.
"""
from typing import List, Dict, Set
from collections import defaultdict
from .base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining multiple algorithms with weighted voting."""
    
    def __init__(self, recommenders: List[BaseRecommender], 
                 weights: List[float] = None):
        """
        Initialize hybrid recommender.
        
        Args:
            recommenders: List of recommender instances
            weights: Optional weights for each recommender (must sum to 1)
        """
        super().__init__(name="Hybrid")
        self.recommenders = recommenders
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(recommenders)] * len(recommenders)
        
        if len(weights) != len(recommenders):
            raise ValueError("Number of weights must match number of recommenders")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.weights = weights
    
    def fit(self, train_data: Dict[int, List[int]]) -> 'HybridRecommender':
        """
        Train all component recommenders.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        print(f"[{self.name}] Training {len(self.recommenders)} component models...")
        
        for i, (recommender, weight) in enumerate(zip(self.recommenders, self.weights)):
            print(f"\n  [{i+1}/{len(self.recommenders)}] Training {recommender.name} (weight={weight:.3f})...")
            recommender.fit(train_data)
        
        self.is_fitted = True
        print(f"\n[{self.name}] All models trained!")
        
        return self
    
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Generate recommendations by combining multiple recommenders.
        
        Uses rank-based voting: items are scored based on their rank position
        in each recommender's output, weighted by recommender weight.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get recommendations from each model (request more to have better coverage)
        all_recommendations = []
        for recommender in self.recommenders:
            try:
                recs = recommender.recommend(user_id, n=n*2, exclude_items=exclude_items)
                all_recommendations.append(recs)
            except Exception as e:
                print(f"[WARNING] {recommender.name} failed: {e}")
                all_recommendations.append([])
        
        # Score items using weighted rank-based voting
        item_scores = defaultdict(float)
        
        for recs, weight in zip(all_recommendations, self.weights):
            for rank, item_id in enumerate(recs):
                # Higher rank = lower position score (exponential decay)
                # Score = weight * (1 / (rank + 1))
                score = weight * (1.0 / (rank + 1))
                item_scores[item_id] += score
        
        # Sort by score and return top-n
        recommendations = sorted(item_scores.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, score in recommendations[:n]]
    
    def get_model_names(self) -> List[str]:
        """Get names of component models."""
        return [rec.name for rec in self.recommenders]

