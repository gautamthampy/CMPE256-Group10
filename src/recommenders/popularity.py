"""
Popularity-based recommender.
"""
from typing import List, Dict, Set
from collections import Counter
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends most popular items globally."""
    
    def __init__(self):
        """Initialize popularity recommender."""
        super().__init__(name="Popularity")
        self.popular_items = []
    
    def fit(self, train_data: Dict[int, List[int]]) -> 'PopularityRecommender':
        """
        Train by counting item frequencies.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        # Count item frequencies
        item_counts = Counter()
        for items in train_data.values():
            item_counts.update(items)
        
        # Sort by popularity (count, then by item_id for stability)
        self.popular_items = [item for item, count in 
                            sorted(item_counts.items(), 
                                   key=lambda x: (-x[1], x[0]))]
        
        self.is_fitted = True
        print(f"[{self.name}] Fitted on {len(item_counts)} items")
        
        return self
    
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Recommend top-N most popular items.
        
        Args:
            user_id: User ID (not used in this method)
            n: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        recommendations = []
        for item in self.popular_items:
            if exclude_items and item in exclude_items:
                continue
            recommendations.append(item)
            if len(recommendations) >= n:
                break
        
        # If we don't have enough recommendations, pad with any available items
        if len(recommendations) < n:
            for item in self.popular_items:
                if item not in recommendations:
                    if not exclude_items or item not in exclude_items:
                        recommendations.append(item)
                        if len(recommendations) >= n:
                            break
        
        return recommendations[:n]

