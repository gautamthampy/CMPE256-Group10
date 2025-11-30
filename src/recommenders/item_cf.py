"""
Item-based Collaborative Filtering.
"""
from typing import List, Dict, Set
import numpy as np
from collections import defaultdict
from .base import BaseRecommender


class ItemCFRecommender(BaseRecommender):
    """Item-based Collaborative Filtering using cosine similarity."""
    
    def __init__(self, k_neighbors: int = 50):
        """
        Initialize Item-CF recommender.
        
        Args:
            k_neighbors: Number of similar items to consider
        """
        super().__init__(name="Item-CF")
        self.k_neighbors = k_neighbors
        self.train_data = {}
        self.item_users = defaultdict(set)
        self.all_items = []
    
    def fit(self, train_data: Dict[int, List[int]]) -> 'ItemCFRecommender':
        """
        Build item-user index.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        self.train_data = train_data
        
        # Build item-user reverse index
        for user_id, items in train_data.items():
            for item in items:
                self.item_users[item].add(user_id)
        
        self.all_items = list(self.item_users.keys())
        
        self.is_fitted = True
        print(f"[{self.name}] Fitted on {len(self.all_items)} items")
        
        return self
    
    def compute_item_similarity(self, item1: int, item2: int) -> float:
        """
        Compute cosine similarity between two items.
        
        Args:
            item1: First item ID
            item2: Second item ID
            
        Returns:
            Similarity score (0-1)
        """
        users1 = self.item_users.get(item1, set())
        users2 = self.item_users.get(item2, set())
        
        if len(users1) == 0 or len(users2) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(users1 & users2)
        union = len(users1 | users2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_similar_items(self, item_id: int, k: int = None) -> List[tuple]:
        """
        Find k most similar items.
        
        Args:
            item_id: Target item ID
            k: Number of similar items (defaults to self.k_neighbors)
            
        Returns:
            List of (item_id, similarity) tuples
        """
        if k is None:
            k = self.k_neighbors
        
        similarities = []
        for other_item in self.all_items:
            if other_item == item_id:
                continue
            
            sim = self.compute_item_similarity(item_id, other_item)
            if sim > 0:
                similarities.append((other_item, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]
    
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Recommend items similar to user's history.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's item history
        user_items = self.train_data.get(user_id, [])
        
        if len(user_items) == 0:
            # Fallback to popular items
            from .popularity import PopularityRecommender
            fallback = PopularityRecommender()
            fallback.fit(self.train_data)
            return fallback.recommend(user_id, n, exclude_items)
        
        # Score items based on similarity to user's history
        item_scores = defaultdict(float)
        for user_item in user_items:
            similar_items = self.find_similar_items(user_item)
            for similar_item, similarity in similar_items:
                if exclude_items and similar_item in exclude_items:
                    continue
                item_scores[similar_item] += similarity
        
        # Sort by score and return top-n
        recommendations = sorted(item_scores.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, score in recommendations[:n]]

