"""
User-based Collaborative Filtering.
"""
from typing import List, Dict, Set
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict
from .base import BaseRecommender


class UserCFRecommender(BaseRecommender):
    """User-based Collaborative Filtering using cosine similarity."""
    
    def __init__(self, k_neighbors: int = 50):
        """
        Initialize User-CF recommender.
        
        Args:
            k_neighbors: Number of similar users to consider
        """
        super().__init__(name="User-CF")
        self.k_neighbors = k_neighbors
        self.train_data = {}
        self.user_items_set = {}
        self.all_items = set()
        self.all_users = []
    
    def fit(self, train_data: Dict[int, List[int]]) -> 'UserCFRecommender':
        """
        Store training data for similarity computation.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        self.train_data = train_data
        self.all_users = list(train_data.keys())
        
        # Convert to sets for faster lookup
        for user_id, items in train_data.items():
            self.user_items_set[user_id] = set(items)
            self.all_items.update(items)
        
        self.is_fitted = True
        print(f"[{self.name}] Fitted on {len(self.all_users)} users, {len(self.all_items)} items")
        
        return self
    
    def compute_user_similarity(self, user1: int, user2: int) -> float:
        """
        Compute cosine similarity between two users.
        
        Args:
            user1: First user ID
            user2: Second user ID
            
        Returns:
            Similarity score (0-1)
        """
        items1 = self.user_items_set.get(user1, set())
        items2 = self.user_items_set.get(user2, set())
        
        if len(items1) == 0 or len(items2) == 0:
            return 0.0
        
        # Jaccard similarity (faster than cosine for sets)
        intersection = len(items1 & items2)
        union = len(items1 | items2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_similar_users(self, user_id: int, k: int = None) -> List[tuple]:
        """
        Find k most similar users.
        
        Args:
            user_id: Target user ID
            k: Number of similar users (defaults to self.k_neighbors)
            
        Returns:
            List of (user_id, similarity) tuples
        """
        if k is None:
            k = self.k_neighbors
        
        similarities = []
        for other_user in self.all_users:
            if other_user == user_id:
                continue
            
            sim = self.compute_user_similarity(user_id, other_user)
            if sim > 0:
                similarities.append((other_user, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]
    
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Recommend items based on similar users.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get similar users
        similar_users = self.find_similar_users(user_id)
        
        if len(similar_users) == 0:
            # Fallback to popular items
            from .popularity import PopularityRecommender
            fallback = PopularityRecommender()
            fallback.fit(self.train_data)
            return fallback.recommend(user_id, n, exclude_items)
        
        # Score items based on similar users
        item_scores = defaultdict(float)
        for other_user, similarity in similar_users:
            for item in self.user_items_set.get(other_user, set()):
                if exclude_items and item in exclude_items:
                    continue
                item_scores[item] += similarity
        
        # Sort by score and return top-n
        recommendations = sorted(item_scores.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, score in recommendations[:n]]

