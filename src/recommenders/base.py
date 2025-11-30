"""
Base recommender class.
"""
from typing import List, Dict, Set
from abc import ABC, abstractmethod
import numpy as np


class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""
    
    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize recommender.
        
        Args:
            name: Name of the recommender
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data: Dict[int, List[int]]) -> 'BaseRecommender':
        """
        Train the recommender on the training data.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations to generate
            exclude_items: Set of items to exclude from recommendations
            
        Returns:
            List of recommended item IDs
        """
        pass
    
    def recommend_all(self, user_ids: List[int], n: int = 20,
                     exclude_train: bool = True,
                     train_data: Dict[int, List[int]] = None) -> Dict[int, List[int]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            exclude_train: Whether to exclude items from training set
            train_data: Training data (required if exclude_train=True)
            
        Returns:
            Dictionary mapping user_id to list of recommended items
        """
        recommendations = {}
        
        for user_id in user_ids:
            exclude = None
            if exclude_train and train_data and user_id in train_data:
                exclude = set(train_data[user_id])
            
            recommendations[user_id] = self.recommend(user_id, n, exclude)
        
        return recommendations
    
    def get_name(self) -> str:
        """Get recommender name."""
        return self.name

