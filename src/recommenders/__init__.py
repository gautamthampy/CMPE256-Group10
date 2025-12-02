"""
Recommender system implementations.
"""
from .base import BaseRecommender
from .user_cf import UserCFRecommender

__all__ = [
    'BaseRecommender',
    'UserCFRecommender',
]

