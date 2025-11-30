"""
Data loading and preprocessing module for recommendation systems.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
from typing import Tuple, Dict, List, Set
import random


class DataLoader:
    """Load and preprocess user-item interaction data."""
    
    def __init__(self, filepath: str, seed: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            filepath: Path to the training data file
            seed: Random seed for reproducibility
        """
        self.filepath = filepath
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Data structures
        self.user_items = {}  # user_id -> list of item_ids
        self.item_users = defaultdict(set)  # item_id -> set of user_ids
        self.all_users = set()
        self.all_items = set()
        
        # Statistics
        self.n_users = 0
        self.n_items = 0
        self.n_interactions = 0
        
    def load_data(self) -> Dict[int, List[int]]:
        """
        Load data from file.
        
        Returns:
            Dictionary mapping user_id to list of item_ids
        """
        print(f"Loading data from {self.filepath}...")
        
        with open(self.filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                user_id = int(parts[0])
                item_ids = [int(x) for x in parts[1:]]
                
                self.user_items[user_id] = item_ids
                self.all_users.add(user_id)
                
                for item_id in item_ids:
                    self.all_items.add(item_id)
                    self.item_users[item_id].add(user_id)
                    self.n_interactions += 1
        
        self.n_users = len(self.all_users)
        self.n_items = len(self.all_items)
        
        print(f"Loaded {self.n_users} users, {self.n_items} items, {self.n_interactions} interactions")
        print(f"Sparsity: {100 * (1 - self.n_interactions / (self.n_users * self.n_items)):.4f}%")
        
        return self.user_items
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        interactions_per_user = [len(items) for items in self.user_items.values()]
        interactions_per_item = [len(users) for users in self.item_users.values()]
        
        stats = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_interactions': self.n_interactions,
            'sparsity': 100 * (1 - self.n_interactions / (self.n_users * self.n_items)),
            'avg_interactions_per_user': np.mean(interactions_per_user),
            'median_interactions_per_user': np.median(interactions_per_user),
            'min_interactions_per_user': np.min(interactions_per_user),
            'max_interactions_per_user': np.max(interactions_per_user),
            'avg_interactions_per_item': np.mean(interactions_per_item),
            'median_interactions_per_item': np.median(interactions_per_item),
            'min_interactions_per_item': np.min(interactions_per_item),
            'max_interactions_per_item': np.max(interactions_per_item),
        }
        
        return stats
    
    def create_train_test_split(self, test_ratio: float = 0.2, 
                                min_items_test: int = 1) -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train and test sets.
        
        For each user, randomly select test_ratio of items for testing,
        ensuring at least min_items_test items in test set if possible.
        
        Args:
            test_ratio: Ratio of items to use for testing
            min_items_test: Minimum number of items in test set per user
            
        Returns:
            (train_data, test_data, validation_data)
        """
        print(f"Splitting data (test_ratio={test_ratio})...")
        
        train_data = {}
        test_data = {}
        validation_data = {}
        
        for user_id, items in self.user_items.items():
            items = list(items)
            n_items = len(items)
            
            # Need at least 3 items to split into train/val/test
            if n_items < 3:
                train_data[user_id] = items
                test_data[user_id] = []
                validation_data[user_id] = []
                continue
            
            # Shuffle items
            random.shuffle(items)
            
            # Calculate split sizes
            n_test = max(min_items_test, int(n_items * test_ratio))
            n_val = max(1, int(n_items * test_ratio))
            
            # Ensure we have items left for training
            if n_test + n_val >= n_items:
                n_test = max(1, int(n_items * 0.2))
                n_val = max(1, int(n_items * 0.2))
                if n_test + n_val >= n_items:
                    n_test = 1
                    n_val = 1
            
            # Split
            test_data[user_id] = items[:n_test]
            validation_data[user_id] = items[n_test:n_test+n_val]
            train_data[user_id] = items[n_test+n_val:]
        
        # Statistics
        train_interactions = sum(len(items) for items in train_data.values())
        val_interactions = sum(len(items) for items in validation_data.values())
        test_interactions = sum(len(items) for items in test_data.values())
        
        print(f"Train: {train_interactions} interactions")
        print(f"Validation: {val_interactions} interactions")
        print(f"Test: {test_interactions} interactions")
        
        return train_data, validation_data, test_data
    
    def create_user_item_matrix(self, user_items_dict: Dict[int, List[int]], 
                                implicit_value: float = 1.0) -> csr_matrix:
        """
        Create sparse user-item interaction matrix.
        
        Args:
            user_items_dict: Dictionary mapping user_id to list of item_ids
            implicit_value: Value to use for positive interactions
            
        Returns:
            Sparse CSR matrix of shape (n_users, n_items)
        """
        # Create mapping from user/item IDs to indices
        user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(self.all_users))}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(sorted(self.all_items))}
        
        # Create matrix
        matrix = lil_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        for user_id, item_ids in user_items_dict.items():
            user_idx = user_to_idx[user_id]
            for item_id in item_ids:
                if item_id in item_to_idx:
                    item_idx = item_to_idx[item_id]
                    matrix[user_idx, item_idx] = implicit_value
        
        return matrix.tocsr(), user_to_idx, item_to_idx
    
    def get_item_popularity(self, user_items_dict: Dict[int, List[int]] = None) -> Dict[int, int]:
        """
        Get item popularity (number of users who interacted with each item).
        
        Args:
            user_items_dict: Optional specific dataset to use
            
        Returns:
            Dictionary mapping item_id to popularity count
        """
        if user_items_dict is None:
            user_items_dict = self.user_items
        
        item_counts = defaultdict(int)
        for items in user_items_dict.values():
            for item_id in items:
                item_counts[item_id] += 1
        
        return dict(item_counts)
    
    def get_user_history(self, user_id: int, dataset: Dict[int, List[int]] = None) -> Set[int]:
        """
        Get set of items a user has interacted with.
        
        Args:
            user_id: User ID
            dataset: Optional specific dataset to use
            
        Returns:
            Set of item IDs
        """
        if dataset is None:
            dataset = self.user_items
        
        return set(dataset.get(user_id, []))


def print_statistics(stats: Dict):
    """Print dataset statistics in a formatted way."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nBasic Info:")
    print(f"  Users:        {stats['n_users']:,}")
    print(f"  Items:        {stats['n_items']:,}")
    print(f"  Interactions: {stats['n_interactions']:,}")
    print(f"  Sparsity:     {stats['sparsity']:.4f}%")
    
    print(f"\nUser Interaction Stats:")
    print(f"  Average:      {stats['avg_interactions_per_user']:.2f}")
    print(f"  Median:       {stats['median_interactions_per_user']:.2f}")
    print(f"  Min:          {stats['min_interactions_per_user']}")
    print(f"  Max:          {stats['max_interactions_per_user']}")
    
    print(f"\nItem Interaction Stats:")
    print(f"  Average:      {stats['avg_interactions_per_item']:.2f}")
    print(f"  Median:       {stats['median_interactions_per_item']:.2f}")
    print(f"  Min:          {stats['min_interactions_per_item']}")
    print(f"  Max:          {stats['max_interactions_per_item']}")
    
    print("="*50 + "\n")

