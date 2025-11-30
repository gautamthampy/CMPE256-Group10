"""
Matrix Factorization recommenders (ALS and SVD).
"""
from typing import List, Dict, Set
import numpy as np
from scipy.sparse import csr_matrix
from .base import BaseRecommender


class ALSRecommender(BaseRecommender):
    """Alternating Least Squares for implicit feedback."""
    
    def __init__(self, factors: int = 50, regularization: float = 0.01, 
                 iterations: int = 15, alpha: float = 1.0):
        """
        Initialize ALS recommender.
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of ALS iterations
            alpha: Confidence scaling parameter for implicit feedback
        """
        super().__init__(name="ALS")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.all_items = []
    
    def fit(self, train_data: Dict[int, List[int]]) -> 'ALSRecommender':
        """
        Train ALS model using implicit library.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        try:
            from implicit.als import AlternatingLeastSquares
            
            # Create mappings
            all_users = sorted(set(train_data.keys()))
            all_items_set = set()
            for items in train_data.values():
                all_items_set.update(items)
            self.all_items = sorted(all_items_set)
            
            self.user_to_idx = {user: idx for idx, user in enumerate(all_users)}
            self.item_to_idx = {item: idx for idx, item in enumerate(self.all_items)}
            self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
            
            n_users = len(all_users)
            n_items = len(self.all_items)
            
            # Build sparse matrix (items x users for implicit library)
            from scipy.sparse import lil_matrix
            matrix = lil_matrix((n_items, n_users), dtype=np.float32)
            
            for user_id, items in train_data.items():
                user_idx = self.user_to_idx[user_id]
                for item_id in items:
                    if item_id in self.item_to_idx:
                        item_idx = self.item_to_idx[item_id]
                        matrix[item_idx, user_idx] = 1.0
            
            matrix = matrix.tocsr()
            
            # Train model
            print(f"[{self.name}] Training with {self.factors} factors, {self.iterations} iterations...")
            model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42
            )
            model.fit(matrix * self.alpha)
            
            self.user_factors = model.user_factors
            self.item_factors = model.item_factors
            
            self.is_fitted = True
            print(f"[{self.name}] Fitted on {n_users} users, {n_items} items")
            
        except ImportError:
            print("[WARNING] implicit library not available. Using basic ALS implementation.")
            self._fit_basic(train_data)
        
        return self
    
    def _fit_basic(self, train_data: Dict[int, List[int]]):
        """Basic ALS implementation without implicit library."""
        # Create mappings
        all_users = sorted(set(train_data.keys()))
        all_items_set = set()
        for items in train_data.values():
            all_items_set.update(items)
        self.all_items = sorted(all_items_set)
        
        self.user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.all_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users = len(all_users)
        n_items = len(self.all_items)
        
        # Initialize factors randomly
        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.factors))
        
        # Build interaction matrix
        from scipy.sparse import lil_matrix
        R = lil_matrix((n_users, n_items), dtype=np.float32)
        for user_id, items in train_data.items():
            user_idx = self.user_to_idx[user_id]
            for item_id in items:
                if item_id in self.item_to_idx:
                    item_idx = self.item_to_idx[item_id]
                    R[user_idx, item_idx] = 1.0
        R = R.tocsr()
        
        print(f"[{self.name}] Training basic ALS...")
        # Simple ALS iterations
        for iteration in range(self.iterations):
            # Fix item factors, update user factors
            for u in range(n_users):
                items_u = R[u].indices
                if len(items_u) == 0:
                    continue
                
                I_u = self.item_factors[items_u]
                A = I_u.T @ I_u + self.regularization * np.eye(self.factors)
                b = I_u.T @ np.ones(len(items_u))
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Fix user factors, update item factors
            R_T = R.T.tocsr()
            for i in range(n_items):
                users_i = R_T[i].indices
                if len(users_i) == 0:
                    continue
                
                U_i = self.user_factors[users_i]
                A = U_i.T @ U_i + self.regularization * np.eye(self.factors)
                b = U_i.T @ np.ones(len(users_i))
                self.item_factors[i] = np.linalg.solve(A, b)
            
            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration + 1}/{self.iterations}")
        
        self.is_fitted = True
        print(f"[{self.name}] Fitted on {n_users} users, {n_items} items")
    
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Generate recommendations using learned factors.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_to_idx:
            # New user - return popular items
            item_counts = {}
            for item_id in self.all_items:
                if not exclude_items or item_id not in exclude_items:
                    item_idx = self.item_to_idx[item_id]
                    # Use item popularity as proxy
                    item_counts[item_id] = np.sum(self.item_factors[item_idx])
            sorted_items = sorted(item_counts.items(), key=lambda x: -x[1])
            return [item for item, _ in sorted_items[:n]]
        
        user_idx = self.user_to_idx[user_id]
        user_vec = self.user_factors[user_idx]
        
        # Score all items
        scores = self.item_factors @ user_vec
        
        # Get sorted item indices
        item_indices = np.argsort(-scores)
        
        # Convert to item IDs and filter
        recommendations = []
        for idx in item_indices:
            item_id = self.idx_to_item[idx]
            if exclude_items and item_id in exclude_items:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= n:
                break
        
        return recommendations


class SVDRecommender(BaseRecommender):
    """SVD-based recommender using Surprise library."""
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr_all: float = 0.005,
                 reg_all: float = 0.02):
        """
        Initialize SVD recommender.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            lr_all: Learning rate
            reg_all: Regularization parameter
        """
        super().__init__(name="SVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        self.model = None
        self.train_data = {}
        self.all_items = []
        self.trainset = None
    
    def fit(self, train_data: Dict[int, List[int]]) -> 'SVDRecommender':
        """
        Train SVD model using Surprise library.
        
        Args:
            train_data: Dictionary mapping user_id to list of item_ids
            
        Returns:
            self
        """
        try:
            from surprise import Dataset, Reader, SVD
            from surprise.model_selection import PredictionImpossible
            
            self.train_data = train_data
            
            # Get all items
            all_items_set = set()
            for items in train_data.values():
                all_items_set.update(items)
            self.all_items = sorted(all_items_set)
            
            # Convert to Surprise format (user, item, rating)
            ratings = []
            for user_id, items in train_data.items():
                for item_id in items:
                    ratings.append((user_id, item_id, 1.0))  # Implicit feedback = 1
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(
                pd.DataFrame(ratings, columns=['user', 'item', 'rating']),
                reader
            )
            
            # Train model
            print(f"[{self.name}] Training with {self.n_factors} factors, {self.n_epochs} epochs...")
            self.trainset = data.build_full_trainset()
            
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                random_state=42
            )
            self.model.fit(self.trainset)
            
            self.is_fitted = True
            print(f"[{self.name}] Fitted on {len(train_data)} users, {len(self.all_items)} items")
            
        except ImportError:
            print("[WARNING] Surprise library not available. Using ALS instead.")
            als = ALSRecommender(factors=self.n_factors)
            als.fit(train_data)
            # Copy ALS attributes
            self.__dict__.update(als.__dict__)
            self.name = "SVD (ALS fallback)"
        
        return self
    
    def recommend(self, user_id: int, n: int = 20, 
                 exclude_items: Set[int] = None) -> List[int]:
        """
        Generate recommendations using SVD.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            List of recommended item IDs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Score all items
        item_scores = []
        for item_id in self.all_items:
            if exclude_items and item_id in exclude_items:
                continue
            
            try:
                pred = self.model.predict(user_id, item_id)
                item_scores.append((item_id, pred.est))
            except:
                # If prediction fails, give low score
                item_scores.append((item_id, 0.0))
        
        # Sort by score and return top-n
        item_scores.sort(key=lambda x: -x[1])
        return [item for item, score in item_scores[:n]]


# Import pandas for SVD
try:
    import pandas as pd
except ImportError:
    print("[WARNING] pandas not available")

