# CMPE 256 Recommender Systems - Final Project

## Team Information
- **Team**: Team 5/6/7/8 (Dataset 2)
- **Dataset**: train-2.txt
- **Evaluation Metric**: NDCG@20

## Project Overview
This project implements multiple recommendation algorithms to generate top-20 item recommendations for users based on implicit feedback (positive interactions).

## Dataset
- **Format**: Each row contains User_ID followed by space-separated Item_IDs
- **Type**: Implicit feedback (positive interactions only)
- **Task**: Generate 20 item recommendations per user

## Project Structure
```
CMPE_256_FINAL_PROJECT/
├── data/
│   └── train-2.txt                 # Training dataset
├── src/
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── evaluation.py               # Evaluation metrics (NDCG@20, etc.)
│   ├── recommenders/
│   │   ├── __init__.py
│   │   ├── base.py                 # Base recommender class
│   │   ├── popularity.py           # Popularity-based recommender
│   │   ├── user_cf.py              # User-based collaborative filtering
│   │   ├── item_cf.py              # Item-based collaborative filtering
│   │   ├── matrix_factorization.py # SVD/ALS implementations
│   │   └── hybrid.py               # Hybrid/ensemble methods
│   └── utils.py                    # Utility functions
├── notebooks/
│   └── exploration.ipynb           # Data exploration
├── outputs/
│   └── submission.txt              # Final predictions for leaderboard
├── models/
│   └── saved_models/               # Saved trained models
├── main.py                         # Main training and evaluation script
├── generate_submission.py          # Generate submission file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Implemented Algorithms

### 1. Popularity-Based Recommender
- Recommends most popular items globally
- Simple baseline approach
- Fast and interpretable

### 2. User-Based Collaborative Filtering
- Finds similar users based on interaction history
- Recommends items that similar users liked
- Uses cosine similarity

### 3. Item-Based Collaborative Filtering
- Finds similar items based on user interactions
- Recommends items similar to user's history
- More stable than user-based for sparse data

### 4. Matrix Factorization (ALS)
- Alternating Least Squares for implicit feedback
- Learns latent factors for users and items
- Handles sparsity well

### 5. SVD (Surprise Library)
- Singular Value Decomposition
- Optimized for recommendation tasks
- Good for explicit/implicit feedback

### 6. Hybrid/Ensemble Methods
- Combines multiple algorithms
- Weighted averaging of predictions
- Improves robustness

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Exploration
```bash
jupyter notebook notebooks/exploration.ipynb
```

### 2. Train and Evaluate Models
```bash
python main.py
```

This will:
- Load and preprocess the data
- Split into train/validation/test sets
- Train all implemented algorithms
- Evaluate using NDCG@20 and other metrics
- Display comparison results

### 3. Generate Submission File
```bash
python generate_submission.py --model hybrid
```

Output will be saved to `outputs/submission.txt` in the required format.

## Evaluation Metrics

- **NDCG@20**: Normalized Discounted Cumulative Gain at 20 (primary metric for leaderboard)
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **MAP@K**: Mean Average Precision
- **Coverage**: Percentage of items recommended across all users

## Results

| Algorithm | NDCG@20 | Precision@20 | Recall@20 | Training Time |
|-----------|---------|--------------|-----------|---------------|
| Popularity | TBD | TBD | TBD | TBD |
| User-CF | TBD | TBD | TBD | TBD |
| Item-CF | TBD | TBD | TBD | TBD |
| ALS | TBD | TBD | TBD | TBD |
| SVD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD |

## Key Findings
(To be filled after running experiments)

## Leaderboard Submission
- **URL**: http://coe-clp.sjsu.edu/
- **Authentication**: MySJSU credentials via VPN
- **Submission Format**: 20 items per user (space-separated)

## References
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets
- Sarwar, B., et al. (2001). Item-based collaborative filtering recommendation algorithms

## License
MIT License

