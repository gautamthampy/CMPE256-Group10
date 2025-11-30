# How to Get Top 20 Recommendations and NDCG@20

## Quick Answer

### Method 1: Automatic (Recommended) ⭐
```bash
# This does everything automatically:
# - Trains models
# - Gets top 20 for each user
# - Calculates NDCG@20
# - Shows comparison table
python main.py
```

### Method 2: Generate Submission File
```bash
# This creates a file with top 20 recommendations for ALL users
python generate_submission.py --model hybrid

# Output: outputs/submission.txt
# Format: user_id item1 item2 item3 ... item20
```

### Method 3: Interactive Examples
```bash
# Run detailed examples showing step-by-step process
python example_usage.py
```

---

## Understanding the Code

### 1️⃣ Getting Top 20 Recommendations

**Code Location**: Any recommender in `src/recommenders/`

```python
import sys
sys.path.insert(0, 'src')

from data_loader import DataLoader
from recommenders import ALSRecommender

# Load data
loader = DataLoader('train-2.txt')
train_data = loader.load_data()

# Train model
model = ALSRecommender(factors=50, iterations=15)
model.fit(train_data)

# Get top 20 for user 5
user_id = 5
top_20_items = model.recommend(
    user_id=user_id,
    n=20,  # Number of recommendations
    exclude_items=set(train_data[user_id])  # Don't recommend what they have
)

print(f"Top 20 for User {user_id}: {top_20_items}")
# Output: [item_10, item_25, item_3, item_47, ...]  (20 items)
```

### 2️⃣ Calculating NDCG@20

**Code Location**: `src/evaluation.py`

```python
from evaluation import ndcg_at_k

# Your model's recommendations (ordered by score)
recommended = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
               11, 21, 31, 41, 51, 61, 71, 81, 91, 101]  # 20 items

# Ground truth: items the user actually liked
relevant = {20, 30, 100, 200, 300}  # Set of relevant items

# Calculate NDCG@20
score = ndcg_at_k(recommended, relevant, k=20)

print(f"NDCG@20: {score:.4f}")
# Output: NDCG@20: 0.3521 (example)
```

### 3️⃣ Evaluating All Users

**Code Location**: `src/evaluation.py`

```python
from evaluation import Evaluator

# Get recommendations for all users
recommendations = {}
for user_id in test_data.keys():
    recommendations[user_id] = model.recommend(
        user_id=user_id,
        n=20,
        exclude_items=set(train_data.get(user_id, []))
    )

# Calculate average NDCG@20 across all users
evaluator = Evaluator(k=20)
results = evaluator.evaluate_all(
    recommendations=recommendations,
    test_data=test_data,  # Ground truth
    n_items=loader.n_items
)

print(f"Average NDCG@20: {results['ndcg']:.4f}")
print(f"Average Precision@20: {results['precision']:.4f}")
print(f"Average Recall@20: {results['recall']:.4f}")
```

---

## What is NDCG@20?

**NDCG = Normalized Discounted Cumulative Gain**

It measures how good your top 20 recommendations are, with:
- ✅ Higher weight for relevant items at the top
- ✅ Normalized to 0-1 range (1 = perfect)
- ✅ Takes into account ranking position

### Formula Breakdown

```python
# For each recommended item at position i:
DCG = sum(relevance[i] / log2(i + 2))

# If we ranked perfectly:
iDCG = sum(1 / log2(i + 2) for i in range(num_relevant_items))

# Normalized score:
NDCG = DCG / iDCG
```

### Example Calculation

```
Recommended:  [10, 5, 20, 15, 30, 25, ...]  (20 items)
Relevant:     {5, 20, 30}                     (ground truth)

Position  Item  Relevant?  Contribution to DCG
   1       10      ❌         0 / log2(3) = 0
   2        5      ✅         1 / log2(4) = 0.5000
   3       20      ✅         1 / log2(5) = 0.4307
   4       15      ❌         0 / log2(6) = 0
   5       30      ✅         1 / log2(7) = 0.3562
   ...

DCG = 0.5000 + 0.4307 + 0.3562 = 1.2869

Ideal ranking: [5, 20, 30, ...] (all relevant first)
iDCG = 1/log2(3) + 1/log2(4) + 1/log2(5) = 1.5614

NDCG@20 = 1.2869 / 1.5614 = 0.8242
```

**Interpretation**:
- **0.8242** = Very good! Found 3/3 relevant items, mostly near top
- **0.5000** = Okay - found some relevant items but not well-ranked
- **0.0000** = Bad - found no relevant items in top 20
- **1.0000** = Perfect - all relevant items at the very top

---

## Practical Examples

### Example 1: Check One User

```python
import sys
sys.path.insert(0, 'src')

from data_loader import DataLoader
from recommenders import HybridRecommender, ItemCFRecommender, ALSRecommender
from evaluation import ndcg_at_k

# Load and split data
loader = DataLoader('train-2.txt')
user_items = loader.load_data()
train, val, test = loader.create_train_test_split()

# Train hybrid model
components = [
    ItemCFRecommender(k_neighbors=50),
    ALSRecommender(factors=50, iterations=15)
]
model = HybridRecommender(components, weights=[0.4, 0.6])
model.fit(train)

# Pick a user
user_id = 10

# Get recommendations
user_history = set(train.get(user_id, []))
top_20 = model.recommend(user_id, n=20, exclude_items=user_history)

# Get ground truth
ground_truth = set(val.get(user_id, []))

# Calculate NDCG@20
ndcg = ndcg_at_k(top_20, ground_truth, k=20)

print(f"User {user_id}:")
print(f"  Training items: {len(user_history)}")
print(f"  Validation items: {len(ground_truth)}")
print(f"  Top 20 recommendations: {top_20[:5]}...")
print(f"  Hits in top 20: {len(set(top_20) & ground_truth)}")
print(f"  NDCG@20: {ndcg:.4f}")
```

### Example 2: Compare Models

```python
from recommenders import PopularityRecommender, ItemCFRecommender, ALSRecommender
from evaluation import Evaluator

# Train models
models = {
    'Popularity': PopularityRecommender(),
    'Item-CF': ItemCFRecommender(k_neighbors=50),
    'ALS': ALSRecommender(factors=50, iterations=15),
}

for name, model in models.items():
    model.fit(train)

# Evaluate each
evaluator = Evaluator(k=20)

for name, model in models.items():
    # Generate recommendations
    recs = model.recommend_all(
        user_ids=list(val.keys()),
        n=20,
        exclude_train=True,
        train_data=train
    )
    
    # Evaluate
    results = evaluator.evaluate_all(recs, val, loader.n_items)
    
    print(f"{name:15} NDCG@20: {results['ndcg']:.4f}")
```

Output:
```
Popularity      NDCG@20: 0.0823
Item-CF         NDCG@20: 0.1756
ALS             NDCG@20: 0.2341
```

---

## Understanding the Output

### When you run `python main.py`:

```
==============================================================================
VALIDATION SET EVALUATION
==============================================================================

Evaluating Popularity on validation set...
============================================================
EVALUATION RESULTS: Popularity
============================================================

Ranking Metrics (@ 20):
  NDCG@20:      0.0823    ← This is what you care about!
  Precision@20: 0.0156
  Recall@20:    0.0234
  MAP@20:       0.0089
  Hit Rate@20:  0.2341

Diversity:
  Coverage:    45.32%
============================================================
```

**What each metric means:**

- **NDCG@20: 0.0823** 
  - Overall quality of top 20 recommendations (0-1 scale)
  - **This is your leaderboard score!** 📊
  
- **Precision@20: 0.0156**
  - Of the 20 items recommended, 1.56% were relevant
  - Higher = fewer irrelevant recommendations
  
- **Recall@20: 0.0234**
  - Of all relevant items, 2.34% were in top 20
  - Higher = found more relevant items
  
- **MAP@20: 0.0089**
  - Mean Average Precision
  - Considers precision at each relevant item position
  
- **Hit Rate@20: 0.2341**
  - 23.41% of users got at least 1 relevant item
  - Binary: either found something or didn't
  
- **Coverage: 45.32%**
  - 45.32% of all items were recommended to someone
  - Higher = more diverse recommendations

---

## Submission File Format

When you run `python generate_submission.py`, it creates:

**File**: `outputs/submission.txt`

```
0 45 23 78 12 56 89 34 67 90 11 44 77 22 55 88 33 66 99 10 43
1 12 34 56 78 90 11 33 55 77 99 22 44 66 88 10 32 54 76 98 20
2 89 67 45 23 12 90 78 56 34 22 88 66 44 33 77 55 44 32 21 10
...
```

**Format**:
- First number = user_id
- Next 20 numbers = recommended item_ids (in order of relevance)
- One line per user

---

## Tips for Better NDCG@20

### 1. Tune Hyperparameters
```python
# Try higher values:
ALSRecommender(
    factors=100,      # Default: 50, Try: 100, 150, 200
    iterations=30,    # Default: 15, Try: 20, 30, 40
    regularization=0.001  # Default: 0.01, Try: 0.001, 0.1
)
```

### 2. Use Hybrid Model
```python
# Hybrid usually performs best
# Adjust weights based on validation results
HybridRecommender(
    recommenders=[popularity, item_cf, als],
    weights=[0.2, 0.3, 0.5]  # More weight to better models
)
```

### 3. Check Validation Results
```bash
# main.py shows you which model works best
python main.py

# Then use that model for submission
python generate_submission.py --model als  # if ALS was best
```

---

## Common Issues

### Issue: NDCG@20 is 0 for some users
**Cause**: User has no items in validation set  
**Solution**: This is normal, evaluation skips these users

### Issue: NDCG@20 is very low (< 0.05)
**Cause**: Model not learning well  
**Solution**: 
- Increase factors/iterations
- Try different model (ALS usually works well)
- Use hybrid approach

### Issue: All models have similar NDCG@20
**Cause**: Dataset might be very sparse  
**Solution**: 
- This is normal for sparse data
- Hybrid can still improve slightly
- Focus on relative improvements

---

## Summary

✅ **Get Top 20**: `model.recommend(user_id, n=20, exclude_items=user_history)`  
✅ **Calculate NDCG@20**: `ndcg_at_k(recommended, relevant, k=20)`  
✅ **Evaluate All**: `evaluator.evaluate_all(recommendations, test_data)`  
✅ **Best Practice**: Run `python main.py` to see everything automatically  

**NDCG@20 Range**:
- 0.00 - 0.10: Poor
- 0.10 - 0.20: Fair  
- 0.20 - 0.30: Good ⭐
- 0.30 - 0.40: Very Good ⭐⭐
- 0.40+: Excellent ⭐⭐⭐

Your goal: Get the highest NDCG@20 possible for the leaderboard! 🎯

