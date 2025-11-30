"""
Example: How to get top 20 recommendations and calculate NDCG@20
"""
import sys
sys.path.insert(0, 'src')

from data_loader import DataLoader
from evaluation import Evaluator, ndcg_at_k
from recommenders import ALSRecommender, HybridRecommender, ItemCFRecommender

# ============================================================================
# EXAMPLE 1: Get Top 20 Recommendations for a Single User
# ============================================================================
print("="*80)
print("EXAMPLE 1: Getting Top 20 Recommendations")
print("="*80)

# Load data
loader = DataLoader('train-2.txt', seed=42)
user_items = loader.load_data()

# Split data (so we can test)
train_data, val_data, test_data = loader.create_train_test_split(test_ratio=0.2)

# Train a model
print("\nTraining ALS model...")
model = ALSRecommender(factors=50, iterations=10)
model.fit(train_data)

# Get recommendations for user 0
user_id = 0
user_history = set(train_data.get(user_id, []))

print(f"\nUser {user_id} history (training): {len(user_history)} items")
print(f"Sample items: {list(user_history)[:10]}")

# Get top 20 recommendations
top_20 = model.recommend(
    user_id=user_id,
    n=20,
    exclude_items=user_history  # Don't recommend what they already have
)

print(f"\nTop 20 Recommendations for User {user_id}:")
print(top_20)

# ============================================================================
# EXAMPLE 2: Calculate NDCG@20 for One User
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Calculating NDCG@20 for One User")
print("="*80)

# Get ground truth (what user actually liked in validation set)
ground_truth = set(val_data.get(user_id, []))
print(f"\nGround truth (validation): {len(ground_truth)} items")
print(f"Items: {ground_truth}")

# Calculate NDCG@20
ndcg_score = ndcg_at_k(
    recommended=top_20,
    relevant=ground_truth,
    k=20
)

print(f"\nNDCG@20 Score: {ndcg_score:.4f}")

# How many relevant items did we find?
hits = len(set(top_20) & ground_truth)
print(f"Relevant items in top 20: {hits}/{len(ground_truth)}")

# ============================================================================
# EXAMPLE 3: Calculate NDCG@20 for All Users (Like main.py does)
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: Calculating Average NDCG@20 Across All Users")
print("="*80)

# Get recommendations for multiple users
print("\nGenerating recommendations for all users...")
all_users = list(val_data.keys())[:100]  # Just first 100 for demo

recommendations = {}
for uid in all_users:
    user_hist = set(train_data.get(uid, []))
    recommendations[uid] = model.recommend(uid, n=20, exclude_items=user_hist)

# Evaluate
evaluator = Evaluator(k=20)
results = evaluator.evaluate_all(
    recommendations=recommendations,
    test_data={uid: val_data[uid] for uid in all_users if uid in val_data},
    n_items=loader.n_items
)

print(f"\nResults for {len(all_users)} users:")
print(f"  Average NDCG@20:     {results['ndcg']:.4f}")
print(f"  Average Precision@20: {results['precision']:.4f}")
print(f"  Average Recall@20:    {results['recall']:.4f}")
print(f"  Average MAP@20:       {results['map']:.4f}")
print(f"  Hit Rate@20:          {results['hit_rate']:.4f}")

# ============================================================================
# EXAMPLE 4: Manual NDCG@20 Calculation (Step by Step)
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 4: NDCG@20 Calculation - Step by Step")
print("="*80)

# Example data
recommended = [10, 5, 20, 15, 30, 25]  # Top 6 recommendations
relevant = {5, 20, 30}                  # Ground truth: items user liked

print(f"\nRecommended: {recommended}")
print(f"Relevant (ground truth): {relevant}")

# Step 1: Create relevance list
relevance = [1 if item in relevant else 0 for item in recommended]
print(f"\nStep 1 - Relevance list: {relevance}")
print("  (1 = relevant, 0 = not relevant)")

# Step 2: Calculate DCG
import numpy as np
dcg = 0
for i, rel in enumerate(relevance):
    position = i + 1
    discount = np.log2(position + 1)
    contribution = rel / discount
    dcg += contribution
    if rel == 1:
        print(f"  Position {position}: relevant item, contribution = 1/log2({position+1}) = {contribution:.4f}")

print(f"\nStep 2 - DCG = {dcg:.4f}")

# Step 3: Calculate Ideal DCG (if all relevant items were at top)
ideal_relevance = [1] * len(relevant) + [0] * (len(recommended) - len(relevant))
ideal_relevance = ideal_relevance[:len(recommended)]
print(f"\nStep 3 - Ideal relevance: {ideal_relevance}")

idcg = 0
for i, rel in enumerate(ideal_relevance):
    if rel == 1:
        idcg += rel / np.log2(i + 2)

print(f"  Ideal DCG = {idcg:.4f}")

# Step 4: Calculate NDCG
ndcg = dcg / idcg if idcg > 0 else 0
print(f"\nStep 4 - NDCG = DCG / iDCG = {dcg:.4f} / {idcg:.4f} = {ndcg:.4f}")

# Verify with built-in function
ndcg_builtin = ndcg_at_k(recommended, relevant, k=6)
print(f"\nVerification with built-in function: {ndcg_builtin:.4f}")

# ============================================================================
# EXAMPLE 5: Compare Multiple Models by NDCG@20
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE 5: Comparing Models by NDCG@20")
print("="*80)

from recommenders import PopularityRecommender

print("\nTraining multiple models...")

# Model 1: Popularity
pop_model = PopularityRecommender()
pop_model.fit(train_data)

# Model 2: Item-CF
itemcf_model = ItemCFRecommender(k_neighbors=30)
itemcf_model.fit(train_data)

# Model 3: ALS (already trained above)

# Evaluate each
models = {
    'Popularity': pop_model,
    'Item-CF': itemcf_model,
    'ALS': model
}

print("\nGenerating recommendations and calculating NDCG@20...\n")

for name, m in models.items():
    # Get recommendations
    recs = {}
    for uid in all_users:
        user_hist = set(train_data.get(uid, []))
        recs[uid] = m.recommend(uid, n=20, exclude_items=user_hist)
    
    # Evaluate
    results = evaluator.evaluate_all(
        recommendations=recs,
        test_data={uid: val_data[uid] for uid in all_users if uid in val_data},
        n_items=loader.n_items
    )
    
    print(f"{name:15} NDCG@20: {results['ndcg']:.4f}")

print("\n" + "="*80)
print("Examples Complete!")
print("="*80)
print("\nKey Takeaways:")
print("1. Use model.recommend(user_id, n=20) to get top 20 items")
print("2. Use ndcg_at_k(recommended, relevant, k=20) to calculate NDCG@20")
print("3. Use Evaluator.evaluate_all() to get average NDCG@20 across users")
print("4. Higher NDCG@20 = better recommendations (range: 0 to 1)")
print("="*80)

