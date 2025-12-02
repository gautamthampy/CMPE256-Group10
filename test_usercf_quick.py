"""
Quick test script for User-CF to verify NDCG@20 performance.
Tests on a smaller subset (1000 users) for fast results.
"""
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from evaluation import Evaluator
from recommenders import UserCFRecommender


def main():
    print("="*80)
    print("QUICK USER-CF TEST (Subset of 1000 Users)")
    print("="*80)
    print("This will take ~5-10 minutes instead of 60+ minutes\n")
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("[1/5] Loading data...")
    loader = DataLoader('train-2.txt', seed=42)
    user_items = loader.load_data()
    
    print(f"Total dataset: {loader.n_users} users, {loader.n_items} items")
    
    # =========================================================================
    # 2. CREATE SUBSET FOR TESTING
    # =========================================================================
    print("\n[2/5] Creating test subset (1000 users)...")
    
    # Take first 1000 users
    all_users = sorted(user_items.keys())
    test_users = all_users[:1000]
    
    # Create subset
    subset_data = {uid: user_items[uid] for uid in test_users}
    
    print(f"Test subset: {len(subset_data)} users")
    
    # Split the subset
    train_data, val_data, test_data = loader.create_train_test_split(
        test_ratio=0.2,
        min_items_test=1
    )
    
    # Keep only test users in validation
    val_data_subset = {uid: val_data[uid] for uid in test_users if uid in val_data and len(val_data[uid]) > 0}
    
    print(f"Validation users: {len(val_data_subset)}")
    
    # =========================================================================
    # 3. TRAIN USER-CF
    # =========================================================================
    print("\n[3/5] Training User-CF model...")
    print("Note: This uses same algorithm as full submission")
    
    model = UserCFRecommender(k_neighbors=50)
    
    start = time.time()
    model.fit(train_data)
    train_time = time.time() - start
    
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    # =========================================================================
    # 4. GENERATE RECOMMENDATIONS
    # =========================================================================
    print("\n[4/5] Generating recommendations for test users...")
    
    recommendations = {}
    start = time.time()
    
    for i, user_id in enumerate(val_data_subset.keys()):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(val_data_subset)} users...")
        
        user_history = set(train_data.get(user_id, []))
        recommendations[user_id] = model.recommend(
            user_id=user_id,
            n=20,
            exclude_items=user_history
        )
    
    rec_time = time.time() - start
    print(f"✓ Recommendations generated in {rec_time:.2f} seconds ({rec_time/60:.2f} minutes)")
    
    # =========================================================================
    # 5. EVALUATE
    # =========================================================================
    print("\n[5/5] Evaluating NDCG@20...")
    
    evaluator = Evaluator(k=20)
    results = evaluator.evaluate_all(
        recommendations=recommendations,
        test_data=val_data_subset,
        n_items=loader.n_items
    )
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("TEST RESULTS (1000 users)")
    print("="*80)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  NDCG@20:      {results['ndcg']:.4f} ⭐")
    print(f"  Precision@20: {results['precision']:.4f}")
    print(f"  Recall@20:    {results['recall']:.4f}")
    print(f"  MAP@20:       {results['map']:.4f}")
    print(f"  Hit Rate@20:  {results['hit_rate']:.4f}")
    
    print(f"\n⏱️  Timing:")
    print(f"  Training:      {train_time:.2f}s")
    print(f"  Recommendations: {rec_time:.2f}s ({rec_time/60:.2f} min)")
    print(f"  Total:         {train_time + rec_time:.2f}s ({(train_time + rec_time)/60:.2f} min)")
    
    print(f"\n🔮 Extrapolation to Full Dataset ({loader.n_users} users):")
    estimated_time = (rec_time / len(val_data_subset)) * loader.n_users
    print(f"  Estimated time: {estimated_time:.2f}s ({estimated_time/60:.2f} min)")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    ndcg = results['ndcg']
    
    if ndcg >= 0.08:
        print("✅ EXCELLENT! NDCG@20 >= 0.08")
        print("   This is a good score for this dataset.")
        print("   Recommendation: Use User-CF for submission!")
    elif ndcg >= 0.05:
        print("✅ GOOD! NDCG@20 between 0.05-0.08")
        print("   This is acceptable for this sparse dataset.")
        print("   Recommendation: Use User-CF for submission.")
    elif ndcg >= 0.03:
        print("⚠️  FAIR. NDCG@20 between 0.03-0.05")
        print("   Not great but better than popularity (0.0077).")
        print("   Consider using User-CF or investigating further.")
    else:
        print("❌ POOR. NDCG@20 < 0.03")
        print("   Something might be wrong. Check implementation.")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if ndcg >= 0.05:
        print("\n✅ User-CF looks good! To generate full submission:")
        print("   python generate_submission_usercf.py")
        print(f"   (Estimated time: ~{estimated_time/60:.0f} minutes)")
        print("\n✅ Or push your code to GitHub first:")
        print("   git add .")
        print("   git commit -m 'Add working User-CF implementation'")
        print("   git push -u origin alshamams-recommender-system")
    else:
        print("\n⚠️  NDCG@20 is lower than expected.")
        print("   You may want to:")
        print("   1. Check if there's an issue with the implementation")
        print("   2. Try different k_neighbors values")
        print("   3. Compare with other team members' results")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

