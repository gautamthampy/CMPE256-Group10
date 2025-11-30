"""
Generate submission file for leaderboard.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from recommenders import (
    PopularityRecommender,
    UserCFRecommender,
    ItemCFRecommender,
    ALSRecommender,
    SVDRecommender,
    HybridRecommender
)
from utils import timer


def generate_submission(model_type: str = "hybrid", output_file: str = "outputs/submission.txt"):
    """
    Generate submission file with 20 recommendations per user.
    
    Args:
        model_type: Type of model to use ('popularity', 'user_cf', 'item_cf', 'als', 'svd', 'hybrid')
        output_file: Path to output file
    """
    
    print("="*80)
    print("GENERATING SUBMISSION FILE")
    print("="*80)
    print(f"Model: {model_type}")
    print(f"Output: {output_file}\n")
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    data_path = "train-2.txt"
    
    with timer("Loading data"):
        loader = DataLoader(data_path, seed=42)
        user_items = loader.load_data()
    
    # =========================================================================
    # 2. TRAIN MODEL
    # =========================================================================
    print(f"\nTraining {model_type} model...")
    
    if model_type.lower() == "popularity":
        model = PopularityRecommender()
    elif model_type.lower() == "user_cf":
        model = UserCFRecommender(k_neighbors=50)
    elif model_type.lower() == "item_cf":
        model = ItemCFRecommender(k_neighbors=50)
    elif model_type.lower() == "als":
        model = ALSRecommender(factors=100, iterations=20, regularization=0.01)
    elif model_type.lower() == "svd":
        model = SVDRecommender(n_factors=100, n_epochs=30)
    elif model_type.lower() == "hybrid":
        # Train component models
        print("\nTraining component models for hybrid...")
        
        popularity = PopularityRecommender()
        popularity.fit(user_items)
        
        item_cf = ItemCFRecommender(k_neighbors=50)
        item_cf.fit(user_items)
        
        als = ALSRecommender(factors=100, iterations=20, regularization=0.01)
        als.fit(user_items)
        
        # Create hybrid with optimized weights
        model = HybridRecommender(
            recommenders=[popularity, item_cf, als],
            weights=[0.15, 0.35, 0.50]  # Adjust based on validation results
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    with timer(f"Training {model_type}"):
        model.fit(user_items)
    
    # =========================================================================
    # 3. GENERATE RECOMMENDATIONS
    # =========================================================================
    print("\nGenerating recommendations for all users...")
    
    all_users = sorted(user_items.keys())
    
    with timer("Generating recommendations"):
        recommendations = model.recommend_all(
            user_ids=all_users,
            n=20,
            exclude_train=True,
            train_data=user_items
        )
    
    # =========================================================================
    # 4. WRITE SUBMISSION FILE
    # =========================================================================
    print(f"\nWriting submission to {output_file}...")
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for user_id in all_users:
            recs = recommendations[user_id]
            
            # Ensure we have exactly 20 recommendations
            if len(recs) < 20:
                print(f"WARNING: User {user_id} has only {len(recs)} recommendations")
                # Pad with popular items if needed
                all_items = sorted(loader.all_items)
                user_history = set(user_items[user_id])
                for item in all_items:
                    if item not in user_history and item not in recs:
                        recs.append(item)
                        if len(recs) >= 20:
                            break
            
            # Write: user_id followed by 20 item_ids
            line = f"{user_id} " + " ".join(map(str, recs[:20]))
            f.write(line + "\n")
    
    print(f"\n✓ Submission file generated successfully!")
    print(f"  Users: {len(all_users)}")
    print(f"  Recommendations per user: 20")
    print(f"  Total recommendations: {len(all_users) * 20}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Connect to SJSU VPN")
    print("2. Go to http://coe-clp.sjsu.edu/")
    print("3. Login with MySJSU credentials")
    print(f"4. Upload {output_file}")
    print("5. Check your NDCG@20 score on the leaderboard!")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate submission file for recommender system")
    parser.add_argument(
        '--model',
        type=str,
        default='hybrid',
        choices=['popularity', 'user_cf', 'item_cf', 'als', 'svd', 'hybrid'],
        help='Model type to use for recommendations'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/submission.txt',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    generate_submission(model_type=args.model, output_file=args.output)


if __name__ == "__main__":
    main()

