"""
Generate submission file using User-CF (best performing model).
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from recommenders import UserCFRecommender
from utils import timer


def generate_submission(output_file: str = "outputs/submission_usercf.txt"):
    """
    Generate submission file with 20 recommendations per user using User-CF.
    
    Args:
        output_file: Path to output file
    """
    
    print("="*80)
    print("GENERATING SUBMISSION FILE WITH USER-CF")
    print("="*80)
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
    print("\nTraining User-CF model...")
    print("Note: This will be slow but gives best results (NDCG@20: 0.0846)")
    
    model = UserCFRecommender(k_neighbors=50)
    
    with timer("Training User-CF"):
        model.fit(user_items)
    
    # =========================================================================
    # 3. GENERATE RECOMMENDATIONS
    # =========================================================================
    print("\nGenerating recommendations for all users...")
    print("Note: This takes ~60+ minutes but produces quality recommendations")
    
    all_users = sorted(user_items.keys())
    
    recommendations = {}
    
    with timer("Generating recommendations"):
        from tqdm import tqdm
        for user_id in tqdm(all_users, desc="Users"):
            recommendations[user_id] = model.recommend(
                user_id=user_id,
                n=20,
                exclude_items=set(user_items[user_id])
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
    print("5. Expected NDCG@20: ~0.08-0.09 (much better than 0.0077!)")
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_submission()

