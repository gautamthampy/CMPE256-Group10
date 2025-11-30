"""
Main script for training and evaluating recommender systems.
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader, print_statistics
from evaluation import Evaluator, compare_models
from recommenders import (
    PopularityRecommender,
    UserCFRecommender,
    ItemCFRecommender,
    ALSRecommender,
    SVDRecommender,
    HybridRecommender
)
from utils import timer


def main():
    """Main training and evaluation pipeline."""
    
    print("="*80)
    print("CMPE 256 - RECOMMENDER SYSTEMS PROJECT")
    print("="*80)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    data_path = "train-2.txt"
    
    with timer("Data Loading"):
        loader = DataLoader(data_path, seed=42)
        user_items = loader.load_data()
        
        # Print statistics
        stats = loader.get_statistics()
        print_statistics(stats)
    
    # =========================================================================
    # 2. SPLIT DATA
    # =========================================================================
    with timer("Data Splitting"):
        train_data, val_data, test_data = loader.create_train_test_split(
            test_ratio=0.2,
            min_items_test=1
        )
    
    # =========================================================================
    # 3. INITIALIZE MODELS
    # =========================================================================
    print("\n" + "="*80)
    print("INITIALIZING MODELS")
    print("="*80 + "\n")
    
    models = {
        'Popularity': PopularityRecommender(),
        'User-CF': UserCFRecommender(k_neighbors=50),
        'Item-CF': ItemCFRecommender(k_neighbors=50),
        'ALS': ALSRecommender(factors=50, iterations=15, regularization=0.01),
    }
    
    # Note: SVD is commented out initially as it's slower
    # Uncomment if you want to train it
    # models['SVD'] = SVDRecommender(n_factors=50, n_epochs=20)
    
    # =========================================================================
    # 4. TRAIN MODELS
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80 + "\n")
    
    trained_models = {}
    training_times = {}
    
    for model_name, model in models.items():
        with timer(f"Training {model_name}"):
            start = time.time()
            model.fit(train_data)
            training_times[model_name] = time.time() - start
            trained_models[model_name] = model
    
    # =========================================================================
    # 5. TRAIN HYBRID MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING HYBRID MODEL")
    print("="*80 + "\n")
    
    # Create hybrid from best performing models
    # Adjust weights based on your validation results
    hybrid_components = [
        trained_models['Popularity'],
        trained_models['Item-CF'],
        trained_models['ALS'],
    ]
    hybrid_weights = [0.2, 0.4, 0.4]  # Adjust based on validation performance
    
    hybrid = HybridRecommender(hybrid_components, hybrid_weights)
    
    with timer("Training Hybrid"):
        start = time.time()
        hybrid.fit(train_data)
        training_times['Hybrid'] = time.time() - start
        trained_models['Hybrid'] = hybrid
    
    # =========================================================================
    # 6. EVALUATE ON VALIDATION SET
    # =========================================================================
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80 + "\n")
    
    evaluator = Evaluator(k=20)
    val_results = {}
    
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name} on validation set...")
        
        with timer(f"Generating recommendations for {model_name}"):
            recommendations = model.recommend_all(
                user_ids=list(val_data.keys()),
                n=20,
                exclude_train=True,
                train_data=train_data
            )
        
        results = evaluator.evaluate_all(
            recommendations=recommendations,
            test_data=val_data,
            n_items=loader.n_items
        )
        
        results['training_time'] = training_times[model_name]
        val_results[model_name] = results
        
        evaluator.print_results(results, model_name)
    
    # Compare all models
    compare_models(val_results, k=20)
    
    # =========================================================================
    # 7. EVALUATE ON TEST SET (BEST MODEL)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST SET EVALUATION (BEST MODEL)")
    print("="*80 + "\n")
    
    # Find best model based on validation NDCG
    best_model_name = max(val_results.items(), key=lambda x: x[1]['ndcg'])[0]
    best_model = trained_models[best_model_name]
    
    print(f"Best model: {best_model_name} (NDCG@20: {val_results[best_model_name]['ndcg']:.4f})")
    print(f"\nEvaluating {best_model_name} on test set...")
    
    with timer(f"Test set evaluation"):
        test_recommendations = best_model.recommend_all(
            user_ids=list(test_data.keys()),
            n=20,
            exclude_train=True,
            train_data=train_data
        )
        
        test_results = evaluator.evaluate_all(
            recommendations=test_recommendations,
            test_data=test_data,
            n_items=loader.n_items
        )
        
        evaluator.print_results(test_results, f"{best_model_name} (Test Set)")
    
    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    print(f"Dataset: {loader.n_users} users, {loader.n_items} items")
    print(f"Training: {sum(len(v) for v in train_data.values())} interactions")
    print(f"Validation: {sum(len(v) for v in val_data.values())} interactions")
    print(f"Test: {sum(len(v) for v in test_data.values())} interactions")
    print(f"\nBest Model: {best_model_name}")
    print(f"  Validation NDCG@20: {val_results[best_model_name]['ndcg']:.4f}")
    print(f"  Test NDCG@20: {test_results['ndcg']:.4f}")
    print(f"  Training Time: {training_times[best_model_name]:.2f}s")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review validation results and adjust hyperparameters if needed")
    print("2. Run generate_submission.py to create submission file")
    print("3. Submit to leaderboard: http://coe-clp.sjsu.edu/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

