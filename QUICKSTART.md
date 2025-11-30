# Quick Start Guide

## Installation

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Train and Evaluate Multiple Models (Recommended First)

This will train all models and show you which performs best:

```bash
python main.py
```

Expected output:
- Data loading and statistics
- Training progress for each model
- Validation results with NDCG@20 scores
- Model comparison table
- Best model identification

This takes approximately 5-15 minutes depending on your hardware.

### Option 2: Generate Submission File

Once you've identified the best model, generate the submission:

```bash
# Generate with hybrid model (recommended)
python generate_submission.py --model hybrid

# Or specify a different model
python generate_submission.py --model als
python generate_submission.py --model item_cf
```

Available models:
- `popularity`: Baseline popularity-based
- `user_cf`: User-based collaborative filtering
- `item_cf`: Item-based collaborative filtering
- `als`: Alternating Least Squares matrix factorization
- `svd`: SVD-based matrix factorization
- `hybrid`: Ensemble of multiple models (recommended)

Output: `outputs/submission.txt`

### Option 3: Explore Data

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Submitting to Leaderboard

1. Connect to SJSU VPN
2. Go to http://coe-clp.sjsu.edu/
3. Login with MySJSU credentials
4. Upload `outputs/submission.txt`
5. Check your NDCG@20 score!

## Troubleshooting

**Issue: "implicit library not available"**
- The code will automatically fall back to basic ALS implementation
- For better performance: `pip install implicit`

**Issue: "Surprise library not available"**  
- The code will use ALS instead of SVD
- To enable SVD: `pip install scikit-surprise`

**Issue: Out of memory**
- Reduce `k_neighbors` parameter in collaborative filtering models
- Reduce `factors` in matrix factorization models
- Process data in smaller batches

## Tips for Better Performance

1. **Hyperparameter Tuning**: Edit the parameters in `main.py`:
   - Increase `factors` for ALS (50 → 100)
   - Increase `iterations` for ALS (15 → 20-30)
   - Adjust `k_neighbors` for CF models (50 → 100)

2. **Hybrid Model Weights**: Adjust weights in `generate_submission.py` based on validation results

3. **Try Different Models**: Some datasets work better with different algorithms

## File Structure

```
CMPE_256_FINAL_PROJECT/
├── train-2.txt              # Your dataset
├── main.py                  # Training and evaluation
├── generate_submission.py   # Create submission file
├── requirements.txt         # Dependencies
├── src/                     # Source code
│   ├── data_loader.py
│   ├── evaluation.py
│   └── recommenders/
└── outputs/                 # Generated submissions
```
