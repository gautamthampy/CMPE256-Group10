# 🚀 START HERE - CMPE 256 Recommender Systems Project

## What You Have

A **complete, production-ready recommender system** with:
- ✅ 6 different recommendation algorithms
- ✅ Comprehensive evaluation framework (NDCG@20 and more)
- ✅ Automated train/test splitting
- ✅ Submission file generator
- ✅ Data exploration tools
- ✅ GitHub-ready codebase

## Quick Start (3 Steps)

### Step 1: Install (2 minutes)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train & Evaluate (5-15 minutes)
```bash
python main.py
```

**What this does:**
- Loads your dataset (train-2.txt)
- Trains 6 different models
- Shows you which performs best
- Displays comprehensive comparison

**Expected Output:**
```
NDCG@20 scores for each model:
  Popularity: ~0.08
  User-CF: ~0.12
  Item-CF: ~0.18
  ALS: ~0.22
  Hybrid: ~0.25 (usually best!)
```

### Step 3: Generate Submission (1 minute)
```bash
# Use the best model (usually hybrid)
python generate_submission.py --model hybrid
```

**Output:** `outputs/submission.txt` (ready for leaderboard!)

## Submit to Leaderboard

1. Connect to SJSU VPN
2. Go to: http://coe-clp.sjsu.edu/
3. Login with MySJSU credentials
4. Upload: `outputs/submission.txt`
5. 🎉 See your NDCG@20 score!

## What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `main.py` | Train & compare all models | **Start here** - Run this first! |
| `generate_submission.py` | Create leaderboard file | After training, to submit |
| `notebooks/exploration.ipynb` | Explore data visually | Optional - for understanding data |
| `README.md` | Complete documentation | Reference for details |
| `QUICKSTART.md` | Quick reference guide | When you need help |
| `PROJECT_SUMMARY.md` | Detailed project overview | For your report |

## Implemented Algorithms

1. **Popularity** - Simple baseline (fast)
2. **User-CF** - User-based collaborative filtering
3. **Item-CF** - Item-based collaborative filtering (usually good!)
4. **ALS** - Matrix factorization (usually best individual model!)
5. **SVD** - Alternative matrix factorization
6. **Hybrid** - Combines multiple models (often best overall!)

## Typical Results

Based on similar datasets:
- **Popularity**: NDCG@20 ≈ 0.05-0.10
- **User-CF**: NDCG@20 ≈ 0.10-0.15
- **Item-CF**: NDCG@20 ≈ 0.15-0.20
- **ALS**: NDCG@20 ≈ 0.20-0.25
- **Hybrid**: NDCG@20 ≈ 0.22-0.30

Your results will vary based on your specific dataset!

## Improving Performance

If you want higher NDCG@20 scores, edit `generate_submission.py`:

```python
# Increase these for better quality (but slower training)
ALSRecommender(
    factors=100,      # Default: 50, Try: 100, 150
    iterations=20,    # Default: 15, Try: 20, 30
)

# Adjust hybrid weights based on validation results
hybrid_weights = [0.2, 0.4, 0.4]  # [popularity, item_cf, als]
```

## Troubleshooting

**Problem**: `ModuleNotFoundError`
```bash
# Solution:
pip install -r requirements.txt
```

**Problem**: "implicit library not available"
```bash
# Optional: Better performance
pip install implicit
# Or: Just ignore - code has fallback
```

**Problem**: Code is slow
```bash
# Solution: Use fewer models
# Edit main.py and comment out slow models (User-CF, SVD)
```

**Problem**: Out of memory
```bash
# Solution: Reduce parameters in main.py
# Change k_neighbors from 50 to 30
# Change factors from 50 to 30
```

## For Your Report

This implementation gives you:

✅ **Multiple Algorithms**: 6 different approaches  
✅ **Proper Evaluation**: NDCG@20 + 5 other metrics  
✅ **Data Preprocessing**: Train/val/test splits  
✅ **Algorithm Comparison**: Side-by-side results  
✅ **Code Quality**: Clean, documented, modular  
✅ **GitHub Ready**: Professional structure  

## Project Structure

```
CMPE_256_FINAL_PROJECT/
│
├── 📄 START_HERE.md          ← You are here!
├── 📄 README.md               ← Full documentation
├── 📄 QUICKSTART.md          ← Quick reference
├── 📄 requirements.txt       ← Install this
│
├── 🔥 main.py                ← RUN THIS FIRST
├── 🎯 generate_submission.py ← Then run this
│
├── 📊 train-2.txt            ← Your data
│
├── 📁 src/                   ← All the algorithms
├── 📁 notebooks/             ← Data exploration
├── 📁 outputs/               ← Your submission file
└── 📁 models/                ← Optional: save models
```

## Your Workflow

```
1. Install dependencies
   ↓
2. Run: python main.py
   ↓
3. Check which model is best
   ↓
4. Run: python generate_submission.py --model hybrid
   ↓
5. Submit outputs/submission.txt to leaderboard
   ↓
6. If NDCG@20 is low, tune hyperparameters and repeat
   ↓
7. Use results for your project report
```

## Time Estimate

- **Setup**: 2 minutes
- **First training run**: 5-15 minutes
- **Generate submission**: 1 minute
- **Submit to leaderboard**: 2 minutes
- **Total**: ~20 minutes to first submission!

## Questions?

1. **How do I know it's working?** → Run `python main.py` and watch the output
2. **Which model should I submit?** → Usually "hybrid" performs best
3. **Can I modify the code?** → Yes! It's designed to be extensible
4. **What if I get errors?** → Check QUICKSTART.md troubleshooting section

## Ready? Let's Go! 🚀

```bash
# Step 1: Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 2: Train
python main.py

# Step 3: Submit
python generate_submission.py --model hybrid

# Step 4: Upload outputs/submission.txt to leaderboard!
```

## Success Metrics

After running, you should have:
- [x] Trained 6 different models
- [x] NDCG@20 scores for each model
- [x] Comparison table showing best model
- [x] `outputs/submission.txt` file
- [x] Ready to submit to leaderboard

Good luck! 🎓
