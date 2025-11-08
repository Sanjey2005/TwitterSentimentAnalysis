# 📚 Twitter Sentiment Analysis - Setup & Run Guide

This guide will walk you through running the entire project from scratch.

## Prerequisites

- Python 3.8 or higher
- Git (optional, if cloning from repository)
- At least 8GB RAM (16GB recommended for BERT training)
- Internet connection (for downloading pre-trained models)

---

## Step 1: Environment Setup

### 1.1 Navigate to Project Directory
```bash
cd C:\Users\acer\Desktop\ML\ml_proj
```

### 1.2 Create Virtual Environment (if not already created)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
# source .venv/bin/activate
```

### 1.3 Install Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Download NLTK data (required for text processing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Step 2: Data Preparation

### 2.1 Check Dataset File
- Ensure `sentiment140.csv` exists in the project root directory
- If not, download it from: http://help.sentiment140.com/for-students
- Place it in: `C:\Users\acer\Desktop\ML\ml_proj\sentiment140.csv`

### 2.2 Verify Data Structure
The CSV should have columns:
- `sentiment` (0=negative, 4=positive)
- `tweet_id`
- `date`
- `query`
- `username`
- `tweet_text`

---

## Step 3: Run Notebooks in Order

**Important:** Run notebooks sequentially as each depends on outputs from previous notebooks.

### Step 3.1: Data Exploration
```bash
# Open Jupyter Notebook
jupyter notebook

# Or use VS Code/other IDE to open:
notebooks/01_data_exploration.ipynb
```

**What it does:**
- Loads and explores the Sentiment140 dataset
- Analyzes sentiment distribution
- Visualizes text characteristics
- Saves exploration reports

**Expected Output:**
- Figures saved in `reports/figures/sentiment_distribution.png`
- Dataset statistics and insights

**Run all cells** (Shift + Enter through each cell)

---

### Step 3.2: Data Preprocessing
```bash
# Open notebook:
notebooks/02_preprocessing.ipynb
```

**What it does:**
- Cleans and preprocesses text data
- Handles Twitter-specific elements (hashtags, mentions, URLs)
- Creates TF-IDF features
- Creates Word2Vec embeddings
- Balances classes using SMOTE
- Saves processed data

**Expected Output:**
- Processed data in `data/processed/`:
  - `X_tfidf_train.npy`, `X_tfidf_test.npy`
  - `y_train.npy`, `y_test.npy`
  - `text_train.pkl`, `text_test.pkl`
- Models in `models/saved_models/`:
  - `tfidf_vectorizer.pkl`

**Run all cells** - This may take 10-30 minutes depending on dataset size.

---

### Step 3.3: Train Baseline Models
```bash
# Open notebook:
notebooks/03_baseline_models.ipynb
```

**What it does:**
- Trains Logistic Regression model
- Trains Naive Bayes model
- Trains Multi-Layer Perceptron (MLP) model
- Performs hyperparameter tuning
- Compares model performance
- Saves trained models

**Expected Output:**
- Models in `models/saved_models/`:
  - `logistic_regression_model.pkl`
  - `naive_bayes_model.pkl`
  - `mlp_model.pkl`
  - `scaler.pkl` (for Logistic Regression and MLP)
- Figures in `reports/figures/`:
  - `baseline_models_comparison.png`
  - `feature_importance.png`

**Run all cells** - Training may take 15-45 minutes.

---

### Step 3.4: Train BERT Model
```bash
# Open notebook:
notebooks/04_deep_learning.ipynb
```

**What it does:**
- Loads DistilBERT (lightweight BERT)
- Fine-tunes BERT for sentiment analysis
- Evaluates BERT performance
- Saves BERT model and tokenizer

**Expected Output:**
- Model in `models/saved_models/bert_model/`:
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer_config.json`
  - `vocab.txt`
- Figures in `reports/figures/`:
  - `bert_training_history.png`
  - `bert_evaluation.png`

**Important Notes:**
- BERT training is CPU-friendly but still takes time (30-60 minutes)
- Uses DistilBERT which is 60% smaller and faster than full BERT
- Dataset is automatically reduced for faster training (8000 train, 2000 test)
- Set `RUN_BERT = True` in the notebook (already set by default)

**Run all cells** - BERT training is the longest step.

---

### Step 3.5: Explainability Analysis (Optional)
```bash
# Open notebook:
notebooks/05_explainability.ipynb
```

**What it does:**
- Uses SHAP for model explainability
- Analyzes feature importance
- Creates explainability visualizations
- Compares model explanations

**Expected Output:**
- Figures in `reports/figures/`:
  - `shap_summary_lr.png`
  - `shap_summary_nb.png` (if implemented)
  - `shap_waterfall_lr.png`
- Reports in `reports/`:
  - `feature_importance.json`

**Run all cells** - SHAP analysis may take 10-20 minutes.

---

## Step 4: Run the Streamlit App

### 4.1 Start the Application
```bash
# Make sure you're in the project root directory
cd C:\Users\acer\Desktop\ML\ml_proj

# Activate virtual environment (if not already activated)
.venv\Scripts\activate

# Run Streamlit app
streamlit run app.py
```

### 4.2 Access the App
- The app will open automatically in your browser
- If not, navigate to: `http://localhost:8501`

### 4.3 App Features
- **Home**: Project overview and capabilities
- **Model Information**: View available models and their metrics
- **Predict Sentiment**: Make predictions with any of the 4 models:
  - Logistic Regression
  - Naive Bayes
  - MLP
  - BERT
- **Explainability**: Understand model predictions with feature importance

---

## Step 5: Verify Everything Works

### 5.1 Check Model Files
Ensure these files exist in `models/saved_models/`:
```
✅ logistic_regression_model.pkl
✅ naive_bayes_model.pkl
✅ mlp_model.pkl
✅ scaler.pkl
✅ tfidf_vectorizer.pkl
✅ bert_model/ (directory with BERT files)
```

### 5.2 Test Predictions
1. Open the Streamlit app
2. Go to "🔮 Predict Sentiment"
3. Enter a test tweet: "I love this product! It's amazing!"
4. Select each model and verify predictions work
5. Check explainability section

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** 
- Activate virtual environment: `.venv\Scripts\activate`
- Reinstall requirements: `pip install -r requirements.txt`

### Issue: BERT model not loading
**Solution:**
- Check if `models/saved_models/bert_model/` directory exists
- Re-run notebook 04 if BERT model is missing
- Ensure transformers library is installed: `pip install transformers`

### Issue: Out of Memory during training
**Solution:**
- Reduce dataset size in notebooks (set `SMALL_RUN = True`)
- Close other applications
- Use smaller batch sizes in BERT training

### Issue: SHAP not working
**Solution:**
- Install SHAP: `pip install shap>=0.42.0`
- Some models may not support SHAP (MLP requires special handling)

### Issue: Streamlit app shows "No models available"
**Solution:**
- Check that all model files exist in `models/saved_models/`
- Re-run notebooks 03 and 04 to generate models
- Check file paths in `app.py` are correct

---

## Quick Start (If Models Already Trained)

If you already have trained models, you can skip to Step 4:

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Run Streamlit app
streamlit run app.py
```

---

## Project Structure

```
ml_proj/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Step 3.1
│   ├── 02_preprocessing.ipynb          # Step 3.2
│   ├── 03_baseline_models.ipynb        # Step 3.3
│   ├── 04_deep_learning.ipynb          # Step 3.4
│   └── 05_explainability.ipynb         # Step 3.5
├── data/
│   └── processed/                      # Processed data (from notebook 02)
├── models/
│   └── saved_models/                   # Trained models (from notebooks 03, 04)
├── reports/
│   └── figures/                        # Visualizations
├── src/                                # Source code modules
├── app.py                              # Streamlit app (Step 4)
├── requirements.txt                    # Dependencies (Step 1.3)
└── sentiment140.csv                    # Dataset (Step 2.1)
```

---

## Time Estimates

- **Step 1 (Setup):** 5-10 minutes
- **Step 2 (Data Prep):** 2 minutes (if data exists)
- **Step 3.1 (Exploration):** 5-10 minutes
- **Step 3.2 (Preprocessing):** 10-30 minutes
- **Step 3.3 (Baseline Models):** 15-45 minutes
- **Step 3.4 (BERT Training):** 30-60 minutes
- **Step 3.5 (Explainability):** 10-20 minutes (optional)
- **Step 4 (Run App):** Instant

**Total Time:** ~2-3 hours for full pipeline (depending on hardware)

---

## Next Steps

After completing all steps:
1. ✅ All models trained and saved
2. ✅ Streamlit app running
3. ✅ Make predictions with any model
4. ✅ View model explanations
5. ✅ Analyze feature importance

**You're all set! 🎉**

---

## Need Help?

- Check `TROUBLESHOOTING.md` for common issues
- Review notebook outputs for errors
- Verify all dependencies are installed
- Check that dataset file exists and is valid


