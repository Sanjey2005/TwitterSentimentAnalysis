# Streamlit App - Quick Start Guide

## 🚀 Running the Streamlit App

### 1. Install Dependencies

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

### 2. Run the App

Navigate to the project directory and run:

**Windows (PowerShell/CMD):**
```bash
python -m streamlit run app.py
```

**Linux/Mac:**
```bash
streamlit run app.py
```

Or use the provided scripts:
- **Windows**: Double-click `run_app.bat` or run `run_app.bat` from command line
- **Linux/Mac**: `bash run_app.sh`

The app will automatically open in your default web browser at `http://localhost:8501`

**Note**: If `streamlit` command is not recognized on Windows, use `python -m streamlit run app.py` instead.

### 3. Using the App

#### Navigation
The app has 4 main sections accessible from the sidebar:

1. **🏠 Home**: Overview of the project and quick start guide
2. **🤖 Model Information**: Details about available models and their performance
3. **🔮 Predict Sentiment**: Interactive sentiment prediction interface
4. **📈 Project Capabilities**: Comprehensive feature list and use cases

#### Making Predictions

1. Go to **"🔮 Predict Sentiment"** section
2. Choose input method:
   - **Type your text**: Enter text directly in the text area
   - **Upload a file**: Upload a CSV file with a 'text' column
3. Select a model (Logistic Regression, Random Forest, or Ensemble)
4. Click **"🚀 Predict Sentiment"** button
5. View results with sentiment, confidence, and probabilities

#### Features

- **Single Text Prediction**: Get sentiment analysis for individual texts
- **Batch Processing**: Upload CSV files to analyze multiple texts at once
- **Multiple Models**: Choose from different trained models
- **Visual Results**: Color-coded sentiment indicators and confidence scores
- **Export Results**: Download predictions as CSV file

### 4. Model Requirements

The app requires the following saved models in `models/saved_models/`:

- `tfidf_vectorizer.pkl` - TF-IDF vectorizer (required)
- `logistic_regression_model.pkl` - Logistic Regression model
- `random_forest_model.pkl` - Random Forest model
- `ensemble_model.pkl` - Ensemble model
- `scaler.pkl` - Feature scaler (for Logistic Regression)
- Preprocessing pipeline (automatically initialized)

### 5. Troubleshooting

**Models not loading?**
- Check that model files exist in `models/saved_models/` directory
- Ensure model files are properly saved and not corrupted

**Prediction errors?**
- Make sure the text input is not empty
- Check that the TF-IDF vectorizer is properly loaded
- Verify model compatibility with the vectorizer

**Import errors?**
- Install all dependencies: `pip install -r requirements.txt`
- Ensure you're using Python 3.8 or higher

**Streamlit command not found (Windows)?**
- Use `python -m streamlit run app.py` instead of `streamlit run app.py`
- This is the recommended way on Windows systems

### 6. Customization

You can customize the app by:

- Modifying the styling in the CSS section
- Adding more models or features
- Changing the layout or adding new pages
- Integrating additional analysis features

---

**Note**: The app uses cached model loading for better performance. Models are loaded once when the app starts.
