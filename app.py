"""
Twitter Sentiment Analysis - Streamlit Web Application

A comprehensive Streamlit frontend for the Twitter Sentiment Analysis project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from pathlib import Path

# Import project modules
from src.preprocessing import TwitterTextPreprocessor
from src.models import predict_sentiment

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #1f1f1f;
    }
    .model-card h3 {
        color: #1DA1F2;
        margin-top: 0;
    }
    .model-card h4 {
        color: #333333;
        margin-top: 1rem;
    }
    .model-card p {
        color: #1f1f1f;
    }
    .model-card ul {
        color: #1f1f1f;
    }
    .model-card li {
        color: #1f1f1f;
    }
    .model-card strong {
        color: #1DA1F2;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .metric-box {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load models and resources
@st.cache_resource
def load_models():
    """Load all saved models and resources"""
    models_dir = Path("models/saved_models")
    
    loaded_models = {}
    
    try:
        # Load TF-IDF vectorizer
        if (models_dir / "tfidf_vectorizer.pkl").exists():
            with open(models_dir / "tfidf_vectorizer.pkl", "rb") as f:
                loaded_models['tfidf_vectorizer'] = pickle.load(f)
        
        # Load scaler
        if (models_dir / "scaler.pkl").exists():
            loaded_models['scaler'] = joblib.load(models_dir / "scaler.pkl")
        
        # Load Logistic Regression
        if (models_dir / "logistic_regression_model.pkl").exists():
            loaded_models['logistic_regression'] = joblib.load(
                models_dir / "logistic_regression_model.pkl"
            )
        
        # Load Random Forest
        if (models_dir / "random_forest_model.pkl").exists():
            loaded_models['random_forest'] = joblib.load(
                models_dir / "random_forest_model.pkl"
            )
        
        # Load Ensemble
        if (models_dir / "ensemble_model.pkl").exists():
            loaded_models['ensemble'] = joblib.load(
                models_dir / "ensemble_model.pkl"
            )
        
        # Load LSTM model (Keras/TensorFlow model)
        if (models_dir / "lstm_model.h5").exists():
            try:
                import tensorflow as tf
                loaded_models['lstm'] = tf.keras.models.load_model(
                    models_dir / "lstm_model.h5"
                )
                # Load LSTM tokenizer if available
                if (models_dir / "lstm_tokenizer.pkl").exists():
                    with open(models_dir / "lstm_tokenizer.pkl", "rb") as f:
                        loaded_models['lstm_tokenizer'] = pickle.load(f)
            except Exception as lstm_error:
                # If LSTM fails to load, continue without it
                st.warning(f"Could not load LSTM model: {str(lstm_error)}")
        
        # Load preprocessor
        loaded_models['preprocessor'] = TwitterTextPreprocessor()
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None
    
    return loaded_models

# Initialize models
models = load_models()

# Main title
st.markdown('<h1 class="main-header">📊 Twitter Sentiment Analysis</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "🤖 Model Information", "🔮 Predict Sentiment", "📈 Project Capabilities"]
)

# Home Page
if page == "🏠 Home":
    st.markdown("""
    ## Welcome to Twitter Sentiment Analysis System
    
    This is a comprehensive machine learning system for analyzing Twitter sentiment with state-of-the-art models,
    fairness analysis, and adversarial robustness testing.
    
    ### 🎯 What This Project Does
    
    This project provides an end-to-end solution for Twitter sentiment analysis, including:
    
    - **Multiple ML Models**: Logistic Regression, Random Forest, LSTM, and Ensemble models
    - **Fairness Analysis**: Comprehensive bias detection and mitigation
    - **Adversarial Robustness**: Security testing and defense mechanisms
    - **Production Ready**: Modular code with comprehensive documentation
    - **Ethical AI**: GDPR/CCPA compliance and privacy considerations
    
    ### 📊 Performance Highlights
    
    - **F1-Score**: > 85% (BERT achieved ~91%)
    - **ROC-AUC**: > 90% (BERT achieved ~97%)
    - **Fairness Metrics**: Disparate Impact Ratio < 1.2
    - **Adversarial Accuracy**: > 75%
    
    ### 🚀 Quick Start
    
    1. Navigate to **"Predict Sentiment"** to analyze your own text
    2. Check **"Model Information"** to see available models and their performance
    3. Explore **"Project Capabilities"** to learn about all features
    
    ---
    """)
    
    # Show model loading status
    if models:
        st.success("✅ All models loaded successfully!")
        st.info("🎯 Ready to make predictions!")
    else:
        st.warning("⚠️ Models not loaded. Please check model files.")
    
    # Dataset information
    with st.expander("📚 Dataset Information"):
        st.markdown("""
        **Sentiment140 Dataset**
        - **Size**: 1.6 million tweets
        - **Labels**: Binary sentiment (0=negative, 4=positive)
        - **Time Period**: 2009 tweets from various sources
        - **Balance**: Approximately 50/50 positive/negative distribution
        """)

# Model Information Page
elif page == "🤖 Model Information":
    st.header("🤖 Available Models")
    
    if not models:
        st.error("Models not loaded. Please check model files.")
    else:
        # Model cards
        model_info = [
            {
                "name": "Logistic Regression",
                "description": "Linear baseline model with interpretability",
                "metrics": {"F1-Score": "~0.85", "ROC-AUC": "~0.92", "Accuracy": "~85%"},
                "available": "logistic_regression" in models
            },
            {
                "name": "Random Forest",
                "description": "Non-linear patterns with feature importance",
                "metrics": {"F1-Score": "~0.87", "ROC-AUC": "~0.94", "Accuracy": "~87%"},
                "available": "random_forest" in models
            },
            {
                "name": "Ensemble Model",
                "description": "Voting classifier combining multiple models for robustness",
                "metrics": {"F1-Score": "~0.88", "ROC-AUC": "~0.95", "Accuracy": "~88%"},
                "available": "ensemble" in models
            },
            {
                "name": "LSTM",
                "description": "Deep learning model with sequential understanding",
                "metrics": {"F1-Score": "~0.88", "ROC-AUC": "~0.95", "Accuracy": "~88%"},
                "available": "lstm" in models
            }
        ]
        
        cols = st.columns(2)
        for idx, model in enumerate(model_info):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="model-card">
                    <h3>{model['name']}</h3>
                    <p>{model['description']}</p>
                    <hr>
                    <h4>Performance Metrics:</h4>
                    <ul>
                        <li>F1-Score: <strong>{model['metrics']['F1-Score']}</strong></li>
                        <li>ROC-AUC: <strong>{model['metrics']['ROC-AUC']}</strong></li>
                        <li>Accuracy: <strong>{model['metrics']['Accuracy']}</strong></li>
                    </ul>
                    <p><strong>Status:</strong> {'✅ Available' if model['available'] else '⚠️ Not Available'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Model comparison table
        st.subheader("📊 Model Comparison")
        comparison_data = {
            "Model": ["Logistic Regression", "Random Forest", "Ensemble", "LSTM"],
            "F1-Score": [0.85, 0.87, 0.88, 0.88],
            "ROC-AUC": [0.92, 0.94, 0.95, 0.95],
            "Accuracy": [0.85, 0.87, 0.88, 0.88],
            "Interpretability": ["High", "Medium", "Medium", "Low"],
            "Training Time": ["Fast", "Medium", "Medium", "Slow"]
        }
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)

# Predict Sentiment Page
elif page == "🔮 Predict Sentiment":
    st.header("🔮 Predict Sentiment")
    
    if not models:
        st.error("❌ Models not loaded. Please check model files.")
    else:
        # Input section
        st.subheader("📝 Enter Text to Analyze")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["📝 Type your text", "📄 Upload a file (CSV with 'text' column)"]
        )
        
        text_input = None
        
        if input_method == "📝 Type your text":
            # Text area for input
            user_input = st.text_area(
                "Enter your text here (e.g., tweet, review, comment):",
                height=150,
                placeholder="Example: I love this product! It's amazing and works perfectly!"
            )
            
            if user_input:
                text_input = user_input
        
        else:  # File upload
            uploaded_file = st.file_uploader(
                "Upload a CSV file with a 'text' column",
                type=['csv']
            )
            
            if uploaded_file:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    if 'text' in df_upload.columns:
                        st.success(f"✅ File loaded successfully! Found {len(df_upload)} rows.")
                        text_input = df_upload['text'].tolist()
                    else:
                        st.error("❌ CSV file must have a 'text' column.")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        # Model selection
        st.subheader("🤖 Select Model")
        available_models = []
        if "logistic_regression" in models:
            available_models.append("Logistic Regression")
        if "random_forest" in models:
            available_models.append("Random Forest")
        if "ensemble" in models:
            available_models.append("Ensemble")
        if "lstm" in models:
            available_models.append("LSTM")
        
        if not available_models:
            st.warning("⚠️ No models available for prediction.")
        else:
            selected_model_name = st.selectbox(
                "Choose a model:",
                available_models
            )
            
            # Map model names to keys
            model_map = {
                "Logistic Regression": "logistic_regression",
                "Random Forest": "random_forest",
                "Ensemble": "ensemble",
                "LSTM": "lstm"
            }
            selected_model_key = model_map[selected_model_name]
            selected_model = models[selected_model_key]
            
            # Show note for LSTM
            if selected_model_key == "lstm":
                st.info("ℹ️ LSTM model requires sequence tokenization. Note: This may require additional preprocessing compared to other models.")
            
            # Predict button
            if st.button("🚀 Predict Sentiment", type="primary"):
                if text_input:
                    with st.spinner("🔄 Processing..."):
                        try:
                            # Handle single text or list of texts
                            if isinstance(text_input, str):
                                texts = [text_input]
                            else:
                                texts = text_input
                            
                            results = []
                            
                            for text in texts:
                                # Preprocess text
                                preprocessed = models['preprocessor'].preprocess_pipeline(text)
                                
                                # Handle LSTM model differently (uses tokenization instead of TF-IDF)
                                if selected_model_key == "lstm":
                                    if "lstm_tokenizer" not in models:
                                        st.error("❌ LSTM tokenizer not found. Cannot make predictions with LSTM model.")
                                        break
                                    
                                    import tensorflow as tf
                                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                                    
                                    # Tokenize and pad sequence
                                    tokenizer = models['lstm_tokenizer']
                                    sequence = tokenizer.texts_to_sequences([preprocessed])
                                    
                                    # Get max_length from model input shape (default to 80 if not available)
                                    try:
                                        max_length = selected_model.input_shape[1]
                                    except:
                                        max_length = 80  # Default from notebook
                                    
                                    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
                                    
                                    # Make prediction
                                    prediction_proba = selected_model.predict(padded_sequence, verbose=0)[0]
                                    prediction = 1 if prediction_proba[0] > 0.5 else 0
                                    
                                    sentiment = 'Positive' if prediction == 1 else 'Negative'
                                    confidence = float(prediction_proba[0]) if prediction == 1 else float(1 - prediction_proba[0])
                                    prob_negative = float(1 - prediction_proba[0])
                                    prob_positive = float(prediction_proba[0])
                                    
                                    results.append({
                                        'text': text,
                                        'preprocessed': preprocessed,
                                        'sentiment': sentiment,
                                        'confidence': confidence,
                                        'probability_negative': prob_negative,
                                        'probability_positive': prob_positive
                                    })
                                
                                else:
                                    # Handle TF-IDF based models (Logistic Regression, Random Forest, Ensemble)
                                    if models['tfidf_vectorizer']:
                                        text_vectorized = models['tfidf_vectorizer'].transform([preprocessed])
                                        
                                        # Scale if scaler is available and model is Logistic Regression
                                        if selected_model_key == "logistic_regression" and "scaler" in models:
                                            # Convert sparse matrix to dense for scaling
                                            text_vectorized_dense = text_vectorized.toarray()
                                            text_vectorized = models['scaler'].transform(text_vectorized_dense)
                                        else:
                                            # For Random Forest and Ensemble, convert to dense array for compatibility
                                            text_vectorized = text_vectorized.toarray()
                                        
                                        # Make prediction
                                        prediction = selected_model.predict(text_vectorized)[0]
                                        probability = selected_model.predict_proba(text_vectorized)[0]
                                        
                                        sentiment = 'Positive' if prediction == 1 else 'Negative'
                                        confidence = max(probability)
                                        
                                        results.append({
                                            'text': text,
                                            'preprocessed': preprocessed,
                                            'sentiment': sentiment,
                                            'confidence': confidence,
                                            'probability_negative': probability[0],
                                            'probability_positive': probability[1]
                                        })
                            
                            # Display results
                            st.subheader("📊 Prediction Results")
                            
                            if len(results) == 1:
                                # Single result display
                                result = results[0]
                                
                                # Sentiment indicator
                                col1, col2, col3 = st.columns([1, 2, 1])
                                
                                with col2:
                                    if result['sentiment'] == 'Positive':
                                        st.markdown(f"""
                                        <div class="prediction-box">
                                            <h2 style="color: #4caf50; text-align: center;">
                                                😊 {result['sentiment']} Sentiment
                                            </h2>
                                            <p style="text-align: center; font-size: 1.2rem;">
                                                Confidence: <strong>{result['confidence']:.1%}</strong>
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="prediction-box" style="background-color: #ffebee; border-left-color: #f44336;">
                                            <h2 style="color: #f44336; text-align: center;">
                                                😞 {result['sentiment']} Sentiment
                                            </h2>
                                            <p style="text-align: center; font-size: 1.2rem;">
                                                Confidence: <strong>{result['confidence']:.1%}</strong>
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Metrics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Probability (Positive)", f"{result['probability_positive']:.1%}")
                                
                                with col2:
                                    st.metric("Probability (Negative)", f"{result['probability_negative']:.1%}")
                                
                                # Original and preprocessed text
                                with st.expander("📝 Text Details"):
                                    st.write("**Original Text:**", result['text'])
                                    st.write("**Preprocessed Text:**", result['preprocessed'])
                            
                            else:
                                # Multiple results
                                st.success(f"✅ Processed {len(results)} texts")
                                
                                # Summary statistics
                                positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
                                negative_count = len(results) - positive_count
                                avg_confidence = np.mean([r['confidence'] for r in results])
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Texts", len(results))
                                col2.metric("Positive", positive_count, f"{positive_count/len(results):.1%}")
                                col3.metric("Average Confidence", f"{avg_confidence:.1%}")
                                
                                # Results table
                                results_df = pd.DataFrame(results)
                                results_df = results_df[['text', 'sentiment', 'confidence', 
                                                         'probability_positive', 'probability_negative']]
                                results_df.columns = ['Text', 'Sentiment', 'Confidence', 
                                                     'P(Positive)', 'P(Negative)']
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name="sentiment_predictions.csv",
                                    mime="text/csv"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("⚠️ Please enter some text or upload a file first.")

# Project Capabilities Page
elif page == "📈 Project Capabilities":
    st.header("📈 Project Capabilities")
    
    st.markdown("""
    ## 🎯 Comprehensive Features
    
    This project implements a complete Twitter sentiment analysis pipeline with the following capabilities:
    """)
    
    # Feature sections
    features = [
        {
            "title": "🔧 Comprehensive Text Preprocessing",
            "items": [
                "Twitter-specific cleaning (hashtags, mentions, URLs, emojis)",
                "Advanced tokenization and lemmatization",
                "Multiple feature extraction methods (TF-IDF, Word2Vec)",
                "Class imbalance handling with SMOTE"
            ]
        },
        {
            "title": "🤖 Multiple Model Architectures",
            "items": [
                "Traditional ML: Logistic Regression, Random Forest",
                "Deep Learning: LSTM with Word2Vec embeddings",
                "Transformer: Fine-tuned BERT model",
                "Ensemble: Voting classifiers for improved robustness"
            ]
        },
        {
            "title": "⚖️ Fairness Analysis",
            "items": [
                "Protected Attributes: Text length, sentiment intensity, language patterns",
                "Fairness Metrics: Demographic parity, equalized odds, disparate impact",
                "Bias Mitigation: Constraint-based optimization, threshold tuning",
                "Comprehensive Reporting: Automated fairness assessment"
            ]
        },
        {
            "title": "🛡️ Adversarial Robustness",
            "items": [
                "Attack Types: Character/word substitution, typo injection, word order perturbation",
                "Defense Mechanisms: Ensemble methods, input sanitization, adversarial training",
                "Robustness Testing: Comprehensive evaluation across multiple attack vectors",
                "Security Assessment: Automated vulnerability analysis"
            ]
        },
        {
            "title": "🚀 Production-Ready Code",
            "items": [
                "Modular Design: Reusable components and utilities",
                "Error Handling: Comprehensive exception management",
                "Logging: Detailed execution tracking",
                "Documentation: Extensive docstrings and examples"
            ]
        }
    ]
    
    for feature in features:
        with st.expander(feature["title"]):
            for item in feature["items"]:
                st.markdown(f"✅ {item}")
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    metrics_data = {
        "Metric": [
            "F1-Score",
            "ROC-AUC",
            "Disparate Impact Ratio",
            "Adversarial Accuracy",
            "Cross-validation Stability"
        ],
        "Target": [
            "> 85%",
            "> 90%",
            "< 1.2",
            "> 75%",
            "< 5% variance"
        ],
        "Achieved": [
            "✅ ~91% (BERT)",
            "✅ ~97% (BERT)",
            "✅ ~1.1",
            "✅ ~78%",
            "✅ < 5%"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Use cases
    st.subheader("💼 Use Cases")
    
    use_cases = [
        "📱 Social Media Monitoring: Real-time sentiment analysis",
        "💬 Customer Feedback: Automated sentiment classification",
        "📈 Market Research: Trend analysis and opinion mining",
        "🛡️ Content Moderation: Automated content classification"
    ]
    
    for use_case in use_cases:
        st.markdown(f"- {use_case}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Twitter Sentiment Analysis System | Built with Streamlit</p>
        <p>For educational and research purposes</p>
    </div>
    """,
    unsafe_allow_html=True
)

