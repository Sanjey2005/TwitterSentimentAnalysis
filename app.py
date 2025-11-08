"""
Twitter Sentiment Analysis - Streamlit Web Application
A simple and informative interface for Twitter sentiment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Import project modules
from src.preprocessing import TwitterTextPreprocessor

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1DA1F2;
        margin: 1rem 0;
        color: #1f1f1f !important;
    }
    .info-box h3 {
        color: #1DA1F2 !important;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .info-box p {
        color: #1f1f1f !important;
        margin: 0.5rem 0;
    }
    .info-box strong {
        color: #1DA1F2 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
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
        
        # Load ML models
        model_files = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'naive_bayes': 'naive_bayes_model.pkl',
            'mlp': 'mlp_model.pkl'
        }
        
        for key, filename in model_files.items():
            if (models_dir / filename).exists():
                loaded_models[key] = joblib.load(models_dir / filename)
        
        # Load BERT model
        bert_model_path = models_dir / "bert_model"
        if bert_model_path.exists() and (bert_model_path / "config.json").exists():
            try:
                from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
                loaded_models['bert_tokenizer'] = DistilBertTokenizer.from_pretrained(str(bert_model_path))
                loaded_models['bert'] = DistilBertForSequenceClassification.from_pretrained(str(bert_model_path))
            except Exception as e:
                st.warning(f"Could not load BERT: {str(e)}")
        
        # Load preprocessor
        loaded_models['preprocessor'] = TwitterTextPreprocessor()
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None
    
    return loaded_models

# Prediction functions
def predict_with_bert(text, tokenizer, model):
    """Predict sentiment using BERT model"""
    model.eval()
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    model = model.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    prediction = torch.argmax(logits, dim=1).item()
    probability = probs[0].cpu().numpy()
    return prediction, probability

def predict_with_ml(text, model, vectorizer, scaler=None):
    """Predict sentiment using ML models (LR, NB, MLP)"""
    preprocessed = models['preprocessor'].preprocess_pipeline(text)
    text_vector = vectorizer.transform([preprocessed])
    
    if scaler and model.__class__.__name__ in ['LogisticRegression', 'MLPClassifier']:
        text_vector = scaler.transform(text_vector.toarray())
    else:
        text_vector = text_vector.toarray()
    
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    return prediction, probability, preprocessed

# Initialize models
models = load_models()

# Main title
st.markdown('<h1 class="main-header">📊 Twitter Sentiment Analysis</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "🔮 Predict", "🤖 Models", "🔍 Explain"]
)

# Home Page
if page == "🏠 Home":
    st.markdown("""
    ## Welcome! 👋
    
    A comprehensive **Twitter Sentiment Analysis** system powered by machine learning.
    Analyze text sentiment with multiple models and understand how predictions are made.
    """)
    
    # Project features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 4 ML Models</h3>
            <p>Logistic Regression, Naive Bayes, MLP, and BERT (DistilBERT)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🔍 Explainable AI</h3>
            <p>SHAP-based feature importance and prediction explanations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 High Accuracy</h3>
            <p>BERT achieves >85% F1-score and >90% ROC-AUC</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick info
    st.subheader("📋 What This Project Offers")
    
    features = [
        "**Text Preprocessing**: Twitter-specific cleaning (hashtags, mentions, URLs, emojis)",
        "**Multiple Models**: Choose from 4 different ML models for prediction",
        "**Model Explainability**: Understand predictions with SHAP values and feature importance",
        "**Production Ready**: Clean, modular code with comprehensive documentation",
        "**Batch Processing**: Analyze single texts or upload CSV files",
        "**Visual Analytics**: Interactive charts and visualizations"
    ]
    
    for feature in features:
        st.markdown(f"✅ {feature}")
    
    # Model status
    st.markdown("---")
    st.subheader("📦 Model Status")
    
    if models:
        available_models = []
        if 'logistic_regression' in models:
            available_models.append("✅ Logistic Regression")
        if 'naive_bayes' in models:
            available_models.append("✅ Naive Bayes")
        if 'mlp' in models:
            available_models.append("✅ MLP")
        if 'bert' in models:
            available_models.append("✅ BERT (DistilBERT)")
        
        if available_models:
            for model in available_models:
                st.markdown(f"- {model}")
        else:
            st.warning("⚠️ No models loaded. Please train models first.")
    else:
        st.error("❌ Models not loaded. Please check model files.")
    
    # Dataset info
    with st.expander("📚 About the Dataset"):
        st.markdown("""
        **Sentiment140 Dataset**
        - **Size**: 1.6 million tweets
        - **Labels**: Binary sentiment (0=negative, 4=positive)
        - **Time Period**: 2009 tweets
        - **Balance**: ~50/50 positive/negative distribution
        """)

# Predict Sentiment Page
elif page == "🔮 Predict":
    st.header("🔮 Predict Sentiment")
    
    if not models:
        st.error("❌ Models not loaded. Please check model files.")
    else:
        # Input section
        input_method = st.radio(
            "Input method:",
            ["📝 Type text", "📄 Upload CSV"]
        )
        
        text_input = None
        
        if input_method == "📝 Type text":
            user_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Example: I love this product! It's amazing!"
            )
            if user_input:
                text_input = user_input
        
        else:
            uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        st.success(f"✅ Loaded {len(df)} rows")
                        text_input = df['text'].tolist()
                    else:
                        st.error("❌ CSV must have 'text' column")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Model selection
        available_models = []
        model_map = {}
        if "logistic_regression" in models:
            available_models.append("Logistic Regression")
            model_map["Logistic Regression"] = "logistic_regression"
        if "naive_bayes" in models:
            available_models.append("Naive Bayes")
            model_map["Naive Bayes"] = "naive_bayes"
        if "mlp" in models:
            available_models.append("MLP")
            model_map["MLP"] = "mlp"
        if "bert" in models:
            available_models.append("BERT")
            model_map["BERT"] = "bert"
        
        if not available_models:
            st.warning("⚠️ No models available")
        else:
            selected_model_name = st.selectbox("Select model:", available_models)
            selected_model_key = model_map[selected_model_name]
            
            if selected_model_key == "bert":
                st.info("ℹ️ BERT uses transformer architecture. May be slower but more accurate.")
            
            # Predict button
            if st.button("🚀 Predict", type="primary"):
                if text_input:
                    with st.spinner("Processing..."):
                        try:
                            texts = [text_input] if isinstance(text_input, str) else text_input
                            results = []
                            
                            for text in texts:
                                if selected_model_key == "bert":
                                    prediction, probability = predict_with_bert(
                                        models['preprocessor'].preprocess_pipeline(text),
                                        models['bert_tokenizer'],
                                        models['bert']
                                    )
                                    preprocessed = models['preprocessor'].preprocess_pipeline(text)
                                else:
                                    prediction, probability, preprocessed = predict_with_ml(
                                        text,
                                        models[selected_model_key],
                                        models['tfidf_vectorizer'],
                                        models.get('scaler')
                                    )
                                
                                sentiment = 'Positive' if prediction == 1 else 'Negative'
                                confidence = max(probability)
                                
                                results.append({
                                    'text': text,
                                    'sentiment': sentiment,
                                    'confidence': confidence,
                                    'prob_positive': probability[1],
                                    'prob_negative': probability[0]
                                })
                            
                            # Display results
                            st.subheader("📊 Results")
                            
                            if len(results) == 1:
                                r = results[0]
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Sentiment", r['sentiment'], 
                                            f"{r['confidence']:.1%} confidence")
                                with col2:
                                    st.metric("Positive Prob", f"{r['prob_positive']:.1%}")
                                
                                # Visual
                                fig, ax = plt.subplots(figsize=(8, 2))
                                colors = ['#4caf50' if r['sentiment'] == 'Positive' else '#f44336']
                                ax.barh(['Sentiment'], [r['confidence']], color=colors[0])
                                ax.set_xlim(0, 1)
                                ax.set_xlabel('Confidence')
                                st.pyplot(fig)
                                
                                with st.expander("📝 Text Details"):
                                    st.write("**Original:**", r['text'])
                                    st.write("**Preprocessed:**", preprocessed)
                            
                            else:
                                # Batch results
                                positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
                                avg_conf = np.mean([r['confidence'] for r in results])
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total", len(results))
                                col2.metric("Positive", positive_count, f"{positive_count/len(results):.1%}")
                                col3.metric("Avg Confidence", f"{avg_conf:.1%}")
                                
                                # Results table
                                df_results = pd.DataFrame(results)
                                df_results = df_results[['text', 'sentiment', 'confidence']]
                                st.dataframe(df_results, use_container_width=True)
                                
                                # Download
                                csv = df_results.to_csv(index=False)
                                st.download_button(
                                    "📥 Download CSV",
                                    csv,
                                    "predictions.csv",
                                    "text/csv"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter text or upload file")

# Model Information Page
elif page == "🤖 Models":
    st.header("🤖 Available Models")
    
    if not models:
        st.error("❌ Models not loaded")
    else:
        # Model info
        model_info = [
            {
                "name": "Logistic Regression",
                "desc": "Linear baseline model, fast and interpretable",
                "metrics": {"F1": "~68%", "ROC-AUC": "~72%", "Speed": "Very Fast"},
                "available": "logistic_regression" in models
            },
            {
                "name": "Naive Bayes",
                "desc": "Probabilistic classifier, efficient for text",
                "metrics": {"F1": "~70%", "ROC-AUC": "~75%", "Speed": "Very Fast"},
                "available": "naive_bayes" in models
            },
            {
                "name": "MLP",
                "desc": "Neural network with hidden layers",
                "metrics": {"F1": "~72%", "ROC-AUC": "~78%", "Speed": "Medium"},
                "available": "mlp" in models
            },
            {
                "name": "BERT (DistilBERT)",
                "desc": "Lightweight transformer, state-of-the-art",
                "metrics": {"F1": "~85%", "ROC-AUC": "~90%", "Speed": "Slow"},
                "available": "bert" in models
            }
        ]
        
        # Display models in a 2x2 grid using Streamlit native components
        cols = st.columns(2)
        for idx, model in enumerate(model_info):
            with cols[idx % 2]:
                status = "✅ Available" if model['available'] else "⚠️ Not Available"
                status_icon = "🟢" if model['available'] else "🔴"
                
                # Create a card-like container
                st.markdown("---")
                st.markdown(f"### {model['name']}")
                st.markdown(f"{status_icon} **Status:** {status}")
                st.markdown(f"*{model['desc']}*")
                st.markdown("")
                
                # Metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("F1-Score", model['metrics']['F1'])
                with metric_col2:
                    st.metric("ROC-AUC", model['metrics']['ROC-AUC'])
                
                st.metric("Speed", model['metrics']['Speed'])
                st.markdown("")
        
        # Comparison table
        st.subheader("📊 Model Comparison")
        comparison = pd.DataFrame({
            "Model": ["Logistic Regression", "Naive Bayes", "MLP", "BERT"],
            "F1-Score": ["~68%", "~70%", "~72%", "~85%"],
            "ROC-AUC": ["~72%", "~75%", "~78%", "~90%"],
            "Interpretability": ["High", "High", "Medium", "Low"],
            "Speed": ["Fast", "Very Fast", "Medium", "Slow"]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

# Explainability Page
elif page == "🔍 Explain":
    st.header("🔍 Model Explainability")
    
    st.markdown("""
    Understand how models make predictions using SHAP (SHapley Additive exPlanations) values.
    """)
    
    if not models:
        st.error("❌ Models not loaded")
    else:
        explain_type = st.radio(
            "Analysis type:",
            ["🔍 Explain Prediction", "📈 Feature Importance"]
        )
        
        if explain_type == "🔍 Explain Prediction":
            explain_text = st.text_area(
                "Enter text to explain:",
                height=100,
                placeholder="Example: I love this product!"
            )
            
            # Get available models for explanation (MLP excluded as it requires SHAP)
            explain_models = []
            explain_model_map = {}
            if "logistic_regression" in models:
                explain_models.append("Logistic Regression")
                explain_model_map["Logistic Regression"] = "logistic_regression"
            if "naive_bayes" in models:
                explain_models.append("Naive Bayes")
                explain_model_map["Naive Bayes"] = "naive_bayes"
            
            if not explain_models:
                st.warning("⚠️ No models available for explanation")
            else:
                model_choice = st.selectbox("Select model:", explain_models)
                
                if st.button("🔍 Explain", type="primary"):
                    if explain_text:
                        try:
                            model_key = explain_model_map.get(model_choice)
                            if not model_key or model_key not in models:
                                st.error("Model not available")
                            else:
                                # Preprocess and predict
                                preprocessed = models['preprocessor'].preprocess_pipeline(explain_text)
                                text_vector = models['tfidf_vectorizer'].transform([preprocessed])
                                
                                if model_key in ["logistic_regression", "mlp"] and "scaler" in models:
                                    text_vector = models['scaler'].transform(text_vector.toarray())
                                else:
                                    text_vector = text_vector.toarray()
                                
                                model = models[model_key]
                                prediction = model.predict(text_vector)[0]
                                probability = model.predict_proba(text_vector)[0]
                                
                                sentiment = 'Positive' if prediction == 1 else 'Negative'
                                
                                st.subheader("📊 Prediction")
                                col1, col2 = st.columns(2)
                                col1.metric("Sentiment", sentiment)
                                col2.metric("Confidence", f"{max(probability):.1%}")
                                
                                # Feature contributions
                                st.subheader("🔍 Top Contributing Features")
                                
                                if model_choice == "Logistic Regression":
                                    coefficients = model.coef_[0]
                                    feature_names = models['tfidf_vectorizer'].get_feature_names_out()
                                    feature_values = text_vector[0]
                                    contributions = coefficients * feature_values
                                    top_indices = np.argsort(np.abs(contributions))[-15:][::-1]
                                    
                                    contrib_df = pd.DataFrame({
                                        'Feature': [feature_names[i] for i in top_indices],
                                        'Contribution': [contributions[i] for i in top_indices]
                                    })
                                    st.dataframe(contrib_df, use_container_width=True)
                                    
                                    # Visualize
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    colors = ['green' if x > 0 else 'red' for x in contrib_df['Contribution']]
                                    ax.barh(range(len(contrib_df)), contrib_df['Contribution'], color=colors)
                                    ax.set_yticks(range(len(contrib_df)))
                                    ax.set_yticklabels(contrib_df['Feature'])
                                    ax.set_xlabel('Contribution')
                                    ax.set_title('Feature Contributions')
                                    ax.invert_yaxis()
                                    st.pyplot(fig)
                                
                                elif model_choice == "Naive Bayes":
                                    feature_names = models['tfidf_vectorizer'].get_feature_names_out()
                                    feature_log_probs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                                    feature_values = text_vector[0]
                                    contributions = feature_log_probs * feature_values
                                    top_indices = np.argsort(np.abs(contributions))[-15:][::-1]
                                    
                                    contrib_df = pd.DataFrame({
                                        'Feature': [feature_names[i] for i in top_indices],
                                        'Contribution': [contributions[i] for i in top_indices]
                                    })
                                    st.dataframe(contrib_df, use_container_width=True)
                                    
                                    # Visualize
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    colors = ['green' if x > 0 else 'red' for x in contrib_df['Contribution']]
                                    ax.barh(range(len(contrib_df)), contrib_df['Contribution'], color=colors)
                                    ax.set_yticks(range(len(contrib_df)))
                                    ax.set_yticklabels(contrib_df['Feature'])
                                    ax.set_xlabel('Log Probability Contribution')
                                    ax.set_title('Feature Contributions (Naive Bayes)')
                                    ax.invert_yaxis()
                                    st.pyplot(fig)
                                
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter text")
        
        else:
            st.subheader("📈 Global Feature Importance")
            st.info("""
            Feature importance shows which words/terms are most important for model predictions.
            """)
            
            # Check for SHAP plots
            shap_plots = {
                "Logistic Regression": "reports/figures/shap_summary_lr.png",
                "Naive Bayes": "reports/figures/shap_summary_nb.png"
            }
            
            selected_plot = st.selectbox("Select model:", list(shap_plots.keys()))
            plot_path = Path(shap_plots[selected_plot])
            
            if plot_path.exists():
                st.image(str(plot_path), caption=f"SHAP Summary - {selected_plot}")
            else:
                st.warning("⚠️ Plot not found. Run the explainability notebook to generate plots.")
                st.info("💡 Run `notebooks/05_explainability.ipynb` to generate visualizations.")
            
            # About SHAP
            with st.expander("ℹ️ About SHAP Values"):
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)** explains model predictions:
                - Shows contribution of each feature to a prediction
                - Positive values push toward positive sentiment
                - Negative values push toward negative sentiment
                - Magnitude indicates strength of contribution
                """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Twitter Sentiment Analysis System | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
