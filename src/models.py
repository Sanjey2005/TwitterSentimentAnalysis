"""
Twitter Sentiment Analysis - Models Module

This module contains model training, evaluation, and comparison functions
for Twitter sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Comprehensive model evaluation"""
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return results, model
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, ax):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, ax):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
    
    def compare_models(self, results_list, y_test, save_path=None):
        """Compare multiple models"""
        
        # Create comparison dataframe
        comparison_data = []
        for result in results_list:
            comparison_data.append({
                'Model': result['model_name'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'],
                'CV F1-Score': result['cv_mean']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion matrices
        for i, result in enumerate(results_list):
            row, col = i // 2, i % 2
            if row < 2 and col < 2:
                self.plot_confusion_matrix(y_test, result['y_pred'], result['model_name'], axes[row, col])
        
        # ROC curves
        for result in results_list:
            self.plot_roc_curve(y_test, result['y_pred_proba'], result['model_name'], axes[1, 0])
        
        # Performance metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(metrics))
        width = 0.8 / len(results_list)
        
        for i, result in enumerate(results_list):
            scores = [result[metric.lower().replace('-', '_')] for metric in metrics]
            axes[1, 1].bar(x + i * width, scores, width, label=result['model_name'], alpha=0.8)
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_xticks(x + width * (len(results_list) - 1) / 2)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return comparison_df


class BaselineModels:
    """
    Baseline machine learning models for sentiment analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def train_logistic_regression(self, X_train, X_test, y_train, y_test, 
                                hyperparameter_tuning=True, save_path=None):
        """Train Logistic Regression model"""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                lr_model, param_grid, cv=3, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            lr_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Train and evaluate
        evaluator = ModelEvaluator()
        results, trained_model = evaluator.evaluate_model(
            lr_model, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression"
        )
        
        # Save model
        if save_path:
            joblib.dump(trained_model, f"{save_path}/logistic_regression_model.pkl")
            joblib.dump(self.scaler, f"{save_path}/scaler.pkl")
        
        self.models['logistic_regression'] = trained_model
        
        return results, trained_model
    
    def train_random_forest(self, X_train, X_test, y_train, y_test,
                          hyperparameter_tuning=True, save_path=None):
        """Train Random Forest model"""
        
        # Initialize model
        rf_model = RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1
        )
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                rf_model, param_grid, cv=3, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            rf_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Train and evaluate
        evaluator = ModelEvaluator()
        results, trained_model = evaluator.evaluate_model(
            rf_model, X_train, X_test, y_train, y_test, "Random Forest"
        )
        
        # Save model
        if save_path:
            joblib.dump(trained_model, f"{save_path}/random_forest_model.pkl")
        
        self.models['random_forest'] = trained_model
        
        return results, trained_model
    
    def create_ensemble(self, X_train, X_test, y_train, y_test, save_path=None):
        """Create ensemble model"""
        
        if 'logistic_regression' not in self.models or 'random_forest' not in self.models:
            raise ValueError("Both Logistic Regression and Random Forest models must be trained first")
        
        # Create ensemble
        ensemble_model = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft'
        )
        
        # Train ensemble
        ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        evaluator = ModelEvaluator()
        results, trained_ensemble = evaluator.evaluate_model(
            ensemble_model, X_train, X_test, y_train, y_test, "Ensemble"
        )
        
        # Save model
        if save_path:
            joblib.dump(trained_ensemble, f"{save_path}/ensemble_model.pkl")
        
        self.models['ensemble'] = trained_ensemble
        
        return results, trained_ensemble


class DeepLearningModels:
    """
    Deep learning models for sentiment analysis
    """
    
    def __init__(self):
        self.models = {}
        
    def train_lstm(self, X_train, X_test, y_train, y_test, 
                   embedding_matrix, vocab_size, max_length, save_path=None):
        """Train LSTM model"""
        
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        
        # Build LSTM model
        model = Sequential([
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_matrix.shape[1],
                input_length=max_length,
                weights=[embedding_matrix],
                trainable=False
            ),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ]
        
        if save_path:
            callbacks.append(ModelCheckpoint(f"{save_path}/lstm_model.h5", 
                                           monitor='val_loss', save_best_only=True))
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'model_name': 'LSTM',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba.flatten(),
            'history': history
        }
        
        self.models['lstm'] = model
        
        return results, model
    
    def train_bert(self, train_dataloader, test_dataloader, y_test, 
                   epochs=3, save_path=None):
        """Train BERT model"""
        
        import torch
        from transformers import BertForSequenceClassification, AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # Load BERT model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Set up training
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_train_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_masks, labels = batch
                
                model.zero_grad()
                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_masks, labels = batch
                
                outputs = model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        results = {
            'model_name': 'BERT',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': np.array(predictions),
            'y_pred_proba': None  # Would need additional computation
        }
        
        # Save model
        if save_path:
            model.save_pretrained(f"{save_path}/bert_model")
        
        self.models['bert'] = model
        
        return results, model


def load_model(model_path, model_type='sklearn'):
    """
    Load a trained model
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model ('sklearn', 'keras', 'pytorch')
    
    Returns:
        Loaded model
    """
    if model_type == 'sklearn':
        return joblib.load(model_path)
    elif model_type == 'keras':
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    elif model_type == 'pytorch':
        import torch
        return torch.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def predict_sentiment(text, model, vectorizer, preprocessor=None):
    """
    Predict sentiment for a single text
    
    Args:
        text (str): Input text
        model: Trained model
        vectorizer: Fitted vectorizer
        preprocessor: Text preprocessor (optional)
    
    Returns:
        dict: Prediction results
    """
    if preprocessor:
        text = preprocessor.preprocess_pipeline(text)
    
    # Transform text
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    confidence = max(probability)
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probability_negative': probability[0],
        'probability_positive': probability[1]
    }


if __name__ == "__main__":
    print("Twitter Sentiment Analysis - Models Module")
    print("This module provides model training, evaluation, and comparison functions.")
    print("Use the classes to train and evaluate sentiment analysis models.")
