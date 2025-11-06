"""
Twitter Sentiment Analysis - Fairness Analysis Module

This module provides comprehensive fairness analysis and bias detection
for Twitter sentiment analysis models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# Fairness analysis libraries
from fairlearn.metrics import (
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio,
    selection_rate, false_positive_rate, false_negative_rate
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity, EqualizedOdds

# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Text analysis
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')


class ProtectedAttributeExtractor:
    """
    Extract protected attributes from text data
    """
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def extract_text_length_groups(self, texts, bins=[0, 50, 100, 200, float('inf')], 
                                 labels=['short', 'medium', 'long', 'very_long']):
        """Extract text length groups"""
        text_lengths = [len(text) for text in texts]
        return pd.cut(text_lengths, bins=bins, labels=labels)
    
    def extract_sentiment_intensity_groups(self, texts, bins=[0, 0.1, 0.3, 0.6, 1.0],
                                        labels=['low', 'medium', 'high', 'very_high']):
        """Extract sentiment intensity groups"""
        intensities = [abs(self.sia.polarity_scores(text)['compound']) for text in texts]
        return pd.cut(intensities, bins=bins, labels=labels)
    
    def extract_language_patterns(self, texts):
        """Extract language pattern groups"""
        patterns = []
        for text in texts:
            # Count non-ASCII characters
            non_ascii_count = len([c for c in text if ord(c) > 127])
            total_chars = len(text)
            
            if total_chars == 0:
                patterns.append('unknown')
                continue
            
            non_ascii_ratio = non_ascii_count / total_chars
            
            if non_ascii_ratio > 0.3:
                patterns.append('non_english')
            elif any(word in text.lower() for word in ['lol', 'omg', 'wtf']):
                patterns.append('casual')
            elif any(word in text.lower() for word in ['please', 'thank', 'appreciate']):
                patterns.append('formal')
            else:
                patterns.append('standard')
        
        return patterns
    
    def extract_gender_patterns(self, texts):
        """Extract gender pattern groups (simplified heuristic)"""
        patterns = []
        for text in texts:
            text_lower = text.lower()
            
            # Common patterns (simplified heuristic)
            feminine_patterns = ['she', 'her', 'girl', 'woman', 'lady', 'beautiful', 'cute']
            masculine_patterns = ['he', 'him', 'boy', 'man', 'guy', 'handsome', 'dude']
            
            feminine_count = sum(1 for pattern in feminine_patterns if pattern in text_lower)
            masculine_count = sum(1 for pattern in masculine_patterns if pattern in text_lower)
            
            if feminine_count > masculine_count:
                patterns.append('feminine')
            elif masculine_count > feminine_count:
                patterns.append('masculine')
            else:
                patterns.append('neutral')
        
        return patterns
    
    def extract_all_attributes(self, texts):
        """Extract all protected attributes"""
        return {
            'length_group': self.extract_text_length_groups(texts),
            'intensity_group': self.extract_sentiment_intensity_groups(texts),
            'language_group': self.extract_language_patterns(texts),
            'gender_group': self.extract_gender_patterns(texts)
        }


class FairnessAnalyzer:
    """
    Comprehensive fairness analysis for machine learning models
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_fairness_metrics(self, y_true, y_pred, sensitive_feature, model_name):
        """Calculate comprehensive fairness metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Fairness metrics
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
        eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_feature)
        
        # Selection rates by group
        selection_rates = {}
        for group in np.unique(sensitive_feature):
            group_mask = sensitive_feature == group
            if np.sum(group_mask) > 0:
                selection_rates[group] = selection_rate(y_true[group_mask], y_pred[group_mask])
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'demographic_parity_difference': dp_diff,
            'demographic_parity_ratio': dp_ratio,
            'equalized_odds_difference': eo_diff,
            'equalized_odds_ratio': eo_ratio,
            'selection_rates': selection_rates
        }
    
    def plot_fairness_analysis(self, df, protected_attr, model_pred_col, title, save_path=None):
        """Plot fairness analysis for a protected attribute"""
        
        # Calculate metrics by group
        group_metrics = []
        for group in df[protected_attr].unique():
            group_data = df[df[protected_attr] == group]
            if len(group_data) > 0:
                accuracy = accuracy_score(group_data['true_label'], group_data[model_pred_col])
                precision = precision_score(group_data['true_label'], group_data[model_pred_col])
                recall = recall_score(group_data['true_label'], group_data[model_pred_col])
                f1 = f1_score(group_data['true_label'], group_data[model_pred_col])
                
                group_metrics.append({
                    'group': group,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'count': len(group_data)
                })
        
        group_df = pd.DataFrame(group_metrics)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy by group
        axes[0, 0].bar(group_df['group'], group_df['accuracy'], alpha=0.7)
        axes[0, 0].set_title(f'{title} - Accuracy by {protected_attr}')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision by group
        axes[0, 1].bar(group_df['group'], group_df['precision'], alpha=0.7)
        axes[0, 1].set_title(f'{title} - Precision by {protected_attr}')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall by group
        axes[1, 0].bar(group_df['group'], group_df['recall'], alpha=0.7)
        axes[1, 0].set_title(f'{title} - Recall by {protected_attr}')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score by group
        axes[1, 1].bar(group_df['group'], group_df['f1_score'], alpha=0.7)
        axes[1, 1].set_title(f'{title} - F1-Score by {protected_attr}')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return group_df
    
    def identify_fairness_issues(self, fairness_df, dp_ratio_threshold=0.8, 
                               dp_diff_threshold=0.1, eo_ratio_threshold=0.8, 
                               eo_diff_threshold=0.1):
        """Identify fairness issues in the results"""
        
        issues = []
        
        for _, row in fairness_df.iterrows():
            # Check demographic parity
            if row['DP_Ratio'] < dp_ratio_threshold:
                issues.append(f"{row['Model']} - {row['Attribute']}: Low demographic parity ratio ({row['DP_Ratio']:.3f})")
            
            if row['DP_Difference'] > dp_diff_threshold:
                issues.append(f"{row['Model']} - {row['Attribute']}: High demographic parity difference ({row['DP_Difference']:.3f})")
            
            # Check equalized odds
            if row['EO_Ratio'] < eo_ratio_threshold:
                issues.append(f"{row['Model']} - {row['Attribute']}: Low equalized odds ratio ({row['EO_Ratio']:.3f})")
            
            if row['EO_Difference'] > eo_diff_threshold:
                issues.append(f"{row['Model']} - {row['Attribute']}: High equalized odds difference ({row['EO_Difference']:.3f})")
        
        return issues
    
    def comprehensive_fairness_analysis(self, df, protected_attributes, model_pred_cols, 
                                      model_names, save_path=None):
        """Perform comprehensive fairness analysis"""
        
        fairness_results = []
        
        for attr in protected_attributes:
            for model_name, pred_col in zip(model_names, model_pred_cols):
                # Calculate fairness metrics
                metrics = self.calculate_fairness_metrics(
                    df['true_label'], 
                    df[pred_col], 
                    df[attr], 
                    f"{model_name} - {attr}"
                )
                
                fairness_results.append(metrics)
                
                # Create visualizations
                group_df = self.plot_fairness_analysis(df, attr, pred_col, model_name)
        
        # Create fairness summary
        fairness_summary = []
        
        for result in fairness_results:
            model_attr = result['model_name'].split(' - ')
            model_name = model_attr[0]
            attribute = model_attr[1]
            
            fairness_summary.append({
                'Model': model_name,
                'Attribute': attribute,
                'DP_Difference': result['demographic_parity_difference'],
                'DP_Ratio': result['demographic_parity_ratio'],
                'EO_Difference': result['equalized_odds_difference'],
                'EO_Ratio': result['equalized_odds_ratio'],
                'Accuracy': result['accuracy'],
                'F1_Score': result['f1_score']
            })
        
        fairness_df = pd.DataFrame(fairness_summary)
        
        # Identify issues
        issues = self.identify_fairness_issues(fairness_df)
        
        return fairness_df, issues


class BiasMitigator:
    """
    Bias mitigation using fairlearn techniques
    """
    
    def __init__(self):
        self.mitigated_models = {}
        
    def apply_demographic_parity_constraint(self, model, X_train, y_train, 
                                          sensitive_features, eps=0.01):
        """Apply demographic parity constraint"""
        
        dp_constraint = DemographicParity()
        
        mitigator = ExponentiatedGradient(
            estimator=model,
            constraints=dp_constraint,
            eps=eps
        )
        
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        return mitigator
    
    def apply_equalized_odds_constraint(self, model, X_train, y_train, 
                                      sensitive_features, eps=0.01):
        """Apply equalized odds constraint"""
        
        eo_constraint = EqualizedOdds()
        
        mitigator = ExponentiatedGradient(
            estimator=model,
            constraints=eo_constraint,
            eps=eps
        )
        
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        return mitigator
    
    def apply_threshold_optimization(self, model, X_train, y_train, 
                                   sensitive_features, constraints='demographic_parity'):
        """Apply threshold optimization for bias mitigation"""
        
        if constraints == 'demographic_parity':
            constraint = DemographicParity()
        elif constraints == 'equalized_odds':
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unsupported constraint: {constraints}")
        
        threshold_optimizer = ThresholdOptimizer(
            estimator=model,
            constraints=constraint
        )
        
        threshold_optimizer.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        return threshold_optimizer
    
    def compare_mitigation_techniques(self, model, X_train, X_test, y_train, y_test,
                                    sensitive_features, save_path=None):
        """Compare different bias mitigation techniques"""
        
        # Original model performance
        original_pred = model.predict(X_test)
        original_metrics = FairnessAnalyzer().calculate_fairness_metrics(
            y_test, original_pred, sensitive_features, "Original"
        )
        
        # Apply different mitigation techniques
        dp_mitigator = self.apply_demographic_parity_constraint(
            model, X_train, y_train, sensitive_features
        )
        dp_pred = dp_mitigator.predict(X_test)
        dp_metrics = FairnessAnalyzer().calculate_fairness_metrics(
            y_test, dp_pred, sensitive_features, "DP Mitigated"
        )
        
        eo_mitigator = self.apply_equalized_odds_constraint(
            model, X_train, y_train, sensitive_features
        )
        eo_pred = eo_mitigator.predict(X_test)
        eo_metrics = FairnessAnalyzer().calculate_fairness_metrics(
            y_test, eo_pred, sensitive_features, "EO Mitigated"
        )
        
        # Compare results
        comparison_data = {
            'Method': ['Original', 'DP Mitigated', 'EO Mitigated'],
            'DP_Difference': [
                original_metrics['demographic_parity_difference'],
                dp_metrics['demographic_parity_difference'],
                eo_metrics['demographic_parity_difference']
            ],
            'DP_Ratio': [
                original_metrics['demographic_parity_ratio'],
                dp_metrics['demographic_parity_ratio'],
                eo_metrics['demographic_parity_ratio']
            ],
            'EO_Difference': [
                original_metrics['equalized_odds_difference'],
                dp_metrics['equalized_odds_difference'],
                eo_metrics['equalized_odds_difference']
            ],
            'EO_Ratio': [
                original_metrics['equalized_odds_ratio'],
                dp_metrics['equalized_odds_ratio'],
                eo_metrics['equalized_odds_ratio']
            ],
            'Accuracy': [
                original_metrics['accuracy'],
                dp_metrics['accuracy'],
                eo_metrics['accuracy']
            ],
            'F1_Score': [
                original_metrics['f1_score'],
                dp_metrics['f1_score'],
                eo_metrics['f1_score']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # DP Difference comparison
        axes[0, 0].bar(comparison_df['Method'], comparison_df['DP_Difference'], 
                      color=['red', 'green', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Demographic Parity Difference')
        axes[0, 0].set_ylabel('DP Difference')
        axes[0, 0].axhline(y=0.1, color='orange', linestyle='--', label='Target (<0.1)')
        axes[0, 0].legend()
        
        # DP Ratio comparison
        axes[0, 1].bar(comparison_df['Method'], comparison_df['DP_Ratio'], 
                      color=['red', 'green', 'blue'], alpha=0.7)
        axes[0, 1].set_title('Demographic Parity Ratio')
        axes[0, 1].set_ylabel('DP Ratio')
        axes[0, 1].axhline(y=0.8, color='orange', linestyle='--', label='Target (>0.8)')
        axes[0, 1].legend()
        
        # EO Difference comparison
        axes[1, 0].bar(comparison_df['Method'], comparison_df['EO_Difference'], 
                      color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 0].set_title('Equalized Odds Difference')
        axes[1, 0].set_ylabel('EO Difference')
        axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', label='Target (<0.1)')
        axes[1, 0].legend()
        
        # Accuracy comparison
        axes[1, 1].bar(comparison_df['Method'], comparison_df['Accuracy'], 
                      color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 1].set_title('Accuracy Comparison')
        axes[1, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return comparison_df, dp_mitigator, eo_mitigator


def generate_fairness_report(fairness_df, issues, save_path=None):
    """
    Generate comprehensive fairness report
    
    Args:
        fairness_df (pd.DataFrame): Fairness analysis results
        issues (list): List of identified fairness issues
        save_path (str, optional): Path to save the report
    
    Returns:
        str: Generated report
    """
    
    report = []
    report.append("# Fairness Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append(f"Total models analyzed: {fairness_df['Model'].nunique()}")
    report.append(f"Total attributes analyzed: {fairness_df['Attribute'].nunique()}")
    report.append(f"Total fairness issues identified: {len(issues)}")
    report.append("")
    
    # Fairness metrics summary
    report.append("## Fairness Metrics Summary")
    report.append(fairness_df.to_string(index=False))
    report.append("")
    
    # Issues identified
    report.append("## Fairness Issues Identified")
    if issues:
        for i, issue in enumerate(issues, 1):
            report.append(f"{i}. {issue}")
    else:
        report.append("No significant fairness issues detected.")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    if len(issues) > 0:
        report.append("1. Implement bias mitigation techniques")
        report.append("2. Use demographic parity or equalized odds constraints")
        report.append("3. Consider threshold optimization")
        report.append("4. Regular fairness monitoring recommended")
    else:
        report.append("1. Continue monitoring for fairness issues")
        report.append("2. Regular fairness audits recommended")
        report.append("3. Consider expanding protected attributes")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    print("Twitter Sentiment Analysis - Fairness Analysis Module")
    print("This module provides comprehensive fairness analysis and bias detection.")
    print("Use the classes to analyze and mitigate bias in sentiment analysis models.")
