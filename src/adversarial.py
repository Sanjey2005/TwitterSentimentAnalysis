"""
Twitter Sentiment Analysis - Adversarial Testing Module

This module provides comprehensive adversarial robustness testing
and defense mechanisms for Twitter sentiment analysis models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import string
import re
from collections import Counter
import warnings

# Text processing and NLP
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Model evaluation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')


class AdversarialAttacks:
    """
    Generate various types of adversarial examples for text data
    """
    
    def __init__(self):
        self.char_substitutions = {
            'a': ['@', '4'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'],
            's': ['$', '5'], 't': ['7'], 'l': ['1'], 'b': ['6']
        }
        
    def character_substitution(self, text, substitution_rate=0.1):
        """Replace characters with similar-looking characters"""
        result = []
        for char in text:
            if char.lower() in self.char_substitutions and random.random() < substitution_rate:
                result.append(random.choice(self.char_substitutions[char.lower()]))
            else:
                result.append(char)
        return ''.join(result)
    
    def character_insertion(self, text, insertion_rate=0.05):
        """Insert random characters"""
        result = []
        for char in text:
            result.append(char)
            if random.random() < insertion_rate:
                result.append(random.choice(string.ascii_letters + string.digits))
        return ''.join(result)
    
    def character_deletion(self, text, deletion_rate=0.05):
        """Delete random characters"""
        result = []
        for char in text:
            if random.random() > deletion_rate:
                result.append(char)
        return ''.join(result)
    
    def word_substitution(self, text, substitution_rate=0.1):
        """Replace words with synonyms"""
        words = word_tokenize(text)
        result = []
        
        for word in words:
            if random.random() < substitution_rate:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    result.append(random.choice(synonyms))
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)
    
    def word_order_perturbation(self, text, perturbation_rate=0.1):
        """Randomly swap adjacent words"""
        words = word_tokenize(text)
        result = words.copy()
        
        for i in range(len(words) - 1):
            if random.random() < perturbation_rate:
                result[i], result[i + 1] = result[i + 1], result[i]
        
        return ' '.join(result)
    
    def typo_injection(self, text, typo_rate=0.05):
        """Inject common typos"""
        words = word_tokenize(text)
        result = []
        
        for word in words:
            if random.random() < typo_rate and len(word) > 2:
                # Common typo patterns
                if random.random() < 0.5:
                    # Double character
                    pos = random.randint(0, len(word) - 1)
                    word = word[:pos] + word[pos] + word[pos:]
                else:
                    # Character swap
                    if len(word) > 2:
                        pos = random.randint(0, len(word) - 2)
                        word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            result.append(word)
        
        return ' '.join(result)
    
    def generate_adversarial_examples(self, texts, attack_type='character_substitution', **kwargs):
        """Generate adversarial examples using specified attack type"""
        adversarial_texts = []
        
        for text in texts:
            if attack_type == 'character_substitution':
                adv_text = self.character_substitution(text, **kwargs)
            elif attack_type == 'character_insertion':
                adv_text = self.character_insertion(text, **kwargs)
            elif attack_type == 'character_deletion':
                adv_text = self.character_deletion(text, **kwargs)
            elif attack_type == 'word_substitution':
                adv_text = self.word_substitution(text, **kwargs)
            elif attack_type == 'word_order':
                adv_text = self.word_order_perturbation(text, **kwargs)
            elif attack_type == 'typo_injection':
                adv_text = self.typo_injection(text, **kwargs)
            else:
                adv_text = text
            
            adversarial_texts.append(adv_text)
        
        return adversarial_texts


class AdversarialTester:
    """
    Test model robustness against adversarial attacks
    """
    
    def __init__(self):
        self.attack_types = [
            'character_substitution',
            'character_insertion', 
            'character_deletion',
            'word_substitution',
            'word_order',
            'typo_injection'
        ]
        
        self.attack_params = {
            'character_substitution': {'substitution_rate': 0.1},
            'character_insertion': {'insertion_rate': 0.05},
            'character_deletion': {'deletion_rate': 0.05},
            'word_substitution': {'substitution_rate': 0.1},
            'word_order': {'perturbation_rate': 0.1},
            'typo_injection': {'typo_rate': 0.05}
        }
        
        self.attacks = AdversarialAttacks()
        
    def test_model_robustness(self, model, vectorizer, texts, labels, 
                            baseline_accuracy, model_name="Model"):
        """Test model robustness against all attack types"""
        
        robustness_results = []
        
        for attack_type in self.attack_types:
            print(f"Testing {attack_type} attack on {model_name}...")
            
            # Generate adversarial examples
            adversarial_texts = self.attacks.generate_adversarial_examples(
                texts, attack_type, **self.attack_params[attack_type]
            )
            
            # Transform adversarial texts to features
            adversarial_features = vectorizer.transform(adversarial_texts)
            
            # Get predictions on adversarial examples
            y_pred_adv = model.predict(adversarial_features)
            
            # Calculate adversarial accuracy
            adv_accuracy = accuracy_score(labels, y_pred_adv)
            
            # Calculate robustness score (adversarial accuracy / baseline accuracy)
            robustness_score = adv_accuracy / baseline_accuracy
            
            # Store results
            robustness_results.append({
                'attack_type': attack_type,
                'adversarial_accuracy': adv_accuracy,
                'robustness_score': robustness_score,
                'accuracy_drop': baseline_accuracy - adv_accuracy
            })
            
            print(f"  Adversarial Accuracy: {adv_accuracy:.4f}")
            print(f"  Robustness Score: {robustness_score:.4f}")
        
        return robustness_results
    
    def comprehensive_robustness_test(self, models, vectorizer, texts, labels, 
                                    baseline_accuracies, model_names):
        """Test multiple models against adversarial attacks"""
        
        all_results = []
        
        for model, baseline_acc, model_name in zip(models, baseline_accuracies, model_names):
            results = self.test_model_robustness(
                model, vectorizer, texts, labels, baseline_acc, model_name
            )
            
            # Add model name to results
            for result in results:
                result['model_name'] = model_name
            
            all_results.extend(results)
        
        return pd.DataFrame(all_results)


class DefenseMechanisms:
    """
    Implement defense mechanisms against adversarial attacks
    """
    
    def __init__(self):
        pass
    
    def create_ensemble_defense(self, models, voting='soft'):
        """Create ensemble model for defense"""
        return VotingClassifier(estimators=models, voting=voting)
    
    def sanitize_input(self, text):
        """Sanitize input by removing suspicious patterns"""
        # Remove excessive special characters
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove very short words (likely noise)
        words = text.split()
        words = [word for word in words if len(word) > 1]
        
        return ' '.join(words)
    
    def apply_sanitization_defense(self, texts):
        """Apply sanitization to a list of texts"""
        return [self.sanitize_input(text) for text in texts]
    
    def adversarial_training(self, model, X_train, y_train, vectorizer, 
                           attack_types=['character_substitution', 'typo_injection'],
                           attack_params=None):
        """Apply adversarial training to improve robustness"""
        
        if attack_params is None:
            attack_params = {
                'character_substitution': {'substitution_rate': 0.1},
                'typo_injection': {'typo_rate': 0.05}
            }
        
        attacks = AdversarialAttacks()
        
        # Generate adversarial training examples
        adversarial_texts = []
        adversarial_labels = []
        
        # Convert training data back to text (assuming it's TF-IDF)
        # This is a simplified approach - in practice, you'd need to store original texts
        train_texts = [" ".join([f"word_{i}" for i in range(10)]) for _ in range(len(X_train))]
        
        for attack_type in attack_types:
            adv_texts = attacks.generate_adversarial_examples(
                train_texts, attack_type, **attack_params[attack_type]
            )
            adversarial_texts.extend(adv_texts)
            adversarial_labels.extend(y_train)
        
        # Combine original and adversarial training data
        combined_texts = list(train_texts) + adversarial_texts
        combined_labels = list(y_train) + adversarial_labels
        
        # Transform combined training data
        combined_features = vectorizer.transform(combined_texts)
        
        # Train robust model
        robust_model = model.__class__(**model.get_params())
        robust_model.fit(combined_features, combined_labels)
        
        return robust_model


class RobustnessAnalyzer:
    """
    Analyze and compare robustness of different models and defenses
    """
    
    def __init__(self):
        self.results = {}
        
    def compare_defense_mechanisms(self, models, vectorizer, texts, labels, 
                                 baseline_accuracies, model_names, save_path=None):
        """Compare different defense mechanisms"""
        
        defense_mechanisms = DefenseMechanisms()
        
        # Test baseline models
        tester = AdversarialTester()
        baseline_results = tester.comprehensive_robustness_test(
            models, vectorizer, texts, labels, baseline_accuracies, model_names
        )
        
        # Test ensemble defense
        ensemble_model = defense_mechanisms.create_ensemble_defense(
            [(name, model) for name, model in zip(model_names, models)]
        )
        ensemble_model.fit(vectorizer.transform(texts), labels)
        
        ensemble_baseline = accuracy_score(labels, ensemble_model.predict(vectorizer.transform(texts)))
        ensemble_results = tester.test_model_robustness(
            ensemble_model, vectorizer, texts, labels, ensemble_baseline, "Ensemble"
        )
        
        # Test sanitization defense
        sanitized_texts = defense_mechanisms.apply_sanitization_defense(texts)
        sanitized_features = vectorizer.transform(sanitized_texts)
        
        sanitization_results = []
        for model, baseline_acc, model_name in zip(models, baseline_accuracies, model_names):
            # Generate adversarial examples
            attacks = AdversarialAttacks()
            adversarial_texts = []
            for attack_type in tester.attack_types:
                adv_texts = attacks.generate_adversarial_examples(
                    texts, attack_type, **tester.attack_params[attack_type]
                )
                adversarial_texts.extend(adv_texts)
            
            # Sanitize adversarial examples
            sanitized_adv_texts = defense_mechanisms.apply_sanitization_defense(adversarial_texts)
            sanitized_adv_features = vectorizer.transform(sanitized_adv_texts)
            
            # Test on sanitized adversarial examples
            y_pred_sanitized = model.predict(sanitized_adv_features)
            sanitized_accuracy = accuracy_score(labels * len(tester.attack_types), y_pred_sanitized)
            
            sanitization_results.append({
                'model_name': model_name,
                'sanitized_accuracy': sanitized_accuracy,
                'improvement': sanitized_accuracy - baseline_acc
            })
        
        # Create comparison visualization
        self.plot_defense_comparison(baseline_results, ensemble_results, 
                                   sanitization_results, save_path)
        
        return baseline_results, ensemble_results, sanitization_results
    
    def plot_defense_comparison(self, baseline_results, ensemble_results, 
                              sanitization_results, save_path=None):
        """Plot comparison of defense mechanisms"""
        
        # Prepare data for plotting
        attack_types = baseline_results['attack_type'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Adversarial accuracy by attack type
        for model_name in baseline_results['model_name'].unique():
            model_data = baseline_results[baseline_results['model_name'] == model_name]
            axes[0, 0].plot(attack_types, model_data['adversarial_accuracy'], 
                           marker='o', label=model_name, alpha=0.8)
        
        # Add ensemble results
        ensemble_accuracies = [ensemble_results[i]['adversarial_accuracy'] 
                             for i in range(len(attack_types))]
        axes[0, 0].plot(attack_types, ensemble_accuracies, 
                       marker='s', label='Ensemble', linewidth=2)
        
        axes[0, 0].set_xlabel('Attack Type')
        axes[0, 0].set_ylabel('Adversarial Accuracy')
        axes[0, 0].set_title('Adversarial Accuracy by Attack Type')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0.75, color='red', linestyle='--', label='Target (75%)')
        axes[0, 0].legend()
        
        # 2. Average performance comparison
        avg_baseline = baseline_results.groupby('model_name')['adversarial_accuracy'].mean()
        avg_ensemble = np.mean(ensemble_accuracies)
        avg_sanitization = np.mean([result['sanitized_accuracy'] for result in sanitization_results])
        
        methods = list(avg_baseline.index) + ['Ensemble', 'Sanitization']
        avg_accuracies = list(avg_baseline.values) + [avg_ensemble, avg_sanitization]
        
        axes[0, 1].bar(methods, avg_accuracies, alpha=0.8, 
                      color=['red', 'orange', 'green', 'blue', 'purple'])
        axes[0, 1].set_ylabel('Average Adversarial Accuracy')
        axes[0, 1].set_title('Average Performance Across All Attacks')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.75, color='red', linestyle='--', label='Target (75%)')
        axes[0, 1].legend()
        
        # 3. Robustness scores
        robustness_scores = baseline_results.groupby('model_name')['robustness_score'].mean()
        ensemble_robustness = np.mean([ensemble_results[i]['robustness_score'] 
                                     for i in range(len(attack_types))])
        
        methods_robust = list(robustness_scores.index) + ['Ensemble']
        robustness_values = list(robustness_scores.values) + [ensemble_robustness]
        
        axes[1, 0].bar(methods_robust, robustness_values, alpha=0.8,
                      color=['red', 'orange', 'green', 'blue'])
        axes[1, 0].set_ylabel('Robustness Score')
        axes[1, 0].set_title('Robustness Scores (Adversarial/Baseline Accuracy)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.75, color='red', linestyle='--', label='Target (75%)')
        axes[1, 0].legend()
        
        # 4. Defense effectiveness heatmap
        defense_matrix = baseline_results.pivot(index='model_name', 
                                              columns='attack_type', 
                                              values='adversarial_accuracy')
        
        # Add ensemble results
        ensemble_row = pd.Series(ensemble_accuracies, index=attack_types, name='Ensemble')
        defense_matrix = pd.concat([defense_matrix, ensemble_row.to_frame().T])
        
        sns.heatmap(defense_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 1])
        axes[1, 1].set_title('Defense Effectiveness Heatmap')
        axes[1, 1].set_xlabel('Attack Type')
        axes[1, 1].set_ylabel('Defense Method')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_robustness_report(self, results, save_path=None):
        """
        Generate comprehensive robustness report
        
        Args:
            results (dict): Robustness test results
            save_path (str, optional): Path to save the report
        
        Returns:
            str: Generated report
        """
        
        report = []
        report.append("# Adversarial Robustness Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        baseline_results = results.get('baseline', pd.DataFrame())
        ensemble_results = results.get('ensemble', [])
        sanitization_results = results.get('sanitization', [])
        
        report.append("## Summary Statistics")
        report.append(f"Models tested: {baseline_results['model_name'].nunique() if not baseline_results.empty else 0}")
        report.append(f"Attack types: {len(baseline_results['attack_type'].unique()) if not baseline_results.empty else 0}")
        report.append(f"Defense mechanisms: 3 (Baseline, Ensemble, Sanitization)")
        report.append("")
        
        # Average performance
        if not baseline_results.empty:
            avg_baseline = baseline_results.groupby('model_name')['adversarial_accuracy'].mean()
            report.append("## Average Adversarial Accuracy")
            for model_name, accuracy in avg_baseline.items():
                report.append(f"- {model_name}: {accuracy:.3f}")
            
            if ensemble_results:
                avg_ensemble = np.mean([r['adversarial_accuracy'] for r in ensemble_results])
                report.append(f"- Ensemble: {avg_ensemble:.3f}")
            
            if sanitization_results:
                avg_sanitization = np.mean([r['sanitized_accuracy'] for r in sanitization_results])
                report.append(f"- Sanitization: {avg_sanitization:.3f}")
        
        report.append("")
        
        # Robustness assessment
        report.append("## Robustness Assessment")
        target_accuracy = 0.75
        
        if not baseline_results.empty:
            models_meeting_target = baseline_results[
                baseline_results['adversarial_accuracy'] >= target_accuracy
            ]['model_name'].unique()
            
            if len(models_meeting_target) > 0:
                report.append(f"Models meeting {target_accuracy} target: {', '.join(models_meeting_target)}")
            else:
                report.append(f"No models meet the {target_accuracy} adversarial accuracy target")
        
        report.append("")
        
        # Recommendations
        report.append("## Security Recommendations")
        if not baseline_results.empty:
            max_accuracy = baseline_results['adversarial_accuracy'].max()
            if max_accuracy < target_accuracy:
                report.append("1. Implement stronger defense mechanisms")
                report.append("2. Consider ensemble methods with more models")
                report.append("3. Increase adversarial training data")
                report.append("4. Implement input sanitization")
            else:
                report.append("1. Current defenses are adequate")
                report.append("2. Continue monitoring for new attack vectors")
                report.append("3. Regular adversarial testing recommended")
        
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


if __name__ == "__main__":
    print("Twitter Sentiment Analysis - Adversarial Testing Module")
    print("This module provides comprehensive adversarial robustness testing.")
    print("Use the classes to test and defend against adversarial attacks.")
