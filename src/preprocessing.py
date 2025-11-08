"""
Twitter Sentiment Analysis - Preprocessing Module

This module contains comprehensive text preprocessing functions for Twitter data,
including cleaning, tokenization, feature extraction, and class balancing.
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import pickle

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from textblob import TextBlob

# Feature extraction libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Word embeddings
import gensim
from gensim.models import Word2Vec

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"NLTK download issue: {e}")


class TwitterTextPreprocessor:
    """
    Comprehensive text preprocessing for Twitter data
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.tweet_tokenizer = TweetTokenizer()
        
        # Twitter-specific patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\\w+')
        self.hashtag_pattern = re.compile(r'#\\w+')
        self.emoji_pattern = re.compile(r'[\\U0001F600-\\U0001F64F\\U0001F300-\\U0001F5FF\\U0001F680-\\U0001F6FF\\U0001F1E0-\\U0001F1FF]')
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('URL', text)
        
        # Remove mentions
        text = self.mention_pattern.sub('MENTION', text)
        
        # Handle hashtags (keep the word, remove #)
        text = self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
        
        # Remove emojis
        text = self.emoji_pattern.sub('', text)
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove punctuation except for some important ones
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text):
        """Tokenize text using TweetTokenizer"""
        return self.tweet_tokenizer.tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens):
        """Stem tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_pipeline(self, text, use_lemmatization=True, use_stemming=False):
        """Complete preprocessing pipeline"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply lemmatization or stemming
        if use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        elif use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)


class FeatureExtractor:
    """
    Feature extraction for text data
    """
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.w2v_model = None
        
    def create_tfidf_features(self, texts, max_features=10000, ngram_range=(1, 2), 
                            min_df=5, max_df=0.95, save_path=None):
        """Create TF-IDF features"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        return tfidf_matrix
    
    def create_word2vec_features(self, texts, vector_size=100, window=5, 
                               min_count=5, workers=4, epochs=10, save_path=None):
        """Create Word2Vec features"""
        # Prepare text data for Word2Vec
        texts_for_w2v = [text.split() for text in texts]
        
        # Train Word2Vec model
        self.w2v_model = Word2Vec(
            sentences=texts_for_w2v,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=0,  # Use CBOW
            epochs=epochs
        )
        
        if save_path:
            self.w2v_model.save(save_path)
        
        # Create sentence embeddings
        sentence_embeddings = []
        for text in texts:
            words = text.split()
            vectors = []
            for word in words:
                if word in self.w2v_model.wv:
                    vectors.append(self.w2v_model.wv[word])
            
            if vectors:
                sentence_embeddings.append(np.mean(vectors, axis=0))
            else:
                sentence_embeddings.append(np.zeros(vector_size))
        
        return np.array(sentence_embeddings)
    
    def get_sentence_embedding(self, text, model):
        """Get sentence embedding by averaging word vectors"""
        words = text.split()
        vectors = []
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)


class ClassBalancer:
    """
    Handle class imbalance in datasets
    """
    
    def __init__(self):
        self.smote = None
        self.under_sampler = None
        
    def apply_smote(self, X, y, random_state=42):
        """Apply SMOTE for oversampling"""
        self.smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = self.smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def apply_undersampling(self, X, y, random_state=42):
        """Apply random undersampling"""
        self.under_sampler = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = self.under_sampler.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def apply_smoteenn(self, X, y, random_state=42):
        """Apply SMOTE + Edited Nearest Neighbors"""
        smoteenn = SMOTEENN(random_state=random_state)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        return X_balanced, y_balanced


def load_sentiment140_data(file_path, sample_size=None, balanced=True):
    """
    Load Sentiment140 dataset
    
    Args:
        file_path (str): Path to the CSV file
        sample_size (int, optional): Number of samples to load for faster processing
        balanced (bool): If True and sample_size is specified, ensures equal samples from each class
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    columns = ['sentiment', 'tweet_id', 'date', 'query', 'username', 'tweet_text']
    
    if sample_size:
        if balanced:
            # Load balanced sample: equal number from each class
            samples_per_class = sample_size // 2
            
            # Read file in chunks and collect balanced samples
            negative_samples = []
            positive_samples = []
            negative_count = 0
            positive_count = 0
            chunk_size = 10000  # Read in chunks for efficiency
            
            print(f"Loading balanced sample: {samples_per_class:,} from each class...")
            
            for chunk in pd.read_csv(
                file_path, 
                header=None, 
                names=columns,
                chunksize=chunk_size,
                encoding='latin-1'
            ):
                # Collect negative samples (sentiment = 0)
                if negative_count < samples_per_class:
                    neg_chunk = chunk[chunk['sentiment'] == 0]
                    needed = samples_per_class - negative_count
                    neg_selected = neg_chunk.head(needed)
                    if len(neg_selected) > 0:
                        negative_samples.append(neg_selected)
                        negative_count += len(neg_selected)
                
                # Collect positive samples (sentiment = 4)
                if positive_count < samples_per_class:
                    pos_chunk = chunk[chunk['sentiment'] == 4]
                    needed = samples_per_class - positive_count
                    pos_selected = pos_chunk.head(needed)
                    if len(pos_selected) > 0:
                        positive_samples.append(pos_selected)
                        positive_count += len(pos_selected)
                
                # Stop if we have enough samples from both classes
                if negative_count >= samples_per_class and positive_count >= samples_per_class:
                    break
            
            # Combine samples
            df_negative = pd.concat(negative_samples, ignore_index=True).head(samples_per_class) if negative_samples else pd.DataFrame()
            df_positive = pd.concat(positive_samples, ignore_index=True).head(samples_per_class) if positive_samples else pd.DataFrame()
            
            # Combine and shuffle
            df = pd.concat([df_negative, df_positive], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"Loaded {len(df_negative):,} negative and {len(df_positive):,} positive samples")
        else:
            # Simple sequential sampling
            df = pd.read_csv(
                file_path, 
                header=None, 
                names=columns, 
                nrows=sample_size,
                encoding='latin-1'
            )
    else:
        df = pd.read_csv(
            file_path, 
            header=None, 
            names=columns,
            encoding='latin-1'
        )
    
    # Convert sentiment labels
    sentiment_mapping = {0: 'Negative', 4: 'Positive'}
    df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
    df['sentiment_binary'] = (df['sentiment'] == 4).astype(int)
    
    return df


def preprocess_dataset(df, preprocessor, feature_extractor, save_paths=None):
    """
    Complete dataset preprocessing pipeline
    
    Args:
        df (pd.DataFrame): Input dataset
        preprocessor (TwitterTextPreprocessor): Text preprocessor
        feature_extractor (FeatureExtractor): Feature extractor
        save_paths (dict, optional): Paths to save processed data
    
    Returns:
        dict: Processed data and features
    """
    # Apply text preprocessing
    print("Applying text preprocessing...")
    df['cleaned_text'] = df['tweet_text'].apply(
        lambda x: preprocessor.preprocess_pipeline(x, use_lemmatization=True)
    )
    
    # Remove empty tweets
    df = df[df['cleaned_text'].str.strip() != '']
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_path = save_paths['tfidf_vectorizer'] if save_paths else None
    X_tfidf = feature_extractor.create_tfidf_features(
        df['cleaned_text'], save_path=tfidf_path
    )
    
    # Create Word2Vec features
    print("Creating Word2Vec features...")
    w2v_path = save_paths['word2vec_model'] if save_paths else None
    X_w2v = feature_extractor.create_word2vec_features(
        df['cleaned_text'], save_path=w2v_path
    )
    
    # Get labels
    y = df['sentiment_binary'].values
    
    # Train-test split
    print("Splitting data...")
    X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_w2v_train, X_w2v_test, _, _ = train_test_split(
        X_w2v, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    if save_paths:
        print("Saving processed data...")
        np.save(save_paths['X_tfidf_train'], X_tfidf_train.toarray())
        np.save(save_paths['X_tfidf_test'], X_tfidf_test.toarray())
        np.save(save_paths['X_w2v_train'], X_w2v_train)
        np.save(save_paths['X_w2v_test'], X_w2v_test)
        np.save(save_paths['y_train'], y_train)
        np.save(save_paths['y_test'], y_test)
        
        # Save text data
        text_train, text_test, _, _ = train_test_split(
            df['cleaned_text'], y, test_size=0.2, random_state=42, stratify=y
        )
        
        with open(save_paths['text_train'], 'wb') as f:
            pickle.dump(text_train.tolist(), f)
        
        with open(save_paths['text_test'], 'wb') as f:
            pickle.dump(text_test.tolist(), f)
    
    return {
        'X_tfidf_train': X_tfidf_train,
        'X_tfidf_test': X_tfidf_test,
        'X_w2v_train': X_w2v_train,
        'X_w2v_test': X_w2v_test,
        'y_train': y_train,
        'y_test': y_test,
        'tfidf_vectorizer': feature_extractor.tfidf_vectorizer,
        'w2v_model': feature_extractor.w2v_model
    }


if __name__ == "__main__":
    # Example usage
    print("Twitter Sentiment Analysis - Preprocessing Module")
    print("This module provides comprehensive text preprocessing functions.")
    print("Use the classes and functions to preprocess your Twitter data.")
