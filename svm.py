import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import normalize
import pickle
import os
import re

class OneClassSVMTextDetector:
    def __init__(self, 
                 kernel='rbf', 
                 nu=0.1,
                 max_features=10000,
                 min_df=2,
                 max_df=0.95,
                 ngram_range=(1, 2)):
        """
        Initialize the detector with TF-IDF and One-Class SVM
        
        Parameters:
        kernel: SVM kernel ('rbf', 'linear', 'poly')
        nu: SVM nu parameter (approximate proportion of outliers)
        max_features: Maximum number of features for TF-IDF
        min_df: Minimum document frequency for TF-IDF features
        max_df: Maximum document frequency for TF-IDF features
        ngram_range: Range of n-grams to use
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        self.svm = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma='scale'
        )
        
        self.train_vectors = None
        self.train_sentences = None
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep structure
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def fit(self, sentences):
        """
        Fit the model on training sentences
        
        Returns:
        self for method chaining
        """
        self.train_sentences = sentences
        
        # Preprocess sentences
        processed_sentences = [self.preprocess_text(s) for s in sentences]
        
        # Fit and transform with TF-IDF
        print("Fitting TF-IDF vectorizer...")
        tfidf_vectors = self.tfidf.fit_transform(processed_sentences)
        
        # Normalize vectors
        print("Normalizing vectors...")
        self.train_vectors = normalize(tfidf_vectors.toarray())
        
        # Fit One-Class SVM
        print("Training One-Class SVM...")
        self.svm.fit(self.train_vectors)
        
        # Print training summary
        n_features = self.train_vectors.shape[1]
        print(f"\nTraining Summary:")
        print(f"Number of training documents: {len(sentences)}")
        print(f"Vocabulary size: {n_features}")
        print(f"Top features by IDF weight:")
        self._print_top_features()
        
        return self
    
    def predict(self, sentences, return_score=False):
        """
        Predict if new sentences are normal (-1) or anomalous (1)
        
        Parameters:
        sentences: List of sentences to predict
        return_score: If True, return decision scores as well
        
        Returns:
        predictions or (predictions, scores) if return_score=True
        """
        # Preprocess new sentences
        processed_sentences = [self.preprocess_text(s) for s in sentences]
        
        # Transform to TF-IDF vectors
        tfidf_vectors = self.tfidf.transform(processed_sentences)
        vectors = normalize(tfidf_vectors.toarray())
        
        # Get predictions
        predictions = self.svm.predict(vectors)
        
        if return_score:
            scores = self.svm.score_samples(vectors)
            return predictions, scores
        
        return predictions
    
    def _print_top_features(self, top_n=10):
        """Print top features by IDF weight"""
        vocab = self.tfidf.get_feature_names_out()
        idf_scores = dict(zip(vocab, self.tfidf.idf_))
        top_terms = sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        for term, score in top_terms:
            print(f"{term}: {score:.4f}")
    
    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'svm': self.svm,
                'train_vectors': self.train_vectors,
                'train_sentences': self.train_sentences
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            components = pickle.load(f)
        
        model = cls()
        model.tfidf = components['tfidf']
        model.svm = components['svm']
        model.train_vectors = components['train_vectors']
        model.train_sentences = components['train_sentences']
        return model

# Example usage
def example():
    # Generate some example data
    normal_sentences = [
        "How to install a database",
        "Tutorial on configuring servers",
        "Guide for optimizing network",
        "Best practices for server setup",
        "Database configuration guide"
    ]
    
    anomaly_sentences = [
        "Recipe for chocolate cake",
        "Best movies of 2024",
        "How to plant tomatoes",
    ]
    
    # Initialize and train model
    detector = OneClassSVMTextDetector(nu=0.1)
    detector.fit(normal_sentences)
    
    # Make predictions
    test_sentences = normal_sentences[:2] + anomaly_sentences[:2]
    predictions, scores = detector.predict(test_sentences, return_score=True)
    
    # Print results
    print("\nPrediction Results:")
    for sentence, pred, score in zip(test_sentences, predictions, scores):
        status = "Normal" if pred == 1 else "Anomaly"
        print(f"\nSentence: {sentence}")
        print(f"Prediction: {status}")
        print(f"Score: {score:.4f}")

if __name__ == "__main__":
    example()
