from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import numpy as np
import re
import os
import pickle


class OneClassSVMTextDetector:
    def __init__(self,
                 vectorizer_type='tfidf',  # Options: 'tfidf' or 'doc2vec'
                 vector_size=100,  # For Doc2Vec
                 min_count=2,  # For Doc2Vec
                 epochs=20,  # For Doc2Vec
                 kernel='rbf',
                 nu=0.1,
                 max_features=10000,
                 min_df=2,
                 max_df=0.95,
                 ngram_range=(1, 2)):
        """
        Initialize the detector with choice of TF-IDF or Doc2Vec vectorization

        Parameters:
        vectorizer_type: Type of vectorizer to use ('tfidf' or 'doc2vec')
        vector_size: Size of document vectors for Doc2Vec
        min_count: Minimum word count for Doc2Vec
        epochs: Number of training epochs for Doc2Vec
        kernel: SVM kernel ('rbf', 'linear', 'poly')
        nu: SVM nu parameter (approximate proportion of outliers)
        max_features: Maximum number of features for TF-IDF
        min_df: Minimum document frequency for TF-IDF features
        max_df: Maximum document frequency for TF-IDF features
        ngram_range: Range of n-grams to use for TF-IDF
        """
        self.vectorizer_type = vectorizer_type

        # Initialize TF-IDF vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                norm='l2',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
        # Initialize Doc2Vec model
        elif vectorizer_type == 'doc2vec':
            self.vectorizer = Doc2Vec(
                vector_size=vector_size,
                min_count=min_count,
                epochs=epochs,
                dm=1,  # Use PV-DM
                workers=4
            )
        else:
            raise ValueError("vectorizer_type must be either 'tfidf' or 'doc2vec'")

        self.svm = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma='scale'
        )

        self.train_vectors = None
        self.train_sentences = None
        self.vector_size = vector_size  # For Doc2Vec

    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep structure
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _vectorize_text(self, sentences, training=False):
        """Vectorize text using either TF-IDF or Doc2Vec"""
        processed_sentences = [self.preprocess_text(s) for s in sentences]

        if self.vectorizer_type == 'tfidf':
            if training:
                vectors = self.vectorizer.fit_transform(processed_sentences)
            else:
                vectors = self.vectorizer.transform(processed_sentences)
            return normalize(vectors.toarray())

        elif self.vectorizer_type == 'doc2vec':
            if training:
                # Prepare tagged documents for Doc2Vec
                tagged_data = [TaggedDocument(words=doc.split(), tags=[i])
                               for i, doc in enumerate(processed_sentences)]
                # Train Doc2Vec model
                self.vectorizer.build_vocab(tagged_data)
                self.vectorizer.train(
                    tagged_data,
                    total_examples=self.vectorizer.corpus_count,
                    epochs=self.vectorizer.epochs
                )
                # Get document vectors
                vectors = np.array([self.vectorizer.dv[i]
                                    for i in range(len(processed_sentences))])
            else:
                # Infer vectors for new documents
                vectors = np.array([self.vectorizer.infer_vector(doc.split())
                                    for doc in processed_sentences])
            return normalize(vectors)

    def fit(self, sentences):
        """Train the model with automatic validation split"""
        # Split data into train and validation
        train_sentences, val_sentences = train_test_split(
            sentences, test_size=0.2, random_state=42
        )

        # Store training sentences
        self.train_sentences = train_sentences

        # Vectorize training data
        print(f"Fitting {self.vectorizer_type.upper()} vectorizer...")
        self.train_vectors = self._vectorize_text(train_sentences, training=True)

        # Fit One-Class SVM
        print("Training One-Class SVM...")
        self.svm.fit(self.train_vectors)

        # Validate the model
        validation_results = self.validate_model(val_sentences)

        # Print training and validation summary
        self._print_training_summary(validation_results)

        return self

    def validate_model(self, val_sentences, anomaly_sentences=None):
        """Validate model performance"""
        metrics = {}

        # Get vectors for validation data
        val_vectors = self._vectorize_text(val_sentences)

        # Get scores
        train_scores = self.svm.score_samples(self.train_vectors)
        val_scores = self.svm.score_samples(val_vectors)

        # Calculate basic metrics
        metrics['train_mean'] = np.mean(train_scores)
        metrics['train_std'] = np.std(train_scores)
        metrics['val_mean'] = np.mean(val_scores)
        metrics['val_std'] = np.std(val_scores)
        metrics['score_diff'] = abs(metrics['train_mean'] - metrics['val_mean'])

        # For TF-IDF, check vocabulary stability
        if self.vectorizer_type == 'tfidf':
            val_vectorizer = TfidfVectorizer(
                max_features=self.vectorizer.max_features,
                min_df=self.vectorizer.min_df,
                max_df=self.vectorizer.max_df
            )
            val_vectorizer.fit([self.preprocess_text(s) for s in val_sentences])

            train_vocab = set(self.vectorizer.get_feature_names_out())
            val_vocab = set(val_vectorizer.get_feature_names_out())
            vocab_overlap = len(train_vocab.intersection(val_vocab)) / len(train_vocab)
            metrics['vocab_stability'] = vocab_overlap

        # If anomaly data provided, calculate separation
        if anomaly_sentences is not None:
            anomaly_vectors = self._vectorize_text(anomaly_sentences)
            anomaly_scores = self.svm.score_samples(anomaly_vectors)

            metrics['anomaly_mean'] = np.mean(anomaly_scores)
            metrics['anomaly_std'] = np.std(anomaly_scores)
            metrics['normal_anomaly_separation'] = metrics['val_mean'] - metrics['anomaly_mean']

        # Overall assessment
        metrics['assessment'] = self._assess_metrics(metrics)

        return metrics

    def predict(self, sentences, return_score=False):
        """Predict if new sentences are normal (-1) or anomalous (1)"""
        # Get vectors for new sentences
        vectors = self._vectorize_text(sentences)

        # Get predictions
        predictions = self.svm.predict(vectors)

        if return_score:
            scores = self.svm.score_samples(vectors)
            return predictions, scores

        return predictions

    def predict_single(self, sentence):
        """Predict if a single sentence is normal or anomalous"""
        # Process and vectorize single sentence
        vectors = self._vectorize_text([sentence])

        # Get prediction and score
        prediction = self.svm.predict(vectors)[0]
        score = self.svm.score_samples(vectors)[0]

        return {
            'sentence': sentence,
            'processed': self.preprocess_text(sentence),
            'prediction': 'Normal' if prediction == 1 else 'Anomaly',
            'score': float(score)
        }

    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer_type': self.vectorizer_type,
                'vectorizer': self.vectorizer,
                'svm': self.svm,
                'train_vectors': self.train_vectors,
                'train_sentences': self.train_sentences
            }, f)

    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            components = pickle.load(f)

        model = cls(vectorizer_type=components['vectorizer_type'])
        model.vectorizer = components['vectorizer']
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

    # Test with TF-IDF
    print("\nTesting with TF-IDF vectorization:")
    detector_tfidf = OneClassSVMTextDetector(vectorizer_type='tfidf', nu=0.1)
    detector_tfidf.fit(normal_sentences)

    # Test with Doc2Vec
    print("\nTesting with Doc2Vec vectorization:")
    detector_doc2vec = OneClassSVMTextDetector(vectorizer_type='doc2vec', nu=0.1)
    detector_doc2vec.fit(normal_sentences)

    # Compare results
    test_sentence = "How to optimize database performance"

    print("\nComparing results for:", test_sentence)
    result_tfidf = detector_tfidf.predict_single(test_sentence)
    result_doc2vec = detector_doc2vec.predict_single(test_sentence)

    print("\nTF-IDF Results:")
    print(f"Prediction: {result_tfidf['prediction']}")
    print(f"Score: {result_tfidf['score']:.4f}")

    print("\nDoc2Vec Results:")
    print(f"Prediction: {result_doc2vec['prediction']}")
    print(f"Score: {result_doc2vec['score']:.4f}")
