from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import re

class BM25Vectorizer:
    """BM25 vectorizer with custom tokenization"""
    def __init__(self, tokenizer, k1=1.5, b=0.75):
        self.tokenizer = tokenizer
        self.k1 = k1  # term frequency saturation parameter
        self.b = b    # length normalization parameter
        self.vocabulary_ = {}
        self._doc_freq = None
        self._avgdl = None
        
    def fit_transform(self, raw_documents):
        # Tokenize documents
        tokenized_docs = [self.tokenizer(doc) for doc in raw_documents]
        
        # Build vocabulary
        vocab = {}
        for doc in tokenized_docs:
            for term in doc:
                if term not in vocab:
                    vocab[term] = len(vocab)
        self.vocabulary_ = vocab
        
        # Calculate document frequencies
        self._doc_freq = np.zeros(len(vocab))
        doc_lengths = []
        
        # Create term-document matrix
        X = np.zeros((len(tokenized_docs), len(vocab)))
        
        for i, doc in enumerate(tokenized_docs):
            term_freq = {}
            for term in doc:
                if term in vocab:
                    term_freq[vocab[term]] = term_freq.get(vocab[term], 0) + 1
            
            # Update document frequencies
            for term_idx in term_freq:
                self._doc_freq[term_idx] += 1
            
            # Update term frequencies in matrix
            for term_idx, freq in term_freq.items():
                X[i, term_idx] = freq
            
            doc_lengths.append(len(doc))
        
        # Calculate average document length
        self._avgdl = np.mean(doc_lengths)
        
        # Apply BM25 transformation
        return self._bm25_transform(X, doc_lengths)
    
    def transform(self, raw_documents):
        tokenized_docs = [self.tokenizer(doc) for doc in raw_documents]
        X = np.zeros((len(tokenized_docs), len(self.vocabulary_)))
        doc_lengths = []
        
        for i, doc in enumerate(tokenized_docs):
            term_freq = {}
            for term in doc:
                if term in self.vocabulary_:
                    term_freq[self.vocabulary_[term]] = term_freq.get(self.vocabulary_[term], 0) + 1
            
            for term_idx, freq in term_freq.items():
                X[i, term_idx] = freq
            
            doc_lengths.append(len(doc))
        
        return self._bm25_transform(X, doc_lengths)
    
    def _bm25_transform(self, X, doc_lengths):
        """Apply BM25 transformation to term frequency matrix"""
        n_docs = X.shape[0]
        
        # Calculate IDF scores
        idf = np.log((n_docs - self._doc_freq + 0.5) / (self._doc_freq + 0.5) + 1.0)
        
        # Calculate BM25 scores
        doc_lengths = np.array(doc_lengths)[:, np.newaxis]
        len_norm = (1 - self.b + self.b * doc_lengths / self._avgdl)
        
        # BM25 formula
        numerator = X * (self.k1 + 1)
        denominator = X + self.k1 * len_norm
        
        return (numerator / denominator) * idf

class BM25LSAModel:
    """BM25 + LSA for text embeddings with custom tokenization."""
    def __init__(self, n_components=50, k1=1.5, b=0.75):
        self.tokenizer = CustomTokenizer()
        self.bm25 = BM25Vectorizer(
            tokenizer=self.tokenizer,
            k1=k1,
            b=b
        )
        self.n_components = n_components
        self.lsa = None
        self.train_vectors = None
        self.train_sentences = None
        
    def analyze_tokens(self, text):
        """Analyze how the text is being tokenized."""
        tokens = self.tokenizer(text)
        return {
            'original_text': text,
            'tokens': tokens,
            'token_count': len(tokens)
        }
        
    def fit(self, sentences):
        # Print example tokenization for first few sentences
        print("Tokenization examples:")
        for sentence in sentences[:3]:
            analysis = self.analyze_tokens(sentence)
            print(f"\nOriginal: {analysis['original_text']}")
            print(f"Tokens: {analysis['tokens']}")
        
        self.train_sentences = sentences
        bm25_vectors = self.bm25.fit_transform(sentences)
        n_features = bm25_vectors.shape[1]
        
        # Print vocabulary information
        print(f"\nVocabulary size: {len(self.bm25.vocabulary_)}")
        print("Sample vocabulary items:")
        sample_vocab = list(self.bm25.vocabulary_.items())[:10]
        for term, index in sample_vocab:
            print(f"  {term}: {index}")
        
        # Adjust n_components if necessary
        n_components = min(self.n_components, n_features - 1)
        print(f"\nUsing {n_components} components for LSA")
        
        self.lsa = TruncatedSVD(n_components=n_components)
        self.train_vectors = self.lsa.fit_transform(bm25_vectors)
        return self.train_vectors
    
    def transform(self, sentences):
        bm25_vectors = self.bm25.transform(sentences)
        return self.lsa.transform(bm25_vectors)
    
    def find_similar_sentences(self, query_sentence, top_k=5):
        """Find most similar sentences to the query sentence."""
        if self.train_vectors is None or self.train_sentences is None:
            raise ValueError("Model must be trained before finding similar sentences")
        
        # Transform query sentence
        query_vector = self.transform([query_sentence])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.train_vectors)[0]
        
        # Get top_k similar sentences
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                'sentence': self.train_sentences[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'lsa': self.lsa,
                'train_vectors': self.train_vectors,
                'train_sentences': self.train_sentences
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            components = pickle.load(f)
        model = cls()
        model.bm25 = components['bm25']
        model.lsa = components['lsa']
        model.train_vectors = components['train_vectors']
        model.train_sentences = components['train_sentences']
        return model
        
    def find_similar_sentences(self, query_sentence, top_k=5):
        """Find most similar sentences to the query sentence."""
        if self.train_vectors is None or self.train_sentences is None:
            raise ValueError("Model must be trained before finding similar sentences")
        
        # Transform query sentence
        query_vector = self.transform([query_sentence])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.train_vectors)[0]
        
        # Get top_k similar sentences
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                'sentence': self.train_sentences[idx],
                'score': float(similarities[idx])  # Convert to float for better printing
            })
        
        return results
    
    def transform(self, sentences):
        tfidf_vectors = self.tfidf.transform(sentences)
        return self.lsa.transform(tfidf_vectors)
        
    def save(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf, 
                'lsa': self.lsa,
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
        model.lsa = components['lsa']
        model.train_vectors = components['train_vectors']
        model.train_sentences = components['train_sentences']
        return model
