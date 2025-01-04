# [Previous imports remain the same]
import pandas as pd
import numpy as np
import re
from thefuzz import fuzz
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import time
from typing import Dict, List, Union

# [Previous NLTK setup and preprocessing functions remain the same]
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def basic_preprocess(text):
    """Basic preprocessing - just lowercase and punctuation removal."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    for punct in ['.', ',', '!', '?', '(', ')', '[', ']', '{', '}', '/', '\\', '-', '_']:
        text = text.replace(punct, ' ')
    return ' '.join(text.split())


def advanced_preprocess(text):
    """Advanced preprocessing - includes stop word removal and thorough cleaning."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


class SimilarityMatcher:
    def __init__(self, preprocessing_method='basic'):
        """
        Initialize the matcher with specified preprocessing method.

        Args:
            preprocessing_method (str): 'basic' or 'advanced'
        """
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.preprocessing_method = preprocessing_method

    def preprocess_text(self, text):
        """Preprocess text using the selected method."""
        if self.preprocessing_method == 'advanced':
            return advanced_preprocess(text)
        return basic_preprocess(text)

    def calculate_fuzzy_similarity(self, text1, text2):
        """Calculate similarity using fuzzy matching."""
        proc_text1 = self.preprocess_text(text1)
        proc_text2 = self.preprocess_text(text2)

        if not proc_text1 or not proc_text2:
            return 0

        ratio = fuzz.ratio(proc_text1, proc_text2)
        partial_ratio = fuzz.partial_ratio(proc_text1, proc_text2)
        token_sort_ratio = fuzz.token_sort_ratio(proc_text1, proc_text2)
        token_set_ratio = fuzz.token_set_ratio(proc_text1, proc_text2)

        weighted_score = (
                ratio * 0.25 +
                partial_ratio * 0.25 +
                token_sort_ratio * 0.25 +
                token_set_ratio * 0.25
        )

        return weighted_score / 100

    def calculate_tfidf_similarity(self, texts):
        """Calculate similarity matrix using TF-IDF."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        return cosine_similarity(tfidf_matrix)

    def calculate_transformer_similarity(self, texts):
        """Calculate similarity matrix using Sentence Transformers."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        embeddings = self.sentence_transformer.encode(processed_texts)
        return cosine_similarity(embeddings)

    def calculate_hybrid_similarity(self,
                                    text1: str,
                                    text2: str,
                                    methods: Dict[str, float],
                                    similarity_matrices: Dict[str, np.ndarray] = None,
                                    indices: tuple = None) -> float:
        """
        Calculate hybrid similarity using multiple methods with weights.

        Args:
            text1: First text
            text2: Second text
            methods: Dictionary of method names and their weights
                    e.g., {'fuzzy': 0.3, 'tfidf': 0.3, 'transformer': 0.4}
            similarity_matrices: Pre-computed similarity matrices for tfidf and transformer
            indices: Tuple of indices for looking up values in similarity matrices

        Returns:
            float: Weighted similarity score
        """
        similarity_scores = []
        weights = []

        for method, weight in methods.items():
            if method == 'fuzzy':
                score = self.calculate_fuzzy_similarity(text1, text2)
            elif similarity_matrices and method in similarity_matrices and indices:
                i, j = indices
                score = similarity_matrices[method][i, j]
            else:
                continue

            similarity_scores.append(score)
            weights.append(weight)

        if not similarity_scores:
            return 0

        # Normalize weights to sum to 1
        weights = np.array(weights) / sum(weights)
        return np.average(similarity_scores, weights=weights)

    def group_problems(self,
                       df: pd.DataFrame,
                       method: Union[str, Dict[str, float]] = 'fuzzy',
                       threshold: float = 0.85) -> pd.DataFrame:
        """
        Group similar problems using specified method(s).

        Args:
            df: Input DataFrame
            method: Either a string ('fuzzy', 'tfidf', 'transformer') or a dictionary
                   of methods and their weights for hybrid approach
            threshold: Similarity threshold (0-1)

        Returns:
            pd.DataFrame: Processed DataFrame with grouped codes
        """
        result_df = df.copy()
        result_df['processed_description'] = df['Problem Description'].apply(self.preprocess_text)

        # Handle hybrid method
        if isinstance(method, dict):
            # Pre-compute similarity matrices for tfidf and transformer if needed
            similarity_matrices = {}
            texts = result_df['Problem Description'].tolist()

            if 'tfidf' in method:
                similarity_matrices['tfidf'] = self.calculate_tfidf_similarity(texts)
            if 'transformer' in method:
                similarity_matrices['transformer'] = self.calculate_transformer_similarity(texts)

            # Use hybrid similarity calculation
            problem_groups = defaultdict(set)
            processed_indices = set()

            for i, row1 in df.iterrows():
                if i in processed_indices:
                    continue

                desc1 = row1['Problem Description']
                code1 = row1['Resolution Code']
                current_group = {code1}

                for j, row2 in df.iloc[i + 1:].iterrows():
                    desc2 = row2['Problem Description']
                    code2 = row2['Resolution Code']

                    similarity = self.calculate_hybrid_similarity(
                        desc1, desc2, method, similarity_matrices, (i, j)
                    )

                    if similarity >= threshold:
                        current_group.add(code2)
                        processed_indices.add(j)

                problem_groups[i] = current_group

        else:
            # Original single method logic
            if method in ['tfidf', 'transformer']:
                texts = result_df['Problem Description'].tolist()
                if method == 'tfidf':
                    similarity_matrix = self.calculate_tfidf_similarity(texts)
                else:
                    similarity_matrix = self.calculate_transformer_similarity(texts)

            problem_groups = defaultdict(set)
            processed_indices = set()

            for i, row1 in df.iterrows():
                if i in processed_indices:
                    continue

                desc1 = row1['Problem Description']
                code1 = row1['Resolution Code']
                current_group = {code1}

                for j, row2 in df.iloc[i + 1:].iterrows():
                    desc2 = row2['Problem Description']
                    code2 = row2['Resolution Code']

                    if method == 'fuzzy':
                        similarity = self.calculate_fuzzy_similarity(desc1, desc2)
                    else:
                        similarity = similarity_matrix[i, j]

                    if similarity >= threshold:
                        current_group.add(code2)
                        processed_indices.add(j)

                problem_groups[i] = current_group

        # Assign resolution codes
        grouped_codes = []
        for i, row in df.iterrows():
            if i in problem_groups:
                grouped_codes.append(sorted(list(problem_groups[i])))
            else:
                found_group = False
                for group_idx, codes in problem_groups.items():
                    if isinstance(method, dict):
                        similarity = self.calculate_hybrid_similarity(
                            row['Problem Description'],
                            df.loc[group_idx, 'Problem Description'],
                            method,
                            similarity_matrices,
                            (i, group_idx)
                        )
                    elif method == 'fuzzy':
                        similarity = self.calculate_fuzzy_similarity(
                            row['Problem Description'],
                            df.loc[group_idx, 'Problem Description']
                        )
                    else:
                        similarity = similarity_matrix[i, group_idx]

                    if similarity >= threshold:
                        grouped_codes.append(sorted(list(codes)))
                        found_group = True
                        break
                if not found_group:
                    grouped_codes.append([row['Resolution Code']])

        result_df['Resolution Codes'] = grouped_codes
        return result_df[['Problem Description', 'processed_description', 'Resolution Code', 'Resolution Codes']]


def compare_methods(df, thresholds=[0.85], preprocessing_methods=['basic', 'advanced'],
                    hybrid_configs=None):
    """
    Compare different similarity methods, preprocessing approaches, and hybrid configurations.

    Args:
        df: Input DataFrame
        thresholds: List of threshold values to try
        preprocessing_methods: List of preprocessing methods to try
        hybrid_configs: List of dictionaries containing hybrid method configurations
                       e.g., [{'fuzzy': 0.3, 'transformer': 0.7}, ...]
    """
    methods = ['fuzzy', 'tfidf', 'transformer']
    results = defaultdict(list)

    # Add hybrid configurations if provided
    if hybrid_configs is None:
        hybrid_configs = []

    for prep_method in preprocessing_methods:
        matcher = SimilarityMatcher(preprocessing_method=prep_method)

        # Test individual methods
        for method in methods:
            for threshold in thresholds:
                print(f"\nRunning {method} method with {prep_method} preprocessing, threshold {threshold}:")

                start_time = time.time()
                result_df = matcher.group_problems(df, method=method, threshold=threshold)
                execution_time = time.time() - start_time

                avg_group_size = np.mean([len(codes) for codes in result_df['Resolution Codes']])
                num_groups = len(set(tuple(codes) for codes in result_df['Resolution Codes']))

                results['Preprocessing'].append(prep_method)
                results['Method'].append(method)
                results['Threshold'].append(threshold)
                results['Execution Time (s)'].append(execution_time)
                results['Avg Group Size'].append(avg_group_size)
                results['Num Groups'].append(num_groups)

                print(f"\nSample groupings ({method}, {prep_method} preprocessing, threshold={threshold}):")
                print(result_df.head())
                print(f"\nExecution time: {execution_time:.2f}s")
                print(f"Average group size: {avg_group_size:.2f}")
                print(f"Number of unique groups: {num_groups}")

        # Test hybrid configurations
        for hybrid_config in hybrid_configs:
            config_name = '+'.join([f"{k}:{v}" for k, v in hybrid_config.items()])

            for threshold in thresholds:
                print(
                    f"\nRunning hybrid method ({config_name}) with {prep_method} preprocessing, threshold {threshold}:")

                start_time = time.time()
                result_df = matcher.group_problems(df, method=hybrid_config, threshold=threshold)
                execution_time = time.time() - start_time

                avg_group_size = np.mean([len(codes) for codes in result_df['Resolution Codes']])
                num_groups = len(set(tuple(codes) for codes in result_df['Resolution Codes']))

                results['Preprocessing'].append(prep_method)
                results['Method'].append(f"hybrid_{config_name}")
                results['Threshold'].append(threshold)
                results['Execution Time (s)'].append(execution_time)
                results['Avg Group Size'].append(avg_group_size)
                results['Num Groups'].append(num_groups)

                print(f"\nSample groupings (hybrid_{config_name}, {prep_method} preprocessing, threshold={threshold}):")
                print(result_df.head())
                print(f"\nExecution time: {execution_time:.2f}s")
                print(f"Average group size: {avg_group_size:.2f}")
                print(f"Number of unique groups: {num_groups}")

    return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    # Sample data
    data = {
        'Problem Description': [
            'System crash during startup',
            'System crashes on boot',
            'Network connection lost',
            'Unable to connect to network',
            'Printer not responding',
            'System crashes when booting up',
            'Network connectivity issues'
        ],
        'Resolution Code': ['CR001', 'CR002', 'NT001', 'NT002', 'PR001', 'CR003', 'NT003']
    }
    df = pd.DataFrame(data)

    # Define hybrid configurations to test
    hybrid_configs = [
        {'fuzzy': 0.3, 'transformer': 0.7},  # Fuzzy + Transformer
        {'fuzzy': 0.2, 'tfidf': 0.3, 'transformer': 0.5},  # All three methods
        {'tfidf': 0.4, 'transformer': 0.6}  # TF-IDF + Transformer
    ]

    # Compare all methods including hybrid approaches
    comparison_results = compare_methods(
        df,
        thresholds=[0.90],
        preprocessing_methods=['basic', 'advanced'],
        hybrid_configs=hybrid_configs
    )

    print("\nMethod Comparison Summary:")
    # Display full DataFrame without truncation
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None,
                           'display.expand_frame_repr', False,
                           'display.precision', 3):
        print(comparison_results)

    # Save results to CSV for easier viewing if needed
    comparison_results.to_csv('method_comparison_results.csv', index=False)
    print("\nResults have been saved to 'method_comparison_results.csv'")
