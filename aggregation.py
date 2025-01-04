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

# NLTK setup
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
        self.tfidf_vectorizer = TfidfVectorizer()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.preprocessing_method = preprocessing_method

    def preprocess_text(self, text):
        if self.preprocessing_method == 'advanced':
            return advanced_preprocess(text)
        return basic_preprocess(text)

    def calculate_fuzzy_similarity(self, text1, text2):
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
        processed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        return cosine_similarity(tfidf_matrix)

    def calculate_transformer_similarity(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        embeddings = self.sentence_transformer.encode(processed_texts)
        return cosine_similarity(embeddings)

    def calculate_hybrid_similarity(self, text1, text2, methods, similarity_matrices=None, indices=None):
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

        weights = np.array(weights) / sum(weights)
        return np.average(similarity_scores, weights=weights)

    def group_problems(self, df, method='fuzzy', threshold=0.85):
        """
        Group similar problems based on Short Description and Resolution Type.

        Args:
            df: DataFrame with 'Short Description' and 'Resolution Type' columns
            method: Similarity method or hybrid configuration
            threshold: Similarity threshold

        Returns:
            DataFrame with grouped resolution types and matching descriptions
        """
        result_df = df.copy()
        result_df['processed_description'] = df['Short Description'].apply(self.preprocess_text)

        # Initialize tracking dictionaries
        similarity_matrices = {}
        problem_groups = defaultdict(set)
        matching_details = defaultdict(list)  # Track matching descriptions and their indices
        processed_indices = set()

        # Pre-compute similarity matrices for hybrid method
        if isinstance(method, dict):
            texts = result_df['Short Description'].tolist()
            if 'tfidf' in method:
                similarity_matrices['tfidf'] = self.calculate_tfidf_similarity(texts)
            if 'transformer' in method:
                similarity_matrices['transformer'] = self.calculate_transformer_similarity(texts)

        # Main grouping logic
        for i, row1 in df.iterrows():
            if i in processed_indices:
                continue

            desc1 = row1['Short Description']
            type1 = row1['Resolution Type']
            current_group = {type1}
            current_matches = []  # Track matches for current description

            for j, row2 in df.iloc[i + 1:].iterrows():
                desc2 = row2['Short Description']
                type2 = row2['Resolution Type']

                # Calculate similarity based on method
                if isinstance(method, dict):
                    similarity = self.calculate_hybrid_similarity(
                        desc1, desc2, method, similarity_matrices, (i, j)
                    )
                elif method == 'fuzzy':
                    similarity = self.calculate_fuzzy_similarity(desc1, desc2)
                else:
                    similarity = similarity_matrices.get(method,
                                                         self.calculate_tfidf_similarity([desc1, desc2]))[0, 1]

                if similarity >= threshold:
                    current_group.add(type2)
                    processed_indices.add(j)
                    current_matches.append({
                        'index': j,
                        'description': desc2,
                        'similarity': similarity
                    })

            problem_groups[i] = current_group
            matching_details[i] = current_matches

        # Create output columns
        grouped_types = []
        matching_info = []

        for i, row in df.iterrows():
            if i in problem_groups:
                grouped_types.append(sorted(list(problem_groups[i])))
                matches = matching_details[i]
                match_info = [
                    f"Index {m['index']}: '{m['description']}' (similarity: {m['similarity']:.2f})"
                    for m in matches
                ]
                matching_info.append(match_info if match_info else ['No matches'])
            else:
                grouped_types.append([row['Resolution Type']])
                matching_info.append(['No matches'])

        result_df['Grouped Resolution Types'] = grouped_types
        result_df['Matching Descriptions'] = matching_info

        return result_df[['Short Description', 'processed_description', 'Resolution Type',
                          'Grouped Resolution Types', 'Matching Descriptions']]


def compare_methods(df, save_path,
                    thresholds=[0.85],
                    preprocessing_methods=['basic', 'advanced'],
                    hybrid_configs=None,
                    methods=['fuzzy', 'tfidf', 'transformer']):
    """
    Compare different similarity methods with the updated column names.

    Args:
        df: DataFrame with 'Short Description' and 'Resolution Type' columns
        thresholds: List of threshold values
        preprocessing_methods: List of preprocessing methods
        hybrid_configs: List of hybrid method configurations
    """

    results = defaultdict(list)

    for prep_method in preprocessing_methods:
        matcher = SimilarityMatcher(preprocessing_method=prep_method)

        # Test individual methods
        for method in methods:
            for threshold in thresholds:
                print(f"\nRunning {method} method with {prep_method} preprocessing, threshold {threshold}:")

                start_time = time.time()
                result_df = matcher.group_problems(df, method=method, threshold=threshold)
                execution_time = time.time() - start_time

                avg_group_size = np.mean([len(codes) for codes in result_df['Grouped Resolution Types']])
                num_groups = len(set(tuple(codes) for codes in result_df['Grouped Resolution Types']))

                results['Preprocessing'].append(prep_method)
                results['Method'].append(method)
                results['Threshold'].append(threshold)
                results['Execution Time (s)'].append(execution_time)
                results['Avg Group Size'].append(avg_group_size)
                results['Num Groups'].append(num_groups)

                print(f"\nSample groupings:")
                result_df.to_csv(save_path + f"/{method}_{prep_method}_{threshold}.csv", index=False)
                print(f"Execution time: {execution_time:.2f}s")
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

                avg_group_size = np.mean([len(codes) for codes in result_df['Grouped Resolution Types']])
                num_groups = len(set(tuple(codes) for codes in result_df['Grouped Resolution Types']))

                results['Preprocessing'].append(prep_method)
                results['Method'].append(f"hybrid_{config_name}")
                results['Threshold'].append(threshold)
                results['Execution Time (s)'].append(execution_time)
                results['Avg Group Size'].append(avg_group_size)
                results['Num Groups'].append(num_groups)

                print(f"\nSample groupings:")
                config_name_replaced = config_name.replace(":", "")
                result_df.to_csv(save_path + f"/{config_name_replaced}_{prep_method}_{threshold}.csv", index=False)
                print(f"Execution time: {execution_time:.2f}s")
                print(f"Average group size: {avg_group_size:.2f}")
                print(f"Number of unique groups: {num_groups}")

    return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    # Sample data
    data = {
        'Short Description': [
            'System crash during startup',
            'System crashes on boot',
            'Network connection lost',
            'Unable to connect to network',
            'Printer not responding',
            'System crashes when booting up',
            'Network connectivity issues'
        ],
        'Resolution Type': ['CR001', 'CR002', 'NT001', 'NT002', 'PR001', 'CR003', 'NT003']
    }
    df = pd.DataFrame(data)

    hybrid_configs = [
        {'fuzzy': 0.3, 'transformer': 0.7},
        # {'fuzzy': 0.2, 'tfidf': 0.3, 'transformer': 0.5},
        # {'tfidf': 0.4, 'transformer': 0.6}
    ]

    methods = [(conf.keys()) for conf in hybrid_configs]
    methods = sorted(methods, key=lambda x: -len(x))[0]

    # Run comparison with different methods and configurations
    comparison_results = compare_methods(
        df,
        save_path="./tmp",
        thresholds=[0.90],
        preprocessing_methods=['basic', 'advanced'],
        hybrid_configs=hybrid_configs,
        methods=methods
    )

    print("\nMethod Comparison Summary:")
    print(comparison_results)
