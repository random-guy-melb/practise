import spacy
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import re, pickle, os
from typing import Dict, List


class DependencyGraphClassifier:
    def __init__(self, spacy_model='en_core_web_sm', transformer_model='all-MiniLM-L6-v2'):
        # Load SpaCy model for dependency parsing
        self.spacy_model_name = 'en_core_web_sm'
        self.nlp = spacy.load(spacy_model)

        # Load sentence transformer for semantic embeddings
        self.transformer_model_name = 'all-MiniLM-L6-v2'
        self.sentence_transformer = SentenceTransformer(transformer_model)

        # Initialize one-class SVM for outlier detection
        self.svm = OneClassSVM(kernel='rbf', nu=0.1)
        self.scaler = StandardScaler()

        # Initialize patterns dictionary for IT-specific entity recognition
        self.patterns = {
            'DEVICE_ID': [
                (r'ACO\s*\d+(?:,\d+)?', 'ACO ID'),
                (r'LANE\s*\d+(?:\s*,\s*\d+)*', 'Lane ID'),
                (r'(?:LANE|2LS)\s*\d+', 'Lane ID'),
                (r'S[VSOW]\d+SC\d+', 'Scale ID'),
                (r'SQ\d+(?:BP|NW|IS)\d+', 'Workstation ID'),  # Added NW/IS variants
                (r'SN\d+NW\d+', 'Network Device ID')
            ],
            'LOCATION': [
                (r'CS\d+(?:Co)?', 'Store Code'),
                (r'CS\d+\s+[A-Za-z\s]+', 'Store Location'),  # e.g., CS4786 Wanniassa
                (r'\d+\s+[A-Za-z\s]+(?:Mall|City)', 'Store Location'),
                (r'[A-Za-z]+\s+City', 'City')
            ],
            'NETWORK': [
                (r'(?:IP address|IP):\s*(?:\d{1,3}\.){3}\d{1,3}', 'IP Address'),
                (r'(?:\d{1,3}\.){3}\d{1,3}', 'IP Address'),
                (r'Suspicious traffic', 'Security Alert'),
                (r'Offline', 'Network Status')
            ],
            'SYSTEM': [
                (r'SMKTS\s+BOS\s+Workstation', 'BOS System'),
                (r'Supply\s+Chain', 'Supply Chain'),
                (r'WMS', 'Warehouse System'),
                (r'EFT\s+Operator', 'Payment System'),
                (r'BOS', 'BOS System')
            ],
            'DEPARTMENT': [
                (r'Bakery', 'Department'),
                (r'Deli', 'Department'),
                (r'Supermarket', 'Department'),
                (r'COL', 'Department')
            ],
            'ERROR_TYPE': [
                (r'unhandled\s+exception', 'System Error'),
                (r'Missing\s+ANS\s+number', 'Data Error'),
                (r'Power\s+outage', 'Power Issue'),
                (r'Payments\s+not\s+going\s+through', 'Payment Error'),
                (r'payment\s+cannot\s+be\s+processed', 'Payment Error'),
                (r'Order\s+payment', 'Payment Process')
            ],
            'STATUS': [
                (r'\[.*?\]', 'Status Tag'),
                (r'(?:Not Trading|Trading|Non-Operational)', 'Operational Status'),
                (r'offline', 'Network Status')
            ],
            'ERROR_CODE': [
                (r'\b\d{4,6}\b(?!\s*(?:Mall|South|North|East|West))', 'Numeric Code')
            ],
            'EQUIPMENT': [
                (r'(?:Scale|Printer|UPS|POS|Software|Camera)', 'Equipment Type'),
                (r'(?:Bakery\s+Scale|bakery\s+scale)', 'Bakery Scale')
            ],
            'ISSUE_TYPE': [
                (r'non\s*operational', 'Operational Status'),
                (r'cannot\s+read', 'Error Type'),
                (r'piece\s+of\s+plastic\s+cam\s+off', 'Physical Issue'),
                (r'printer\s+says', 'Printer Issue')
            ]
        }

        # Store training data
        self.fitted = False
        self.training_texts = None

    def extract_graph_features(self, doc) -> np.ndarray:
        """Extract features from dependency parse graph"""
        # Create dependency graph
        edges = []
        edge_types = []
        for token in doc:
            edges.append((token.i, token.head.i))
            edge_types.append(token.dep_)

        G = nx.Graph(edges)

        # Extract graph-theoretical features
        graph_features = []

        # Basic graph metrics
        try:
            graph_features.extend([
                nx.average_clustering(G),
                nx.degree_pearson_correlation_coefficient(G),
                len(list(nx.connected_components(G))),
                nx.average_shortest_path_length(G) if nx.is_connected(G) else 0,
                G.number_of_edges() / G.number_of_nodes(),  # Edge density
                len(set(edge_types))  # Unique dependency types
            ])
        except:
            # Fallback values if graph metrics calculation fails
            graph_features.extend([0, 0, 1, 0, 0, 0])

        # Dependency-specific features
        dep_counts = {}
        for dep in edge_types:
            dep_counts[dep] = dep_counts.get(dep, 0) + 1

        # Common dependency types in error messages
        key_deps = ['nsubj', 'dobj', 'prep', 'compound', 'amod']
        for dep in key_deps:
            graph_features.append(dep_counts.get(dep, 0))

        return np.array(graph_features)

    def extract_pattern_features(self, text: str) -> np.ndarray:
        """Extract features based on IT-specific patterns"""
        pattern_features = []

        for category, patterns in self.patterns.items():
            category_matches = 0
            for pattern, _ in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    category_matches += 1
            pattern_features.append(category_matches)

        return np.array(pattern_features)

    def extract_features(self, text: str) -> np.ndarray:
        # Parse text using SpaCy
        doc = self.nlp(text)

        # Get graph features from dependency parse
        graph_features = self.extract_graph_features(doc)

        # Get pattern-based features
        pattern_features = self.extract_pattern_features(text)

        # Get semantic embeddings
        semantic_features = self.sentence_transformer.encode(text)

        # Additional structural features
        structural_features = np.array([
            len(text),  # Text length
            len(doc),  # Token count
            len([token for token in doc if token.is_punct]),  # Punctuation count
            len([token for token in doc if token.like_num]),  # Number count
            len([token for token in doc if token.is_upper]),  # Uppercase token count
            len([token for token in doc if token.pos_ == 'VERB']),  # Verb count
            len([token for token in doc if token.pos_ == 'NOUN'])  # Noun count
        ])

        # Combine all features
        return np.concatenate([
            graph_features,
            pattern_features,
            semantic_features,
            structural_features
        ])

    def fit(self, texts: List[str]) -> None:
        """Train the classifier on normal examples"""
        # Extract features for all texts
        features = np.vstack([self.extract_features(text) for text in texts])

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit one-class SVM
        self.svm.fit(features_scaled)
        self.fitted = True

    def predict(self, text: str) -> int:
        """Predict if a text is an outlier (-1) or normal (1)"""
        features = self.extract_features(text)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.svm.predict(features_scaled)[0]

    def save(self, path: str) -> None:
        """Save model to disk

        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Create a dictionary of components to save
        save_dict = {
            'spacy_model_name': self.spacy_model_name,
            'transformer_model_name': self.transformer_model_name,
            'svm': self.svm,
            'scaler': self.scaler,
            'patterns': self.patterns,
            'fitted': self.fitted,
            'training_texts': self.training_texts
        }

        # Save to disk
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str) -> 'DependencyGraphClassifier':
        """Load model from disk

        Args:
            path: Path to load the model from

        Returns:
            DependencyGraphClassifier: Loaded classifier
        """
        # Load components from disk
        with open(path, 'rb') as f:
            components = pickle.load(f)

        # Create new instance with saved model names
        instance = cls(
            spacy_model=components['spacy_model_name'],
            transformer_model=components['transformer_model_name']
        )

        # Restore components
        instance.svm = components['svm']
        instance.scaler = components['scaler']
        instance.patterns = components['patterns']
        instance.fitted = components['fitted']
        instance.training_texts = components['training_texts']

        return instance

    def analyze_structure(self, text: str) -> Dict:
        """Analyze the syntactic structure of the text"""
        doc = self.nlp(text)

        analysis = {
            'dependency_structure': [],
            'key_phrases': [],
            'entities': []
        }

        # Extract dependency relationships
        for token in doc:
            if token.dep_ != 'punct':
                analysis['dependency_structure'].append({
                    'token': token.text,
                    'dependency': token.dep_,
                    'head': token.head.text
                })

        # Extract key phrases (noun phrases and verb phrases)
        for chunk in doc.noun_chunks:
            analysis['key_phrases'].append({
                'text': chunk.text,
                'type': 'noun_phrase'
            })

        # Extract IT-specific entities
        for category, patterns in self.patterns.items():
            for pattern, label in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    analysis['entities'].append({
                        'text': match.group(),
                        'type': label,
                        'category': category
                    })

        return analysis


if __name__ == "__main__":
    # Initialize classifier
    classifier = DependencyGraphClassifier()

    # Training data
    normal_incidents = [
        'ACO130,132 - SMKTS ACO Software Stuck on Error Message - "Unhandled Exception has Occurred" - [Not Trading ]',
        'LANE123 â€" Cash Management â€" not registering cash declaration â€"[Not Trading]',
        "Bakery Scale- SV7501SC401- not printing correctly - [non-operational]"
    ]

    # Train the classifier
    classifier.fit(normal_incidents)

    classifier.save('saved_models/model.pkl')

    # Load a model
    loaded_classifier = DependencyGraphClassifier.load('saved_models/model.pkl')

    # Test new incident
    new_incident = 'LANE114 - Software - "Stuck on Enter ID screen - [Not Trading ]'
    result = loaded_classifier.predict(new_incident)

    # Analyze structure
    analysis = loaded_classifier.analyze_structure(new_incident)

    print("Prediction (1=normal, -1=outlier):", result)
    print("\nStructural Analysis:")
    print(analysis)
