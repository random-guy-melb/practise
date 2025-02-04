import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc
from spacy.training import Example
import random

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

class TechnicalEntityExtractor:
    """Extract technical entities from incident reports using rule-based and ML approaches"""
    
    def __init__(self):
        """Initialize the extractor with entity patterns"""
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
            ]
            'LOCATION': [
                (r'CE\d+', 'Store Code'),
                (r'\d+\s+\w+\s+(?:Mall|South|North|East|West)', 'Store Location')
            ],
            'ERROR_CODE': [
                (r'\b\d{4,6}\b(?!\s*(?:Mall|South|North|East|West))', 'Numeric Code')
            ],
            'STATUS': [
                (r'\[.*?\]', 'Status Tag'),
                (r'(?:Not Trading|Trading|Non-Operational)', 'Operational Status')
            ],
            'EQUIPMENT': [
                (r'(?:Scale|Printer|UPS|POS|Software|Camera)', 'Equipment Type'),
                (r'(?:Bakery\s+Scale|bakery\s+scale)', 'Bakery Scale')
            ],
            'DEPARTMENT': [
                (r'Bakery', 'Department'),
                (r'Deli', 'Department')
            ],
            'ISSUE_TYPE': [
                (r'non\s*operational', 'Operational Status'),
                (r'cannot\s+read', 'Error Type'),
                (r'piece\s+of\s+plastic\s+cam\s+off', 'Physical Issue'),
                (r'printer\s+says', 'Printer Issue')
            ]
        }
        
        # Initialize spaCy model
        self.nlp = spacy.blank('en')
        
        # Add entity ruler for rule-based matching
        self.setup_entity_ruler()
        
    def setup_entity_ruler(self):
        """Set up the entity ruler component with patterns"""
        ruler = self.nlp.add_pipe('entity_ruler')
        patterns = []
        
        # Convert regex patterns to spaCy patterns
        for label, pattern_list in self.patterns.items():
            for pattern, _ in pattern_list:
                patterns.append({
                    "label": label,
                    "pattern": [{"TEXT": {"REGEX": f"^{pattern}$"}}]
                })
        
        ruler.add_patterns(patterns)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using both rule-based and ML approaches"""
        doc = self.nlp(text)
        entities = []
        
        # Get entities from spaCy pipeline
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))
        
        # Additional regex-based extraction for cases spaCy might miss
        for entity_type, patterns in self.patterns.items():
            for pattern, sublabel in patterns:
                for match in re.finditer(pattern, text):
                    # Check if this span overlaps with existing entities
                    overlap = False
                    for e in entities:
                        if (match.start() < e.end and match.end() > e.start):
                            overlap = True
                            break
                    
                    if not overlap:
                        entities.append(Entity(
                            text=match.group(),
                            label=f"{entity_type}_{sublabel}",
                            start=match.start(),
                            end=match.end()
                        ))
        
        # Sort entities by start position
        entities.sort(key=lambda x: x.start)
        return entities

    def format_entities(self, text: str, entities: List[Entity]) -> str:
        """Format text with highlighted entities for display"""
        result = []
        last_end = 0
        
        for entity in entities:
            # Add text before entity
            result.append(text[last_end:entity.start])
            # Add highlighted entity
            result.append(f"[{entity.text}]({entity.label})")
            last_end = entity.end
        
        # Add remaining text
        result.append(text[last_end:])
        return ''.join(result)

    def prepare_training_data(self, texts: List[str]) -> List[Example]:
        """Prepare training data for fine-tuning spaCy model"""
        training_data = []
        
        for text in texts:
            # Get entities using current rules
            entities = self.extract_entities(text)
            
            # Convert to spaCy's format
            spans = []
            for ent in entities:
                spans.append((ent.start, ent.end, ent.label))
            
            # Create Example
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": spans})
            training_data.append(example)
        
        return training_data

    def train_model(self, texts: List[str], iterations: int = 30):
        """Train the NER model on provided texts"""
        # Configure pipeline
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe('ner')
        
        # Prepare training data
        training_data = self.prepare_training_data(texts)
        
        # Train the model
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for itn in range(iterations):
                random.shuffle(training_data)
                losses = {}
                
                for example in training_data:
                    self.nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
                
                print(f'Iteration {itn+1}, Losses:', losses)

# Example usage
def example():
    # Initialize extractor
    extractor = TechnicalEntityExtractor()
    
    # Example incidents with new patterns
    incidents = []
    
    # Extract and display entities
    print("\nExtracted Entities from New Patterns:")
    for incident in incidents:
        entities = extractor.extract_entities(incident)
        formatted = extractor.format_entities(incident, entities)
        print("\nOriginal:", incident)
        print("Parsed:", formatted)
        print("Entities:")
        for entity in entities:
            print(f"- {entity.label}: {entity.text}")
            
if __name__ == "__main__":
    example()
    
    # Extract and display entities
    print("Extracted Entities:")
    for incident in incidents:
        entities = extractor.extract_entities(incident)
        formatted = extractor.format_entities(incident, entities)
        print("\nOriginal:", incident)
        print("Parsed:", formatted)
        print("\nEntities:")
        for entity in entities:
            print(f"- {entity.label}: {entity.text}")
    
    # Train model with examples
    print("\nTraining model...")
    extractor.train_model(incidents)

if __name__ == "__main__":
    example()
