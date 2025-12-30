"""
Training script for the line classifier.
Run: python -m model.train
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from .features import features_to_vector, extract_features
from .classifier import LineClassifier

def load_dataset(path: str) -> tuple:
    """Load JSONL dataset returning (X, y). Computes features if missing."""
    X, y = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Use existing features or compute them from text
            if 'entropy' in item:
                features = {
                    'entropy': item.get('entropy', 0),
                    'digit_ratio': item.get('digit_ratio', 0),
                    'punctuation_ratio': item.get('punctuation_ratio', 0),
                    'leading_whitespace_ratio': item.get('leading_whitespace_ratio', 0),
                    'repetition_ratio': item.get('repetition_ratio', 0),
                    'avg_word_length': item.get('avg_word_length', 0),
                    'long_token_ratio': item.get('long_token_ratio', 0),
                    'shape_entropy': item.get('shape_entropy', 0),
                    'shape_symbol_ratio': item.get('shape_symbol_ratio', 0),
                }
            elif 'text' in item:
                features = extract_features(item['text'])
            else:
                continue # Skip if no features and no text
                
            X.append(features_to_vector(features))
            y.append(item['label'])
    return np.array(X), y


def train(dataset_path: str = None, output_path: str = None, test_size: float = 0.2):
    """Train model with 80/20 train/validation split."""
    model_dir = Path(__file__).parent
    
    if dataset_path is None:
        dataset_path = model_dir.parent / 'dataset' / 'full-set.jsonl'
    if output_path is None:
        output_path = model_dir / 'trained_model.pkl'
    
    print(f"Loading: {dataset_path}")
    X, y = load_dataset(str(dataset_path))
    print(f"Samples: {len(y)}")
    
    # 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(y_train)}, Validation: {len(y_val)}")
    
    # Train
    classifier = LineClassifier.__new__(LineClassifier)
    classifier.model = None
    classifier.label_encoder = classifier.__class__.__dict__['LABELS']
    from sklearn.preprocessing import LabelEncoder
    classifier.label_encoder = LabelEncoder()
    classifier.label_encoder.fit(LineClassifier.LABELS)
    
    classifier.train(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.label_encoder.inverse_transform(classifier.model.predict(X_val))
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save
    classifier.save(str(output_path))
    print(f"Model saved: {output_path}")
    
    return classifier, accuracy


if __name__ == '__main__':
    train()
