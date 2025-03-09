# Named Entity Recognition (NER) Model Comparison
# Conditional Random Fields (CRF) vs spaCy

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import spacy
from spacy.tokens import Doc
from collections import Counter
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model: en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 1. Load the CoNLL-2003 dataset
logger.info("Loading CoNLL-2003 dataset")
dataset = load_dataset("conll2003")
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

logger.info(f"Dataset loaded successfully:")
logger.info(f"- Train set: {len(train_dataset)} examples")
logger.info(f"- Validation set: {len(validation_dataset)} examples")
logger.info(f"- Test set: {len(test_dataset)} examples")

# Display sample from the dataset
sample_idx = 0
print("\nSample from the dataset:")
print(f"Tokens: {train_dataset[sample_idx]['tokens']}")
print(f"NER tags: {train_dataset[sample_idx]['ner_tags']}")
print(f"NER tags (mapped): {[dataset.features['ner_tags'].feature.names[tag] for tag in train_dataset[sample_idx]['ner_tags']]}")

# 2. Feature Extraction for CRF
def word2features(sent, i):
    """
    Extract features for a given token in a sentence.
    
    Args:
        sent (list): List of tokens in a sentence
        i (int): Index of the current token
        
    Returns:
        dict: Features for the token
    """
    word = sent[i]
    
    # Basic features
    features = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word[-3:]': word[-3:] if len(word) >= 3 else word,
        'word[-2:]': word[-2:] if len(word) >= 2 else word,
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
    }
    
    # Add POS tag and dependency features using spaCy
    doc = nlp(' '.join(sent))
    features['pos'] = doc[i].pos_ if i < len(doc) else 'NONE'
    features['dep'] = doc[i].dep_ if i < len(doc) else 'NONE'
    
    # Features for previous token
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower': word1.lower(),
            '-1:word.istitle': word1.istitle(),
            '-1:word.isupper': word1.isupper(),
            '-1:pos': doc[i-1].pos_ if i-1 < len(doc) else 'NONE',
            '-1:dep': doc[i-1].dep_ if i-1 < len(doc) else 'NONE',
        })
    else:
        features['BOS'] = True
    
    # Features for next token
    if i < len(sent) - 1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower': word1.lower(),
            '+1:word.istitle': word1.istitle(),
            '+1:word.isupper': word1.isupper(),
            '+1:pos': doc[i+1].pos_ if i+1 < len(doc) else 'NONE',
            '+1:dep': doc[i+1].dep_ if i+1 < len(doc) else 'NONE',
        })
    else:
        features['EOS'] = True
    
    return features

def sent2features(sent):
    """Convert a sentence to a list of features."""
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent_labels, tag_names):
    """Convert numeric labels to BIO tag strings."""
    return [tag_names[label] for label in sent_labels]

# 3. Prepare data for CRF
logger.info("Preparing data for CRF model")

# Get tag names from dataset
tag_names = dataset.features['ner_tags'].feature.names
print(f"\nNER tag names: {tag_names}")

# Function to prepare dataset for CRF
def prepare_data_for_crf(dataset_split, tag_names, max_samples=None):
    """
    Prepare features and labels from a dataset split for CRF training.
    
    Args:
        dataset_split: Dataset split (train, validation, test)
        tag_names: List of tag names
        max_samples: Maximum number of samples to process (for debugging)
        
    Returns:
        tuple: (X_features, y_labels)
    """
    X = []
    y = []
    
    samples = dataset_split if max_samples is None else dataset_split[:max_samples]
    
    for i, sample in enumerate(samples):
        try:
            # Process tokens and labels
            tokens = sample['tokens']
            ner_tags = sample['ner_tags']
            
            # Extract features and convert labels
            X.append(sent2features(tokens))
            y.append(sent2labels(ner_tags, tag_names))
            
            # Print progress
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
    
    return X, y

# Process smaller subsets for demonstration purposes
# In a real scenario, you would use the full dataset
max_train_samples = 1000  # Limit for demonstration
max_val_samples = 200
max_test_samples = 200

X_train, y_train = prepare_data_for_crf(train_dataset, tag_names, max_train_samples)
X_val, y_val = prepare_data_for_crf(validation_dataset, tag_names, max_val_samples)
X_test, y_test = prepare_data_for_crf(test_dataset, tag_names, max_test_samples)

logger.info(f"Data prepared for CRF:")
logger.info(f"- Training set: {len(X_train)} sentences")
logger.info(f"- Validation set: {len(X_val)} sentences")
logger.info(f"- Test set: {len(X_test)} sentences")

# 4. Train CRF model
logger.info("Training CRF model")

# Initialize CRF model
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# Train model
crf.fit(X_train, y_train)

# 5. Evaluate CRF model
logger.info("Evaluating CRF model")

# Make predictions
y_pred = crf.predict(X_test)

# Print evaluation metrics
print("\nCRF Model Evaluation:")
print(metrics.flat_classification_report(y_test, y_pred, labels=list(set(tag for tags in y_test for tag in tags) - {'O'})))

# Calculate metrics
crf_accuracy = accuracy_score(y_test, y_pred)
crf_precision = precision_score(y_test, y_pred)
crf_recall = recall_score(y_test, y_pred)
crf_f1 = f1_score(y_test, y_pred)

print(f"CRF Model Metrics:")
print(f"- Accuracy: {crf_accuracy:.4f}")
print(f"- Precision: {crf_precision:.4f}")
print(f"- Recall: {crf_recall:.4f}")
print(f"- F1 Score: {crf_f1:.4f}")

# 6. Evaluate spaCy NER
logger.info("Evaluating spaCy NER model")

def convert_conll_to_spacy_format(tokens, ner_tags, tag_names):
    """Convert CoNLL format to spaCy format for evaluation."""
    entities = []
    current_entity = None
    
    for i, (token, tag_idx) in enumerate(zip(tokens, ner_tags)):
        tag = tag_names[tag_idx]
        
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'start': i, 'end': i + 1, 'label': tag[2:]}
        elif tag.startswith('I-') and current_entity and current_entity['label'] == tag[2:]:
            current_entity['end'] = i + 1
        elif tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def evaluate_spacy_ner(test_dataset, tag_names, max_samples=None):
    """Evaluate spaCy NER on the test dataset."""
    true_entities_list = []
    pred_entities_list = []
    
    samples = test_dataset if max_samples is None else test_dataset[:max_samples]
    
    for i, sample in enumerate(samples):
        try:
            tokens = sample['tokens']
            text = ' '.join(tokens)
            ner_tags = sample['ner_tags']
            
            # Get ground truth entities
            true_entities = convert_conll_to_spacy_format(tokens, ner_tags, tag_names)
            
            # Get spaCy predictions
            doc = nlp(text)
            pred_entities = []
            for ent in doc.ents:
                # Map spaCy entity types to CoNLL types
                label = map_spacy_to_conll(ent.label_)
                if label:
                    # Find token indices for this entity
                    start = -1
                    end = -1
                    for j, token in enumerate(tokens):
                        if ent.start_char <= sum(len(t) + 1 for t in tokens[:j+1]):
                            if start == -1:
                                start = j
                        if ent.end_char <= sum(len(t) + 1 for t in tokens[:j+1]):
                            end = j + 1
                            break
                    
                    if start != -1 and end != -1:
                        pred_entities.append({'start': start, 'end': end, 'label': label})
            
            # Convert to BIO format
            true_bio = ['O'] * len(tokens)
            pred_bio = ['O'] * len(tokens)
            
            for ent in true_entities:
                true_bio[ent['start']] = f"B-{ent['label']}"
                for j in range(ent['start'] + 1, ent['end']):
                    true_bio[j] = f"I-{ent['label']}"
            
            for ent in pred_entities:
                if ent['start'] < len(pred_bio):
                    pred_bio[ent['start']] = f"B-{ent['label']}"
                    for j in range(ent['start'] + 1, min(ent['end'], len(pred_bio))):
                        pred_bio[j] = f"I-{ent['label']}"
            
            true_entities_list.append(true_bio)
            pred_entities_list.append(pred_bio)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples for spaCy evaluation")
                
        except Exception as e:
            logger.error(f"Error evaluating sample {i} with spaCy: {e}")
    
    return true_entities_list, pred_entities_list

def map_spacy_to_conll(spacy_label):
    """Map spaCy entity types to CoNLL types."""
    mapping = {
        'PERSON': 'PER',
        'ORG': 'ORG',
        'GPE': 'LOC',  # GPE (countries, cities, etc.) maps to LOC in CoNLL
        'LOC': 'LOC',
        'PRODUCT': 'MISC',
        'EVENT': 'MISC',
        'WORK_OF_ART': 'MISC',
        'LANGUAGE': 'MISC',
        'FAC': 'LOC',  # Facilities often map to LOC
        'NORP': 'MISC'  # Nationalities, religious and political groups
    }
    return mapping.get(spacy_label, None)

# Evaluate spaCy on test set
true_entities, spacy_predictions = evaluate_spacy_ner(test_dataset, tag_names, max_test_samples)

# Print evaluation metrics for spaCy
print("\nspaCy NER Model Evaluation:")
print(classification_report(true_entities, spacy_predictions))

# Calculate metrics for spaCy
spacy_accuracy = accuracy_score(true_entities, spacy_predictions)
spacy_precision = precision_score(true_entities, spacy_predictions)
spacy_recall = recall_score(true_entities, spacy_predictions)
spacy_f1 = f1_score(true_entities, spacy_predictions)

print(f"spaCy NER Model Metrics:")
print(f"- Accuracy: {spacy_accuracy:.4f}")
print(f"- Precision: {spacy_precision:.4f}")
print(f"- Recall: {spacy_recall:.4f}")
print(f"- F1 Score: {spacy_f1:.4f}")

# 7. Compare models
print("\nModel Comparison:")
print(f"{'Metric':<10} {'CRF':<10} {'spaCy':<10}")
print(f"{'-'*30}")
print(f"{'Accuracy':<10} {crf_accuracy:.4f}     {spacy_accuracy:.4f}")
print(f"{'Precision':<10} {crf_precision:.4f}     {spacy_precision:.4f}")
print(f"{'Recall':<10} {crf_recall:.4f}     {spacy_recall:.4f}")
print(f"{'F1 Score':<10} {crf_f1:.4f}     {spacy_f1:.4f}")

# 8. Visualize comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
crf_scores = [crf_accuracy, crf_precision, crf_recall, crf_f1]
spacy_scores = [spacy_accuracy, spacy_precision, spacy_recall, spacy_f1]

# Set up plot
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(metrics_names))

# Create bars
plt.bar(index, crf_scores, bar_width, label='CRF')
plt.bar(index + bar_width, spacy_scores, bar_width, label='spaCy')

# Customize plot
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('NER Models Comparison (CRF vs spaCy)')
plt.xticks(index + bar_width / 2, metrics_names)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, v in enumerate(crf_scores):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

for i, v in enumerate(spacy_scores):
    plt.text(i + bar_width, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('ner_model_comparison.png')
plt.show()

# 9. Feature importance for CRF
top_features = 20
logger.info(f"Extracting top {top_features} features from CRF model")

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print(f"{label:6} {attr:35} {weight:.6f}")

print("\nTop positive features:")
print_state_features(Counter(crf.state_features_).most_common(top_features))

print("\nTop negative features:")
print_state_features(sorted(
    Counter(crf.state_features_).items(),
    key=lambda x: x[1],
    reverse=False
)[:top_features])

# 10. Save CRF model for later use
import pickle
import os

# Create output directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save CRF model
with open('models/crf_ner_model.pkl', 'wb') as f:
    pickle.dump(crf, f)

logger.info("CRF model saved to models/crf_ner_model.pkl")

# 11. Usage example with the trained CRF model
print("\nExample usage of trained CRF model:")

def extract_features_for_text(text):
    """Extract features for all tokens in a text."""
    tokens = text.split()
    return sent2features(tokens)

def predict_entities_with_crf(text, crf_model, tag_names):
    """Predict entities in text using the trained CRF model."""
    features = extract_features_for_text(text)
    predictions = crf_model.predict([features])[0]
    
    entities = []
    current_entity = None
    
    for i, (token, tag) in enumerate(zip(text.split(), predictions)):
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {'text': token, 'type': tag[2:], 'start': i}
        elif tag.startswith('I-') and current_entity and current_entity['type'] == tag[2:]:
            current_entity['text'] += ' ' + token
        elif tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

# Example text
example_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
print(f"Text: {example_text}")

# Predict with CRF
crf_entities = predict_entities_with_crf(example_text, crf, tag_names)
print("CRF entities:")
for entity in crf_entities:
    print(f"- {entity['text']} ({entity['type']})")

# Predict with spaCy
doc = nlp(example_text)
print("spaCy entities:")
for entity in doc.ents:
    print(f"- {entity.text} ({entity.label_})")

print("\nThis demonstrates how to train a CRF model for NER and compare it with spaCy's pre-trained model.")