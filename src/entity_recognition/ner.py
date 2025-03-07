#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Named Entity Recognition (NER) module.
This module provides functions to extract named entities from text.
"""

import os
import pickle
import logging
import spacy
import sklearn_crfsuite
from spacy.tokens import Doc
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Initialize logging
logger = logging.getLogger(__name__)

class NERExtractor:
    """Base class for named entity recognition."""
    
    def __init__(self):
        """Initialize the NER extractor."""
        pass
    
    def extract_entities(self, text):
        """
        Extract entities from text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of (entity_text, entity_type) tuples
        """
        raise NotImplementedError("Subclasses must implement extract_entities")

class SpacyNERExtractor(NERExtractor):
    """Named entity recognition using spaCy."""
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize the spaCy NER extractor.
        
        Args:
            model_name (str): Name of the spaCy model to use
        """
        super().__init__()
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.info(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
    
    def extract_entities(self, text):
        """
        Extract entities from text using spaCy.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of (entity_text, entity_type) tuples
        """
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities with spaCy: {e}")
            return []

class CRFExtractor(NERExtractor):
    """Named entity recognition using Conditional Random Fields (CRF)."""
    
    def __init__(self, model_path=None):
        """
        Initialize the CRF extractor.
        
        Args:
            model_path (str, optional): Path to saved CRF model
        """
        super().__init__()
        self.model = None
        self.nlp = spacy.load("en_core_web_sm")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("No CRF model provided or file not found")
    
    def load_model(self, model_path):
        """
        Load a trained CRF model.
        
        Args:
            model_path (str): Path to saved CRF model
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded CRF model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading CRF model: {e}")
            return False
    
    def save_model(self, model_path):
        """
        Save the trained CRF model.
        
        Args:
            model_path (str): Path to save CRF model
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved CRF model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving CRF model: {e}")
            return False
    
    def _get_features(self, sentence, index):
        """
        Extract features for CRF from a sentence at a specific index.
        
        Args:
            sentence (list): List of tokens
            index (int): Current token index
            
        Returns:
            dict: Features for CRF
        """
        word = sentence[index].text
        
        # Basic features
        features = {
            'bias': 1.0,
            'word.lower': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper': word.isupper(),
            'word.istitle': word.istitle(),
            'word.isdigit': word.isdigit(),
            'pos': sentence[index].pos_,
            'dep': sentence[index].dep_,
        }
        
        # Features for previous token
        if index > 0:
            prev_word = sentence[index-1].text
            features.update({
                '-1:word.lower': prev_word.lower(),
                '-1:word.istitle': prev_word.istitle(),
                '-1:word.isupper': prev_word.isupper(),
                '-1:pos': sentence[index-1].pos_,
                '-1:dep': sentence[index-1].dep_,
            })
        else:
            features['BOS'] = True
        
        # Features for next token
        if index < len(sentence) - 1:
            next_word = sentence[index+1].text
            features.update({
                '+1:word.lower': next_word.lower(),
                '+1:word.istitle': next_word.istitle(),
                '+1:word.isupper': next_word.isupper(),
                '+1:pos': sentence[index+1].pos_,
                '+1:dep': sentence[index+1].dep_,
            })
        else:
            features['EOS'] = True
        
        return features
    
    def _get_features_for_sentence(self, sentence):
        """
        Extract features for a complete sentence.
        
        Args:
            sentence (spacy.tokens.Doc): spaCy document for a sentence
            
        Returns:
            list: List of feature dictionaries
        """
        return [self._get_features(sentence, i) for i in range(len(sentence))]
    
    def _get_labels(self, sentence):
        """
        Extract entity labels for a sentence using BIO format.
        
        Args:
            sentence (spacy.tokens.Doc): spaCy document for a sentence
            
        Returns:
            list: List of entity labels
        """
        labels = ['O'] * len(sentence)
        
        for ent in sentence.ents:
            labels[ent.start] = f'B-{ent.label_}'
            for i in range(ent.start + 1, ent.end):
                labels[i] = f'I-{ent.label_}'
        
        return labels
    
    def load_conll_dataset(self):
        """
        Load the CoNLL-2003 dataset for NER training.
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        try:
            logger.info("Loading CoNLL-2003 dataset")
            dataset = load_dataset("conll2003")
            
            train_texts = [' '.join(tokens) for tokens in dataset['train']['tokens']]
            val_texts = [' '.join(tokens) for tokens in dataset['validation']['tokens']]
            test_texts = [' '.join(tokens) for tokens in dataset['test']['tokens']]
            
            # Process with spaCy for feature extraction
            train_docs = list(self.nlp.pipe(train_texts))
            val_docs = list(self.nlp.pipe(val_texts))
            test_docs = list(self.nlp.pipe(test_texts))
            
            # Convert to features and labels
            X_train = [self._get_features_for_sentence(doc) for doc in train_docs]
            y_train = [self._get_labels(doc) for doc in train_docs]
            
            X_val = [self._get_features_for_sentence(doc) for doc in val_docs]
            y_val = [self._get_labels(doc) for doc in val_docs]
            
            X_test = [self._get_features_for_sentence(doc) for doc in test_docs]
            y_test = [self._get_labels(doc) for doc in test_docs]
            
            logger.info("Dataset loaded successfully")
            return X_train, y_train, X_val, y_val, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None, None, None, None, None, None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the CRF model.
        
        Args:
            X_train (list): Training features
            y_train (list): Training labels
            X_val (list, optional): Validation features
            y_val (list, optional): Validation labels
            **kwargs: Additional parameters for CRF
            
        Returns:
            bool: Success flag
        """
        try:
            # Default parameters for CRF
            params = {
                'algorithm': 'lbfgs',
                'c1': 0.1,
                'c2': 0.1,
                'max_iterations': 100,
                'all_possible_transitions': True
            }
            
            # Update with user-provided parameters
            params.update(kwargs)
            
            logger.info("Training CRF model with parameters: %s", params)
            
            # Initialize and train CRF model
            self.model = sklearn_crfsuite.CRF(**params)
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set if provided
            if X_val and y_val:
                y_pred = self.model.predict(X_val)
                logger.info("Validation completed")
            
            logger.info("CRF model trained successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error training CRF model: {e}")
            return False
    
    def extract_entities(self, text):
        """
        Extract entities from text using the trained CRF model.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of (entity_text, entity_type) tuples
        """
        if self.model is None:
            logger.error("No CRF model loaded")
            return []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract features
            features = self._get_features_for_sentence(doc)
            
            # Predict labels
            labels = self.model.predict([features])[0]
            
            # Convert BIO tags to entity spans
            entities = []
            entity_text = ""
            entity_type = ""
            in_entity = False
            
            for i, (token, label) in enumerate(zip(doc, labels)):
                if label.startswith('B-'):
                    # End previous entity if any
                    if in_entity:
                        entities.append((entity_text.strip(), entity_type))
                    
                    # Start new entity
                    entity_text = token.text
                    entity_type = label[2:]
                    in_entity = True
                
                elif label.startswith('I-') and in_entity and label[2:] == entity_type:
                    # Continue current entity
                    entity_text += " " + token.text
                
                elif in_entity:
                    # End entity
                    entities.append((entity_text.strip(), entity_type))
                    entity_text = ""
                    entity_type = ""
                    in_entity = False
            
            # Add last entity if exists
            if in_entity:
                entities.append((entity_text.strip(), entity_type))
            
            return entities
        
        except Exception as e:
            logger.error(f"Error extracting entities with CRF: {e}")
            return []

def extract_entities(text, method="spacy", model_path=None):
    """
    Extract named entities from text.
    
    Args:
        text (str): Input text
        method (str): Method to use ('spacy' or 'crf')
        model_path (str, optional): Path to CRF model
        
    Returns:
        list: List of (entity_text, entity_type) tuples
    """
    if method.lower() == "spacy":
        extractor = SpacyNERExtractor()
    elif method.lower() == "crf":
        extractor = CRFExtractor(model_path)
    else:
        logger.error(f"Unsupported method: {method}")
        return []
    
    return extractor.extract_entities(text)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage - SpaCy NER
    spacy_extractor = SpacyNERExtractor()
    sample_text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California."
    entities = spacy_extractor.extract_entities(sample_text)
    print("SpaCy NER Entities:")
    for entity in entities:
        print(f"- {entity[0]} ({entity[1]})")
    
    # Example usage - CRF NER (training would be required first)
    print("\nTo train and use CRF NER:")
    print("1. Load dataset")
    print("2. Train model")
    print("3. Save model")
    print("4. Extract entities")