#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relation Extraction (RE) module.
This module provides functions to extract relations between entities.
"""

import logging
import spacy
from collections import defaultdict

# Initialize logging
logger = logging.getLogger(__name__)

class RelationExtractor:
    """Base class for relation extraction."""
    
    def __init__(self):
        """Initialize the relation extractor."""
        pass
    
    def extract_relations(self, text, entities=None):
        """
        Extract relations from text.
        
        Args:
            text (str): Input text
            entities (list, optional): Pre-extracted entities
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        raise NotImplementedError("Subclasses must implement extract_relations")

class SpacyRelationExtractor(RelationExtractor):
    """Relation extraction using spaCy's dependency parsing."""
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize the spaCy relation extractor.
        
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
    
    def _find_entity_for_token(self, token, entities):
        """
        Find the entity that contains a token.
        
        Args:
            token (spacy.tokens.Token): The token to check
            entities (list): List of (start, end, text, type) entity tuples
            
        Returns:
            tuple: (entity_text, entity_type) or None
        """
        for start, end, text, type_ in entities:
            if token.i >= start and token.i < end:
                return (text, type_)
        return None
    
    def extract_simple_relations(self, doc, entities):
        """
        Extract simple subject-verb-object relations.
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            entities (list): List of (start, end, text, type) entity tuples
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        relations = []
        
        for token in doc:
            # Check if token is a verb
            if token.pos_ == "VERB":
                # Find subject and object
                subj, obj = None, None
                predicate = token.lemma_
                
                for child in token.children:
                    # Subject
                    if child.dep_ in ("nsubj", "nsubjpass") and not subj:
                        subj_entity = self._find_entity_for_token(child, entities)
                        if subj_entity:
                            subj = subj_entity
                    
                    # Object - direct object or prepositional object
                    elif child.dep_ in ("dobj", "pobj", "attr") and not obj:
                        obj_entity = self._find_entity_for_token(child, entities)
                        if obj_entity:
                            obj = obj_entity
                    
                    # Handle preposition + object (e.g., "located in London")
                    elif child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj" and not obj:
                                obj_entity = self._find_entity_for_token(grandchild, entities)
                                if obj_entity:
                                    obj = obj_entity
                                    # Include preposition in the predicate
                                    predicate = f"{token.lemma_} {child.text}"
                
                # If we found both subject and object, add the relation
                if subj and obj:
                    relations.append((subj, predicate, obj))
        
        return relations
    
    def extract_compound_relations(self, doc, entities):
        """
        Extract relations from compound nouns (e.g., "Apple CEO Tim Cook").
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            entities (list): List of (start, end, text, type) entity tuples
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        relations = []
        
        for token in doc:
            if token.dep_ == "appos":
                # Find the entity for the token and its head
                token_entity = self._find_entity_for_token(token, entities)
                head_entity = self._find_entity_for_token(token.head, entities)
                
                if token_entity and head_entity:
                    # Create relations in both directions
                    if token_entity[1] == "PERSON" and head_entity[1] == "ORG":
                        relations.append((head_entity, "has_member", token_entity))
                        relations.append((token_entity, "member_of", head_entity))
                    elif head_entity[1] == "PERSON" and token_entity[1] == "ORG":
                        relations.append((token_entity, "has_member", head_entity))
                        relations.append((head_entity, "member_of", token_entity))
            
            # Check for compound relations like "Apple CEO Tim Cook"
            elif token.dep_ == "compound" and token.head.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj"):
                token_entity = self._find_entity_for_token(token, entities)
                head_entity = self._find_entity_for_token(token.head, entities)
                
                if token_entity and head_entity:
                    relations.append((token_entity, "related_to", head_entity))
        
        return relations
    
    def extract_possessive_relations(self, doc, entities):
        """
        Extract possessive relations (e.g., "Apple's headquarters").
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            entities (list): List of (start, end, text, type) entity tuples
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        relations = []
        
        for token in doc:
            if token.dep_ == "poss":
                # Find the entity for the token and its head
                token_entity = self._find_entity_for_token(token, entities)
                head_entity = self._find_entity_for_token(token.head, entities)
                
                if token_entity and head_entity:
                    relations.append((token_entity, "owns", head_entity))
                    relations.append((head_entity, "owned_by", token_entity))
        
        return relations
    
    def extract_preposition_relations(self, doc, entities):
        """
        Extract relations through prepositions (e.g., "headquartered in Cupertino").
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            entities (list): List of (start, end, text, type) entity tuples
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        relations = []
        
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ("VERB", "NOUN"):
                # Get the head entity
                head_entity = None
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        head_entity = self._find_entity_for_token(child, entities)
                        break
                
                # If no subject found, try the head itself
                if not head_entity:
                    head_entity = self._find_entity_for_token(token.head, entities)
                
                # Get the object entity
                obj_entity = None
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj_entity = self._find_entity_for_token(child, entities)
                        break
                
                # Create relation if both entities found
                if head_entity and obj_entity:
                    predicate = f"{token.head.lemma_}_{token.text}"
                    relations.append((head_entity, predicate, obj_entity))
        
        return relations
    
    def _convert_spacy_entities_to_spans(self, doc):
        """
        Convert spaCy entities to span format.
        
        Args:
            doc (spacy.tokens.Doc): Processed spaCy document
            
        Returns:
            list: List of (start, end, text, type) entity tuples
        """
        entities = []
        for ent in doc.ents:
            entities.append((ent.start, ent.end, ent.text, ent.label_))
        return entities
    
    def extract_relations(self, text, entities=None):
        """
        Extract relations from text.
        
        Args:
            text (str): Input text
            entities (list, optional): Pre-extracted entities
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # If entities not provided, use spaCy's NER
            if not entities:
                entities = self._convert_spacy_entities_to_spans(doc)
            
            # Extract different types of relations
            relations = []
            relations.extend(self.extract_simple_relations(doc, entities))
            relations.extend(self.extract_compound_relations(doc, entities))
            relations.extend(self.extract_possessive_relations(doc, entities))
            relations.extend(self.extract_preposition_relations(doc, entities))
            
            # Remove duplicates while preserving order
            unique_relations = []
            seen = set()
            for relation in relations:
                relation_tuple = (relation[0][0], relation[1], relation[2][0])
                if relation_tuple not in seen:
                    seen.add(relation_tuple)
                    unique_relations.append(relation)
            
            return unique_relations
        
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []

class PatternBasedRelationExtractor(RelationExtractor):
    """Relation extraction using pattern matching."""
    
    def __init__(self):
        """Initialize the pattern-based relation extractor."""
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define patterns for relation extraction
        self.patterns = [
            # Founding pattern
            {"subject": {"POS": "PROPN"}, "relation": {"LEMMA": {"IN": ["found", "establish", "start", "launch", "create"]}}, "object": {"POS": "PROPN"}},
            
            # Acquisition pattern
            {"subject": {"POS": "PROPN"}, "relation": {"LEMMA": {"IN": ["acquire", "buy", "purchase"]}}, "object": {"POS": "PROPN"}},
            
            # Location pattern
            {"subject": {"POS": "PROPN"}, "relation": {"LEMMA": {"IN": ["locate", "base", "headquarter"]}}, "object": {"POS": "PROPN"}},
            
            # Employment pattern
            {"subject": {"POS": "PROPN"}, "relation": {"LEMMA": {"IN": ["work", "join", "lead", "head"]}}, "object": {"POS": "PROPN"}}
        ]
    
    def extract_relations(self, text, entities=None):
        """
        Extract relations from text using pattern matching.
        
        Args:
            text (str): Input text
            entities (list, optional): Pre-extracted entities
            
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # If entities not provided, use spaCy's NER
            if not entities:
                entities = [(ent.start, ent.end, ent.text, ent.label_) for ent in doc.ents]
            
            # Extract relations using patterns
            relations = []
            
            # For simplicity, we'll just use a rule-based approach
            # In a full implementation, you would iterate through each pattern
            # and check if it matches the text
            
            # For now, we'll just use a basic implementation
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ in ["found", "establish", "acquire", "work", "lead"]:
                    # Find subject and object
                    subj, obj = None, None
                    
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass") and not subj:
                            for ent in doc.ents:
                                if child.i >= ent.start and child.i < ent.end:
                                    subj = (ent.text, ent.label_)
                                    break
                        
                        elif child.dep_ in ("dobj", "pobj", "attr") and not obj:
                            for ent in doc.ents:
                                if child.i >= ent.start and child.i < ent.end:
                                    obj = (ent.text, ent.label_)
                                    break
                        
                        elif child.dep_ == "prep":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj" and not obj:
                                    for ent in doc.ents:
                                        if grandchild.i >= ent.start and grandchild.i < ent.end:
                                            obj = (ent.text, ent.label_)
                                            break
                    
                    # If we found both subject and object, add the relation
                    if subj and obj:
                        relations.append((subj, token.lemma_, obj))
            
            return relations
        
        except Exception as e:
            logger.error(f"Error extracting relations with patterns: {e}")
            return []

def extract_relations(text, method="spacy", entities=None):
    """
    Extract relations from text.
    
    Args:
        text (str): Input text
        method (str): Method to use ('spacy' or 'pattern')
        entities (list, optional): Pre-extracted entities
        
    Returns:
        list: List of (subject, predicate, object) tuples
    """
    if method.lower() == "spacy":
        extractor = SpacyRelationExtractor()
    elif method.lower() == "pattern":
        extractor = PatternBasedRelationExtractor()
    else:
        logger.error(f"Unsupported method: {method}")
        return []
    
    return extractor.extract_relations(text, entities)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    sample_text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California. Tim Cook is the current CEO of Apple. Microsoft was acquired by Bill Gates."
    
    # Extract relations using spaCy
    spacy_extractor = SpacyRelationExtractor()
    relations = spacy_extractor.extract_relations(sample_text)
    
    print("Relations extracted with spaCy:")
    for subject, predicate, obj in relations:
        print(f"- {subject[0]} ({subject[1]}) --[{predicate}]--> {obj[0]} ({obj[1]})")
    
    # Extract relations using pattern matching
    pattern_extractor = PatternBasedRelationExtractor()
    relations = pattern_extractor.extract_relations(sample_text)
    
    print("\nRelations extracted with pattern matching:")
    for subject, predicate, obj in relations:
        print(f"- {subject[0]} ({subject[1]}) --[{predicate}]--> {obj[0]} ({obj[1]})")