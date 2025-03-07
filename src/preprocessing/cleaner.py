#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text cleaning and preprocessing module.
This module provides functions to clean and normalize text data.
"""

import re
import logging
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize logging
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def remove_html_tags(text):
    """Remove HTML tags from text."""
    try:
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    except Exception as e:
        logger.error(f"Error removing HTML tags: {e}")
        return text

def remove_special_characters(text, keep_hyphens=True):
    """
    Remove special characters from text.
    
    Args:
        text (str): Input text
        keep_hyphens (bool): Whether to keep hyphens for compound words
        
    Returns:
        str: Cleaned text
    """
    try:
        if keep_hyphens:
            pattern = r'[^a-zA-Z0-9\s\-]'
        else:
            pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)
    except Exception as e:
        logger.error(f"Error removing special characters: {e}")
        return text

def remove_extra_whitespace(text):
    """Remove extra whitespace from text."""
    try:
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Error removing extra whitespace: {e}")
        return text

def remove_stopwords(text, language='english'):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        language (str): Language for stopwords
        
    Returns:
        str: Text without stopwords
    """
    try:
        stop_words = set(stopwords.words(language))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)
    except Exception as e:
        logger.error(f"Error removing stopwords: {e}")
        return text

def lemmatize_text(text):
    """
    Lemmatize text using NLTK's WordNetLemmatizer.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lemmatized text
    """
    try:
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    except Exception as e:
        logger.error(f"Error lemmatizing text: {e}")
        return text

def lemmatize_with_spacy(text):
    """
    Lemmatize text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lemmatized text
    """
    try:
        doc = nlp(text)
        lemmatized_text = [token.lemma_ for token in doc]
        return ' '.join(lemmatized_text)
    except Exception as e:
        logger.error(f"Error lemmatizing text with spaCy: {e}")
        return text

def clean_text(text, lowercase=True, remove_html=True, remove_special_chars=True, 
               keep_hyphens=True, remove_extra_spaces=True, remove_stops=True, 
               lemmatize=True, use_spacy=False):
    """
    Clean and preprocess text.
    
    Args:
        text (str): Input text
        lowercase (bool): Convert text to lowercase
        remove_html (bool): Remove HTML tags
        remove_special_chars (bool): Remove special characters
        keep_hyphens (bool): Keep hyphens for compound words
        remove_extra_spaces (bool): Remove extra whitespace
        remove_stops (bool): Remove stopwords
        lemmatize (bool): Lemmatize text
        use_spacy (bool): Use spaCy for lemmatization
        
    Returns:
        str: Cleaned and preprocessed text
    """
    logger.info("Cleaning text...")
    
    if not text:
        logger.warning("Empty text provided")
        return ""
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove HTML tags
    if remove_html:
        text = remove_html_tags(text)
    
    # Remove special characters
    if remove_special_chars:
        text = remove_special_characters(text, keep_hyphens)
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = remove_extra_whitespace(text)
    
    # Remove stopwords
    if remove_stops:
        text = remove_stopwords(text)
    
    # Lemmatize text
    if lemmatize:
        if use_spacy:
            text = lemmatize_with_spacy(text)
        else:
            text = lemmatize_text(text)
    
    logger.info("Text cleaning completed")
    return text

if __name__ == "__main__":
    # Example usage
    sample_text = """<p>This is a sample text with some <b>HTML tags</b> and special characters: !@#$%^&*(). 
    It also contains some extra  whitespace and stop words like the, is, and a. 
    We'll also include some compound words like state-of-the-art technology.</p>"""
    
    cleaned_text = clean_text(sample_text)
    print(f"Original text:\n{sample_text}\n")
    print(f"Cleaned text:\n{cleaned_text}")