#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main module for the Graphify_AI project.
This script runs the complete pipeline from data collection to knowledge graph construction.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graphify_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_arg_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description='Graphify_AI: Knowledge Graph Builder')
    
    parser.add_argument('--scrape', action='store_true', 
                        help='Scrape news articles from websites')
    
    parser.add_argument('--clean', action='store_true',
                        help='Clean and preprocess the collected text data')
    
    parser.add_argument('--ner', action='store_true',
                        help='Perform Named Entity Recognition on the preprocessed data')
    
    parser.add_argument('--compare', action='store_true',
                        help='Compare CRF model with spaCy\'s NER model')
    
    parser.add_argument('--re', action='store_true',
                        help='Extract relations between entities')
    
    parser.add_argument('--kg', action='store_true',
                        help='Build knowledge graph from entities and relations')
    
    parser.add_argument('--all', action='store_true',
                        help='Run the complete pipeline')
    
    parser.add_argument('--source', type=str, default='reuters',
                        help='Source website for scraping (default: reuters)')
    
    parser.add_argument('--articles', type=int, default=10,
                        help='Number of articles to scrape (default: 10)')
    
    parser.add_argument('--output', type=str, default='knowledge_graph.rdf',
                        help='Output file for the knowledge graph (default: knowledge_graph.rdf)')
    
    return parser

def main():
    """Main function to run the complete pipeline."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    logger.info("Starting Graphify_AI")
    
    # Import modules only when needed to avoid unnecessary imports
    if args.all or args.scrape:
        from data_collection.scraper import scrape_articles
        logger.info("Scraping articles from %s", args.source)
        # TODO: Implement scrape_articles function
    
    if args.all or args.clean:
        from preprocessing.cleaner import clean_text
        logger.info("Cleaning and preprocessing text data")
        # TODO: Implement clean_text function
    
    if args.all or args.ner:
        from entity_recognition.ner import extract_entities
        logger.info("Extracting entities using NER")
        # TODO: Implement extract_entities function
    
    if args.all or args.compare:
        from entity_recognition.comparison import compare_models
        logger.info("Comparing NER models")
        # TODO: Implement compare_models function
    
    if args.all or args.re:
        from relation_extraction.extractor import extract_relations
        logger.info("Extracting relations between entities")
        # TODO: Implement extract_relations function
    
    if args.all or args.kg:
        from knowledge_graph.builder import build_graph
        logger.info("Building knowledge graph")
        # TODO: Implement build_graph function
        logger.info(f"Knowledge graph saved to {args.output}")
    
    logger.info("Graphify_AI completed successfully")

if __name__ == "__main__":
    main()