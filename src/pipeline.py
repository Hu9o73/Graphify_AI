#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Construction Pipeline.
This script orchestrates the complete process from text cleaning to knowledge graph embedding.
"""

import os
import logging
import argparse
import json
from datetime import datetime

# Import project modules
from src.data_collection.scraper import scrape_articles
from src.preprocessing.cleaner import clean_text
from src.entity_recognition.ner import SpacyNERExtractor, CRFExtractor
from src.entity_recognition.comparison import compare_models as compare_ner_models
from src.relation_extraction.extractor import SpacyRelationExtractor
from src.knowledge_graph.builder import build_graph
from src.knowledge_graph.augmentation import augment_knowledge_graph
from src.knowledge_graph.embeddings import create_and_train_embeddings

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(args):
    """
    Run the complete knowledge graph construction pipeline.
    
    Args:
        args: Command-line arguments
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    raw_data_dir = os.path.join(output_dir, 'raw')
    processed_data_dir = os.path.join(output_dir, 'processed')
    kg_dir = os.path.join(output_dir, 'kg')
    model_dir = os.path.join(output_dir, 'models')
    visualization_dir = os.path.join(output_dir, 'visualization')
    
    # Create output directories
    for directory in [raw_data_dir, processed_data_dir, kg_dir, model_dir, visualization_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Data Collection (optional)
    if args.scrape:
        logger.info("Step 1: Data Collection")
        articles = scrape_articles(
            source=args.scrape_source,
            num_articles=args.num_articles,
            category=args.category,
            output_dir=raw_data_dir
        )
        logger.info(f"Scraped {len(articles)} articles")
    
    # Step 2: Text Preprocessing
    logger.info("Step 2: Text Preprocessing")
    
    # Load articles from raw data directory
    import glob
    import json
    
    article_files = glob.glob(f"{raw_data_dir}/*.json")
    if not article_files:
        logger.error("No article files found in raw data directory")
        return
    
    logger.info(f"Found {len(article_files)} article files")
    
    articles = []
    for file in article_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                article = json.load(f)
                if 'content' in article and article['content']:
                    articles.append(article)
        except Exception as e:
            logger.error(f"Error loading article file {file}: {e}")
    
    logger.info(f"Loaded {len(articles)} articles with content")
    
    # Preprocess text
    processed_articles = []
    for article in articles:
        # Clean the text
        cleaned_text = clean_text(
            article['content'],
            lowercase=False,  # Keep case for NER
            remove_stops=False,  # Keep stop words for context
            lemmatize=False  # Don't lemmatize to preserve entities
        )
        
        # Create processed article
        processed_article = {
            'id': article.get('id', ''),
            'title': article.get('title', ''),
            'source': article.get('source', ''),
            'category': article.get('category', ''),
            'published_date': article.get('published_date', ''),
            'original_content': article['content'],
            'processed_content': cleaned_text
        }
        
        processed_articles.append(processed_article)
    
    # Save processed articles
    processed_file = os.path.join(processed_data_dir, 'processed_articles.json')
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Saved {len(processed_articles)} processed articles to {processed_file}")
    
    # Step 3: Named Entity Recognition
    logger.info("Step 3: Named Entity Recognition")
    
    # Initialize NER extractors
    spacy_extractor = SpacyNERExtractor()
    
    # Extract entities with spaCy
    all_entities = []
    for article in processed_articles:
        entities = spacy_extractor.extract_entities(article['processed_content'])
        all_entities.append(entities)
        logger.info(f"Extracted {len(entities)} entities from article: {article['title']}")
    
    # Save entities
    entities_file = os.path.join(processed_data_dir, 'entities.json')
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Saved entities to {entities_file}")
    
    # Step 4: Relation Extraction
    logger.info("Step 4: Relation Extraction")
    
    # Initialize relation extractor
    relation_extractor = SpacyRelationExtractor()
    
    # Extract relations
    all_relations = []
    for article in processed_articles:
        relations = relation_extractor.extract_relations(article['processed_content'])
        all_relations.append(relations)
        logger.info(f"Extracted {len(relations)} relations from article: {article['title']}")
    
    # Save relations
    relations_file = os.path.join(processed_data_dir, 'relations.json')
    with open(relations_file, 'w', encoding='utf-8') as f:
        json.dump(all_relations, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Saved relations to {relations_file}")
    
    # Step 5: Knowledge Graph Construction
    logger.info("Step 5: Knowledge Graph Construction")
    
    # Collect all entities and relations
    all_kg_entities = set()
    for entities in all_entities:
        for entity in entities:
            all_kg_entities.add(entity)
    
    all_kg_relations = []
    for relations in all_relations:
        all_kg_relations.extend(relations)
    
    # Build knowledge graph
    kg_file = os.path.join(kg_dir, 'knowledge_graph.ttl')
    vis_file = os.path.join(visualization_dir, 'knowledge_graph.html')
    
    kg_builder = build_graph(
        list(all_kg_entities),
        all_kg_relations,
        namespace=args.namespace,
        output_path=kg_file,
        format="turtle",
        visualize=True,
        vis_output=vis_file
    )
    
    logger.info(f"Built knowledge graph with {len(all_kg_entities)} entities and {len(all_kg_relations)} relations")
    logger.info(f"Saved knowledge graph to {kg_file}")
    logger.info(f"Saved visualization to {vis_file}")
    
    # Step 6: Knowledge Graph Augmentation
    logger.info("Step 6: Knowledge Graph Augmentation")
    
    # Augment knowledge graph
    augmented_kg_file = os.path.join(kg_dir, 'knowledge_graph_augmented.ttl')
    
    augmenter = augment_knowledge_graph(
        input_file=kg_file,
        output_file=augmented_kg_file,
        entity_types=["PERSON", "ORG", "GPE", "LOC"],
        max_entities=args.max_augment_entities,
        use_dbpedia=True,
        use_wikidata=True,
        connection_depth=args.connection_depth,
        namespace=args.namespace
    )
    
    logger.info(f"Augmented knowledge graph saved to {augmented_kg_file}")
    
    # Step 7: Knowledge Graph Embedding
    logger.info("Step 7: Knowledge Graph Embedding")
    
    # Create embeddings
    embeddings_dir = os.path.join(model_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    embedder = create_and_train_embeddings(
        rdf_file=augmented_kg_file,
        models=['TransE', 'DistMult'],
        output_dir=embeddings_dir,
        train_all=False
    )
    
    logger.info(f"Created knowledge graph embeddings and saved to {embeddings_dir}")
    
    # Save pipeline metadata
    metadata = {
        'pipeline_run_date': datetime.now().isoformat(),
        'args': vars(args),
        'stats': {
            'num_articles': len(processed_articles),
            'num_entities': len(all_kg_entities),
            'num_relations': len(all_kg_relations),
        }
    }
    
    metadata_file = os.path.join(output_dir, 'pipeline_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Saved pipeline metadata to {metadata_file}")
    logger.info("Knowledge graph pipeline completed successfully!")

def main():
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Knowledge Graph Construction Pipeline')
    
    # Basic arguments
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for all pipeline artifacts')
    parser.add_argument('--namespace', type=str, default='http://example.org/graphify/',
                        help='Namespace for the knowledge graph')
    
    # Scraping arguments
    parser.add_argument('--scrape', action='store_true',
                        help='Whether to scrape articles')
    parser.add_argument('--scrape-source', type=str, default='reuters',
                        help='Source to scrape articles from')
    parser.add_argument('--num-articles', type=int, default=10,
                        help='Number of articles to scrape')
    parser.add_argument('--category', type=str, default='world',
                        help='Category of articles to scrape')
    
    # Knowledge graph augmentation arguments
    parser.add_argument('--max-augment-entities', type=int, default=50,
                        help='Maximum number of entities to augment')
    parser.add_argument('--connection-depth', type=int, default=1,
                        help='Depth of connections to add during augmentation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(args)

if __name__ == "__main__":
    main()