#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Embedding module.
This module provides functions to create and evaluate embeddings from a knowledge graph.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal
from pykeen.evaluation.rank_based_evaluator import RankBasedEvaluator


# Initialize logging
logger = logging.getLogger(__name__)

class KnowledgeGraphEmbedder:
    """Class for creating and evaluating knowledge graph embeddings."""
    
    def __init__(self, namespace="http://example.org/"):
        """
        Initialize the knowledge graph embedder.
        
        Args:
            namespace (str): Base namespace for the knowledge graph
        """
        self.namespace = namespace
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}
        self.triples_factory = None
        self.model_results = {}
        self.entity_types = {}  # For visualization by entity type
        
        logger.info(f"Initialized knowledge graph embedder with namespace: {namespace}")
    
    def load_from_rdf(self, rdf_file, format="turtle"):
        """
        Load knowledge graph from RDF file.
        
        Args:
            rdf_file (str): Path to RDF file
            format (str): RDF format (turtle, xml, etc.)
                
        Returns:
            bool: Success flag
        """
        try:
            # Load RDF graph
            g = Graph()
            g.parse(rdf_file, format=format)
            logger.info(f"Loaded knowledge graph from {rdf_file}")
            
            # Extract triples where both subject and object are URIs
            uri_triples = []
            for s, p, o in g:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    uri_triples.append((str(s), str(p), str(o)))
            
            if not uri_triples:
                logger.warning("No URI-to-URI triples found in the knowledge graph")
                return False
                
            # Create entity and relation dictionaries
            entity_to_id = {}
            relation_to_id = {}
            
            # Assign IDs to entities and relations
            for s, p, o in uri_triples:
                if s not in entity_to_id:
                    entity_to_id[s] = len(entity_to_id)
                if o not in entity_to_id:
                    entity_to_id[o] = len(entity_to_id)
                if p not in relation_to_id:
                    relation_to_id[p] = len(relation_to_id)
            
            # Create ID to entity/relation mappings
            id_to_entity = {v: k for k, v in entity_to_id.items()}
            id_to_relation = {v: k for k, v in relation_to_id.items()}
            
            # Convert triples to mapped format
            mapped_triples = []
            for s, p, o in uri_triples:
                mapped_triples.append((entity_to_id[s], relation_to_id[p], entity_to_id[o]))
            
            # Convert to numpy array
            mapped_triples = np.array(mapped_triples, dtype=np.int64)
            
            # Store mappings
            self.entity_to_id = entity_to_id
            self.relation_to_id = relation_to_id
            self.id_to_entity = id_to_entity
            self.id_to_relation = id_to_relation
            
            # Create the triples factory
            self.triples_factory = TriplesFactory(
                mapped_triples=mapped_triples,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )
            
            logger.info(f"Extracted {len(mapped_triples)} triples with {len(entity_to_id)} entities and {len(relation_to_id)} relations")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph from RDF: {e}")
            return False
    
    def split_dataset(self, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, random_seed=42):
        try:
            if self.triples_factory is None:
                logger.error("No triples loaded. Call load_from_rdf() first.")
                return None, None, None
                
            # Get the mapped triples
            mapped_triples = self.triples_factory.mapped_triples
            num_triples = len(mapped_triples)
            
            # Shuffle indices
            rng = np.random.RandomState(seed=random_seed)
            indices = np.arange(num_triples)
            rng.shuffle(indices)
            
            # Calculate split sizes
            train_size = int(train_ratio * num_triples)
            valid_size = int(validation_ratio * num_triples)
            
            # Split indices
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:train_size + valid_size]
            test_indices = indices[train_size + valid_size:]
            
            # Create split factories
            training = TriplesFactory(
                mapped_triples=mapped_triples[train_indices],
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id
            )
            
            validation = TriplesFactory(
                mapped_triples=mapped_triples[valid_indices],
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id
            )
            
            testing = TriplesFactory(
                mapped_triples=mapped_triples[test_indices],
                entity_to_id=self.entity_to_id,
                relation_to_id=self.relation_to_id
            )
            
            logger.info(f"Split dataset: {len(train_indices)} training, {len(valid_indices)} validation, {len(test_indices)} testing")
            return training, validation, testing
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return None, None, None
    
    def train_embedding_model(self, model_name='TransE', training=None, validation=None, testing=None, 
                            epochs=100, embedding_dim=50, batch_size=32, learning_rate=0.01,
                            num_negs_per_pos=10, random_seed=42, early_stopping=True,
                            early_stopping_patience=5):
        """
        Train a knowledge graph embedding model.
        """
        try:
            # Use provided triples or the entire dataset if not provided
            if training is None:
                if self.triples_factory is None:
                    logger.error("No triples loaded. Call load_from_rdf() first.")
                    return None
                    
                # Split dataset if not provided
                training, validation, testing = self.split_dataset(random_seed=random_seed)
            
            # Train model with minimal parameters
            logger.info(f"Training {model_name} with {embedding_dim} dimensions for {epochs} epochs")
            result = pipeline(
                training=training,
                validation=validation,
                testing=testing,
                model=model_name,
                model_kwargs={'embedding_dim': embedding_dim},
                epochs=epochs,
                random_seed=random_seed
            )
            
            # Store result
            self.model_results[model_name] = result
            logger.info(f"Trained {model_name} model successfully")
            
            return result
        
        except Exception as e:
            logger.error(f"Error training {model_name} model: {e}")
            return None
    
    def get_entity_embeddings(self, model_name):
        """
        Get entity embeddings from a trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            numpy.ndarray: Entity embeddings matrix
        """
        if model_name not in self.model_results:
            logger.error(f"Model {model_name} not trained yet")
            return None
            
        result = self.model_results[model_name]
        return result.model.entity_embeddings.weight.detach().numpy()
    
    def get_relation_embeddings(self, model_name):
        """
        Get relation embeddings from a trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            numpy.ndarray: Relation embeddings matrix
        """
        if model_name not in self.model_results:
            logger.error(f"Model {model_name} not trained yet")
            return None
            
        result = self.model_results[model_name]
        return result.model.relation_embeddings.weight.detach().numpy()
    
    def find_similar_entities(self, entity_uri, model_name, top_k=5):
        """
        Find entities similar to a given entity.
        
        Args:
            entity_uri (str): URI of the entity
            model_name (str): Name of the model
            top_k (int): Number of similar entities to return
            
        Returns:
            list: List of (entity_uri, similarity) tuples
        """
        if model_name not in self.model_results:
            logger.error(f"Model {model_name} not trained yet")
            return []
            
        # Get entity embeddings
        embeddings = self.get_entity_embeddings(model_name)
        
        # Get entity ID
        if entity_uri not in self.entity_to_id:
            logger.error(f"Entity {entity_uri} not found in knowledge graph")
            return []
            
        entity_id = self.entity_to_id[entity_uri]
        
        # Calculate similarities
        entity_vector = embeddings[entity_id].reshape(1, -1)
        similarities = cosine_similarity(entity_vector, embeddings)[0]
        
        # Get top K similar entities
        most_similar_ids = np.argsort(similarities)[-top_k-1:-1][::-1]
        
        # Return similar entities with similarity scores
        similar_entities = []
        for sim_id in most_similar_ids:
            if sim_id != entity_id:  # Skip the entity itself
                similar_entities.append((self.id_to_entity[sim_id], similarities[sim_id]))
        
        return similar_entities
    
    def predict_tail_entities(self, head_uri, relation_uri, model_name, top_k=5):
        """
        Predict tail entities for a given head and relation.
        
        Args:
            head_uri (str): URI of the head entity
            relation_uri (str): URI of the relation
            model_name (str): Name of the model
            top_k (int): Number of predicted entities to return
            
        Returns:
            list: List of (entity_uri, score) tuples
        """
        if model_name not in self.model_results:
            logger.error(f"Model {model_name} not trained yet")
            return []
            
        result = self.model_results[model_name]
        model = result.model
        
        # Get entity and relation IDs
        if head_uri not in self.entity_to_id:
            logger.error(f"Head entity {head_uri} not found in knowledge graph")
            return []
        if relation_uri not in self.relation_to_id:
            logger.error(f"Relation {relation_uri} not found in knowledge graph")
            return []
            
        head_id = self.entity_to_id[head_uri]
        relation_id = self.relation_to_id[relation_uri]
        
        # Get predictions
        with torch.no_grad():
            predictions = model.predict_with_score_t(
                h=torch.tensor([head_id], device=model.device),
                r=torch.tensor([relation_id], device=model.device)
            )
        
        # Get top K predictions
        top_indices = torch.topk(predictions[0], k=top_k).indices.cpu().numpy()
        top_scores = torch.topk(predictions[0], k=top_k).values.cpu().numpy()
        
        # Return predicted entities with scores
        predicted_entities = []
        for idx, score in zip(top_indices, top_scores):
            predicted_entities.append((self.id_to_entity[idx], score))
        
        return predicted_entities
    
    def predict_head_entities(self, relation_uri, tail_uri, model_name, top_k=5):
        """
        Predict head entities for a given relation and tail.
        
        Args:
            relation_uri (str): URI of the relation
            tail_uri (str): URI of the tail entity
            model_name (str): Name of the model
            top_k (int): Number of predicted entities to return
            
        Returns:
            list: List of (entity_uri, score) tuples
        """
        if model_name not in self.model_results:
            logger.error(f"Model {model_name} not trained yet")
            return []
            
        result = self.model_results[model_name]
        model = result.model
        
        # Get entity and relation IDs
        if tail_uri not in self.entity_to_id:
            logger.error(f"Tail entity {tail_uri} not found in knowledge graph")
            return []
        if relation_uri not in self.relation_to_id:
            logger.error(f"Relation {relation_uri} not found in knowledge graph")
            return []
            
        tail_id = self.entity_to_id[tail_uri]
        relation_id = self.relation_to_id[relation_uri]
        
        # Get predictions
        with torch.no_grad():
            predictions = model.predict_with_score_h(
                r=torch.tensor([relation_id], device=model.device),
                t=torch.tensor([tail_id], device=model.device)
            )
        
        # Get top K predictions
        top_indices = torch.topk(predictions[0], k=top_k).indices.cpu().numpy()
        top_scores = torch.topk(predictions[0], k=top_k).values.cpu().numpy()
        
        # Return predicted entities with scores
        predicted_entities = []
        for idx, score in zip(top_indices, top_scores):
            predicted_entities.append((self.id_to_entity[idx], score))
        
        return predicted_entities
    
    def evaluate_model(self, model_name):
        """Evaluate a trained model using PyKEEN's RankBasedEvaluator."""
        if model_name not in self.model_results:
            logger.error(f"Model {model_name} not trained yet")
            return None

        result = self.model_results[model_name]

        # Initialize metrics
        evaluation = {
            'mean_rank': 0.0,
            'mean_reciprocal_rank': 0.0,
            'hits_at_1': 0.0,
            'hits_at_3': 0.0,
            'hits_at_10': 0.0
        }

        # Get testing data
        _, _, testing = self.split_dataset()

        if testing and hasattr(result, 'model'):
            try:
                model = result.model
                test_triples = testing.mapped_triples.to(model.device)  # Move to the correct device

                with torch.no_grad():
                    evaluator = RankBasedEvaluator()
                    metrics = evaluator.evaluate(model, mapped_triples=test_triples)

                    # Extract relevant metrics
                    evaluation['mean_rank'] = metrics.get_metric('mean_rank')
                    evaluation['mean_reciprocal_rank'] = metrics.get_metric('mean_reciprocal_rank')
                    evaluation['hits_at_1'] = metrics.get_metric('hits_at_1')
                    evaluation['hits_at_3'] = metrics.get_metric('hits_at_3')
                    evaluation['hits_at_10'] = metrics.get_metric('hits_at_10')

            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")

        # Log results
        logger.info(f"Evaluation results for {model_name}:")
        logger.info(f"Mean Rank: {evaluation['mean_rank']:.2f}")
        logger.info(f"Mean Reciprocal Rank: {evaluation['mean_reciprocal_rank']:.4f}")
        logger.info(f"Hits@1: {evaluation['hits_at_1']:.4f}")
        logger.info(f"Hits@3: {evaluation['hits_at_3']:.4f}")
        logger.info(f"Hits@10: {evaluation['hits_at_10']:.4f}")

        return evaluation
    
    def compare_models(self, model_names=None):
        """
        Compare multiple trained models.
        
        Args:
            model_names (list): List of model names to compare
            
        Returns:
            pandas.DataFrame: Comparison table
        """
        if model_names is None:
            model_names = list(self.model_results.keys())
        
        if not model_names:
            logger.error("No models to compare")
            return None
            
        # Metrics to compare
        metrics = ['mean_rank', 'mean_reciprocal_rank', 'hits_at_1', 'hits_at_3', 'hits_at_10']
        
        # Collect metrics for each model
        comparison = {}
        for model_name in model_names:
            if model_name in self.model_results:
                evaluation = self.evaluate_model(model_name)
                comparison[model_name] = {metric: evaluation[metric] for metric in metrics}
        
        # Create comparison table
        df = pd.DataFrame(comparison).T
        logger.info("Model comparison:")
        logger.info("\n" + str(df.round(4)))
        
        return df
    
    def visualize_embeddings(self, model_name, entity_types=None, sample_size=None, figsize=(12, 10), 
                           output_path=None, show=True, title=None):
        """
        Visualize entity embeddings in 2D.
        
        Args:
            model_name (str): Name of the model
            entity_types (dict): Dictionary mapping entity types to lists of entity IDs
            sample_size (int): Number of entities to sample (None for all)
            figsize (tuple): Figure size
            output_path (str): Path to save the visualization
            show (bool): Whether to display the visualization
            title (str): Title for the visualization
            
        Returns:
            bool: Success flag
        """
        try:
            if model_name not in self.model_results:
                logger.error(f"Model {model_name} not trained yet")
                return False
                
            # Get entity embeddings
            embeddings = self.get_entity_embeddings(model_name)
            
            # Sample entities if specified
            if sample_size is not None and sample_size < len(embeddings):
                indices = np.random.choice(len(embeddings), sample_size, replace=False)
                embeddings = embeddings[indices]
                
                # Update entity types if provided
                if entity_types is not None:
                    new_entity_types = {}
                    for type_name, type_indices in entity_types.items():
                        new_indices = [i for i, idx in enumerate(indices) if idx in type_indices]
                        if new_indices:
                            new_entity_types[type_name] = new_indices
                    entity_types = new_entity_types
            
            # Apply t-SNE dimensionality reduction
            logger.info("Applying t-SNE dimensionality reduction...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            reduced_embeddings = tsne.fit_transform(embeddings)
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Plot with or without entity types
            if entity_types is None:
                plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
            else:
                # Create a colormap with distinct colors
                colors = plt.cm.tab10.colors
                
                # Plot entities by type
                for i, (entity_type, indices) in enumerate(entity_types.items()):
                    color = colors[i % len(colors)]
                    plt.scatter(
                        reduced_embeddings[indices, 0],
                        reduced_embeddings[indices, 1],
                        label=entity_type,
                        color=color,
                        alpha=0.7
                    )
                plt.legend()
            
            # Add title
            if title:
                plt.title(title)
            else:
                plt.title(f"{model_name} Entity Embeddings")
            
            # Add axes labels
            plt.xlabel("t-SNE dimension 1")
            plt.ylabel("t-SNE dimension 2")
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {output_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}")
            return False
    
    def group_entities_by_type(self, entity_type_triples):
        """
        Group entities by type for visualization.
        
        Args:
            entity_type_triples (list): List of (entity_uri, rdf:type, type_uri) triples
            
        Returns:
            dict: Dictionary mapping entity types to lists of entity IDs
        """
        entity_types = {}
        
        for entity_uri, _, type_uri in entity_type_triples:
            # Skip if entity not in knowledge graph
            if entity_uri not in self.entity_to_id:
                continue
                
            # Get entity ID
            entity_id = self.entity_to_id[entity_uri]
            
            # Extract type name from URI
            type_name = type_uri.split('/')[-1]
            
            # Add to dictionary
            if type_name not in entity_types:
                entity_types[type_name] = []
            entity_types[type_name].append(entity_id)
        
        # Store entity types for later use
        self.entity_types = entity_types
        
        return entity_types
    
    def save_model(self, model_name, output_dir):
        """
        Save a trained model.
        
        Args:
            model_name (str): Name of the model
            output_dir (str): Directory to save the model
            
        Returns:
            bool: Success flag
        """
        try:
            if model_name not in self.model_results:
                logger.error(f"Model {model_name} not trained yet")
                return False
                
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            result = self.model_results[model_name]
            output_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            torch.save(result.model.state_dict(), output_path)
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"{model_name}_metadata.pth")
            metadata = {
                'entity_to_id': self.entity_to_id,
                'relation_to_id': self.relation_to_id,
                'id_to_entity': self.id_to_entity,
                'id_to_relation': self.id_to_relation,
                'entity_types': self.entity_types
            }
            torch.save(metadata, metadata_path)
            
            logger.info(f"Model {model_name} saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def enrich_with_dbpedia(self, entity_text, entity_type, lang="en"):
        """
        Enrich the knowledge graph with data from DBpedia.
        
        Args:
            entity_text (str): Text of the entity
            entity_type (str): Type of the entity
            lang (str): Language for DBpedia lookup
            
        Returns:
            list: List of (subject, predicate, object) triples
        """
        try:
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            
            # Escape quotes in entity text
            safe_entity_text = entity_text.replace('"', '\\"')
            
            # Map entity type to DBpedia class
            dbpedia_class = ""
            if entity_type in ["PERSON", "PER"]:
                dbpedia_class = "dbo:Person"
            elif entity_type in ["ORG", "ORGANIZATION"]:
                dbpedia_class = "dbo:Organisation"
            elif entity_type in ["LOC", "GPE", "LOCATION"]:
                dbpedia_class = "dbo:Place"
            
            # Base query
            query = f"""
            SELECT DISTINCT ?s ?p ?o WHERE {{
              ?s rdfs:label ?label .
              FILTER(LANG(?label) = '{lang}')
              FILTER(REGEX(?label, "^{safe_entity_text}$", "i"))
              
              # Add type constraint if available
              {f"?s a {dbpedia_class} ." if dbpedia_class else ""}
              
              # Get properties and values
              ?s ?p ?o .
              
              # Filter out complex objects and literals without language
              FILTER(isURI(?o) || (isLiteral(?o) && LANG(?o) = '{lang}'))
              
              # Exclude some common predicates
              FILTER(?p NOT IN (
                rdf:type, rdfs:label, rdfs:comment, owl:sameAs,
                <http://purl.org/dc/terms/subject>, <http://dbpedia.org/ontology/wikiPageID>,
                <http://dbpedia.org/ontology/wikiPageRevisionID>, <http://dbpedia.org/ontology/wikiPageWikiLink>
              ))
            }}
            LIMIT 50
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            # Process results
            triples = []
            for result in results["results"]["bindings"]:
                s = result["s"]["value"]
                p = result["p"]["value"]
                o = result["o"]["value"]
                
                # Add triple
                triples.append((s, p, o))
            
            logger.info(f"Retrieved {len(triples)} triples from DBpedia for entity '{entity_text}'")
            return triples
            
        except Exception as e:
            logger.error(f"Error enriching with DBpedia: {e}")
            return []
    
    def enrich_with_wikidata(self, entity_text, entity_type, lang="en"):
        """
        Enrich the knowledge graph with data from Wikidata.
        
        Args:
            entity_text (str): Text of the entity
            entity_type (str): Type of the entity
            lang (str): Language for Wikidata lookup
            
        Returns:
            list: List of (subject, predicate, object) triples
        """
        try:
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            
            # Escape quotes in entity text
            safe_entity_text = entity_text.replace('"', '\\"')
            
            # Map entity type to Wikidata class
            wikidata_class = ""
            if entity_type in ["PERSON", "PER"]:
                wikidata_class = "wd:Q5"  # human
            elif entity_type in ["ORG", "ORGANIZATION"]:
                wikidata_class = "wd:Q43229"  # organization
            elif entity_type in ["LOC", "GPE", "LOCATION"]:
                wikidata_class = "wd:Q618123"  # geographical object
            
            # Base query
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?s ?p ?o ?oLabel WHERE {{
              ?s rdfs:label ?label .
              FILTER(LANG(?label) = '{lang}')
              FILTER(REGEX(?label, "^{safe_entity_text}$", "i"))
              
              # Add type constraint if available
              {f"?s wdt:P31/wdt:P279* {wikidata_class} ." if wikidata_class else ""}
              
              # Get properties and values
              ?s ?p ?o .
              
              # Only include statements
              FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/"))
              
              # Get labels for objects
              OPTIONAL {{
                ?o rdfs:label ?oLabel .
                FILTER(LANG(?oLabel) = '{lang}')
              }}
            }}
            LIMIT 50
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            # Process results
            triples = []
            for result in results["results"]["bindings"]:
                s = result["s"]["value"]
                p = result["p"]["value"]
                o = result["o"]["value"]
                
                # Add triple
                triples.append((s, p, o))
            
            logger.info(f"Retrieved {len(triples)} triples from Wikidata for entity '{entity_text}'")
            return triples
            
        except Exception as e:
            logger.error(f"Error enriching with Wikidata: {e}")
            return []

def create_and_train_embeddings(rdf_file, models=None, output_dir="output/embeddings", train_all=False):
    """
    Create and train knowledge graph embeddings.
    
    Args:
        rdf_file (str): Path to RDF file
        models (list): List of model names to train
        output_dir (str): Directory to save embeddings
        train_all (bool): Whether to train all models
        
    Returns:
        KnowledgeGraphEmbedder: Knowledge graph embedder instance
    """
    if models is None:
        models = ['TransE'] if not train_all else ['TransE', 'DistMult', 'ComplEx']
    
    # Create embedder
    embedder = KnowledgeGraphEmbedder()
    
    # Load knowledge graph
    if not embedder.load_from_rdf(rdf_file):
        logger.error(f"Failed to load knowledge graph from {rdf_file}")
        return None
    
    # Split dataset
    training, validation, testing = embedder.split_dataset()
    
    # Train models
    for model_name in models:
        logger.info(f"Training {model_name} model...")
        embedder.train_embedding_model(
            model_name=model_name,
            training=training,
            validation=validation,
            testing=testing,
            epochs=100,
            embedding_dim=50
        )
    
    # Compare models
    embedder.compare_models()
    
    # Save models
    for model_name in models:
        embedder.save_model(model_name, output_dir)
    
    return embedder

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )