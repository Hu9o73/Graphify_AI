#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Builder module.
This module provides functions to build a knowledge graph from extracted entities and relations.
"""

import os
import logging
import re
from urllib.parse import quote
import networkx as nx
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, XSD, FOAF, OWL
import matplotlib.pyplot as plt
from pyvis.network import Network

# Initialize logging
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Class for building a knowledge graph from entities and relations."""
    
    def __init__(self, namespace="http://example.org/"):
        """
        Initialize the knowledge graph builder.
        
        Args:
            namespace (str): Base namespace for the knowledge graph
        """
        self.namespace = namespace
        self.graph = Graph()
        self.nx_graph = nx.DiGraph()
        
        # Define namespaces
        self.ns = Namespace(namespace)
        self.graph.bind("ns", self.ns)
        self.graph.bind("foaf", FOAF)
        
        # Standard namespaces
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.graph.bind("owl", OWL)
        
        logger.info(f"Initialized knowledge graph with namespace: {namespace}")
    
    def _sanitize_uri(self, text):
        """
        Sanitize text for use in URIs.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Sanitized text
        """
        # Replace spaces and special characters
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', '_', text.strip())
        return text
    
    def _create_uri(self, entity_text, entity_type):
        """
        Create a URI for an entity.
        
        Args:
            entity_text (str): Entity text
            entity_type (str): Entity type
            
        Returns:
            rdflib.URIRef: URI reference
        """
        sanitized = self._sanitize_uri(entity_text)
        return URIRef(f"{self.namespace}{entity_type.lower()}/{sanitized}")
    
    def _entity_type_to_class(self, entity_type):
        """
        Convert entity type to class URI.
        
        Args:
            entity_type (str): Entity type
            
        Returns:
            rdflib.URIRef: Class URI
        """
        return URIRef(f"{self.namespace}{entity_type}")
    
    def _predicate_to_uri(self, predicate):
        """
        Convert predicate to URI.
        
        Args:
            predicate (str): Predicate text
            
        Returns:
            rdflib.URIRef: Predicate URI
        """
        sanitized = self._sanitize_uri(predicate)
        return URIRef(f"{self.namespace}predicate/{sanitized}")
    
    def add_entity(self, entity_text, entity_type):
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity_text (str): Entity text
            entity_type (str): Entity type
            
        Returns:
            rdflib.URIRef: Entity URI
        """
        try:
            # Create URI
            entity_uri = self._create_uri(entity_text, entity_type)
            
            # Add entity type
            entity_class = self._entity_type_to_class(entity_type)
            self.graph.add((entity_uri, RDF.type, entity_class))
            
            # Add entity label
            self.graph.add((entity_uri, RDFS.label, Literal(entity_text, datatype=XSD.string)))
            
            # Add to NetworkX graph
            self.nx_graph.add_node(entity_uri, label=entity_text, type=entity_type)
            
            logger.debug(f"Added entity: {entity_text} ({entity_type})")
            return entity_uri
        
        except Exception as e:
            logger.error(f"Error adding entity {entity_text}: {e}")
            return None
    
    def add_relation(self, subject, predicate, obj):
        """
        Add a relation to the knowledge graph.
        
        Args:
            subject (tuple): (entity_text, entity_type) for subject
            predicate (str): Predicate text
            obj (tuple): (entity_text, entity_type) for object
            
        Returns:
            bool: Success flag
        """
        try:
            # Extract components
            subj_text, subj_type = subject
            obj_text, obj_type = obj
            
            # Create URIs
            subj_uri = self._create_uri(subj_text, subj_type)
            pred_uri = self._predicate_to_uri(predicate)
            obj_uri = self._create_uri(obj_text, obj_type)
            
            # Add entities if they don't exist
            self.add_entity(subj_text, subj_type)
            self.add_entity(obj_text, obj_type)
            
            # Add predicate label
            self.graph.add((pred_uri, RDFS.label, Literal(predicate, datatype=XSD.string)))
            
            # Add triple
            self.graph.add((subj_uri, pred_uri, obj_uri))
            
            # Add to NetworkX graph
            self.nx_graph.add_edge(subj_uri, obj_uri, label=predicate)
            
            logger.debug(f"Added relation: {subj_text} --[{predicate}]--> {obj_text}")
            return True
        
        except Exception as e:
            logger.error(f"Error adding relation: {e}")
            return False
    
    def add_entities_and_relations(self, entities, relations):
        """
        Add multiple entities and relations to the knowledge graph.
        
        Args:
            entities (list): List of (entity_text, entity_type) tuples
            relations (list): List of (subject, predicate, object) tuples
            
        Returns:
            tuple: (num_entities_added, num_relations_added)
        """
        entity_count = 0
        relation_count = 0
        
        # Add entities
        for entity_text, entity_type in entities:
            if self.add_entity(entity_text, entity_type):
                entity_count += 1
        
        # Add relations
        for subject, predicate, obj in relations:
            if self.add_relation(subject, predicate, obj):
                relation_count += 1
        
        logger.info(f"Added {entity_count} entities and {relation_count} relations")
        return entity_count, relation_count
    
    def get_triples(self):
        """
        Get all triples from the knowledge graph.
        
        Returns:
            list: List of (subject, predicate, object) tuples
        """
        return list(self.graph)
    
    def query_sparql(self, query):
        """
        Execute a SPARQL query on the knowledge graph.
        
        Args:
            query (str): SPARQL query
            
        Returns:
            list: Query results
        """
        try:
            results = self.graph.query(query)
            return list(results)
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return []
    
    def save_to_file(self, output_path, format="turtle"):
        """
        Save the knowledge graph to a file.
        
        Args:
            output_path (str): Output file path
            format (str): Output format (turtle, xml, json-ld, etc.)
            
        Returns:
            bool: Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to file
            self.graph.serialize(destination=output_path, format=format)
            logger.info(f"Saved knowledge graph to {output_path} in {format} format")
            return True
        
        except Exception as e:
            logger.error(f"Error saving knowledge graph to file: {e}")
            return False
    
    def visualize(self, output_path=None, notebook=False):
        """
        Visualize the knowledge graph.
        
        Args:
            output_path (str, optional): Output file path for the visualization
            notebook (bool): Whether to display in a Jupyter notebook
            
        Returns:
            bool: Success flag
        """
        try:
            # Create a PyVis network
            net = Network(notebook=notebook, directed=True)
            
            # Add nodes
            for node, data in self.nx_graph.nodes(data=True):
                label = data.get('label', str(node).split('/')[-1])
                node_type = data.get('type', 'Unknown')
                
                # Color nodes by type
                color = {
                    'PERSON': '#a8e6cf',
                    'ORG': '#ff8b94',
                    'GPE': '#ffd3b6',
                    'LOC': '#dcedc1',
                    'DATE': '#f9f9f9',
                    'MISC': '#d4a5a5'
                }.get(node_type, '#b3b3cc')
                
                net.add_node(str(node), label=label, title=f"{label} ({node_type})", color=color)
            
            # Add edges
            for source, target, data in self.nx_graph.edges(data=True):
                label = data.get('label', '')
                net.add_edge(str(source), str(target), label=label, title=label)
            
            # Set physics layout
            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -100,
                  "centralGravity": 0.01,
                  "springLength": 200,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              },
              "edges": {
                "color": {
                  "inherit": true
                },
                "smooth": {
                  "enabled": false,
                  "type": "continuous"
                },
                "arrows": {
                  "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                  }
                },
                "font": {
                  "size": 10
                }
              },
              "nodes": {
                "font": {
                  "size": 12,
                  "face": "Tahoma"
                }
              }
            }
            """)
            
            # Save or show
            if output_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                net.save_graph(output_path)
                logger.info(f"Saved visualization to {output_path}")
            elif notebook:
                net.show("knowledge_graph.html")
            
            return True
        
        except Exception as e:
            logger.error(f"Error visualizing knowledge graph: {e}")
            return False
    
    def plot_graph(self, output_path=None, figsize=(12, 10)):
        """
        Plot the knowledge graph using matplotlib.
        
        Args:
            output_path (str, optional): Output file path for the plot
            figsize (tuple): Figure size
            
        Returns:
            bool: Success flag
        """
        try:
            # Create labels
            labels = {node: data.get('label', str(node).split('/')[-1]) 
                     for node, data in self.nx_graph.nodes(data=True)}
            
            # Node colors
            node_colors = []
            for node, data in self.nx_graph.nodes(data=True):
                node_type = data.get('type', 'Unknown')
                color = {
                    'PERSON': 'skyblue',
                    'ORG': 'salmon',
                    'GPE': 'lightgreen',
                    'LOC': 'orange',
                    'DATE': 'lightgray',
                    'MISC': 'violet'
                }.get(node_type, 'gray')
                node_colors.append(color)
            
            # Edge labels
            edge_labels = {(u, v): data.get('label', '') 
                          for u, v, data in self.nx_graph.edges(data=True)}
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Create layout
            pos = nx.spring_layout(self.nx_graph, seed=42)
            
            # Draw graph
            nx.draw_networkx_nodes(self.nx_graph, pos, node_color=node_colors, alpha=0.8, node_size=500)
            nx.draw_networkx_edges(self.nx_graph, pos, width=1.0, alpha=0.5, arrows=True)
            nx.draw_networkx_labels(self.nx_graph, pos, labels=labels, font_size=10)
            nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels=edge_labels, font_size=8)
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save or show
            if output_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {output_path}")
            else:
                plt.show()
            
            return True
        
        except Exception as e:
            logger.error(f"Error plotting knowledge graph: {e}")
            return False

def build_graph(entities, relations, namespace="http://example.org/", output_path=None, 
                format="turtle", visualize=False, vis_output=None):
    """
    Build a knowledge graph from entities and relations.
    
    Args:
        entities (list): List of (entity_text, entity_type) tuples
        relations (list): List of (subject, predicate, object) tuples
        namespace (str): Base namespace for the knowledge graph
        output_path (str, optional): Output file path
        format (str): Output format for RDF (turtle, xml, json-ld, etc.)
        visualize (bool): Whether to create a visualization
        vis_output (str, optional): Output file path for visualization
        
    Returns:
        KnowledgeGraphBuilder: Knowledge graph builder object
    """
    # Initialize knowledge graph builder
    kg_builder = KnowledgeGraphBuilder(namespace)
    
    # Add entities and relations
    kg_builder.add_entities_and_relations(entities, relations)
    
    # Save to file if output path provided
    if output_path:
        kg_builder.save_to_file(output_path, format)
    
    # Create visualization if requested
    if visualize:
        if vis_output:
            kg_builder.visualize(vis_output)
        else:
            kg_builder.plot_graph()
    
    return kg_builder

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    entities = [
        ("Apple Inc.", "ORG"),
        ("Steve Jobs", "PERSON"),
        ("Tim Cook", "PERSON"),
        ("Cupertino", "GPE"),
        ("California", "GPE")
    ]
    
    relations = [
        (("Apple Inc.", "ORG"), "founded_by", ("Steve Jobs", "PERSON")),
        (("Apple Inc.", "ORG"), "has_ceo", ("Tim Cook", "PERSON")),
        (("Apple Inc.", "ORG"), "located_in", ("Cupertino", "GPE")),
        (("Cupertino", "GPE"), "located_in", ("California", "GPE"))
    ]
    
    # Build knowledge graph
    kg = build_graph(
        entities, 
        relations,
        namespace="http://example.org/",
        output_path="data/knowledge_graph.ttl",
        visualize=True,
        vis_output="data/knowledge_graph.html"
    )
    
    # Query example
    print("\nExample SPARQL query:")
    query = """
    PREFIX ns: <http://example.org/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?company ?founder
    WHERE {
        ?company rdf:type ns:ORG .
        ?founder rdf:type ns:PERSON .
        ?company ns:predicate/founded_by ?founder .
    }
    """
    
    results = kg.query_sparql(query)
    for row in results:
        company = row[0].split('/')[-1]
        founder = row[1].split('/')[-1]
        print(f"Company: {company}, Founder: {founder}")