#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Augmentation module.
This module provides functions to augment a knowledge graph with external knowledge.
"""

import os
import logging
import time
import json
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

# Initialize logging
logger = logging.getLogger(__name__)

class KnowledgeGraphAugmenter:
    """Class for augmenting a knowledge graph with external knowledge."""
    
    def __init__(self, namespace="http://example.org/"):
        """
        Initialize the knowledge graph augmenter.
        
        Args:
            namespace (str): Base namespace for the knowledge graph
        """
        self.namespace = namespace
        self.graph = Graph()
        self.graph.bind("ns", Namespace(namespace))
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("xsd", XSD)
        
        # Cache for entity linking
        self.entity_cache = {}
        
        logger.info(f"Initialized knowledge graph augmenter with namespace: {namespace}")
    
    def load_graph(self, rdf_file, format="turtle"):
        """
        Load an existing knowledge graph.
        
        Args:
            rdf_file (str): Path to RDF file
            format (str): RDF format (turtle, xml, etc.)
            
        Returns:
            bool: Success flag
        """
        try:
            self.graph.parse(rdf_file, format=format)
            logger.info(f"Loaded knowledge graph from {rdf_file} with {len(self.graph)} triples")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
    
    def save_graph(self, output_file, format="turtle"):
        """
        Save the augmented knowledge graph.
        
        Args:
            output_file (str): Path to output file
            format (str): RDF format (turtle, xml, etc.)
            
        Returns:
            bool: Success flag
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Save the graph
            self.graph.serialize(destination=output_file, format=format)
            logger.info(f"Saved knowledge graph to {output_file} with {len(self.graph)} triples")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def extract_entities(self):
        """
        Extract entities from the knowledge graph.
        
        Returns:
            list: List of (entity_uri, entity_type) tuples
        """
        entities = []
        
        # Look for entities by finding subjects with rdf:type
        for s, p, o in self.graph.triples((None, RDF.type, None)):
            # Get the entity type
            entity_type = str(o).split('/')[-1]
            
            # Add to the list
            entities.append((str(s), entity_type))
        
        logger.info(f"Extracted {len(entities)} entities from knowledge graph")
        return entities
    
    def link_entity_to_dbpedia(self, entity_uri, entity_text, entity_type, lang="en", cache=True):
        """
        Link an entity to DBpedia using a very simple and direct approach.
        """
        try:
            # Check cache first
            cache_key = f"dbpedia:{entity_text}:{entity_type}"
            if cache and cache_key in self.entity_cache:
                return self.entity_cache[cache_key]
            
            # Clean entity text - just basic cleaning
            clean_text = entity_text.replace('"', '').strip()
            
            # Try two main approaches:
            
            # 1. Direct resource lookup (most reliable for well-known entities)
            # Convert "Central Asia" to "Central_Asia" for the URI format
            resource_name = clean_text.replace(' ', '_')
            direct_uri = f"http://dbpedia.org/resource/{resource_name}"
            
            # Check if this resource exists with a very simple query
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setTimeout(5)
            
            direct_check_query = f"""
            ASK {{ <{direct_uri}> ?p ?o }}
            """
            
            sparql.setQuery(direct_check_query)
            sparql.setReturnFormat(JSON)
            
            try:
                result = sparql.query().convert()
                if result.get('boolean', False):
                    # Resource exists!
                    if cache:
                        self.entity_cache[cache_key] = direct_uri
                    logger.debug(f"Linked '{entity_text}' directly to DBpedia: {direct_uri}")
                    return direct_uri
            except Exception as e:
                logger.debug(f"Direct resource check failed: {e}")
            
            # 2. Simple label lookup without constraints
            simple_query = f"""
            SELECT DISTINCT ?s WHERE {{
            ?s rdfs:label ?label .
            FILTER(LCASE(STR(?label)) = LCASE("{clean_text}") && LANG(?label) = '{lang}')
            }}
            LIMIT 1
            """
            
            sparql.setQuery(simple_query)
            sparql.setReturnFormat(JSON)
            
            try:
                results = sparql.query().convert()
                if results["results"]["bindings"]:
                    dbpedia_uri = results["results"]["bindings"][0]["s"]["value"]
                    if cache:
                        self.entity_cache[cache_key] = dbpedia_uri
                    logger.debug(f"Linked '{entity_text}' to DBpedia via label: {dbpedia_uri}")
                    return dbpedia_uri
            except Exception as e:
                logger.debug(f"Label lookup failed: {e}")
            
            # Nothing found
            logger.debug(f"No DBpedia entity found for '{entity_text}'")
            if cache:
                self.entity_cache[cache_key] = None
            return None
            
        except Exception as e:
            logger.warning(f"Error linking entity to DBpedia: {e}")
            return None
    
    def link_entity_to_wikidata(self, entity_uri, entity_text, entity_type, lang="en", cache=True):
        """
        Link an entity to Wikidata.
        
        Args:
            entity_uri (str): URI of the entity
            entity_text (str): Text of the entity
            entity_type (str): Type of the entity
            lang (str): Language for Wikidata lookup
            cache (bool): Whether to use the cache
            
        Returns:
            str: Wikidata URI for the entity, or None if not found
        """
        try:
            # Check cache first
            cache_key = f"wikidata:{entity_text}:{entity_type}"
            if cache and cache_key in self.entity_cache:
                return self.entity_cache[cache_key]
            
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
            
            # Query Wikidata
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setTimeout(5)  
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            
            SELECT DISTINCT ?s WHERE {{
              ?s rdfs:label ?label .
              FILTER(LANG(?label) = '{lang}')
              FILTER(LCASE(STR(?label)) = LCASE("{safe_entity_text}"))
              
              # Add type constraint if available
              {f"?s wdt:P31/wdt:P279* {wikidata_class} ." if wikidata_class else ""}
            }}
            LIMIT 1
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            # Extract results
            if results["results"]["bindings"]:
                wikidata_uri = results["results"]["bindings"][0]["s"]["value"]
                
                # Add to cache
                if cache:
                    self.entity_cache[cache_key] = wikidata_uri
                
                logger.debug(f"Linked '{entity_text}' to Wikidata: {wikidata_uri}")
                return wikidata_uri
            
            logger.debug(f"No Wikidata entity found for '{entity_text}'")
            
            # Add to cache as None
            if cache:
                self.entity_cache[cache_key] = None
                
            return None
            
        except Exception as e:
            logger.error(f"Error linking entity to Wikidata: {e}")
            return None
    
    def get_entity_metadata(self, entity_uri):
        """
        Get metadata for an entity from the graph.
        
        Args:
            entity_uri (str): URI of the entity
            
        Returns:
            tuple: (entity_text, entity_type)
        """
        try:
            entity = URIRef(entity_uri)
            
            # Get entity label
            label = None
            for _, _, o in self.graph.triples((entity, RDFS.label, None)):
                label = str(o)
                break
            
            # Get entity type
            entity_type = None
            for _, _, o in self.graph.triples((entity, RDF.type, None)):
                entity_type = str(o).split('/')[-1]
                break
            
            return label, entity_type
            
        except Exception as e:
            logger.error(f"Error getting entity metadata: {e}")
            return None, None
    
    def add_dbpedia_info(self, entity_uri, dbpedia_uri):
        """
        Add information from DBpedia to an entity.
        
        Args:
            entity_uri (str): URI of the entity
            dbpedia_uri (str): DBpedia URI for the entity
            
        Returns:
            int: Number of triples added
        """
        try:
            # Query DBpedia for properties
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setTimeout(5)  
            query = f"""
            SELECT DISTINCT ?p ?o WHERE {{
              <{dbpedia_uri}> ?p ?o .
              
              # Filter out some common predicates
              FILTER(?p NOT IN (
                rdf:type, rdfs:label, rdfs:comment, owl:sameAs,
                <http://purl.org/dc/terms/subject>, <http://dbpedia.org/ontology/wikiPageID>,
                <http://dbpedia.org/ontology/wikiPageRevisionID>, <http://dbpedia.org/ontology/wikiPageWikiLink>
              ))
              
              # Filter complex objects
              FILTER(isURI(?o) || isLiteral(?o))
            }}
            LIMIT 50
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            # Add triples to graph
            added_count = 0
            entity = URIRef(entity_uri)
            dbpedia_entity = URIRef(dbpedia_uri)
            
            # Add owl:sameAs triple
            self.graph.add((entity, URIRef("http://www.w3.org/2002/07/owl#sameAs"), dbpedia_entity))
            added_count += 1
            
            # Process results
            for result in results["results"]["bindings"]:
                p = result["p"]["value"]
                o = result["o"]["value"]
                
                # Skip triples with predicates we don't want
                if "wiki" in p or "Wiki" in p:
                    continue
                
                # Add triple using original predicate
                predicate = URIRef(p)
                
                # Create appropriate object based on type
                if result["o"]["type"] == "uri":
                    obj = URIRef(o)
                else:
                    # Handle literals with language or datatype
                    if "xml:lang" in result["o"]:
                        obj = Literal(o, lang=result["o"]["xml:lang"])
                    elif "datatype" in result["o"]:
                        obj = Literal(o, datatype=URIRef(result["o"]["datatype"]))
                    else:
                        obj = Literal(o)
                
                # Add triple to graph
                self.graph.add((entity, predicate, obj))
                added_count += 1
            
            logger.debug(f"Added {added_count} triples from DBpedia for {entity_uri}")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding DBpedia info: {e}")
            return 0
    
    def add_wikidata_info(self, entity_uri, wikidata_uri):
        """
        Add information from Wikidata to an entity.
        
        Args:
            entity_uri (str): URI of the entity
            wikidata_uri (str): Wikidata URI for the entity
            
        Returns:
            int: Number of triples added
        """
        try:
            # Query Wikidata for properties
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setTimeout(5)  
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            
            SELECT ?p ?pLabel ?o ?oLabel WHERE {{
              <{wikidata_uri}> ?p ?o .
              
              # Only include direct properties
              FILTER(STRSTARTS(STR(?p), STR(wdt:)))
              
              # Get labels for properties
              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
                ?p rdfs:label ?pLabel .
              }}
              
              # Get labels for objects if they are entities
              OPTIONAL {{
                SERVICE wikibase:label {{
                  bd:serviceParam wikibase:language "en" .
                  ?o rdfs:label ?oLabel .
                }}
              }}
            }}
            LIMIT 50
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            # Add triples to graph
            added_count = 0
            entity = URIRef(entity_uri)
            wikidata_entity = URIRef(wikidata_uri)
            
            # Add owl:sameAs triple
            self.graph.add((entity, URIRef("http://www.w3.org/2002/07/owl#sameAs"), wikidata_entity))
            added_count += 1
            
            # Process results
            for result in results["results"]["bindings"]:
                p = result["p"]["value"]
                p_label = result["pLabel"]["value"] if "pLabel" in result else p.split('/')[-1]
                o = result["o"]["value"]
                o_label = result["oLabel"]["value"] if "oLabel" in result else o
                
                # Create predicate URI in our namespace
                predicate_name = p_label.lower().replace(' ', '_')
                predicate = URIRef(f"{self.namespace}predicate/{predicate_name}")
                
                # Create object
                if "http" in o:  # It's a URI
                    # Create a local entity for this object
                    obj_entity = URIRef(f"{self.namespace}entity/{o.split('/')[-1]}")
                    self.graph.add((obj_entity, RDF.type, URIRef(f"{self.namespace}WIKIDATA")))
                    self.graph.add((obj_entity, RDFS.label, Literal(o_label)))
                    self.graph.add((obj_entity, URIRef("http://www.w3.org/2002/07/owl#sameAs"), URIRef(o)))
                    
                    # Add the triple with our local object
                    self.graph.add((entity, predicate, obj_entity))
                else:  # It's a literal
                    self.graph.add((entity, predicate, Literal(o)))
                
                added_count += 1
            
            logger.debug(f"Added {added_count} triples from Wikidata for {entity_uri}")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding Wikidata info: {e}")
            return 0
    
    def add_dbpedia_connections(self, entity_uri, dbpedia_uri, depth=1):
        """
        Add connections between entities from DBpedia.
        
        Args:
            entity_uri (str): URI of the entity
            dbpedia_uri (str): DBpedia URI for the entity
            depth (int): Depth of connections to add
            
        Returns:
            int: Number of triples added
        """
        try:
            # Query DBpedia for connections
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setTimeout(5)  
            query = f"""
            SELECT DISTINCT ?p ?o ?oLabel WHERE {{
              <{dbpedia_uri}> ?p ?o .
              
              # Only include URIs as objects
              FILTER(isURI(?o))
              
              # Filter out some common predicates
              FILTER(?p NOT IN (
                rdf:type, rdfs:label, rdfs:comment, owl:sameAs,
                <http://purl.org/dc/terms/subject>, <http://dbpedia.org/ontology/wikiPageID>,
                <http://dbpedia.org/ontology/wikiPageRevisionID>, <http://dbpedia.org/ontology/wikiPageWikiLink>
              ))
              
              # Get label for object
              OPTIONAL {{
                ?o rdfs:label ?oLabel .
                FILTER(LANG(?oLabel) = "en")
              }}
            }}
            LIMIT 25
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            # Add triples to graph
            added_count = 0
            entity = URIRef(entity_uri)
            
            # Process results
            connection_entities = []  # To store entities for deeper connections
            
            for result in results["results"]["bindings"]:
                p = result["p"]["value"]
                o = result["o"]["value"]
                o_label = result["oLabel"]["value"] if "oLabel" in result else o.split('/')[-1]
                
                # Skip triples with predicates we don't want
                if "wiki" in p or "Wiki" in p:
                    continue
                
                # Create predicate URI in our namespace
                predicate_name = p.split('/')[-1]
                predicate = URIRef(f"{self.namespace}predicate/{predicate_name}")
                
                # Create a local entity for this object
                obj_entity = URIRef(f"{self.namespace}entity/{o.split('/')[-1]}")
                self.graph.add((obj_entity, RDF.type, URIRef(f"{self.namespace}DBPEDIA")))
                self.graph.add((obj_entity, RDFS.label, Literal(o_label)))
                self.graph.add((obj_entity, URIRef("http://www.w3.org/2002/07/owl#sameAs"), URIRef(o)))
                
                # Add the triple with our local object
                self.graph.add((entity, predicate, obj_entity))
                added_count += 1
                
                # Add to connection entities for deeper exploration
                connection_entities.append((obj_entity, o))
            
            # Recursively add deeper connections if needed
            if depth > 1 and connection_entities:
                # Limit the number of entities to explore for performance
                for i, (conn_entity, conn_dbpedia) in enumerate(connection_entities[:5]):
                    added_count += self.add_dbpedia_connections(
                        str(conn_entity), conn_dbpedia, depth=depth-1
                    )
            
            logger.debug(f"Added {added_count} connection triples from DBpedia for {entity_uri}")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding DBpedia connections: {e}")
            return 0
    
    def enrich_entity(self, entity_uri, use_dbpedia=True, use_wikidata=True, connection_depth=1, 
                 retry_count=3, sleep_time=1):
        """
        Enrich an entity with external knowledge, with early termination for entities
        that are unlikely to yield useful results.
        
        Args:
            entity_uri (str): URI of the entity
            use_dbpedia (bool): Whether to use DBpedia
            use_wikidata (bool): Whether to use Wikidata
            connection_depth (int): Depth of connections to add
            retry_count (int): Number of retries for API calls
            sleep_time (int): Time to sleep between retries
            
        Returns:
            int: Number of triples added
        """
        added_count = 0
        
        # Get entity metadata
        entity_text, entity_type = self.get_entity_metadata(entity_uri)
        if not entity_text or not entity_type:
            logger.warning(f"Could not get metadata for entity {entity_uri}")
            return 0
        
        # Optimizing the enrichment process by skipping certain entities (To go faster, else it's way too slow)

        # Skip singleton/unclear entities (likely to be noise)
        if len(entity_text.split()) < 2 and entity_type not in ['PERSON', 'ORG', 'GPE', 'LOC']:
            # logger.info(f"Skipping likely non-productive entity: {entity_text} ({entity_type})")
            return 0
        
        # Check if entity is already in cache with negative results
        cache_key = f"enrich:{entity_text}:{entity_type}"
        if cache_key in self.entity_cache and self.entity_cache[cache_key] is False:
            # logger.info(f"Skipping previously unproductive entity: {entity_text}")
            return 0
        
        # Try quick entity linking first to see if any sources recognize this entity
        has_potential = False
        
        # Do a quick check with DBpedia
        if use_dbpedia:
            try:
                dbpedia_uri = self.link_entity_to_dbpedia(entity_uri, entity_text, entity_type)
                if dbpedia_uri:
                    has_potential = True
            except Exception as e:
                logger.debug(f"Error in quick DBpedia check: {e}")
        
        # Do a quick check with Wikidata if still no potential
        if use_wikidata and not has_potential:
            try:
                wikidata_uri = self.link_entity_to_wikidata(entity_uri, entity_text, entity_type)
                if wikidata_uri:
                    has_potential = True
            except Exception as e:
                logger.debug(f"Error in quick Wikidata check: {e}")
        
        # If entity has no potential, cache this result and skip
        if not has_potential:
            self.entity_cache[cache_key] = False
            # logger.info(f"Skipping entity with no external knowledge: {entity_text}")
            return 0
        
        # Proceed with full enrichment since entity has potential
        
        # Add DBpedia info
        if use_dbpedia:
            for i in range(retry_count):
                try:
                    dbpedia_uri = self.link_entity_to_dbpedia(entity_uri, entity_text, entity_type)
                    if dbpedia_uri:
                        new_triples = self.add_dbpedia_info(entity_uri, dbpedia_uri)
                        added_count += new_triples
                        
                        # Only add connections if we got some basic info
                        if new_triples > 0 and connection_depth > 0:
                            added_count += self.add_dbpedia_connections(entity_uri, dbpedia_uri, depth=connection_depth)
                    break
                except Exception as e:
                    logger.warning(f"Error enriching with DBpedia (attempt {i+1}): {e}")
                    if i < retry_count - 1:
                        time.sleep(sleep_time)
        
        # Add Wikidata info
        if use_wikidata:
            for i in range(retry_count):
                try:
                    wikidata_uri = self.link_entity_to_wikidata(entity_uri, entity_text, entity_type)
                    if wikidata_uri:
                        added_count += self.add_wikidata_info(entity_uri, wikidata_uri)
                    break
                except Exception as e:
                    logger.warning(f"Error enriching with Wikidata (attempt {i+1}): {e}")
                    if i < retry_count - 1:
                        time.sleep(sleep_time)
        
        # Cache the result
        if added_count == 0:
            self.entity_cache[cache_key] = False
        
        # logger.info(f"Added {added_count} triples for entity {entity_uri}")
        return added_count
    
    def enrich_all_entities(self, entity_types=None, max_entities=None, use_dbpedia=True, 
                           use_wikidata=True, connection_depth=1, save_interval=10, 
                           output_file=None):
        """
        Enrich all entities in the knowledge graph.
        
        Args:
            entity_types (list): List of entity types to enrich
            max_entities (int): Maximum number of entities to enrich
            use_dbpedia (bool): Whether to use DBpedia
            use_wikidata (bool): Whether to use Wikidata
            connection_depth (int): Depth of connections to add
            save_interval (int): Interval at which to save progress
            output_file (str): Path to save the augmented graph
            
        Returns:
            int: Number of triples added
        """
        # Extract entities
        entities = self.extract_entities()
        
        # Filter by entity type if specified
        if entity_types:
            entities = [(uri, t) for uri, t in entities if t in entity_types]
        
        # Limit number of entities if specified
        if max_entities and max_entities < len(entities):
            # Randomly select entities
            indices = np.random.choice(len(entities), max_entities, replace=False)
            entities = [entities[i] for i in indices]
        
        logger.info(f"Enriching {len(entities)} entities...")
        
        # Save original triple count
        original_count = len(self.graph)
        
        # Process entities
        added_count = 0
        for i, (entity_uri, entity_type) in enumerate(tqdm(entities)):
            # Enrich entity
            added = self.enrich_entity(
                entity_uri, 
                use_dbpedia=use_dbpedia, 
                use_wikidata=use_wikidata, 
                connection_depth=connection_depth
            )
            added_count += added
            
            # Save progress at intervals
            if output_file and (i + 1) % save_interval == 0:
                self.save_graph(output_file)
                logger.info(f"Progress saved: {i+1}/{len(entities)} entities processed")
        
        # Final save
        if output_file:
            self.save_graph(output_file)
        
        # Log summary
        logger.info(f"Enrichment complete:")
        logger.info(f"- Original triples: {original_count}")
        logger.info(f"- Added triples: {added_count}")
        logger.info(f"- Total triples: {len(self.graph)}")
        
        return added_count
    
    def save_entity_links(self, output_file):
        """
        Save entity links to a JSON file.
        
        Args:
            output_file (str): Path to output file
            
        Returns:
            bool: Success flag
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Extract sameAs links
            same_as_uri = URIRef("http://www.w3.org/2002/07/owl#sameAs")
            
            entity_links = {}
            for s, p, o in self.graph.triples((None, same_as_uri, None)):
                s_str = str(s)
                o_str = str(o)
                
                if s_str not in entity_links:
                    entity_links[s_str] = []
                
                # Add link with its source (DBpedia or Wikidata)
                if "dbpedia.org" in o_str:
                    entity_links[s_str].append({"type": "dbpedia", "uri": o_str})
                elif "wikidata.org" in o_str:
                    entity_links[s_str].append({"type": "wikidata", "uri": o_str})
                else:
                    entity_links[s_str].append({"type": "other", "uri": o_str})
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(entity_links, f, indent=2)
            
            logger.info(f"Saved entity links for {len(entity_links)} entities to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving entity links: {e}")
            return False
    
    def compute_enrichment_stats(self):
        """
        Compute statistics about the knowledge graph enrichment.
        
        Returns:
            dict: Dictionary of statistics
        """
        try:
            # Get all entities
            entities = self.extract_entities()
            
            # Count entities by type
            entity_type_counts = {}
            for _, entity_type in entities:
                if entity_type not in entity_type_counts:
                    entity_type_counts[entity_type] = 0
                entity_type_counts[entity_type] += 1
            
            # Count linked entities
            same_as_uri = URIRef("http://www.w3.org/2002/07/owl#sameAs")
            linked_entities = set()
            dbpedia_links = 0
            wikidata_links = 0
            
            for s, p, o in self.graph.triples((None, same_as_uri, None)):
                linked_entities.add(str(s))
                
                if "dbpedia.org" in str(o):
                    dbpedia_links += 1
                elif "wikidata.org" in str(o):
                    wikidata_links += 1
            
            # Compute stats
            total_entities = len(entities)
            linked_count = len(linked_entities)
            linked_percentage = (linked_count / total_entities * 100) if total_entities > 0 else 0
            
            # Get total triples
            total_triples = len(self.graph)
            
            stats = {
                "total_entities": total_entities,
                "entity_types": entity_type_counts,
                "linked_entities": linked_count,
                "linked_percentage": linked_percentage,
                "dbpedia_links": dbpedia_links,
                "wikidata_links": wikidata_links,
                "total_triples": total_triples
            }
            
            logger.info("Knowledge Graph Enrichment Statistics:")
            logger.info(f"- Total entities: {total_entities}")
            logger.info(f"- Entity types: {entity_type_counts}")
            logger.info(f"- Linked entities: {linked_count} ({linked_percentage:.2f}%)")
            logger.info(f"- DBpedia links: {dbpedia_links}")
            logger.info(f"- Wikidata links: {wikidata_links}")
            logger.info(f"- Total triples: {total_triples}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing enrichment stats: {e}")
            return {}

def augment_knowledge_graph(input_file, output_file, entity_types=None, max_entities=None, 
                          use_dbpedia=True, use_wikidata=True, connection_depth=1, 
                          namespace="http://example.org/"):
    """
    Augment a knowledge graph with external knowledge.
    
    Args:
        input_file (str): Path to input RDF file
        output_file (str): Path to output RDF file
        entity_types (list): List of entity types to enrich
        max_entities (int): Maximum number of entities to enrich
        use_dbpedia (bool): Whether to use DBpedia
        use_wikidata (bool): Whether to use Wikidata
        connection_depth (int): Depth of connections to add
        namespace (str): Base namespace for the knowledge graph
        
    Returns:
        KnowledgeGraphAugmenter: KnowledgeGraphAugmenter instance
    """
    # Create augmenter
    augmenter = KnowledgeGraphAugmenter(namespace=namespace)
    
    # Load graph
    if not augmenter.load_graph(input_file):
        logger.error(f"Failed to load knowledge graph from {input_file}")
        return None
    
    # Enrich entities
    augmenter.enrich_all_entities(
        entity_types=entity_types,
        max_entities=max_entities,
        use_dbpedia=use_dbpedia,
        use_wikidata=use_wikidata,
        connection_depth=connection_depth,
        output_file=output_file
    )
    
    # Compute and display stats
    stats = augmenter.compute_enrichment_stats()
    
    # Save entity links
    links_file = os.path.splitext(output_file)[0] + "_links.json"
    augmenter.save_entity_links(links_file)
    
    return augmenter

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    input_file = "data/knowledge_graph.ttl"
    output_file = "data/knowledge_graph_augmented.ttl"
    
    augmenter = augment_knowledge_graph(
        input_file=input_file,
        output_file=output_file,
        entity_types=["PERSON", "ORG", "GPE"],
        max_entities=10,  # Limit for demonstration
        use_dbpedia=True,
        use_wikidata=True,
        connection_depth=1
    )