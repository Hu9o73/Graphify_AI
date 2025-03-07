#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for comparing different NER models.
This module provides functions to compare CRF and spaCy NER models.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score as seq_accuracy_score

from .ner import SpacyNERExtractor, CRFExtractor

# Initialize logging
logger = logging.getLogger(__name__)

class NERComparison:
    """Class for comparing NER models."""
    
    def __init__(self, crf_model_path=None):
        """
        Initialize the comparison.
        
        Args:
            crf_model_path (str, optional): Path to saved CRF model
        """
        self.spacy_extractor = SpacyNERExtractor()
        self.crf_extractor = CRFExtractor(crf_model_path)
    
    def _convert_bio_to_tokens(self, doc, labels):
        """
        Convert BIO labels to entity tokens for evaluation.
        
        Args:
            doc (spacy.tokens.Doc): spaCy document
            labels (list): List of BIO labels
            
        Returns:
            list: List of entity tokens
        """
        tokens = []
        entity_tokens = []
        
        for token, label in zip(doc, labels):
            tokens.append(token.text)
            entity_tokens.append(label)
        
        return tokens, entity_tokens
    
    def _convert_to_bio(self, entities, doc_length):
        """
        Convert entity spans to BIO tags.
        
        Args:
            entities (list): List of (start, end, label) entity spans
            doc_length (int): Total number of tokens in document
            
        Returns:
            list: List of BIO tags
        """
        bio_tags = ['O'] * doc_length
        
        for start, end, label in entities:
            bio_tags[start] = f'B-{label}'
            for i in range(start + 1, end):
                bio_tags[i] = f'I-{label}'
        
        return bio_tags
    
    def evaluate_on_dataset(self, texts, gold_entities, return_predictions=False):
        """
        Evaluate models on a dataset.
        
        Args:
            texts (list): List of text strings
            gold_entities (list): List of gold standard entity annotations
            return_predictions (bool): Whether to return model predictions
            
        Returns:
            dict: Evaluation metrics for each model
            (optional) dict: Model predictions if return_predictions is True
        """
        logger.info("Evaluating NER models on dataset")
        
        spacy_predictions = []
        crf_predictions = []
        
        # Process each text
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            
            # Process with spaCy
            doc = self.spacy_extractor.nlp(text)
            spacy_entities = [(ent.start, ent.end, ent.label_) for ent in doc.ents]
            spacy_bio = self._convert_to_bio(spacy_entities, len(doc))
            spacy_predictions.append(spacy_bio)
            
            # Process with CRF (if model is available)
            if self.crf_extractor.model is not None:
                crf_entities = self.crf_extractor.extract_entities(text)
                # Convert to same format as spaCy entities
                # This would require additional processing for CRF output
                # Simplified here - would need to be implemented based on actual CRF output
                crf_bio = ['O'] * len(doc)  # Placeholder
                crf_predictions.append(crf_bio)
        
        # Calculate metrics for spaCy
        spacy_metrics = self._calculate_metrics(gold_entities, spacy_predictions)
        
        # Calculate metrics for CRF (if model is available)
        crf_metrics = None
        if self.crf_extractor.model is not None:
            crf_metrics = self._calculate_metrics(gold_entities, crf_predictions)
        
        results = {
            'spacy': spacy_metrics,
            'crf': crf_metrics
        }
        
        if return_predictions:
            predictions = {
                'spacy': spacy_predictions,
                'crf': crf_predictions if self.crf_extractor.model is not None else None
            }
            return results, predictions
        
        return results
    
    def _calculate_metrics(self, true_labels, pred_labels):
        """
        Calculate evaluation metrics.
        
        Args:
            true_labels (list): List of true BIO labels
            pred_labels (list): List of predicted BIO labels
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Calculate metrics using seqeval for entity-level evaluation
            accuracy = seq_accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels)
            
            # Get detailed report
            report = seq_classification_report(true_labels, pred_labels, output_dict=True)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'report': report
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return None
    
    def compare_models(self, texts, gold_entities):
        """
        Compare NER models on a dataset.
        
        Args:
            texts (list): List of text strings
            gold_entities (list): List of gold standard entity annotations
            
        Returns:
            dict: Comparison results
        """
        logger.info("Comparing NER models")
        
        # Evaluate models
        evaluation_results = self.evaluate_on_dataset(texts, gold_entities)
        
        # Print comparison
        print("\n=== NER Models Comparison ===\n")
        
        # SpaCy metrics
        spacy_metrics = evaluation_results['spacy']
        print("SpaCy NER Metrics:")
        print(f"- Accuracy: {spacy_metrics['accuracy']:.4f}")
        print(f"- Precision: {spacy_metrics['precision']:.4f}")
        print(f"- Recall: {spacy_metrics['recall']:.4f}")
        print(f"- F1 Score: {spacy_metrics['f1']:.4f}")
        
        # CRF metrics (if available)
        crf_metrics = evaluation_results['crf']
        if crf_metrics:
            print("\nCRF NER Metrics:")
            print(f"- Accuracy: {crf_metrics['accuracy']:.4f}")
            print(f"- Precision: {crf_metrics['precision']:.4f}")
            print(f"- Recall: {crf_metrics['recall']:.4f}")
            print(f"- F1 Score: {crf_metrics['f1']:.4f}")
        
        # Plot comparison
        self.plot_comparison(evaluation_results)
        
        return evaluation_results
    
    def plot_comparison(self, evaluation_results, output_path=None):
        """
        Plot model comparison.
        
        Args:
            evaluation_results (dict): Evaluation metrics for each model
            output_path (str, optional): Path to save the plot
        """
        try:
            # Prepare data for plotting
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            spacy_values = [evaluation_results['spacy'][m] for m in metrics]
            
            if evaluation_results['crf']:
                crf_values = [evaluation_results['crf'][m] for m in metrics]
                models = ['SpaCy', 'CRF']
                values = [spacy_values, crf_values]
            else:
                models = ['SpaCy']
                values = [spacy_values]
            
            # Set up plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bar positions
            bar_width = 0.35
            index = np.arange(len(metrics))
            
            # Create bars
            for i, (model, vals) in enumerate(zip(models, values)):
                ax.bar(index + i * bar_width, vals, bar_width, label=model)
            
            # Customize plot
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Scores')
            ax.set_title('NER Models Comparison')
            ax.set_xticks(index + bar_width / 2 if len(models) > 1 else index)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for i, (model, vals) in enumerate(zip(models, values)):
                for j, v in enumerate(vals):
                    ax.text(j + i * bar_width, v + 0.01, f'{v:.2f}', 
                            ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save or show plot
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Plot saved to {output_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting comparison: {e}")

def compare_models(texts, gold_entities, crf_model_path=None, output_plot=None):
    """
    Compare NER models on a dataset.
    
    Args:
        texts (list): List of text strings
        gold_entities (list): List of gold standard entity annotations
        crf_model_path (str, optional): Path to saved CRF model
        output_plot (str, optional): Path to save comparison plot
        
    Returns:
        dict: Comparison results
    """
    comparison = NERComparison(crf_model_path)
    return comparison.compare_models(texts, gold_entities)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    sample_texts = [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.",
        "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque, New Mexico."
    ]
    
    # This is just a placeholder - real gold standard would be needed
    gold_entities = [
        ['O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-PERSON', 'I-PERSON', 'O', 'B-PERSON', 'I-PERSON', 'O', 'O', 'B-PERSON', 'I-PERSON', 'O', 'B-GPE', 'O', 'B-GPE', 'O'],
        ['B-ORG', 'O', 'O', 'O', 'B-PERSON', 'I-PERSON', 'O', 'B-PERSON', 'I-PERSON', 'O', 'B-GPE', 'O', 'B-GPE', 'O']
    ]
    
    print("This is a demonstration. Real gold standard annotations would be needed for actual evaluation.")
    # compare_models(sample_texts, gold_entities)