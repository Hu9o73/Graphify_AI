# Graphify_AI

## AI-Driven Knowledge Graph Builder

Graphify_AI is a project that builds knowledge graphs from raw text data using natural language processing techniques. It extracts entities and relationships from text and organizes them in a structured graph format.

### Features

- Text cleaning and preprocessing
- Named Entity Recognition (NER) using both CRF and pre-trained models
- Relation Extraction (RE) between identified entities
- Knowledge Graph construction using RDF triples
- Web scraping functionality to collect data from news websites

### Project Structure

```
Graphify_AI/
├── Docs/                      # Documents
│   ├── requirements.txt       # Dependencies
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── *.ipynb                # Exploration notebook
│   ├── output/                # Exploration's outputs (data, graphs...)
└── src/                       # Source code
    ├── data_collection/       # Web scraping modules
    ├── preprocessing/         # Text cleaning utilities
    ├── entity_recognition/    # NER implementations
    ├── relation_extraction/   # RE implementations
    └── knowledge_graph/       # Graph building utilities
```

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/Graphify_AI.git
cd Graphify_AI
```

2. Install the dependencies:
```
pip install -r requirements.txt
```

### Usage

You can use the Jupyter notebooks in the `notebooks` directory for step-by-step processing.

### Requirements

- Python 3.6+ (Tested on python 3.12)
- See requirements.txt for package dependencies