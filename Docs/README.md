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
├── data/                      # Data storage
├── models/                    # Trained models storage
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data_collection/       # Web scraping modules
│   ├── preprocessing/         # Text cleaning utilities
│   ├── entity_recognition/    # NER implementations
│   ├── relation_extraction/   # RE implementations
│   └── knowledge_graph/       # Graph building utilities
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # Project documentation
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

1. Collect data through web scraping:
```
python src/data_collection/scraper.py
```

2. Process the data and build knowledge graph:
```
python src/main.py
```

3. Alternatively, you can use the Jupyter notebooks in the `notebooks` directory for step-by-step processing.

### Requirements

- Python 3.6+
- See requirements.txt for package dependencies