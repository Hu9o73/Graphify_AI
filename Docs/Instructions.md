# Knowledge Graph Construction

## Project: Web Scraping & Knowledge Base Construction

**Duration:** 3 sessions (4.5h) / 6 sessions (9h)

### Deliverables:
This lab serves as a preparation phase for your project. You need to integrate this lab session into your code and report for the project. Register your work with the following forms:

- A Python script (or Jupyter Notebook) that performs the complete pipeline from text cleaning to knowledge graph construction.
- A brief report summarizing your methodology, challenges faced, and a few sample queries from your knowledge graph along with their results.

## Objective
In this lab, you will construct a pipeline for knowledge graph construction from raw text data. You will perform the following tasks:

1. **Text Cleaning & Preprocessing**: Clean and normalize the provided text dataset.
2. **Named Entity Recognition (NER)**: Train a CRF model to extract named entities from the cleaned text and compare with spaCy's `en_ner_conll03` pre-trained NER model.
3. **Relation Extraction (RE)**: Use spaCy's `en_core_web_sm` model to extract relations between the identified entities.
4. **Knowledge Graph Building**: Convert the extracted entities and relations into RDF triples and load them into a graph database.
5. **Web Scraping**: Fetch and process web content from websites.

The tasks involve many NLP techniques that you haven't learned so far, so you are encouraged to use generative AI tools (such as ChatGPT).

---

## Environment Setup

Before starting with the tasks, ensure that your environment is properly set up. You will need the following tools and libraries:

- **Python**: Make sure you have Python 3.6 or higher installed.
- **Jupyter Notebook**: Install Jupyter Notebook for running and documenting your code.
- **Google Colab**: Alternatively, use Google Colab, which provides a free cloud-based environment with pre-installed libraries.
- **Visual Studio**: Alternatively, you can use VS code's extensions to deal with ipynb files ! (OPTION I TAKE !)
- **Required Libraries**: Install the necessary Python libraries using `pip`:
  ```sh
  pip install datasets sklearn-crfsuite transformers beautifulsoup4 nltk spacy rdflib
  ```
  If using Google Colab, add `!` at the beginning:
  ```sh
  !pip install datasets sklearn-crfsuite transformers beautifulsoup4 nltk spacy
  ```

---

## Datasets
You will use two datasets for different purposes:

1. **Dataset for NER and RE model training: CoNLL-2003 dataset**
   - A widely used dataset for training and evaluating NER systems.
   - Contains annotated text for four types of entities: PERSON, ORGANIZATION, LOCATION, and MISC.
   - Example code to load the dataset:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("conll2003")
     train_dataset = dataset['train']
     validation_dataset = dataset['validation']
     test_dataset = dataset['test']
     ```

2. **Data for application**
   - You need to build your own web scraping program to obtain data from online websites.

---

## Task 1: Model for NER

### 1. Text Cleaning & Preprocessing
- Remove punctuation and non-text elements (except `-` for compound words).
- Normalize the text (convert to lowercase, remove stop words, apply tokenization and stemming/lemmatization).
- Tools: Python (e.g., BeautifulSoup, NLTK, spaCy).

### 2. Named Entity Recognition (NER)
- Train a **Conditional Random Field (CRF)** model using `sklearn-crfsuite`.
- Use spaCy's `en_ner_conll03` pre-trained NER model.
- Compare CRF model with spaCy's model using accuracy, precision, and F1 score.
- Example code to train a CRF model:
  ```python
  import sklearn_crfsuite
  from sklearn_crfsuite import metrics
  
  crf = sklearn_crfsuite.CRF(
      algorithm='lbfgs',
      c1=0.1,
      c2=0.1,
      max_iterations=100,
      all_possible_transitions=False
  )
  crf.fit(X_train, y_train)
  y_pred = crf.predict(X_test)
  print(metrics.flat_classification_report(y_test, y_pred))
  ```

- Example code for spaCy’s NER model:
  ```python
  import spacy
  nlp = spacy.load("./en_ner_conll03")
  text = "Apple was founded by Steve Jobs."
  doc = nlp(text)
  entities = [(ent.text, ent.label_) for ent in doc.ents]
  print(entities)
  ```

### 3. Relation Extraction (RE)
- Use spaCy’s `en_core_web_sm` model to extract relations.
- Example code:
  ```python
  import spacy
  nlp = spacy.load("en_core_web_sm")
  text = "Apple was founded by Steve Jobs."
  doc = nlp(text)
  relations = []
  for token in doc:
      if (token.dep_ == "nsubj" or token.dep_ == "nsubjpass") and token.head.dep_ == "ROOT":
          subject = token.text
          predicate = token.head.text
          for child in token.head.children:
              if child.dep_ == "prep" or child.dep_ == "agent":
                  for obj in child.children:
                      if obj.dep_ == "pobj":
                          relations.append((subject, predicate, obj.text))
  print(relations)
  ```

### 4. Knowledge Graph Building
- Convert entities and relations into RDF triples.
- Load them into a graph database (e.g., RDFLib, Apache JENA).
- Example code using RDFLib:
  ```python
  from rdflib import Graph, URIRef, Namespace
  from rdflib.namespace import RDF
  g = Graph()
  EX = Namespace("http://example.org/")
  g.add((URIRef(EX.Apple), RDF.type, URIRef(EX.Company)))
  g.add((URIRef(EX.SteveJobs), RDF.type, URIRef(EX.Person)))
  g.add((URIRef(EX.Apple), URIRef(EX.founded_by), URIRef(EX.SteveJobs)))
  print(g.serialize(format="xml"))
  ```

---

## Task 2: Pipeline for Knowledge Graph Construction

### 1. Fetch News Articles
- Write a web scraping script to fetch at least 10 news articles from `reuters.com`.
- Extract title, content, and publication date.
- Use Selenium for dynamic content:
  ```python
  from selenium import webdriver
  from selenium.webdriver.chrome.options import Options
  from bs4 import BeautifulSoup
  options = Options()
  options.add_argument("--headless")
  driver = webdriver.Chrome(options=options)
  driver.get("https://www.reuters.com/world/")
  page_source = driver.page_source
  driver.quit()
  soup = BeautifulSoup(page_source, "html.parser")
  ```

### 2. Apply Methods from Task 1
- Clean and preprocess text.
- Perform NER and RE.
- Convert extracted data into RDF triples.
- Build a knowledge graph.

---

This concludes the instructions for constructing a knowledge graph from raw text data.
