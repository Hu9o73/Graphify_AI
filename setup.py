from setuptools import setup, find_packages

setup(
    name="graphify_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "spacy",
        "transformers",
        "datasets",
        "sklearn-crfsuite",
        "beautifulsoup4",
        "requests",
        "selenium",
        "webdriver-manager",
        "rdflib",
        "matplotlib",
        "networkx",
        "pyvis",
        "tqdm",
    ],
    author="Hugo Bonnell",
    author_email="hugo.bonnell@edu.devinci.fr",
    description="AI-Driven Knowledge Graph Builder",
    keywords="NLP, knowledge graph, entity recognition, relation extraction",
    url="https://github.com/Hu9o73/Graphify_AI",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.12"
    ],
    python_requires=">=3.6",
)