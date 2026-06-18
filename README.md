# Medicine Search Engine

A full-stack information retrieval system for searching 224,000+ medicines by name, condition, therapeutic class, or side effect — with an integrated AI chatbot powered by Gemini.

---

## Overview

This project implements a complete IR pipeline on top of a cleaned medicine dataset. Users can search for medicines using natural language queries and receive ranked results scored by TF-IDF and cosine similarity. A built-in chatbot (MediBot) uses Gemini 2.5 Flash to answer follow-up questions about any medicine.

---

## Features

- Search across 224,013 indexed medicines
- TF-IDF + Cosine Similarity ranking
- Inverted Index for fast candidate retrieval
- Precision & Recall evaluation
- MediBot — an AI assistant powered by Gemini 2.5 Flash + the medicine database
- Flask web application with a clean search UI
- Relevance feedback (mark results as relevant)

---

## How It Works

```
User Query
    │
    ▼
Preprocessing (tokenize → remove stopwords → lemmatize)
    │
    ▼
Inverted Index → candidate documents
    │
    ▼
TF-IDF Vectorizer + Cosine Similarity → ranked results
    │
    ▼
Flask UI → display top results with scores
    │
    ▼
MediBot (Gemini 2.5 Flash) → answer follow-up questions
```

## Dataset

| Property | Details |
|---|---|
| File | `medicine_cleaned.csv` |
| Records | 224,013 medicines |
| Columns used | `name`, `therapeutic_class`, `use`, `side_effects` |

---

## Tech Stack

- **Python** — core logic
- **Flask** — web application server
- **scikit-learn** — TF-IDF vectorization and cosine similarity
- **NLTK** — tokenization, stopword removal, lemmatization
- **Google Gemini 2.5 Flash** — MediBot AI chatbot
- **Pandas** — data loading and handling
- **BeautifulSoup** — web crawler fallback
- **Pickle** — index caching for fast startup

---

## Project Structure

```
├── Untitled19finaaal.ipynb     # Main notebook (Flask app + IR pipeline)
├── medicine_cleaned.csv        # Cleaned medicine dataset
├── cache.pkl                   # Cached inverted index and TF-IDF matrix
├── html_files/
│   ├── index.html              # Search UI
│   └── ...
└── README.md
```

## Retrieval Pipeline

**Preprocessing** — lowercasing, URL/HTML removal, stopword filtering, WordNet lemmatization

**Inverted Index** — maps every token to the set of documents containing it, enabling fast candidate lookup

**TF-IDF + Cosine Similarity** — scores candidates against the query vector and returns the top-ranked results

**Evaluation** — Precision and Recall computed via user relevance feedback
