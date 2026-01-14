# Webtoon Recommendation System

## Overview

This project is an **end-to-end recommendation system** built on top of a Retrieval-Augmented Generation (RAG) pipeline.

It is designed as a **portfolio-grade system** that demonstrates not only RAG and LLM usage, but also **data collection, data cleaning, feature engineering, vector search, and system-level design**.

**The data used here was collected, cleaned, and enriched manually** as part of the project.

<img width="1766" height="846" alt="Ekran görüntüsü 2026-01-14 143150" src="https://github.com/user-attachments/assets/b9e3dcbd-6c35-4284-9c21-26979374c747" />

<img width="1765" height="845" alt="Ekran görüntüsü 2026-01-14 150342" src="https://github.com/user-attachments/assets/b65c23a6-c4cc-489a-8687-1e9073e0ac1f" />

---


## What This Project Demonstrates

This repository showcases experience across the full ML system lifecycle:

- Web scraping and data collection
- Data cleaning and schema design
- Mathematical feature extraction
- Vector embeddings and similarity search
- Hybrid retrieval and query classification
- LLM-based reasoning with structured metadata
- Secure, production-aware project design

The goal is not to provide a public demo, but to **show how such a system can be built correctly from scratch**.

---

## Data Collection & Preparation

### Data Collection

The dataset used in this project was **not publicly distributed**.

I collected the data myself by scraping web pages while **strictly respecting platform rules**:

- Followed `robots.txt`
- Collected only series-level information
- No personal or user-identifiable data was accessed

The scraping pipeline is re-runnable and was built using:

- `requests`
- `lxml`
- `pandas`
- standard Python utilities (`re`, `os`, `time`)

### Dataset Schema

The final dataset includes the following columns:

- `series_id`
- `title`
- `author`
- `genre`
- `summary`
- `view`
- `subscribe`
- `likes`
- `last_likes`
- `total_episodes`
- `released_date`
- `popularity`
- `popularity_number`
- `embedding`
- `cover_url`
- `created_at`

---

## Feature Engineering: Popularity Modeling

Raw engagement signals alone are often misleading.

To address this, I modeled **expected like behavior** as a mathematical function of episode order, capturing how community engagement typically decays over time.

A regression-based function `f(k)` was learned to estimate expected likes per episode:

- Intercept (α): `12.8175`
- Decay coefficient (β): `0.2867`
- R² score: `0.1692`
- 
<img width="1189" height="590" alt="download" src="https://github.com/user-attachments/assets/4b62c443-f494-4360-be78-1c4c6b72c84d" />

<img width="1389" height="490" alt="download (1)" src="https://github.com/user-attachments/assets/d65d0c98-c7b5-417c-912a-ce7c7df60020" />

This allowed me to:

- Normalize popularity across series of different lengths
- Detect over- and under-performing episodes
- Derive a more stable `popularity` signal for retrieval

This step highlights **mathematical reasoning and feature extraction beyond standard scraping**.

---

## System Architecture

The project is organized into three high-level components, with a clear separation of responsibilities:

```
.
├── core/                       # Framework-agnostic RAG engine
│   ├── analysis/               # Query classification, metadata extraction, rejection logic
│   ├── database/               # Hybrid retrieval and vector search
│   ├── embeddings/             # Embedding generation logic
│   ├── llm/                    # LLM client abstractions
│   ├── pipeline/               # End-to-end RAG orchestration
│   ├── utils/                  # Shared helpers and statistics utilities
│   └── validator/              # Input validation and constraints
│
├── web/                        # Web interface layer
│   ├── app.py                  # Flask app factory
│   ├── routes.py               # HTTP routes
│   ├── templates/              # HTML templates
│   └── static/                 # CSS and JS assets
│
├── entrypoints/                # Execution entry points
│   ├── cli.py                  # CLI-based interaction
│   └── web_main.py             # Web server startup
│
├── .env.template               # Environment variable template
├── config.py                   # Centralized configuration
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

This structure makes it explicit which parts of the system are reusable logic, which parts are interface-related, and how execution is orchestrated.

### core/



The project is organized into three high-level components:

```
core/         # RAG engine and domain logic
web/          # Flask-based web interface
entrypoints/  # Execution entry points (CLI & web)
```

### core/

Contains all framework-agnostic logic:

- Data retrieval (hybrid vector search)
- Embedding generation
- Query classification
- Metadata extraction
- Threshold tuning and rejection handling
- RAG pipeline orchestration

This layer can be reused independently of any web framework.

### web/

A lightweight Flask application responsible for:

- User interaction
- Request routing
- Rendering results

No business logic lives here. It only orchestrates calls into `core/`.

### entrypoints/

Defines **how the system is executed**:

- CLI-based usage
- Web application startup

Separating entry points keeps execution concerns isolated from logic.

---

## Retrieval-Augmented Generation Pipeline

At a medium level of abstraction, the pipeline works as follows:

1. User query is validated and classified
2. Query embedding is generated
3. Hybrid retrieval is performed using vector similarity and metadata
4. Results are filtered using tuned thresholds
5. Relevant context is passed to the LLM
6. The LLM generates a grounded recommendation

Design decisions intentionally emphasize:


- Precision over verbosity
- Explicit rejection when confidence is low
- Metadata-aware reasoning

---

## Why This Project Cannot Be Run As-Is

This repository **does not include the dataset or credentials**.

That is intentional.

The system requires:

- A private dataset
- Vector embeddings stored in a database
- API credentials owned by the user

If you import your own suitable data and provide the required credentials, **the system is designed to work without architectural changes**.

---

## What I Learned

Through this project, I gained hands-on experience with:

- Designing re-runnable data pipelines
- Cleaning and normalizing real-world data
- Feature extraction using mathematical models
- Building hybrid retrieval systems
- Orchestrating LLMs with structured context
- Designing secure, modular ML systems

The project intentionally spans **data, infrastructure, and modeling**, rather than focusing on a single layer.

---

## Closing Note

This project represents an end-to-end system built from the ground up:

- Data was collected manually
- Features were engineered deliberately
- Models were chosen and tuned intentionally
- Architecture was designed for clarity and reuse

It is meant to demonstrate **how complex ML systems are built in practice**, not just how they look in tutorials.

