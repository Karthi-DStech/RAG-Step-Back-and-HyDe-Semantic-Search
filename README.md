# RAG-Step-Back-Hyde-Semantic-Search

### Karthi-Dstech RAG – Step-Back & HyDE Semantic Search

This repository provides an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** with two advanced reasoning techniques:

* **Step-Back Prompting** – generates higher-level “explanatory” questions before answering, improving reasoning depth.
* **HyDE (Hypothetical Document Embeddings)** – creates hypothetical answers first, then embeds them to retrieve semantically similar documents.

The codebase is organized for **training, experimentation, and deployment** of hybrid semantic search applications built on LangChain, OpenAI (or other LLMs), and Vector Store.


### 📂 Project Structure

```
karthi-dstech-rag-step-back-and-hyde-semantic-search/
├── README.md                -- Project overview, setup, and usage instructions
├── call_methods.py          -- CLI helper to trigger Step-Back or HyDE retrieval methods
├── main.py                  -- Main entry point to run the RAG pipeline
├── models/                  -- Retrieval & reasoning strategies
│   ├── hyde.py             -- Hypothetical Document Embeddings (HyDE) implementation
│   └── step_back.py        -- Step-Back prompting logic for deeper reasoning
├── options/                 -- Command-line configuration
│   ├── base_options.py     -- Core argparse options (API keys, Neo4j creds, etc.)
│   └── train_options.py    -- Training / indexing specific options
└── utils/                   -- Utility modules
    ├── configuration.py    -- Environment/config loader and helper functions
    └── document_processor.py -- Loads, cleans, and chunks PDFs/text into LangChain Documents

```


