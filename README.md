# Distributed RAG Pipeline with Apache Spark

Distributed Retrieval-Augmented Generation (RAG) pipeline built with Apache Spark, semantic embeddings, and instruction-tuned LLMs.

The project processes PDF course material, generates distributed embeddings, retrieves relevant chunks using similarity search, and uses an LLM to generate grounded exam questions with source-aware context.

## Project Overview

This project implements an end-to-end distributed RAG workflow:

1. PDF ingestion and preprocessing with PySpark
2. Text cleaning and chunking
3. Embedding generation using TF-IDF, Word2Vec, and SBERT
4. Distributed similarity retrieval with Apache Spark
5. Prompt construction and LLM-based question generation
6. Retrieval and generation evaluation

## Architecture

```text
PDF documents
      |
      v
PySpark preprocessing
      |
      v
Cleaning + chunking
      |
      v
Embedding generation
  |      |      |
TF-IDF Word2Vec SBERT
  \      |      /
      Parquet
        |
        v
User query
        |
        v
Query embedding
        |
        v
Distributed Spark retrieval
        |
        v
Top-K relevant chunks
        |
        v
Grounded LLM prompt
        |
        v
Questions + answers + source context
        |
        v
Evaluation
