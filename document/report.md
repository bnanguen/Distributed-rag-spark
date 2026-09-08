# INFO-H-515 — Distributed RAG System for Exam Question Generation
**Team 18** | Project 2025–2026

---

## 1. System Architecture

The system implements a distributed Retrieval-Augmented Generation (RAG) pipeline
split across three tasks. Figure 1 summarises the end-to-end flow.

```
PDF corpus
    │
    ▼
[Task 1 — PySpark]
PDF ingestion (pypdf)
    → page-level text extraction
    → cleaning & filtering
    → word-level chunking (300 words, 20% overlap)
    → embedding (TF-IDF / Word2Vec / SBERT)
    → Parquet storage (partitioned by strategy)
    │
    ▼
[Task 2 — PySpark + HuggingFace]
Query embedding (same strategy as corpus)
    → distributed similarity scoring (PySpark RDD broadcast + map)
    → Top-N chunk retrieval (sortBy + take)
    → grounded prompt construction
    → LLM generation (Mistral-7B-Instruct-v0.3, 4-bit NF4)
    → JSON output with citations
    │
    ▼
[Task 3 — Evaluation]
Retrieval metrics (precision@k, recall@k)
    → generation metrics (faithfulness, citation support)
    → system metrics (latency, throughput, tokens)
    → performance experiments (embedding comparison, distance metric comparison)
```

**Parallelism model.** Task 1 uses PySpark RDDs to distribute PDF ingestion and
chunking across workers (`spark.default.parallelism = 4`). Task 2 broadcasts the
query embedding to all executors and maps the similarity scoring in parallel over
the chunk partitions, avoiding driver-side bottlenecks for large corpora.

---

## 2. Task 1 — Preprocessing, Tokenization, and Embeddings

### 2.1 PDF Ingestion

PDFs are read with **pypdf** (`PdfReader`), producing one dict per page with fields
`{source_pdf, page_num, raw_text}`. The corpus covers all course-material PDFs
provided on the UV page.

### 2.2 Cleaning

Each page goes through `clean_text()`, which applies the following rules in order:

| Rule | Purpose |
|---|---|
| Unicode NFKC normalisation | Resolve ligatures, half-width characters |
| Remove URLs and e-mails | Not useful for retrieval |
| Remove browser timestamps (`4/16/26, 9:03 AM`) | Extraction artefact |
| Remove page counters (`- 3 / 7 -`) | Header/footer noise |
| Re-join hyphenated line-breaks (`distrib-\nuted → distributed`) | PDF line-wrap artefact |
| Collapse repeated whitespace | Formatting noise |

Pages are then filtered by `is_empty_page()`: a page is kept only if it contains
at least 20 words and at least 50% alphabetic characters. This removes cover pages,
table-of-content stubs, and figure-only pages that would produce noisy chunks.

### 2.3 Chunking

Cleaned pages are split by `chunk_page_text()` into overlapping, fixed-size
word-level chunks.

- **Chunk size: 300 words.** A 300-word chunk maps to roughly 400 sub-word tokens,
  which fits within SBERT's 512-token limit and is large enough to preserve
  complete definitions and examples without being truncated.
- **Overlap: 60 words (20%).** The 20% overlap reduces the risk of splitting a
  definition or formula across two consecutive chunks while keeping the index size
  manageable.
- Chunks shorter than 50 words (end-of-page fragments) are discarded.

Each chunk carries the full traceability metadata required by the specification:
`{id, source_pdf, page_num, chunk_id, start_word, end_word, chunk_text,
embedding, embedding_strategy}`.

### 2.4 Embeddings

Three strategies are supported, selectable via the `embedding_strategy` parameter:

| Strategy | Model / Library | Dimension | Notes |
|---|---|---|---|
| `tfidf` | `sklearn` TF-IDF + TruncatedSVD (LSA) | 256 | Sparse lexical baseline; max 10 000 features |
| `word2vec` | `gensim` Word2Vec (skip-gram) | 128 | Trained on the corpus; chunk embedding = mean of in-vocabulary word vectors |
| `sbert` | `sentence-transformers` `all-MiniLM-L6-v2` | 384 | Contextual dense embeddings; best semantic accuracy |

SBERT was chosen as the default because it produces contextual embeddings that
capture long-range semantic relationships, outperforming both TF-IDF (lexical
matching only) and Word2Vec (no context) on retrieval benchmarks.

### 2.5 Storage

Results are written as **Parquet** files (via PySpark) partitioned by
`strategy` (`strategy=tfidf`, `strategy=word2vec`, `strategy=sbert`).
The Parquet format enables efficient columnar reads in Task 2: loading only the
`embedding` and metadata columns without deserialising `chunk_text`.

---

## 3. Task 2 — Retrieval and Generation

### 3.1 Query Embedding

The query is embedded using the **same strategy and model** as the stored corpus
chunks, ensuring the query and chunk vectors live in the same vector space.

- **SBERT**: the pre-trained `all-MiniLM-L6-v2` model is loaded directly.
- **Word2Vec / TF-IDF**: the model is re-fitted on the corpus texts collected
  from the Parquet RDD before transforming the query.

### 3.2 Distributed Similarity Retrieval

Retrieval is implemented entirely with **PySpark RDD operations**:

1. The query embedding is **broadcast** (`SparkContext.broadcast()`) to all
   executors — a single network transfer regardless of corpus size.
2. Each executor runs `score_chunk()` over its local partition, calling the
   pure-Python metric functions `cosine_similarity` / `euclidean_distance`
   captured in the closure.
3. `scored_rdd.sortBy(score, ascending=False).take(top_n)` performs a
   distributed sort and collects only the top-N results to the driver.

**Distance metrics** are implemented from scratch without external libraries
(no `scipy.spatial.distance`, no `sklearn.metrics.pairwise`), as required by
the specification:

- **Cosine similarity**: $\frac{a \cdot b}{\|a\| \cdot \|b\|}$ — default;
  direction-sensitive, scale-invariant.
- **Euclidean distance**: $\sqrt{\sum (a_i - b_i)^2}$ — negated for ranking
  (smaller distance = higher score).

### 3.3 Prompt Construction

A grounded, citation-aware prompt is built from the top-N retrieved chunks.
Each chunk is numbered `[1]`, `[2]`, … and the model is instructed to:

- Use only the provided sources.
- Cite each answer with the corresponding source numbers.
- Follow a strict `Q / A / CITATIONS / ---` format per question.

Chunks are included until `max_context_tokens` (default: 3 000) is reached,
using the approximation 1 word ≈ 1.3 sub-word tokens.

### 3.4 LLM Generation

| Parameter | Value |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Backend | HuggingFace Transformers (local) |
| Quantisation | 4-bit NF4 (`bitsandbytes`) |
| Hardware | Colab T4 GPU (16 GB VRAM) |
| `max_new_tokens` | 512 |

Mistral-7B was chosen because it is publicly available without gating, fits in
4-bit on a T4 GPU (~4.5 GB VRAM), and generates instruction-following output
clearly above the `llama3.2:3b` baseline required by the specification.

**Example output** (query: *"Generate exam questions on Big Data frameworks and
distributed computing with Spark"*, SBERT + cosine, top-5):

> Q: How does Apache Spark differ from Hadoop in terms of data processing?
> A: Apache Spark processes data in-memory, making it faster for iterative ML
> tasks. Hadoop is better suited for batch processing of large datasets. [3]
> CITATIONS: [3]

Generation latency on first call (including model loading): ~585 s.
Subsequent calls (model already in GPU memory): significantly lower.

---

## 4. Task 3 — Evaluation

*To be completed.*

### 4.1 Retrieval Metrics

### 4.2 Generation and Grounding Metrics

### 4.3 Format-Adherence Metrics

### 4.4 System Metrics

### 4.5 Performance Experiments

---

## 5. Scalability Experiments

*To be completed.*

### 5.1 Effect of `parallelism`

### 5.2 Effect of `chunk_size` and `chunk_overlap`

### 5.3 Embedding Strategy Comparison

### 5.4 Distance Metric Comparison

---

## References

*(To be completed.)*
