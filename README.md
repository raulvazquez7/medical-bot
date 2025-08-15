# Medical-Bot

A specialized RAG-based chatbot for answering questions about medications using official Spanish leaflets.

## About The Project

The average patient in Spain spends significant time trying to find specific information, such as contraindications or dosage, within poorly formatted and dense PDF leaflets (`prospectos`). Medical-Bot aims to solve this problem by providing a simple conversational interface to query this information.

**The promise:** "Ask anything about your medication and receive an answer with an exact quote from the official leaflet in < 3 seconds."

This project will initially be built on a knowledge base of ~20 common Spanish medications.

## Tech Stack & Pipeline

The core of this project is a Retrieval-Augmented Generation (RAG) pipeline built with Python and LangChain. The pipeline is orchestrated by a single script (`scripts/_03_ingest.py`) and is designed to be **idempotent**: re-running the process for a document automatically cleans up old data and replaces it with the updated version.

### Step 1: LLM-Powered PDF-to-Markdown Conversion

The first stage of the pipeline (`scripts/_01_pdf_to_markdown.py`) treats the PDF as a visual document. It converts each page into an image and sends the full sequence to a multimodal LLM (Google's Gemini), forcing a structured Markdown output for maximum reliability.

### Step 2: Context-Aware Chunking from Markdown

The second stage (`scripts/_02_markdown_to_chunks.py`) implements a sophisticated, structure-aware chunking logic. It groups content by semantic headers and injects metadata (like medicine name and document path) directly into each chunk to preserve context. The system is currently configured with a chunk size of **800 tokens** and an overlap of **100 tokens**, parameters that are centralized for easy experimentation.

### Step 3: Embedding and Idempotent Vector Storage

The final ingestion step (`scripts/_03_ingest.py`) is designed for robustness. It first cleans any old data for the source document in Supabase, then generates embeddings for the new chunks and uploads them.

## The Chatbot Application (`src/app.py`)

The user-facing application demonstrates a professional-grade RAG implementation.
1.  **Custom Retriever:** A custom `SupabaseRetriever` class directly queries the database using our purpose-built SQL function.
2.  **Reliable Citations:** The final LLM call is the most critical part. Instead of just asking the model to include sources in the text (which is unreliable), we again use **`with_structured_output`**. We force the LLM to return a Pydantic object containing two distinct fields: `answer` (the text) and `cited_sources` (a list of numbers). This makes the citation process deterministic and robust.

## Advanced Features: Observability & Safety

To move from a functional prototype to a reliable application, this project incorporates crucial features for monitoring and safety.

### Observability with LangSmith

The entire application is integrated with [LangSmith](https://smith.langchain.com/) for end-to-end observability. Full tracing allows for detailed debugging and performance monitoring of every component in the RAG chain.

### Multi-Layered Guardrails

Given the sensitive nature of medical information, the chatbot implements three distinct layers of safety checks: a pre-processing guardrail for scope control, a prompt-level guardrail to ensure safe and factual responses, and a retrieval-level guardrail.

1.  **Triage Guardrail (Pre-processing):** Before attempting to answer, a preliminary LLM call analyzes the user's query to identify the mentioned medication and checks if it belongs to the knowledge base. If the medication is unknown, the bot refuses to answer and lists the medications it can discuss.
2.  **Prompt-Level Guardrail:** The final prompt sent to the LLM contains a critical safety rule that strictly forbids providing medical advice and forces it to base its answer exclusively on the provided context.
3.  **Retrieval-Level Guardrail (Metadata Filtering):** To prevent context contamination between different medication leaflets, the retrieval pipeline is enhanced with strict metadata filtering. When the agent identifies the specific medication(s) relevant to a query, it forces the vector search to consider **only** the chunks belonging to those leaflets. This is a critical safety feature that prevents the model from generating answers based on information from the wrong drug.

## Robust Evaluation Framework

To guide development and ensure high quality, this project uses a comprehensive, two-layer evaluation framework based on a manually curated "golden dataset" of questions.

### 1. Retriever Evaluation

We measure the performance of our document retrieval system *before* the LLM sees the data. Using our golden dataset, a dedicated script (`evaluation/evaluate_retriever.py`) calculates key metrics:
*   **Recall@k:** Measures the retriever's ability to find all relevant documents.
*   **Precision@k:** Measures how much "noise" or irrelevant information is retrieved.
*   **F1-Score & MRR:** Provide a holistic view of the retriever's overall performance and ranking quality.

The evaluation framework is also extensible for testing advanced retrieval strategies like two-stage retrieval with a re-ranking layer to rigorously benchmark more complex techniques against the baseline.

### 2. Generation Evaluation

We evaluate the quality of the final, user-facing answer using the **RAGAS** library, which employs an "LLM-as-a-Judge" approach.
*   **Faithfulness:** Measures the degree of hallucination by checking if the answer is strictly based on the provided context.
*   **Answer Relevancy:** Assesses if the answer directly addresses the user's question.

This framework allows us to benchmark our system and make data-driven decisions when implementing improvements. In production, this would be complemented by sampling real user interactions for continuous online evaluation.

## Experimental Research & Data-Driven Decisions

To continuously improve the system, we follow a rigorous process of experimentation. New techniques are tested in isolation and evaluated against our baseline metrics. This data-driven approach ensures that only features with a clear positive impact are integrated.

### Experiment: Query Rewriting for Improved Retrieval
As part of our efforts to enhance retrieval performance, we tested the "Query Rewriting" technique.
*   **Hypothesis:** Rephrasing conversational user questions into optimized, keyword-focused queries before sending them to the vector database would improve retrieval scores.
*   **Implementation:** We added a preliminary LLM call to transform the user's input before the retrieval step.
*   **Results:** Using our evaluation framework, an A/B test showed a negligible impact on retrieval metrics (F1-Score, Precision, Recall) while adding significant latency (~4 seconds per query) to the user's request.
*   **Decision:** Based on this data, a feature that did not improve the user experience was discarded. The performance cost far outweighed the minimal benefits, confirming that our base retriever is already robust enough for direct, conversational queries.

### Experiment: Hybrid Search with Pinecone

To test the hypothesis that hybrid search could improve retrieval for domain-specific terms, a migration from Supabase/pgvector to Pinecone was undertaken.

*   **Hypothesis:** A hybrid approach combining semantic (dense) and keyword (sparse) search would outperform a purely semantic system, especially for technical queries involving specific medication names or chemical compounds.
*   **Implementation:**
    1.  The vector database was migrated to Pinecone to leverage its native support for sparse-dense vectors.
    2.  An advanced retrieval pipeline was built using Pinecone's recommended two-index approach (one for dense vectors, one for sparse) and a custom Reciprocal Rank Fusion (RRF) layer to merge results with a controllable `alpha` weight.
*   **Results:**
    *   The purely semantic search on Pinecone (`alpha = 1.0`) performed on-par with the original Supabase baseline, confirming its effectiveness.
    *   However, any introduction of keyword-based search (`alpha < 1.0`) consistently **degraded performance** across all key metrics (F1-Score, MRR, Precision, Recall) for the existing `golden_dataset`.
*   **Decision:** For the current, predominantly conceptual question set, the added complexity of a hybrid search system did not provide a net benefit and introduced a risk of performance degradation. The experiment was documented, and the project reverted to the simpler, more robust, and equally performant Supabase/pgvector architecture. This provides a valuable baseline for future work, should the system need to handle more keyword-sensitive queries.

### Experiment: Optimizing Embeddings and Re-Ranking

A final experiment was conducted to push the limits of the retrieval pipeline by testing a state-of-the-art embedding model against a powerful re-ranker.

*   **Hypothesis:** Migrating from OpenAI's embeddings to a newer, specialized model (Google's `gemini-embedding-001`) and adding a re-ranking layer (Cohere Rerank) would yield significant improvements.
*   **Implementation:**
    1.  The entire data pipeline was migrated to use `gemini-embedding-001`, including optimizations like specifying `task_type` for queries (`retrieval_query`) and documents (`retrieval_document`).
    2.  A controlled A/B test was performed using the evaluation framework's re-ranking capability.
*   **Results:**
    *   **Google Embeddings Baseline:** The migration to `gemini-embedding-001` alone caused a **dramatic performance increase**, with the **F1-Score jumping from 39.65% to 47.87% (+8.2 points)**. This established a new, powerful baseline, validating the quality of the new embedding model.
    *   **Re-Ranking Impact:** When the Cohere re-ranker was applied on top of this strong baseline, performance metrics consistently **decreased** (F1-Score dropped to 46.20%). The data revealed a "performance ceiling": Google's embedding model's initial ranking is so accurate for this dataset that the secondary re-ranking step introduces noise and degrades the results.
*   **Decision:** The optimal retrieval architecture was determined to be **`gemini-embedding-001` used standalone**. The re-ranker was disabled from the production configuration, proving that for this use case, a simpler, faster, and cheaper architecture is also the most accurate.

### Experiment: Advanced Citation with ContextCite

As part of ongoing research to improve the reliability of the chatbot, we have experimented with advanced citation techniques. Specifically, we explored the `ContextCite` library, which provides a more rigorous form of "contributive attribution".

*   **Our Current Method:** The LLM self-reports which documents it used ("corroborative attribution").
*   **ContextCite's Method:** Uses a scientific approach of ablating (removing) parts of the context to mathematically determine which specific sentences **caused** the model to generate its response ("contributive attribution").

While computationally intensive, this method provides a much more granular and honest view of the model's reasoning process. An experimental script for visual comparison can be found in `evaluation/compare_citations.py`.

## How to Use

### Ingestion Pipeline
To process a new medication leaflet, place the PDF file in the `/data` directory and run the main ingestion script from the root of the `medical-bot` folder:
```bash
python scripts/_03_ingest.py nombre_del_medicamento.pdf
```

### Chatbot Application
To interact with the bot, run the main application script:
```bash
python src/app.py
```
