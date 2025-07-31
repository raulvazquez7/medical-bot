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

The second stage (`scripts/_02_markdown_to_chunks.py`) implements a sophisticated, structure-aware chunking logic. It groups content by semantic headers and injects metadata (like medicine name and document path) directly into each chunk to preserve context.

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

Given the sensitive nature of medical information, the chatbot implements two distinct layers of safety checks: a pre-processing guardrail for scope control and a prompt-level guardrail to ensure safe and factual responses.

### Experimental Research: Advanced Citation with ContextCite

As part of ongoing research to improve the reliability of the chatbot, we have experimented with advanced citation techniques. Specifically, we explored the `ContextCite` library, which provides a more rigorous form of "contributive attribution".

*   **Our Current Method:** The LLM self-reports which documents it used ("corroborative attribution").
*   **ContextCite's Method:** Uses a scientific approach of ablating (removing) parts of the context to mathematically determine which specific sentences **caused** the model to generate its response ("contributive attribution").

While computationally intensive, this method provides a much more granular and honest view of the model's reasoning process. An experimental script for visual comparison can be found in `evaluation/compare_citations.py`.

## Robust Evaluation Framework

To guide development and ensure high quality, this project uses a comprehensive, two-layer evaluation framework based on a manually curated "golden dataset" of questions.

### 1. Retriever Evaluation

We measure the performance of our document retrieval system *before* the LLM sees the data. Using our golden dataset, a dedicated script (`evaluation/evaluate_retriever.py`) calculates key metrics:
*   **Recall@k:** Measures the retriever's ability to find all relevant documents.
*   **Precision@k:** Measures how much "noise" or irrelevant information is retrieved.
*   **F1-Score & MRR:** Provide a holistic view of the retriever's overall performance and ranking quality.

### 2. Generation Evaluation

We evaluate the quality of the final, user-facing answer using the **RAGAS** library, which employs an "LLM-as-a-Judge" approach.
*   **Faithfulness:** Measures the degree of hallucination by checking if the answer is strictly based on the provided context.
*   **Answer Relevancy:** Assesses if the answer directly addresses the user's question.

This framework allows us to benchmark our system and make data-driven decisions when implementing improvements. In production, this would be complemented by sampling real user interactions for continuous online evaluation.

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