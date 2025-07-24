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

## Advanced Features: Observability & Safety

To move from a functional prototype to a reliable application, this project incorporates crucial features for monitoring and safety.

### Observability with LangSmith

The entire application is integrated with [LangSmith](https://smith.langchain.com/) for end-to-end observability.

*   **Full Tracing:** Every time a user asks a question, LangSmith captures a detailed trace of the entire process. This includes the initial query analysis, the documents retrieved from the database, the exact prompt sent to the LLM, and the final structured output.
*   **Debugging & Monitoring:** This visibility is essential for debugging unexpected behavior, monitoring performance (e.g., latency), and evaluating the quality of the retriever and the LLM's responses over time.
*   **Setup:** Integration is enabled by simply adding the LangSmith environment variables (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, etc.) to the `.env` file.

### Multi-Layered Guardrails

Given the sensitive nature of medical information, the chatbot implements two distinct layers of safety checks:

#### 1. Pre-Processing Guardrail: Scope Control

Before attempting to answer a question, a "triage" step ensures the query is within the bot's knowledge base.

*   **Mechanism:** A lightweight LLM call analyzes the user's question to identify the specific medicine being asked about.
*   **Validation:** It then checks if this medicine exists in a pre-loaded list of known medicines from our database.
*   **Action:** If the question is about an unknown medicine, the RAG process is halted, and the bot informs the user about its limitations. This prevents the retriever from fetching irrelevant information and the LLM from attempting to answer out-of-scope questions.

#### 2. Prompt-Level Guardrail: Content & Safety Control

The core RAG prompt has been engineered with strict rules to control the LLM's behavior and ensure user safety.

*   **Behavioral Rules:** The prompt explicitly forbids the LLM from providing medical advice, personal opinions, or any information beyond summarizing the provided text.
*   **Mandatory Disclaimer:** The prompt includes an unbreakable rule forcing the LLM to **always** conclude its response with a specific disclaimer: "Este texto es una guía y no sustituye el consejo médico. Consulta siempre a tu médico o farmacéutico." This ensures the user is consistently reminded of the bot's informational role.

## The Chatbot Application (`src/app.py`)

The user-facing application demonstrates a professional-grade RAG implementation.
1.  **Custom Retriever:** A custom `SupabaseRetriever` class directly queries the database using our purpose-built SQL function.
2.  **Reliable Citations:** The final LLM call is the most critical part. Instead of just asking the model to include sources in the text (which is unreliable), we again use **`with_structured_output`**. We force the LLM to return a Pydantic object containing two distinct fields: `answer` (the text) and `cited_sources` (a list of numbers). This makes the citation process deterministic and robust.

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
```
