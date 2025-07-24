# Medical-Bot

A specialized RAG-based chatbot for answering questions about medications using official Spanish leaflets.

## About The Project

The average patient in Spain spends significant time trying to find specific information, such as contraindications or dosage, within poorly formatted and dense PDF leaflets (`prospectos`). Medical-Bot aims to solve this problem by providing a simple conversational interface to query this information.

**The promise:** "Ask anything about your medication and receive an answer with an exact quote from the official leaflet in < 3 seconds."

This project will initially be built on a knowledge base of ~20 common Spanish medications.

## Tech Stack & Pipeline

The core of this project is a Retrieval-Augmented Generation (RAG) pipeline built with Python and LangChain. The pipeline is orchestrated by a single script (`scripts/_03_ingest.py`) and is designed to be **idempotent**: re-running the process for a document automatically cleans up old data and replaces it with the updated version.

### Step 1: LLM-Powered PDF-to-Markdown Conversion

**Problem:** PDF documents are designed for visual presentation, not for data extraction. Standard text extraction tools often fail to capture the document's inherent structure.

**Solution:** The first stage of the pipeline (`scripts/_01_pdf_to_markdown.py`) treats the PDF as a visual document. It converts each page into an image and sends the full sequence to a multimodal LLM (Google's Gemini). Crucially, it uses LangChain's **`with_structured_output`** feature, forcing the LLM to return a valid JSON object containing a single, perfectly structured Markdown string. This completely eliminates the risk of malformed outputs and makes the parsing process extremely reliable.

### Step 2: Context-Aware Chunking from Markdown

**Problem:** Naive chunking strategies are a primary source of failure in RAG systems, as they can separate a list item from its introductory sentence, losing critical context.

**Solution:** The second stage (`scripts/_02_markdown_to_chunks.py`) implements a sophisticated, structure-aware chunking logic.
1.  **Semantic Grouping:** It parses the Markdown file, grouping content into "semantic blocks" based on the hierarchy of headers (`#`, `##`, etc.).
2.  **Context Injection:** A metadata header, including the medicine's name and the full hierarchical path, is **injected directly into the content** of every resulting chunk. This provides maximum context to the LLM during retrieval.
3.  **Rich Metadata Storage:** Each chunk is also stored with a separate, structured `metadata` field, crucial for filtering and generating accurate citations.

### Step 3: Embedding and Idempotent Vector Storage

The final ingestion step (`scripts/_03_ingest.py`) is designed for robustness.
1.  **Data Cleaning:** Before inserting new data, the script first deletes any existing records belonging to the same source document, ensuring data consistency.
2.  **Embedding:** Each context-rich chunk is converted into a vector embedding using OpenAI's `text-embedding-3-small` model.
3.  **Storage:** The embeddings and their metadata are uploaded to a Supabase PostgreSQL database with `pgvector` for efficient similarity search.

### Step 4: Retrieval and Structured Generation

The user-facing application (`src/app.py`) demonstrates a professional-grade RAG implementation.
1.  **Custom Retriever:** A custom `SupabaseRetriever` class directly queries the database using our purpose-built SQL function.
2.  **Reliable Citations:** The final LLM call is the most critical part. Instead of just asking the model to include sources in the text (which is unreliable), we again use **`with_structured_output`**. We force the LLM to return a Pydantic object containing two distinct fields: `answer` (the text) and `cited_sources` (a list of numbers). This makes the citation process deterministic and robust, eliminating any need for fragile text parsing.

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