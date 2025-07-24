# Medical-Bot

A specialized RAG-based chatbot for answering questions about medications using official Spanish leaflets.

## About The Project

The average patient in Spain spends significant time trying to find specific information, such as contraindications or dosage, within poorly formatted and dense PDF leaflets (`prospectos`). Medical-Bot aims to solve this problem by providing a simple conversational interface to query this information.

**The promise:** "Ask anything about your medication and receive an answer with an exact quote from the official leaflet in < 3 seconds."

This project will initially be built on a knowledge base of ~20 common Spanish medications.

## Tech Stack & Pipeline

The core of this project is a Retrieval-Augmented Generation (RAG) pipeline built with Python and LangChain. The pipeline is orchestrated by a single script (`scripts/_03_ingest.py`) and is designed to be **idempotent**: re-running the process for a document automatically cleans up old data and replaces it with the updated version.

### Step 1: LLM-Powered PDF-to-Markdown Conversion

**Problem:** PDF documents are designed for visual presentation, not for data extraction. Standard text extraction tools often fail to capture the document's inherent structure (headings, lists, nested lists), leading to a flat, context-poor text file.

**Solution:** The first stage of the pipeline (`scripts/_01_pdf_to_markdown.py`) treats the PDF as a visual document. It converts each page into an image and sends the full sequence to a multimodal LLM (Google's Gemini), prompting it to act as a document analysis expert. The LLM interprets the visual layout to generate a semantically correct and perfectly structured Markdown file.

### Step 2: Context-Aware Chunking from Markdown

**Problem:** Naive chunking strategies (e.g., splitting by a fixed character count) are a primary source of failure in RAG systems. They can separate a list item from its introductory sentence or a warning from its heading, losing critical context.

**Solution:** The second stage (`scripts/_02_markdown_to_chunks.py`) implements a sophisticated, structure-aware chunking logic.
1.  **Semantic Grouping:** It first parses the Markdown file, grouping content into "semantic blocks" based on the hierarchy of headers (`#`, `##`, etc.).
2.  **Context Injection:** This is the key innovation. When a block is chunked, a metadata header is **injected directly into the content** of every resulting chunk. This header includes the medicine's name and the full hierarchical path to that section, providing maximum context to the LLM during retrieval.
3.  **Rich Metadata Storage:** Each chunk is also stored with a separate, structured `metadata` field containing the source file, medicine name, and hierarchical path. This is crucial for programmatic filtering and generating accurate citations in the final application.

### Step 3: Embedding and Idempotent Vector Storage

The final step is orchestrated by the main ingestion script (`scripts/_03_ingest.py`).
1.  **Data Cleaning:** Before inserting new data, the script first deletes any existing records in the database that belong to the same source document. This ensures that updating a leaflet always replaces the old information, preventing data duplication and inconsistencies.
2.  **Embedding:** Each context-rich chunk is converted into a vector embedding using OpenAI's `text-embedding-3-small` model.
3.  **Storage:** The embeddings, along with their corresponding text and rich metadata, are uploaded to a Supabase PostgreSQL database equipped with the `pgvector` extension for efficient similarity search.

## How to Use the Ingestion Pipeline

To process a new medication leaflet, place the PDF file in the `/data` directory and run the main ingestion script from the root of the `medical-bot` folder, passing the filename as an argument:

```bash
python scripts/_03_ingest.py nombre_del_medicamento.pdf
```

## Next Steps: Retrieval and Generation

With the knowledge base now built, the final step is to create the user-facing application. This will involve a LangChain-powered retrieval chain that:
1.  Takes a user's question.
2.  Queries the vector database to find the most relevant chunks of text.
3.  Passes the retrieved chunks and the original question to an LLM to synthesize a final, accurate, and cited answer.