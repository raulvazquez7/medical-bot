# Medical-Bot

A specialized RAG-based chatbot for answering questions about medications using official Spanish leaflets.

## About The Project

The average patient in Spain spends significant time trying to find specific information, such as contraindications or dosage, within poorly formatted and dense PDF leaflets (`prospectos`). Medical-Bot aims to solve this problem by providing a simple conversational interface to query this information.

**The promise:** "Ask anything about your medication and receive an answer with an exact quote from the official leaflet in < 3 seconds."

This project will initially be built on a knowledge base of ~20 common Spanish medications.

## Tech Stack & Pipeline

The core of this project is a **Retrieval-Augmented Generation (RAG)** pipeline built with Python and LangChain.

### Architecture Overview (Preliminary)

This project is currently under construction. The planned pipeline is as follows:

1.  **Data Ingestion & Parsing**:
    *   PDF leaflets from CIMA (Centro de Información online de Medicamentos de la AEMPS) are ingested.
    *   A custom parser built with `PyMuPDF` analyzes the document's structure, identifying sections and subsections not just by text, but by formatting cues like **bold fonts**, **text position**, and **layout**.

2.  **Intelligent Chunking Strategy**:
    *   Moving beyond generic text splitting, we are developing a content-aware chunking strategy.
    *   The goal is to create semantically coherent chunks based on the document's logical structure (e.g., a chunk containing the entire "Advertencias y precauciones" subsection).
    *   Each chunk is enriched with detailed, hierarchical metadata, such as:
        ```json
        {
          "source": "eutirox.pdf",
          "medication_name": "Eutirox 25 microgramos",
          "main_section": "2. Qué necesita saber antes de empezar...",
          "subsection": "Advertencias y precauciones" 
        }
        ```
    *   This advanced chunking is key to improving retrieval accuracy and providing contextually aware answers.

3.  **Vector Database**:
    *   The processed chunks are converted into vector embeddings using a sentence-transformer model (e.g., from OpenAI).
    *   These embeddings are stored in a vector database (Supabase with `pgvector`) for efficient similarity search.

4.  **Retrieval & Generation**:
    *   When a user asks a question, a LangChain-powered retrieval chain will query the vector database to find the most relevant chunks.
    *   The retrieved chunks, along with the user's question, will be passed to a Large Language Model (LLM) to generate a precise and helpful answer, with citations to the source document.

## Current Status

The project is in the early development phase. The primary focus is currently on perfecting the data ingestion and intelligent chunking pipeline to ensure the highest quality data foundation for the RAG system.