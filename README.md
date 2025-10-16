# Medical-Bot: Production-Grade RAG Agent for Spanish Medical Leaflets

> **"Ask anything about your medication and receive an answer with an exact quote from the official leaflet in seconds."**

A **production-ready**, **async-first** RAG chatbot specialized in answering questions about medications using official Spanish pharmaceutical leaflets (`prospectos`). Built with modern AI engineering best practices: real timeouts, rate limiting, structured logging, and cost optimization.

---

## 📚 Table of Contents

- [Why This Project?](#why-this-project)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [RAG Pipeline](#rag-pipeline)
- [The Agent (LangGraph)](#the-agent-langgraph)
- [Production Features](#production-features)
- [Experimental Research](#experimental-research)
- [Contributing](#contributing)

---

## Why This Project?

The average patient in Spain spends significant time trying to find specific information (contraindications, dosage, side effects) within poorly formatted and dense PDF leaflets. **Medical-Bot** solves this problem by providing a simple conversational interface powered by state-of-the-art RAG techniques.

This project serves as a **comprehensive case study** in building robust, data-driven, and production-ready RAG systems with LangChain and LangGraph.

---

## Key Features

### 🚀 Production-Ready Architecture
- **Full Async/Await**: All LLM operations use `ainvoke`/`astream` for true concurrency
- **Real Timeouts**: `asyncio.wait_for()` prevents hanging requests
- **Rate Limiting**: `asyncio.Semaphore` controls concurrent LLM calls
- **Parametrized Retry**: Exponential backoff with configurable attempts
- **Structured Logging**: JSON-formatted logs for production observability
- **Cost Optimization**: Smart caching and conditional LLM calls

### 🤖 Intelligent Agent System
- **ReAct Agent**: Multi-step reasoning with tool use
- **Conversational Memory**: Sliding window + summarization
- **Multi-Layered Guardrails**: Prevent hallucinations and out-of-scope queries
- **Context-Aware Query Rewriting**: Improves retrieval quality
- **Recursion Limit**: Prevents infinite loops (configurable)

### 📊 Advanced RAG Pipeline
- **LLM-Powered PDF Parsing**: Multimodal extraction with structure preservation
- **Sentence-Window Chunking**: Precision-optimized chunking strategy (+17.6% precision)
- **Hybrid Search Ready**: Experimented with dense+sparse retrieval
- **Embeddings Cache**: LRU cache reduces redundant API calls
- **Comprehensive Evaluation**: Retriever metrics (Recall@k, MRR) + Generation metrics (RAGAS)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROUTER NODE (Intent Classification)          │
│  - Classifies: medicine question / greeting / unauthorized      │
│  - Validates medicine against known database                    │
└──────────────────┬──────────────────┬─────────────────────────────┘
                   │                  │
        ┌──────────▼────────┐  ┌──────▼──────────┐
        │   CONVERSATIONAL  │  │   UNAUTHORIZED  │
        │       NODE        │  │      NODE       │
        └───────────────────┘  └─────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT NODE (ReAct Core)                       │
│  - Plans steps and decides to use tools                         │
│  - Formulates final response based on retrieved context         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              QUERY REWRITER NODE (Optional)                      │
│  - Enriches query with conversation context                     │
│  - Skips if no context exists (cost optimization)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOOLS NODE (RAG Retrieval)                    │
│  - Searches Supabase vector database                            │
│  - Returns relevant document chunks                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   Docs Found?       │
                └─────┬──────────┬─────┘
                      │ No       │ Yes
          ┌───────────▼───┐      │
          │  RETRIEVAL    │      │
          │   FAILURE     │      │
          │    NODE       │      │
          └───────────────┘      │
                                 ▼
                      ┌────────────────────┐
                      │   BACK TO AGENT    │
                      │  (Iterate or End)  │
                      └────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY MANAGEMENT                             │
│  - PRUNING: Removes tool messages after response                │
│  - END OF TURN: Increments turn counter                         │
│  - SUMMARIZER: Compacts memory every 3 turns                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Core Framework
- **Python 3.11+** - Modern async/await support
- **LangChain** - LLM orchestration and RAG pipeline
- **LangGraph** - Stateful multi-agent workflows
- **Pydantic** - Data validation and settings management

### LLM & Embeddings
- **OpenAI GPT-4o** - Main ReAct agent (reasoning + tool use)
- **Google Gemini 2.5 Flash** - Fast router + query rewriter + summarizer
- **Google Gemini Embedding 001** - Semantic search embeddings

### Storage & Retrieval
- **Supabase (pgvector)** - Vector database for RAG
- **PostgreSQL** - Conversation state persistence (LangGraph checkpointing)

### Observability & Testing
- **LangSmith** - End-to-end tracing and debugging
- **Pytest + pytest-asyncio** - Unit and integration tests
- **Structured Logging** - JSON logs for production monitoring

---

## Project Structure

```
medical-bot/
├── src/
│   ├── config.py                    # Centralized Pydantic settings
│   ├── graph/
│   │   ├── builder.py               # Graph construction
│   │   ├── nodes.py                 # Graph node implementations (async)
│   │   └── edges.py                 # Routing logic between nodes
│   ├── services/
│   │   ├── llm_service.py           # Async LLM with timeout + rate limiting
│   │   ├── retrieval_service.py     # RAG operations + query rewriting
│   │   ├── medicine_service.py      # Intent classification + validation
│   │   └── memory_service.py        # Conversation summarization + pruning
│   ├── models/
│   │   ├── domain.py                # AgentState (TypedDict)
│   │   ├── schemas.py               # Pydantic models for LLM structured output
│   │   └── embeddings.py            # Embeddings factory + caching wrapper
│   ├── database/
│   │   └── supabase.py              # Custom Supabase retriever
│   └── utils/
│       ├── logger.py                # Structured logging setup
│       └── prompts.py               # Centralized prompt loader with @lru_cache
├── config/
│   └── prompts.yaml                 # Externalized system prompts
├── scripts/
│   ├── _01_pdf_to_markdown.py       # PDF → Markdown with Gemini Vision
│   ├── _02_markdown_to_chunks.py    # Sentence-window chunking
│   └── _03_ingest.py                # Idempotent ingestion to Supabase
├── tests/
│   ├── unit/                        # Service-level tests (async)
│   └── integration/                 # Graph flow tests (async)
├── evaluation/
│   ├── evaluate_retriever.py        # Retriever metrics (Recall@k, MRR, F1)
│   └── evaluate_generation.py       # RAGAS metrics (Faithfulness, Relevancy)
├── graph.py                         # Main entry point (async console chat)
└── README.md                        # This file
```

---

## Installation

### Prerequisites
- **Python 3.11+**
- **API Keys**: OpenAI, Google AI, Supabase, Cohere (for evaluation), LangSmith (optional)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medical-bot.git
cd medical-bot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Supabase Database

Before configuring environment variables, you need to initialize the required database tables and functions in Supabase:

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Run the following SQL script:

```sql
-- Create documents table with vector embeddings support
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create vector similarity search index (using ivfflat for performance)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Function to match documents by semantic similarity
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 5
)
RETURNS TABLE (
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        content,
        metadata,
        1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Function to get distinct medicine names from metadata
CREATE OR REPLACE FUNCTION get_distinct_medicine_names()
RETURNS TABLE (medicine_name text)
LANGUAGE sql STABLE
AS $$
    SELECT DISTINCT metadata->>'source' as medicine_name
    FROM documents
    WHERE metadata->>'source' IS NOT NULL
    ORDER BY medicine_name;
$$;
```

**What this does:**
- Creates a `documents` table to store text chunks and their embeddings
- Adds a vector similarity search index for fast retrieval
- Provides `match_documents()` for semantic search
- Provides `get_distinct_medicine_names()` to list available medicines

### 4. Configure Environment Variables
Create a `.env` file in the project root:

```bash
# API Keys (Required)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=...
POSTGRES_CONN_STR=postgresql://...
COHERE_API_KEY=...  # For re-ranking experiments

# LangSmith (Optional - for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=medical-bot

# LLM Service Configuration (Optional - defaults shown)
LLM_TIMEOUT=30              # Timeout in seconds for LLM calls
LLM_MAX_RETRIES=3           # Retry attempts on failure
LLM_RATE_LIMIT=3            # Max concurrent LLM requests
EMBEDDINGS_CACHE_SIZE=100   # LRU cache size for embeddings
MAX_REACT_ITERATIONS=10     # Prevent infinite ReAct loops

# Model Selection (Optional - defaults shown)
AGENT_MODEL=gpt-4o
ROUTER_MODEL=gemini-2.5-flash
EMBEDDINGS_PROVIDER=google
EMBEDDINGS_MODEL=models/gemini-embedding-001
```

---

## Usage

### Running the Chatbot

```bash
python graph.py
```

**Output:**
```
============================================================
Medical Chatbot (Async) - Type 'exit' or 'quit' to stop
============================================================

Your question: ¿Qué es el ibuprofeno?

--- Processing... ---
-> Intent: pregunta_medicamento

Bot Response:
El ibuprofeno es un antiinflamatorio no esteroideo (AINE) que se utiliza...
```

### Ingesting New Medication Leaflets

1. Place PDF in `data/` folder
2. Run ingestion pipeline:

```bash
python scripts/_03_ingest.py nombre_del_medicamento.pdf
```

This will:
- Convert PDF to Markdown (with structure preservation)
- Create sentence-window chunks
- Generate embeddings
- Upload to Supabase (idempotent - replaces old data)

---

## Configuration

All configuration is managed via **Pydantic Settings** in `src/config.py`. Values can be set via:

1. **Environment variables** (`.env` file) - **Recommended**
2. **Direct Python** (defaults in `Settings` class)

### Key Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_TIMEOUT` | 30 | Timeout (seconds) for LLM calls |
| `LLM_MAX_RETRIES` | 3 | Retry attempts on LLM failure |
| `LLM_RATE_LIMIT` | 3 | Max concurrent LLM requests |
| `MAX_REACT_ITERATIONS` | 10 | Prevent infinite ReAct loops |
| `EMBEDDINGS_CACHE_SIZE` | 100 | LRU cache for query embeddings |
| `AGENT_MODEL` | gpt-4o | Main reasoning model |
| `ROUTER_MODEL` | gemini-2.5-flash | Fast classification model |

---

## Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Async tests with verbose output
pytest -v -s tests/
```

### Test Coverage
- **Unit Tests**: Service layer logic (medicine matching, memory management, etc.)
- **Integration Tests**: Full graph flows (greeting, medicine questions, unauthorized queries)
- **All tests use pytest-asyncio** for async support

---

## RAG Pipeline

### Step 1: PDF-to-Markdown Conversion
- **Script**: `scripts/_01_pdf_to_markdown.py`
- **Approach**: Converts each page to an image, sends to **Gemini Vision** for structured Markdown extraction
- **Why**: Preserves hierarchical structure (headings, nested lists) better than text-only parsers
- **Speed**: ~140s per document (high quality, research-oriented)

### Step 2: Sentence-Window Chunking
- **Script**: `scripts/_02_markdown_to_chunks.py`
- **Strategy**: Each sentence/list item becomes a chunk, embedded with N sentences of context
- **Benefit**: **+17.6% precision** vs. fixed-size chunking (proven via A/B testing)
- **Why**: Semantic search finds exact facts, not approximate blocks

### Step 3: Embedding & Idempotent Upload
- **Script**: `scripts/_03_ingest.py`
- **Process**:
  1. Deletes old chunks for the document (by source name)
  2. Generates embeddings with `gemini-embedding-001`
  3. Uploads to Supabase with metadata
- **Result**: Re-running script for a document safely updates its data

---

## The Agent (LangGraph)

### Node Descriptions

| Node | Purpose | Example Logic |
|------|---------|---------------|
| **`router`** | Classifies user intent (medicine/greeting/unauthorized) | "¿Qué es el ibuprofeno?" → `pregunta_medicamento` |
| **`agent`** | ReAct reasoning core - plans steps and uses tools | Decides to search database, then formulates answer |
| **`query_rewriter`** | Enriches query with conversation context | "Is it safe for me?" → "Is Lexatin safe for 29yo male with diabetes?" |
| **`tools`** | Executes database search (RAG retrieval) | Searches Supabase for relevant chunks |
| **`handle_retrieval_failure`** | Safe fallback if no docs found | "I couldn't find that specific information..." |
| **`unauthorized`** | Blocks unknown medicines | "I don't have info about Aspirin. I know about: Ibuprofen, Paracetamol..." |
| **`pruning`** | Removes tool messages from history | Cleans verbose tool outputs to save tokens |
| **`summarizer`** | Compacts conversation every 3 turns | "User asked about Ibuprofen side effects. Agent explained..." |
| **`end_of_turn`** | Increments turn counter (pacemaker) | `turn_count: 2` → `turn_count: 3` |

### ReAct Loop Example

**User**: "Compare Nolotil and Ibuprofen"

1. **Agent**: Plans to search for Nolotil first → Calls `tools`
2. **Tools**: Returns Nolotil information
3. **Agent**: Recognizes task incomplete → Calls `tools` again for Ibuprofen
4. **Tools**: Returns Ibuprofen information
5. **Agent**: Has both contexts → Formulates comparison answer
6. **Pruning**: Removes raw tool outputs
7. **End of Turn**: Increments counter

This `Agent → Tool → Agent` loop enables complex multi-step reasoning.

---

## Production Features

### ⚡ Async-First Design
- **All LLM calls** use `ainvoke`/`astream` (not blocking `invoke`)
- **Benefits**: True concurrency, real timeouts, rate limiting

### ⏱️ Real Timeout Handling
```python
async with self.semaphore:  # Rate limiting
    response = await asyncio.wait_for(
        self.model.ainvoke(messages),
        timeout=30  # Cancels if exceeds 30s
    )
```

### 🚦 Rate Limiting
- **Semaphore**: Limits concurrent LLM requests (default: 3)
- **Why**: Prevents API rate limit errors and controls costs

### 🔄 Parametrized Retry with Exponential Backoff
- **Uses**: `AsyncRetrying` from `tenacity`
- **Config**: `LLM_MAX_RETRIES` (default: 3)
- **Backoff**: `wait_exponential(multiplier=1, min=2, max=10)`

### 💰 Cost Optimization

#### 1. Conditional LLM Calls
- **Query Rewriter skips** if no conversation history (saves $$ on first turn)

#### 2. Embeddings Cache
- **LRU Cache**: Stores last 100 unique query embeddings
- **Benefit**: Repeated queries (e.g., "Ibuprofen side effects") don't regenerate embeddings

#### 3. Centralized Prompt Loading
- **@lru_cache**: Loads `config/prompts.yaml` once, reuses forever
- **Benefit**: No redundant YAML parsing

### 📊 Structured Logging
```json
{
  "timestamp": "2025-01-15T10:23:45Z",
  "level": "INFO",
  "event": "llm_call_started",
  "attempt": 1,
  "timeout": 30,
  "model": "gpt-4o"
}
```
- **JSON format**: Easy to parse and query in production
- **Correlation IDs**: Track requests across services

### 🔒 Multi-Layered Guardrails

1. **Router Guardrail**: Blocks unknown medicines at entry
2. **Retrieval Failure Guardrail**: Safe response if no docs found
3. **Prompt-Level Guardrail**: System prompt forbids medical advice
4. **Recursion Limit**: Prevents infinite ReAct loops (`MAX_REACT_ITERATIONS=10`)

---

## Experimental Research

This project follows a rigorous **data-driven experimentation process**. Each feature is evaluated against baseline metrics before integration.

### Experiment: Query Rewriting for Improved Retrieval
**Hypothesis**: Enriching queries with conversation context improves retrieval quality.

**Implementation**: Added `query_rewriter_node` to enrich queries with context.

**Results**: Dramatic improvement in retrieved document relevance (qualitative LangSmith analysis).

**Decision**: ✅ **Integrated** - Worth the minor latency increase.

---

### Experiment: Hybrid Search with Pinecone
**Hypothesis**: Dense+sparse hybrid search outperforms pure semantic search.

**Implementation**: Migrated to Pinecone with RRF fusion layer.

**Results**:
- Pure semantic (`alpha=1.0`): On-par with Supabase baseline
- Hybrid (`alpha<1.0`): **Performance degraded** across all metrics

**Decision**: ❌ **Reverted** - Stayed with simpler Supabase/pgvector architecture.

---

### Experiment: Optimizing Embeddings and Re-Ranking
**Hypothesis**: Google Gemini embeddings + Cohere re-ranker would improve retrieval.

**Implementation**:
1. Migrated to `gemini-embedding-001` (with task_type optimization)
2. A/B tested Cohere Rerank layer

**Results**:
- **Gemini Embeddings**: **+8.2 F1-Score** (39.65% → 47.87%) 🚀
- **Cohere Rerank**: **-1.67 F1-Score** (47.87% → 46.20%) ❌

**Decision**: ✅ **Gemini standalone** - Simpler, faster, cheaper, more accurate.

---

### Experiment: Sentence-Window Chunking for Precision Boost
**Hypothesis**: Granular sentence-level chunking improves retrieval precision.

**Implementation**: Each sentence embedded with N-sentence window.

**Results**:
- **+17.63 Precision@5** (40.12% → 57.75%) 🚀
- **+8.04 F1-Score**

**Decision**: ✅ **Production standard** - Massive precision improvement.

---

### Experiment: PDF Parser Optimization with Docling
**Hypothesis**: Docling (programmatic) would be faster than Gemini Vision without quality loss.

**Implementation**:
1. **Docling-only**: Fast (~38s) but lost hierarchical structure
2. **Hybrid (Docling + Gemini-Text)**: ~90s with recovered structure

**Results**:
- **Gemini-Visual**: 140s, highest quality
- **Hybrid**: 90s, good quality (~35% faster)

**Decision**: ✅ **Kept Gemini-Visual** - Priority is quality for research; hybrid viable for production.

---

### Experiment: Advanced Citation with ContextCite
**Goal**: Improve attribution transparency.

**Current Method**: LLM self-reports sources (corroborative attribution).

**ContextCite Method**: Ablation-based attribution (removes context parts to find causal sentences).

**Status**: Experimental script available in `evaluation/compare_citations.py` for visual comparison.

---

## License

MIT License - See `LICENSE` file for details.

---

**Built with ❤️ and async/await in Spain 🇪🇸**
