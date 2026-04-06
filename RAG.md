# 🤖 RAG Pipelines — Personal Knowledge Reference

> A concise reference on Retrieval-Augmented Generation (RAG) — architecture, components, use cases, and SRE applications.

---

## 📌 Table of Contents

- [What is RAG?](#what-is-rag)
- [Why RAG over plain LLM?](#why-rag-over-plain-llm)
- [Core Architecture](#core-architecture)
- [Key Components](#key-components)
- [RAG Pipeline — Step by Step](#rag-pipeline--step-by-step)
- [Types of RAG](#types-of-rag)
- [RAG in SRE Context](#rag-in-sre-context)
- [Popular Tools & Stack](#popular-tools--stack)
- [Code Snippet — Basic RAG Flow](#code-snippet--basic-rag-flow)
- [Common Challenges](#common-challenges)
- [Quick Glossary](#quick-glossary)
- [References](#references)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architecture pattern that enhances a Large Language Model (LLM) by injecting **relevant external context** at query time — before generating a response.

Instead of relying solely on what the model was trained on, RAG:
1. **Retrieves** relevant documents/chunks from a knowledge store
2. **Augments** the prompt with that retrieved context
3. **Generates** a grounded, accurate response

> Think of it as giving the LLM a "cheat sheet" pulled from your own data before it answers.

---

## Why RAG over plain LLM?

| Problem with plain LLM | How RAG solves it |
|---|---|
| Knowledge cutoff (stale data) | Retrieves fresh, real-time docs |
| Hallucinations | Grounds answers in retrieved facts |
| No access to private/internal data | Indexes your own knowledge base |
| Generic answers | Context-aware, domain-specific responses |
| Expensive fine-tuning | No retraining needed — just index new docs |

---

## Core Architecture

```
                        ┌─────────────────────────────┐
                        │        Knowledge Base        │
                        │  (Runbooks, Logs, Wikis,     │
                        │   Reports, Confluence, etc.) │
                        └────────────┬────────────────┘
                                     │ Ingest & Chunk
                                     ▼
                        ┌─────────────────────────────┐
                        │       Embedding Model        │
                        │  (text → vector numbers)     │
                        └────────────┬────────────────┘
                                     │ Store vectors
                                     ▼
                        ┌─────────────────────────────┐
                        │        Vector Database       │
                        │  (Pinecone, FAISS, Weaviate, │
                        │   Azure AI Search, etc.)     │
                        └────────────┬────────────────┘
                                     │
User Query ──► [Embed Query] ──► [Similarity Search]
                                     │
                             Top-K Chunks Retrieved
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │         LLM (GPT-4,          │
                        │    Claude, Llama, etc.)      │
                        │  Prompt = Query + Chunks     │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                              ✅ Grounded Answer
```

---

## Key Components

### 1. 📄 Document Loader
Ingests raw data from various sources.
- File types: PDF, DOCX, HTML, CSV, Markdown, JSON
- Sources: Confluence, SharePoint, S3, GitHub, databases
- Tools: LangChain loaders, LlamaIndex readers, custom scripts

### 2. ✂️ Text Chunker / Splitter
Splits large documents into smaller retrievable units.
- **Chunk size**: Typically 256–1024 tokens
- **Overlap**: ~10–20% overlap between chunks to preserve context
- Strategies: Fixed-size, sentence-based, semantic-based

### 3. 🔢 Embedding Model
Converts text chunks into numerical vectors (embeddings).
- Examples: `text-embedding-ada-002` (OpenAI), `sentence-transformers`, `Azure OpenAI Embeddings`
- Same model must be used for indexing AND querying

### 4. 🗄️ Vector Store
Stores and indexes embeddings for fast similarity search.

| Vector DB | Best For |
|---|---|
| **FAISS** | Local / lightweight |
| **Pinecone** | Managed, scalable cloud |
| **Weaviate** | Open source + hybrid search |
| **Azure AI Search** | Azure-native, hybrid search |
| **Chroma** | Dev/prototyping |
| **pgvector** | PostgreSQL extension |

### 5. 🔍 Retriever
Fetches top-K most relevant chunks for a given query.
- **Cosine similarity** is the most common distance metric
- **Hybrid search** = vector search + keyword (BM25) search combined

### 6. 🧠 LLM (Generator)
Takes the query + retrieved chunks as a prompt and generates the final response.
- Examples: GPT-4, Claude 3, Llama 3, Mistral, Azure OpenAI

---

## RAG Pipeline — Step by Step

### Phase 1: Indexing (Offline / One-time)

```
Raw Docs ──► Load ──► Chunk ──► Embed ──► Store in Vector DB
```

### Phase 2: Query (Online / Real-time)

```
User Query
    │
    ▼
Embed the query using same embedding model
    │
    ▼
Similarity search in Vector DB → Top-K chunks
    │
    ▼
Build prompt: [System Instruction] + [Retrieved Chunks] + [User Query]
    │
    ▼
Send to LLM
    │
    ▼
Return grounded response to user
```

---

## Types of RAG

| Type | Description | When to Use |
|---|---|---|
| **Naive RAG** | Basic retrieve → generate | Simple Q&A, prototyping |
| **Advanced RAG** | Re-ranking, query rewriting, better chunking | Production systems |
| **Modular RAG** | Plug-and-play components (routing, fusion) | Complex workflows |
| **Agentic RAG** | LLM decides when/what to retrieve | Multi-step reasoning |
| **Graph RAG** | Knowledge graphs as retrieval store | Relationship-heavy data |

---

## RAG in SRE Context

RAG is highly applicable in SRE workflows. Here are real-world use cases:

### 🚨 Incident Response
- Index past incident reports + runbooks in a vector DB
- Query: *"What caused OOM errors on the Databricks cluster last month?"*
- RAG retrieves relevant past incidents and runbook steps → LLM summarizes and suggests actions

### 📋 Runbook Q&A
- Index all runbooks from Confluence/SharePoint
- On-call engineer asks: *"How do I failover the F5 XC origin pool?"*
- RAG returns the exact procedure without manual searching

### 💰 FinOps Auditing
- Index Azure Cost Management exports, Databricks audit logs, Delta table reports
- Query: *"Which schemas had the highest storage growth in Q1?"*
- RAG over your actual data → grounded cost insights

### 🔧 Pipeline Troubleshooting
- Index YAML pipeline configs, error logs, template files
- Ask: *"Why did the secure file step fail in the mobile pipeline?"*
- RAG retrieves relevant YAML + past error context → actionable fix

### 📊 Log Intelligence
- Stream Azure Monitor / Databricks logs into retrieval store
- Ask: *"What's the common pattern in these job failure alerts?"*
- Contextual correlation across multiple log sources

### 📚 Knowledge Base Assistant
- Index team wikis, architecture docs, POC documentation
- New team members can query: *"What is our Azure Front Door setup?"*
- Reduces onboarding time significantly

---

## Popular Tools & Stack

### Frameworks
| Tool | Description |
|---|---|
| **LangChain** | Most popular RAG framework, many integrations |
| **LlamaIndex** | Focused on data ingestion and indexing for LLMs |
| **Haystack** | Enterprise-grade, production-ready RAG |
| **Azure AI Foundry** | Azure-native RAG with Azure OpenAI + AI Search |
| **Semantic Kernel** | Microsoft SDK for LLM orchestration |

### End-to-End Stack Example (Azure-based)
```
Azure Blob Storage (docs)
    ↓
Azure Document Intelligence (PDF parsing)
    ↓
Azure OpenAI Embeddings (text-embedding-3-small)
    ↓
Azure AI Search (vector + hybrid index)
    ↓
Azure OpenAI GPT-4 (generator)
    ↓
App / API layer
```

---

## Code Snippet — Basic RAG Flow

### Python (LangChain + FAISS)

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Step 1: Load documents
loader = TextLoader("runbook.txt")
documents = loader.load()

# Step 2: Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Embed and store in vector DB
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4: Create retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Step 5: Query
response = qa_chain.run("How do I restart the Databricks cluster?")
print(response)
```

### Prompt Template Pattern

```python
prompt_template = """
You are an SRE assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
"""
```

---

## Common Challenges

| Challenge | Description | Solution |
|---|---|---|
| **Chunking quality** | Bad chunks = poor retrieval | Use semantic chunking, tune overlap |
| **Embedding mismatch** | Different models for index vs query | Always use the same embedding model |
| **Context window limits** | Too many chunks exceed LLM context | Limit top-K, use re-ranking |
| **Stale index** | New docs not indexed | Automate periodic re-indexing |
| **Hallucination** | LLM ignores retrieved context | Strict prompt: "Answer only from context" |
| **Retrieval accuracy** | Wrong chunks retrieved | Use hybrid search (vector + keyword) |
| **Latency** | Slow retrieval + LLM call | Cache embeddings, use ANN indexes |

---

## Quick Glossary

| Term | Meaning |
|---|---|
| **Embedding** | Numerical vector representation of text |
| **Vector DB** | Database optimized for storing and searching vectors |
| **Cosine Similarity** | Metric to measure how similar two vectors are (0 to 1) |
| **Top-K Retrieval** | Fetching the K most similar chunks to a query |
| **Chunking** | Splitting documents into smaller pieces for indexing |
| **Chunk Overlap** | Shared content between adjacent chunks to preserve context |
| **Hybrid Search** | Combining vector search with keyword (BM25) search |
| **Re-ranking** | Scoring retrieved chunks again for relevance before sending to LLM |
| **Grounding** | Anchoring LLM response to specific retrieved facts |
| **ANN** | Approximate Nearest Neighbor — fast vector search algorithm |
| **BM25** | Classic keyword ranking algorithm used in hybrid search |

---

## References

- [LangChain RAG Docs](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Azure AI Search — Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [FAISS by Meta](https://github.com/facebookresearch/faiss)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---
