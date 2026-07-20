# Retrieval-Augmented Generation (RAG) Chatbot Implementation Guide

Welcome to the complete **Retrieval-Augmented Generation (RAG) Chatbot** repository built strictly following the 15-module tutorial specification.

---

## 📐 1. High-Level Architecture Diagram

```
                       +------------------------+
                       |       User Query       |
                       +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       |    Query Embedding     |
                       | (SentenceTransformers) |
                       +-----------+------------+
                                   |
                                   v
+------------------+   +------------------------+
| Knowledge Base   |-->|   FAISS Vector Search  |
| (data/*.txt,pdf) |   |  (Cosine / L2 Metric)  |
+------------------+   +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       |  Top-K Context Chunks  |
                       +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       |  Prompt Construction   |
                       | (Strict Grounding Rule)|
                       +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       | Large Language Model   |
                       | (Grounded Answer Gen)  |
                       +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       |  Final Grounded Answer |
                       +------------------------+
```

---

## 📂 2. Project Directory Structure

```text
RAG_Chatbot/
├── data/
│   └── university.txt         # Sample knowledge base document
├── vector_db/
│   ├── index.faiss            # FAISS index storage file
│   └── index.pkl              # Metadata mapping pickle file
├── templates/
│   └── index.html             # Part 9 Flask Web Chat Interface
├── app.py                     # Main application script (Parts 1 to 9)
├── requirements.txt           # Project dependencies
└── README.md                  # Part 10 Guide, Architecture & Viva Q&A
```

---

## 🛠️ 3. Quick Start & Setup

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

#### A. Run CLI Step-by-Step Demo Mode (Parts 1 to 8)
```bash
python app.py --cli
```

#### B. Launch Web Interface Dashboard (Part 9)
```bash
python app.py
```
Open your browser and visit: **`http://127.0.0.1:5000`**

---

## 🔍 4. Step-by-Step Code Explanation

### Part 1: Imports & Directory Configuration
- Imports `langchain_community` document loaders (`TextLoader`, `PyPDFLoader`), `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, and `FAISS`.
- Sets up standard `data/` and `vector_db/` directory paths automatically.

### Part 2: Document Ingestion & Preprocessing
- Function `load_and_preprocess_documents()` iterates through all files in `data/`.
- Function `preprocess_text()` cleans HTML tags, normalizes whitespace, and removes duplicate punctuation before indexing.

### Part 3: Text Chunking
- Function `split_documents_into_chunks()` uses `RecursiveCharacterTextSplitter`.
- Uses a `chunk_size` of 500 characters and `chunk_overlap` of 50 characters to preserve cross-sentence context boundaries (Sliding Window principle).

### Part 4 & 5: Embedding Generation & FAISS Storage
- Uses `sentence-transformers/all-MiniLM-L6-v2` to map text into 384-dimensional dense semantic vectors.
- Class `RAGVectorDatabase` constructs the FAISS index and persists it to `vector_db/index.faiss` and `vector_db/index.pkl`.

### Part 6: Similarity Search
- Method `similarity_search()` queries the FAISS vector index using L2/Cosine similarity to return the top $K$ closest text chunks along with similarity distance scores.

### Part 7 & 8: Retriever, Prompt Construction & Conversation Memory
- Class `ConversationMemory` stores previous turns in memory.
- Class `RAGChatbotPipeline` builds the strict prompt template:
  ```text
  You are an assistant.
  Use ONLY the supplied context.
  If information is unavailable, say "I don't know."
  ```
- If the question is outside the knowledge base context (e.g. *"Who won the FIFA World Cup in 2030?"*), the pipeline strictly responds **"I don't know."** to prevent hallucinations.

### Part 9: Flask Web Interface & REST API Endpoints
- Provides `/api/chat` for handling chat queries.
- Provides `/api/search` for testing similarity search directly.
- Provides `/api/upload` for adding new documents and dynamically re-indexing the FAISS vector database.

---

## 🎓 5. Viva & Interview Questions & Answers

### Q1: What is Retrieval-Augmented Generation (RAG)?
**Answer:** RAG is an architectural pattern that combines Information Retrieval (vector search over custom documents) with a Large Language Model (LLM). Instead of relying solely on internal weights (which suffer from hallucinations and knowledge cutoffs), RAG retrieves relevant document chunks first and injects them as context into the prompt.

### Q2: Why do LLMs need Retrieval?
**Answer:**
1. **Hallucinations**: LLMs confidently invent false facts.
2. **Knowledge Cutoff**: Pre-trained models lack post-training real-time or recent data.
3. **Domain-Specific / Private Data**: Enterprise docs, APIs, and personal files exceed context limits or are private.

### Q3: What is the purpose of Text Chunking and Overlap?
**Answer:**
- **Chunking** breaks large documents into smaller units so they fit within context windows and yield higher retrieval relevance.
- **Overlap** (e.g., 50 tokens) ensures sentences near chunk boundaries retain full semantic context.

### Q4: What is an Embedding?
**Answer:** An embedding is a high-dimensional numerical vector (e.g., 384 dimensions) representing the semantic meaning of text. Texts with similar meanings are mapped to vectors close to each other in vector space.

### Q5: How does FAISS perform Similarity Search?
**Answer:** FAISS (Facebook AI Similarity Search) indexes vector embeddings and calculates distance metrics (such as Cosine Similarity, Euclidean Distance, or Dot Product). It utilizes Approximate Nearest Neighbor (ANN) algorithms like HNSW or IVF to accelerate search across large datasets.

### Q6: What metrics are used to evaluate RAG systems?
**Answer:**
1. **Retrieval Precision**: Fraction of retrieved chunks that are relevant.
2. **Recall**: Fraction of all relevant chunks in the corpus that were retrieved.
3. **MRR (Mean Reciprocal Rank)**: Measures how early the first relevant chunk appears.
4. **Answer Faithfulness**: Assesses whether the LLM answer is strictly supported by the retrieved context without hallucination.
5. **Answer Relevance**: Assesses whether the response directly addresses the user's prompt.

---

## 🛠️ 6. Troubleshooting Guide

| Issue | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'faiss'` | `faiss-cpu` not installed in environment | Run `pip install faiss-cpu` |
| `ValueError: allow_dangerous_deserialization` | Security check in LangChain FAISS loader | `allow_dangerous_deserialization=True` is enabled in `app.py` |
| Low Retrieval Accuracy | Chunk size too large or small | Adjust `chunk_size` (300–800) and `chunk_overlap` (50–100) in `app.py` |
| OpenAI API Key missing | Environment variable `OPENAI_API_KEY` not set | `app.py` automatically uses grounded context fallback engine |
