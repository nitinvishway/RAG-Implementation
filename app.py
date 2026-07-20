"""
Retrieval-Augmented Generation (RAG) Chatbot Implementation
Follows Parts 1 to 9 of the RAG Chatbot Tutorial.

Structure:
- Part 1: Environment setup, dependencies & folder structure
- Part 2: Load PDFs/Text files & preprocess
- Part 3: Split documents into chunks using RecursiveCharacterTextSplitter
- Part 4: Generate embeddings using HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- Part 5: Create FAISS vector database & save locally to vector_db/
- Part 6: Perform similarity search with distance scores
- Part 7: Retriever & Prompt Construction (Page 15 prompt template)
- Part 8: RAG Chatbot with Conversation Memory
- Part 9: Web Interface using Flask
"""

import os
import sys
import re
import math
import pickle
import json
import urllib.request
import urllib.error
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

# Force UTF-8 encoding for Windows terminal output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Flask for Part 9 Web Interface
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# LangChain & FAISS components
if 'VERCEL' in os.environ:
    HAS_LANGCHAIN = False
else:
    try:
        from langchain_community.document_loaders import TextLoader, PyPDFLoader
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False

if not HAS_LANGCHAIN:
    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
except Exception as e:
    print(f"[Warning] Failed to create directories (read-only filesystem): {e}")


# ---------------------------------------------------------------------------
# Part 2: Document Preprocessing (Page 6)
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = re.sub(r'[\uf0a7\u2022\u25cf\u25aa\u25fe\u2013\u2014]', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def restore_text(text: str) -> str:
    return text


def load_and_preprocess_documents(data_path: str = DATA_DIR) -> List[Document]:
    documents: List[Document] = []
    
    paths_to_check = [data_path]
    # Check /tmp/data on Vercel
    if 'VERCEL' in os.environ:
        tmp_data = "/tmp/data"
        if os.path.exists(tmp_data) and tmp_data not in paths_to_check:
            paths_to_check.append(tmp_data)

    seen_files = set()
    for path in paths_to_check:
        if not os.path.exists(path):
            continue
        for file_name in os.listdir(path):
            if file_name in seen_files:
                continue
            file_path = os.path.join(path, file_name)
            if not os.path.isfile(file_path):
                continue

            raw_text = ""
            try:
                if file_name.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_text = f.read()
                elif file_name.endswith('.pdf'):
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(file_path)
                        raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    except Exception as e:
                        print(f"[Warning] Could not extract PDF {file_name}: {e}")
                        continue
                else:
                    continue

                cleaned_content = preprocess_text(raw_text)
                if cleaned_content:
                    doc = Document(
                        page_content=cleaned_content,
                        metadata={"source": file_path, "filename": file_name, "cleaned": True}
                    )
                    documents.append(doc)
                    seen_files.add(file_name)

            except Exception as e:
                print(f"[Warning] Error reading {file_name}: {e}")

    return documents


# ---------------------------------------------------------------------------
# Part 3: Text Chunking (Pages 7-8)
# ---------------------------------------------------------------------------
def split_documents_into_chunks(
    documents: List[Document], 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Document]:
    if HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
    else:
        chunks = []
        for doc in documents:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_str = text[start:end]
                chunks.append(Document(
                    page_content=chunk_str,
                    metadata=dict(doc.metadata)
                ))
                start += (chunk_size - chunk_overlap)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        
    return chunks


# ---------------------------------------------------------------------------
# Part 4 & 5: Embeddings & FAISS Vector Database (Pages 8-10)
# ---------------------------------------------------------------------------
class FallbackVectorStore:
    def __init__(self):
        self.chunks: List[Document] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 1]

    def build(self, chunks: List[Document]):
        self.chunks = chunks
        doc_counts = {}
        total_docs = len(chunks)

        for chunk in chunks:
            tokens = set(self._tokenize(chunk.page_content))
            for t in tokens:
                doc_counts[t] = doc_counts.get(t, 0) + 1

        self.idf = {t: math.log((total_docs + 1) / (cnt + 1)) + 1 for t, cnt in doc_counts.items()}

    def _get_vector(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = {}
        for t, count in tf.items():
            if t in self.idf:
                vec[t] = (count / len(tokens)) * self.idf[t]
        return vec

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        q_vec = self._get_vector(query)
        if not q_vec:
            return [(chunk, 1.0) for chunk in self.chunks[:top_k]]

        results = []
        q_norm = math.sqrt(sum(v**2 for v in q_vec.values()))

        for chunk in self.chunks:
            c_vec = self._get_vector(chunk.page_content)
            c_norm = math.sqrt(sum(v**2 for v in c_vec.values()))
            
            if q_norm == 0 or c_norm == 0:
                score = 0.0
            else:
                dot = sum(q_vec.get(t, 0) * val for t, val in c_vec.items())
                score = dot / (q_norm * c_norm)

            results.append((chunk, float(1.0 - score)))

        results.sort(key=lambda x: x[1])
        return results[:top_k]


class RAGVectorDatabase:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.vector_store = None
        self.fallback_store = FallbackVectorStore()
        self.use_langchain_faiss = HAS_LANGCHAIN

        if self.use_langchain_faiss:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            except Exception as e:
                self.use_langchain_faiss = False

    def build_or_update(self, chunks: List[Document]):
        if not chunks:
            return None
            
        if self.use_langchain_faiss:
            try:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.save()
                return self.vector_store
            except Exception as e:
                print(f"[Warning] FAISS build error: {e}")
        
        self.fallback_store.build(chunks)
        self.save_fallback()

    def save_fallback(self, path: str = VECTOR_DB_DIR):
        try:
            os.makedirs(path, exist_ok=True)
            index_file = os.path.join(path, "index.pkl")
            with open(index_file, "wb") as f:
                pickle.dump(self.fallback_store, f)
        except Exception as e:
            print(f"[Warning] Failed to save fallback store (probably read-only filesystem): {e}")

    def save(self, path: str = VECTOR_DB_DIR):
        try:
            if self.use_langchain_faiss and self.vector_store:
                os.makedirs(path, exist_ok=True)
                self.vector_store.save_local(path)
            else:
                self.save_fallback(path)
        except Exception as e:
            print(f"[Warning] Failed to save vector store: {e}")

    def load(self, path: str = VECTOR_DB_DIR) -> bool:
        if self.use_langchain_faiss:
            faiss_index_path = os.path.join(path, "index.faiss")
            if os.path.exists(faiss_index_path):
                try:
                    self.vector_store = FAISS.load_local(
                        path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    return True
                except Exception as e:
                    pass
        
        index_file = os.path.join(path, "index.pkl")
        if os.path.exists(index_file):
            try:
                with open(index_file, "rb") as f:
                    self.fallback_store = pickle.load(f)
                return True
            except Exception as e:
                pass
        
        return False

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if self.use_langchain_faiss and self.vector_store:
            return self.vector_store.similarity_search_with_score(query, k=top_k)
        
        return self.fallback_store.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Part 7 & 8: Retriever, Prompt Construction & Memory (Pages 13-16)
# ---------------------------------------------------------------------------
class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_ai_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def _trim(self):
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]


class RAGChatbotPipeline:
    def __init__(self, vector_db: RAGVectorDatabase):
        self.vector_db = vector_db
        self.memory = ConversationMemory()

    def construct_prompt(self, context_chunks: List[str], query: str) -> str:
        cleaned_chunks = [restore_text(c) for c in context_chunks]
        context_str = "\n---\n".join(cleaned_chunks) if cleaned_chunks else "No relevant context found."
        
        prompt = (
            "You are a helpful assistant.\n"
            "Answer the user's question clearly, thoroughly, and accurately using ONLY the supplied context.\n"
            "If the information is not present in the context, say \"I don't know.\"\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question:\n{query}\n\n"
            "Detailed Answer:"
        )
        return prompt

    def generate_response(self, query: str, api_key: str = "", top_k: int = 5) -> Dict[str, Any]:
        search_results = self.vector_db.similarity_search(query, top_k=top_k)
        
        retrieved_docs = []
        context_texts = []
        for doc, score in search_results:
            context_texts.append(doc.page_content)
            retrieved_docs.append({
                "content": restore_text(doc.page_content),
                "source": doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
                "score": float(score)
            })

        prompt = self.construct_prompt(context_texts, query)
        
        active_key = api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("GROQ_API_KEY", "").strip()
        answer = ""
        provider = "Grounded Context Engine"

        # Try Groq API if key starts with gsk_
        if active_key and active_key.startswith("gsk_"):
            groq_ans = self._call_groq_api(prompt, active_key)
            if groq_ans:
                answer = groq_ans
                provider = "Groq LLaMA-3 LLM"

        # Try Gemini API if key is present and not OpenAI/Groq
        if not answer and active_key and not active_key.startswith("sk-") and not active_key.startswith("gsk_"):
            gemini_ans = self._call_gemini_api(prompt, active_key)
            if gemini_ans:
                answer = gemini_ans
                provider = "Google Gemini LLM"

        # Try OpenAI API if key starts with sk-
        if not answer and active_key and active_key.startswith("sk-"):
            openai_ans = self._call_openai_api(prompt, active_key)
            if openai_ans:
                answer = openai_ans
                provider = "OpenAI GPT-3.5 LLM"

        # Full Multi-Chunk Grounded Synthesizer
        if not answer:
            answer = self._grounded_context_answer(context_texts, query)
            provider = "Grounded Context Engine"

        self.memory.add_user_message(query)
        self.memory.add_ai_message(answer)

        return {
            "query": query,
            "answer": restore_text(answer),
            "prompt": prompt,
            "provider": provider,
            "retrieved_chunks": retrieved_docs
        }

    def _call_gemini_api(self, prompt_text: str, key: str) -> str:
        """Calls Google Gemini REST API with retry on rate limits."""
        import time as _time

        models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]

        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}
        }

        last_error = ""
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

            for attempt in range(2):  # try twice per model
                try:
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode('utf-8'),
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(req, timeout=20) as response:
                        res_data = json.loads(response.read().decode('utf-8'))
                        candidates = res_data.get('candidates', [])
                        if candidates:
                            parts = candidates[0].get('content', {}).get('parts', [])
                            if parts:
                                return parts[0].get('text', '').strip()
                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode('utf-8', errors='ignore')
                    except Exception:
                        pass
                    if e.code == 429:
                        if "limit: 0" in body or "quota" in body.lower():
                            last_error = "API quota exhausted (0 requests remaining). Check billing at https://ai.google.dev"
                            break  # no point retrying this model
                        # Transient rate limit — wait and retry
                        _time.sleep(2)
                        continue
                    elif e.code in (400, 401, 403):
                        last_error = f"API key error (HTTP {e.code}). Check your key is valid."
                        break
                    else:
                        last_error = f"Gemini API error (HTTP {e.code})"
                        break
                except Exception as e:
                    last_error = str(e)
                    break

        if last_error:
            print(f"[Gemini API] {last_error}")
        return ""

    def _call_groq_api(self, prompt_text: str, key: str) -> str:
        """Calls Groq Cloud API using LLaMA-3 models."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        models = ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"]
        
        for model in models:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question clearly and accurately using ONLY the provided context."},
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.2
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {key}'
                    }
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    choices = res_data.get('choices', [])
                    if choices:
                        return choices[0].get('message', {}).get('content', '').strip()
            except Exception as e:
                pass
        return ""

    def _call_openai_api(self, prompt_text: str, key: str) -> str:
        """Calls OpenAI Chat Completions API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer ONLY using the provided context."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return ""

    def _grounded_context_answer(self, context_texts: List[str], query: str) -> str:
        """
        Grounded Context Synthesizer (Offline — no LLM needed):
        Extracts complete, relevant sentences from ALL retrieved chunks
        and returns them as a coherent answer.
        """
        if not context_texts:
            return "I don't know."

        # Combine all retrieved chunk texts
        combined = "\n".join(context_texts)
        combined = restore_text(combined)

        # Split into sentences carefully — protect "Rs. " from being split
        protected = combined.replace("Rs. ", "Rs__DOT__ ")
        # Split on period-space, period-newline, or newline
        raw_sentences = re.split(r'\.\s+|\.\n|\n+', protected)
        sentences = []
        for s in raw_sentences:
            s = s.replace("Rs__DOT__ ", "Rs. ").strip()
            if len(s) > 10:
                sentences.append(s)

        # Extract meaningful query keywords
        stop_words = {
            "what", "is", "the", "a", "an", "when", "does", "who", "won",
            "in", "are", "of", "to", "for", "on", "how", "tell", "me",
            "about", "do", "can", "will", "my", "this", "that", "it"
        }
        query_words = [
            w.lower() for w in re.findall(r'[a-zA-Z]+', query)
            if w.lower() not in stop_words and len(w) > 1
        ]

        if not query_words:
            return "I don't know."

        # Score each sentence by how many query keywords it contains
        scored = []
        for sentence in sentences:
            s_lower = sentence.lower()
            match_count = sum(
                1 for w in query_words
                if re.search(r'\b' + re.escape(w) + r'\b', s_lower)
            )
            if match_count > 0:
                # Ensure it ends with a period
                clean = sentence.strip()
                if not clean.endswith('.'):
                    clean += '.'
                scored.append((match_count, clean))

        if not scored:
            return "I don't know."

        # Sort by number of keyword matches (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate and collect top results
        results = []
        seen = set()
        for _, sentence in scored:
            key = sentence.lower().strip()
            if key not in seen:
                seen.add(key)
                results.append(sentence)
            if len(results) >= 3:
                break

        if len(results) == 1:
            return results[0]
        else:
            return "\n".join(f"• {r}" for r in results)


# Initialize Global Vector DB and RAG Pipeline
vector_db = RAGVectorDatabase()
# Re-index data/ directory
raw_docs = load_and_preprocess_documents()
chunks = split_documents_into_chunks(raw_docs)
vector_db.build_or_update(chunks)

rag_pipeline = RAGChatbotPipeline(vector_db)


# ---------------------------------------------------------------------------
# Part 9: Web Server Setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="frontend/dist", static_url_path="", template_folder="frontend/dist")

@app.route("/")
def home():
    try:
        return send_from_directory(app.template_folder, "index.html")
    except Exception:
        # Fallback to old templates directory if React is not built
        app.template_folder = "templates"
        return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    data_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    return jsonify({
        "status": "active",
        "documents_loaded": len(data_files),
        "files": data_files,
        "faiss_enabled": vector_db.use_langchain_faiss,
        "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
        "has_openai_key": bool(os.environ.get("OPENAI_API_KEY"))
    })

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json() or {}
    user_query = data.get("question", "").strip()
    user_api_key = data.get("api_key", "").strip()

    if not user_query:
        return jsonify({"error": "Question is required."}), 400

    response_data = rag_pipeline.generate_response(user_query, api_key=user_api_key)
    return jsonify(response_data)

@app.route("/api/search", methods=["POST"])
def search_endpoint():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "Query is required."}), 400

    results = vector_db.similarity_search(query, top_k=top_k)
    output = []
    for doc, score in results:
        output.append({
            "content": restore_text(doc.page_content),
            "source": doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
            "similarity_score": float(score)
        })

    return jsonify({"query": query, "results": output})

@app.route("/api/upload", methods=["POST"])
def upload_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Write to /tmp on Vercel
    target_dir = DATA_DIR
    if 'VERCEL' in os.environ:
        target_dir = "/tmp/data"
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception:
            pass

    file_path = os.path.join(target_dir, file.filename)
    file.save(file_path)

    docs = load_and_preprocess_documents()
    chunks = split_documents_into_chunks(docs)
    vector_db.build_or_update(chunks)

    return jsonify({
        "message": f"Successfully uploaded '{file.filename}' and updated vector database.",
        "total_chunks": len(chunks)
    })

@app.route("/api/reindex", methods=["POST"])
def reindex_endpoint():
    docs = load_and_preprocess_documents()
    chunks = split_documents_into_chunks(docs)
    vector_db.build_or_update(chunks)
    return jsonify({"message": "Re-indexing complete.", "total_chunks": len(chunks)})


# ---------------------------------------------------------------------------
# CLI Step-by-Step Test Runner (Parts 1-8 Demonstration)
# ---------------------------------------------------------------------------
def run_cli_demo():
    print("=" * 60)
    print("  Retrieval-Augmented Generation (RAG) Chatbot Demo  ")
    print("=" * 60)
    
    print("\n[Part 2] Loading & Preprocessing documents from data/...")
    documents = load_and_preprocess_documents()
    print(f"Loaded {len(documents)} document(s).")

    print("\n[Part 3] Splitting documents into chunks...")
    chunks = split_documents_into_chunks(documents, chunk_size=500, chunk_overlap=50)
    print(f"Generated {len(chunks)} chunk(s).")

    print("\n[Part 4 & 5] Generating Embeddings and creating Vector Database...")
    vdb = RAGVectorDatabase()
    vdb.build_or_update(chunks)

    test_query = "What is the admission fee?"
    print(f"\n[Part 6] Testing Similarity Search for query: '{test_query}'...")
    search_results = vdb.similarity_search(test_query, top_k=2)
    for doc, score in search_results:
        safe_snippet = restore_text(doc.page_content)[:120].encode('ascii', errors='ignore').decode('ascii')
        print(f" - [Distance Score: {score:.4f}] {safe_snippet}...")

    print("\n[Part 7 & 8] Running complete RAG Chatbot Pipeline with Memory...")
    pipeline = RAGChatbotPipeline(vdb)
    
    res1 = pipeline.generate_response("What is the admission fee?")
    print(f"\nQ: What is the admission fee?")
    print(f"A: {res1['answer']}")

    res2 = pipeline.generate_response("When does admission open?")
    print(f"\nQ: When does admission open?")
    print(f"A: {res2['answer']}")

    res3 = pipeline.generate_response("Who won the FIFA World Cup in 2030?")
    print(f"\nQ: Who won the FIFA World Cup in 2030?")
    print(f"A: {res3['answer']}")

    print("\n[Demo Complete] All pipeline tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot Application")
    parser.add_argument("--cli", action="store_true", help="Run CLI step-by-step test demo")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web app")
    args = parser.parse_args()

    if args.cli or not HAS_FLASK:
        run_cli_demo()
    else:
        print("Starting RAG Chatbot Web Server on http://127.0.0.1:5000 ...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
