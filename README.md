# 🤖 NimBot — Personalized AI RAG Document Assistant

An intelligent, privacy-first RAG (Retrieval-Augmented Generation) assistant that allows users to upload any PDF document and chat with it in real-time — featuring zero hallucinations and exact page-level source citations.

---

## 📌 The Business Problem
Reading long documents (contracts, research papers, technical manuals, annual reports) to find specific answers is slow and inefficient:
- **Time Wasted**: Skimming hundreds of pages manually burns valuable engineering and business hours.
- **AI Hallucinations**: Standard LLMs often invent facts when asked about company-specific or private documents.
- **Privacy Concerns**: Uploading confidential enterprise PDFs to public cloud APIs violates data compliance.

---

## 💡 The Solution
**NimBot** solves document overload using a modern, local RAG pipeline:
1. **Metadata-Aware PDF Extraction**: Reads PDFs page-by-page using `pdfplumber` and applies a sliding-window chunking strategy with overlap, preserving exact page numbers.
2. **Local Embedding Engine**: Generates dense 384-dimensional vector embeddings locally using `FastEmbed` (`BAAI/bge-small-en-v1.5`) — 0% API cost, 100% privacy.
3. **Vector Database Persistence**: Indexes text chunks and metadata into a local `ChromaDB` instance for high-speed semantic similarity retrieval.
4. **Strict Context-Only Generation**: Routes relevant context snippets to a local LLM (`qwen2.5-coder:7b` via Ollama) with a strict anti-hallucination system prompt.
5. **Interactive NimBot UI**: A ChatGPT-style Streamlit dashboard featuring custom cyan branding, session state memory, and page citation badges (`📍 Source: Page X`).

---

## 🏗️ System Architecture
[ Upload PDF ]
│
▼
[ app/pdf_processor.py ] ──► Extracts Text & Chunks (Preserves Page #)
│
▼
[ app/embeddings.py ] ──► FastEmbed Vector Generation (Local ONNX)
│
▼
[ app/vector_store.py ] ──► Stores Vectors & Metadata in ChromaDB
│
▼ (User Query)
[ app/qa_engine.py ] ──► 1. Semantic Similarity Search (Top-K)
│ 2. Context Injection + Strict Anti-Hallucination Prompt
│ 3. Local Ollama LLM Inference
▼
[ app/main.py ] ──► NimBot Streamlit UI (Displays Answer + Source Citations)

## 🛠️ Tech Stack & Tools

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.10+ | Core application runtime |
| **Frontend UI** | Streamlit | Customized Light Cyan & White ChatGPT-style dashboard |
| **PDF Extraction** | `pdfplumber` | Page-level text and layout extraction |
| **Vector Embeddings** | `FastEmbed` | Lightweight local ONNX embedding inference (`bge-small-en`) |
| **Vector Database** | `ChromaDB` | Embedded vector database for semantic similarity search |
| **LLM Inference** | Ollama (`qwen2.5-coder:7b`) | Privacy-first local LLM execution via OpenAI SDK |
| **Environment** | python-dotenv | Secure environment variable management |

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running locally

### 1. Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/nimasharifiniko/ai-pdf-rag-assistant.git
cd ai-pdf-rag-assistant

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Pull Local AI Model
Ensure Ollama is running, then pull the model:

Bash

ollama pull qwen2.5-coder:7b
3. Run NimBot Application
Bash

streamlit run app/main.py
👤 Author
Developed by Nima Sharifi Niko as part of an advanced AI Engineering portfolio.

GitHub: github.com/nimasharifiniko
LinkedIn: linkedin.com/in/nimasharifiniko