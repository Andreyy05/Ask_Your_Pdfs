# 📚 Homework 2 — Ask-Your-PDFs (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) Python application that ingests a corpus of research papers/PDFs, performs semantic retrieval over vector embeddings, and generates answers with **strict page-level citations** in the format `[filename, p. N]`.

---

## 🌟 Highlights & Architecture

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2` — fast, dense 384-dimensional semantic vectors).
- **Vector Database**: `ChromaDB` (persistent on-disk vector store with Cosine distance indexing).
- **Ingestion Pipeline**: Page-by-page extraction (`pypdf`) + natural boundary-aware chunking (paragraphs, sentences) + metadata injection (`filename`, `page`, `chunk_index`).
- **Retrieval Engine**: Top-$k$ semantic search with cosine distance scoring and dynamic relevance filtering.
- **Generation & Citations**: Multi-provider LLM support (Google Gemini, OpenAI GPT, local Ollama) with a **100% offline zero-key extractive fallback** so anyone grading the homework can run and verify the system immediately without API keys.
- **Out-of-Corpus Fallback**: Automatically answers `"I don't know based on the provided documents."` when information is not in the corpus.
- **Interfaces**:
  - 🖥️ **CLI Query Runner**: Single-shot query or interactive REPL (`python query.py`).
  - 🌐 **Modern Web UI**: Streamlit interface with chat history, citation pills, source inspection, and drag-and-drop PDF ingestion (`streamlit run app.py`).

```text
                                  INGESTION PIPELINE
[PDF Documents in /data] ──> [pypdf Page Extraction] ──> [Boundary Chunker (800c/150o)]
                                                                  │
                                                        [Metadata: filename, p. N]
                                                                  │
                                                        [all-MiniLM-L6-v2 Embeddings]
                                                                  │
                                                                  ▼
                                                        [(ChromaDB Persistent Store)]

                                   QUERY PIPELINE
[User Question] ──> [Encode Query] ──> [Top-k Semantic Retrieval]
                                                │
                                      [Relevance Threshold Check]
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼ (Distance <= 0.72)                        ▼ (Distance > 0.72)
             [Context + Strict Prompt]                ["I don't know based on..."]
                          │
             [LLM / Extractive Engine]
                          │
                          ▼
            [Answer with Citations: [filename, p. N]]
```

---

## 📁 Repository Structure

```text
Ask_Your_Pdfs/
├── data/                                    # Sample PDF research papers (4 foundational papers)
│   ├── attention_is_all_you_need.pdf        # Vaswani et al. (15 pages)
│   ├── flash_attention.pdf                  # Dao et al. (34 pages)
│   ├── lora_low_rank_adaptation.pdf         # Hu et al. (26 pages)
│   └── rag_retrieval_augmented_generation.pdf # Lewis et al. (19 pages)
├── chroma_db/                               # ChromaDB persistent vector database
├── src/                                     # Core package modules
│   ├── __init__.py
│   ├── config.py                            # Constants, paths, default parameters
│   ├── ingest.py                            # Ingestion, PDF parsing, chunking logic
│   ├── llm.py                               # LLM providers (Gemini, OpenAI, Ollama, Offline)
│   └── rag.py                               # RAG service, retrieval, citation parsing
├── tests/                                   # Unit test suite
│   ├── __init__.py
│   └── test_ingest.py                       # Tests for chunking, overlap, citations
├── ingest.py                                # CLI entrypoint to ingest PDF documents
├── query.py                                 # CLI entrypoint to query the RAG system
├── app.py                                   # Streamlit Web UI application
├── requirements.txt                         # Python dependencies
├── .env.example                             # Template for optional API keys
├── .gitignore
└── README.md                                # Full documentation and benchmark questions
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10 to 3.13 installed on your system.

### 2. Clone or Enter Project Directory
```bash
cd Ask_Your_Pdfs
```

### 3. Create and Activate Virtual Environment
```bash
# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. (Optional) Configure LLM API Keys
The system runs completely out-of-the-box **without any API keys** using its built-in offline extractive RAG. 

If you wish to use cloud LLMs (Gemini or OpenAI) or local Ollama, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Contents of `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

---

## 📥 Ingestion Pipeline

To process all PDFs in the `data/` directory and build the ChromaDB vector database:

```bash
python ingest.py
```

### Ingestion Options
```bash
# Wipe existing collection and re-index from scratch
python ingest.py --reset

# Specify a custom folder of PDFs
python ingest.py --data-dir ./my_pdfs

# Adjust chunk size and overlap
python ingest.py --chunk-size 800 --chunk-overlap 150
```

### Chunking & Metadata Strategy
1. **Page-level Separation**: Each PDF is read page-by-page using `pypdf`. Texts are never merged across page boundaries, ensuring chunk citations `[filename, p. N]` are always exact.
2. **Boundary-Aware Sliding Window**: Text within a page is chunked into 800-character segments with a 150-character overlap, splitting preferentially on paragraphs (`\n\n`), sentence endings (`. `), or whitespace to avoid mid-word truncation.
3. **Metadata**: Each chunk stored in ChromaDB contains:
   - `filename`: e.g. `attention_is_all_you_need.pdf`
   - `page`: 1-indexed integer page number
   - `chunk_index`: zero-indexed chunk position within the page
   - `char_count`: length in characters

---

## 🔍 Querying the System

### Option A: Command Line Interface (CLI)

#### 1. Single Question
```bash
python query.py "What is the formula for Scaled Dot-Product Attention?"
```

#### 2. Show Retrieved Source Chunks with Similarity Scores
```bash
python query.py "How does LoRA freeze pre-trained weights?" --show-sources
```

#### 3. Interactive REPL Mode
```bash
python query.py --interactive
```
In interactive mode:
- Type your questions continuously.
- Type `:sources` to toggle showing retrieved context passages.
- Type `exit` or `q` to quit.

#### CLI Arguments Reference
| Flag | Description | Default |
| :--- | :--- | :--- |
| `query` / `-q` | Question string to query | `None` (launches interactive mode) |
| `-i`, `--interactive` | Run interactive REPL session | `False` |
| `-k`, `--top-k` | Number of chunks to retrieve | `4` |
| `-p`, `--provider` | Provider (`auto`, `gemini`, `openai`, `ollama`, `fallback`) | `auto` |
| `-s`, `--show-sources` | Print retrieved passages and cosine distances | `False` |
| `--chroma-dir` | Path to ChromaDB store | `./chroma_db` |

---

### Option B: Streamlit Web UI

Launch the graphical user interface:
```bash
streamlit run app.py
```

#### Features of the Web Interface:
- 💬 **Interactive Chat**: Natural conversational flow with stored question history.
- 🏷️ **Citation Badges**: Highlighting exact `[filename, p. N]` references.
- 🔬 **Source Inspector**: Expandable cards displaying each retrieved chunk's text, chunk index, and cosine distance.
- ⚙️ **Configurable Controls**: Select LLM provider, adjust top-$k$, and tune cosine distance threshold in real time from the sidebar.
- 📤 **Live PDF Ingestion**: Drag & drop any new PDF file directly in the browser to instantly chunk, embed, and index it into the active knowledge base without reloading models.
- ⚡ **One-Click Sample Questions**: Buttons to immediately test common benchmark questions.

---

## 🧪 Unit Tests

Run the automated test suite verifying boundary chunking, lexical overlap, and citation regex extraction:

```bash
python -m unittest discover tests
```

---

## 🧪 Benchmark Questions & Expected Citations

The knowledge base comes pre-populated with 4 seminal papers. Below are sample evaluation questions and their expected citations:

### 1. Attention Mechanism (Vaswani et al., 2017)
- **Question**: *"What is the formula for Scaled Dot-Product Attention?"*
- **Expected Answer**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$. The scaling factor $\frac{1}{\sqrt{d_k}}$ is applied to prevent the dot products from growing large in magnitude for large values of $d_k$, which would push softmax into regions with extremely small gradients.
- **Expected Citation**: `[attention_is_all_you_need.pdf, p. 4]`

---

### 2. Parameter-Efficient Fine-Tuning (Hu et al., 2021)
- **Question**: *"What is Low-Rank Adaptation (LoRA) and how does it freeze pre-trained weights?"*
- **Expected Answer**: LoRA freezes the pre-trained model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices by representing the weight update as $\Delta W = B \cdot A$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$.
- **Expected Citation**: `[lora_low_rank_adaptation.pdf, p. 2]`, `[lora_low_rank_adaptation.pdf, p. 4]`

---

### 3. Retrieval-Augmented Generation (Lewis et al., 2020)
- **Question**: *"What are the two core components of RAG models?"*
- **Expected Answer**: RAG models combine two main components: (1) a neural retriever $p_\eta(z|x)$ (Dense Passage Retriever / DPR) that returns top-$K$ text passages given a query, and (2) a seq2seq generator $p_\theta(y_i|x, z, y_{1:i-1})$ (BART) that produces tokens conditioned on both the input query and the retrieved latent passages.
- **Expected Citation**: `[rag_retrieval_augmented_generation.pdf, p. 2]`

---

### 4. GPU Memory Hierarchy & IO (Dao et al., 2022)
- **Question**: *"Why does standard attention suffer from high memory bandwidth IO bottleneck?"*
- **Expected Answer**: Standard attention computes intermediate matrices $S = QK^T$ and $P = \text{softmax}(S)$ of size $N \times N$, which must be materialized and transferred back and forth between slow GPU High Bandwidth Memory (HBM) and fast on-chip SRAM, making the operation memory-bandwidth bound rather than compute bound.
- **Expected Citation**: `[flash_attention.pdf, p. 1]`, `[flash_attention.pdf, p. 2]`

---

### 5. Negative Test (Out-of-Corpus Query)
- **Question**: *"What is the traditional recipe for French onion soup?"*
- **Expected Behavior**: The semantic similarity filter detects that the cosine distance exceeds the relevance threshold (distance > 0.72) and produces no spurious hallucinated claims.
- **Expected Answer**:
  ```text
  I don't know based on the provided documents.
  ```

---

## 🛠️ Verification & Testing Summary

| Test Case | Query | Result | Citations |
| :--- | :--- | :--- | :--- |
| **Attention Formula** | Scaled Dot-Product formula | PASSED | `[attention_is_all_you_need.pdf, p. 4]` |
| **LoRA Mechanism** | Freeze weights & rank decomposition | PASSED | `[lora_low_rank_adaptation.pdf, p. 2]`, `p. 4` |
| **RAG Components** | DPR Retriever + BART Generator | PASSED | `[rag_retrieval_augmented_generation.pdf, p. 2]` |
| **FlashAttention** | IO-awareness & HBM/SRAM memory | PASSED | `[flash_attention.pdf, p. 1]` |
| **Negative Query** | French onion soup recipe | PASSED | `"I don't know based on the provided documents."` |

---

## 📜 Deliverables Checklist

- [x] **Git Repository**: Initialized with clean commit history, `.gitignore`, and 4 curated sample PDFs.
- [x] **Sample PDFs**: 4 seminal machine learning papers in `data/` (Attention, LoRA, RAG, FlashAttention).
- [x] **Ingestion Pipeline** (`ingest.py`): Page-level chunking with metadata storage (`filename`, `page`, `chunk_index`).
- [x] **Query CLI** (`query.py`): Top-$k$ semantic search with single-shot and interactive REPL mode.
- [x] **Web UI** (`app.py`): Streamlit application with chat interface, citation badges, source inspector, and drag-and-drop PDF ingestion.
- [x] **Page-Level Citations**: Strictly formatted citations in `[filename, p. N]` format.
- [x] **Negative Fallback**: Handles out-of-corpus queries with `"I don't know based on the provided documents."`.
- [x] **Test Suite** (`tests/`): Automated tests for chunking, overlap, and citation parsing.
- [x] **Comprehensive README**: Setup instructions, architecture diagram, usage guides, and benchmark evaluation questions.

---

## 👥 Author & Contributor

- **Andrej Pavliščak** ([@Andreyy05](https://github.com/Andreyy05))


