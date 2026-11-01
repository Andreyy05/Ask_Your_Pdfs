from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ask_your_pdfs"

# Embedding model: sentence-transformers/all-MiniLM-L6-v2 (fast, high quality, 384-dim)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings
# Page-level chunking: text within a page is chunked while strictly preserving page metadata
CHUNK_SIZE = 800       # Target characters per chunk (~150-200 words)
CHUNK_OVERLAP = 150    # Overlap characters between consecutive chunks

# Retrieval settings
DEFAULT_TOP_K = 4

# LLM keys and endpoints
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

