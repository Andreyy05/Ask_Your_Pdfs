import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def extract_pages_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    Extracts text from each page of a PDF file.
    Returns a list of dicts: [{'page': 1, 'text': '...'}, ...]
    Page numbers are 1-indexed.
    """
    pages_data = []
    reader = pypdf.PdfReader(str(pdf_path))
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        
        # Clean up whitespace while preserving paragraph breaks
        cleaned_text = re.sub(r"[ \t]+", " ", raw_text)
        cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text).strip()
        
        if cleaned_text:
            pages_data.append({
                "page": page_num,
                "text": cleaned_text
            })
            
    return pages_data


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks using paragraph, sentence, and word boundaries.
    Ensures clean word boundaries without mid-word truncation and guarantees proper overlap.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # If remaining text fits within one chunk, emit it and terminate
        if text_len - start <= chunk_size:
            chunk = text[start:].strip()
            if len(chunk) > 20:
                chunks.append(chunk)
            break

        end = start + chunk_size

        # Search backwards from end for a clean semantic break point
        break_point = -1
        # 1. Paragraph boundary
        p_break = text.rfind("\n\n", start + chunk_overlap, end)
        if p_break != -1:
            break_point = p_break
        else:
            # 2. Sentence boundary
            s_break = text.rfind(". ", start + chunk_overlap, end)
            if s_break != -1:
                break_point = s_break + 1  # include period
            else:
                # 3. Line break
                nl_break = text.rfind("\n", start + chunk_overlap, end)
                if nl_break != -1:
                    break_point = nl_break
                else:
                    # 4. Word boundary
                    sp_break = text.rfind(" ", start + chunk_overlap, end)
                    if sp_break != -1:
                        break_point = sp_break

        if break_point != -1 and break_point > start:
            end = break_point

        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)

        # Move forward, respecting overlap
        new_start = max(start + 1, end - chunk_overlap)

        # Snap new_start to a word boundary to avoid slicing words in half
        if new_start < text_len and not text[new_start].isspace():
            if new_start > 0 and not text[new_start - 1].isspace():
                next_space = text.find(" ", new_start)
                if next_space != -1 and next_space < end:
                    new_start = next_space + 1

        # Prevent stalling or backwards motion
        if new_start <= start:
            new_start = end

        start = new_start

    return chunks


def process_pdf_file(pdf_path: Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Extracts pages from a single PDF and chunks each page separately.
    Assigns strict page-level metadata to every chunk: [filename, p. N].
    """
    filename = pdf_path.name
    pages = extract_pages_from_pdf(pdf_path)
    processed_chunks = []

    for page_info in pages:
        page_num = page_info["page"]
        raw_page_text = page_info["text"]
        
        chunks = chunk_text(raw_page_text, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            # Clean safe ID
            clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)
            chunk_id = f"{clean_name}_p{page_num}_c{idx}"
            
            processed_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "page": int(page_num),
                    "chunk_index": int(idx),
                    "char_count": len(chunk),
                }
            })

    return processed_chunks


def ingest_folder(
    data_dir: Path = DEFAULT_DATA_DIR,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    reset: bool = False,
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict:
    """
    Ingests all PDF files from data_dir into ChromaDB.
    Returns a dictionary of ingestion statistics.
    """
    data_dir = Path(data_dir)
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(data_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"[!] No PDF files found in directory: {data_dir.resolve()}")
        return {"files": 0, "pages": 0, "chunks": 0}

    print(f"[*] Found {len(pdf_files)} PDF file(s) in {data_dir.resolve()}")
    for pdf in pdf_files:
        print(f"    - {pdf.name} ({round(pdf.stat().st_size / 1024, 1)} KB)")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"[*] Cleared existing collection '{COLLECTION_NAME}' (reset=True)")
        except ValueError:
            print(f"[*] Collection '{COLLECTION_NAME}' does not exist, creating fresh.")
        except Exception as e:
            print(f"[!] Warning: Could not delete collection '{COLLECTION_NAME}': {e}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Process all PDFs
    all_chunks = []
    total_pages = 0

    for pdf in pdf_files:
        print(f"[*] Parsing: {pdf.name}...")
        chunks = process_pdf_file(pdf, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        # count unique pages
        unique_pages = len(set(c["metadata"]["page"] for c in chunks))
        total_pages += unique_pages
        print(f"    -> Extracted {unique_pages} pages, {len(chunks)} chunks")

    if not all_chunks:
        print("[!] No text chunks were extracted from PDFs.")
        return {"files": len(pdf_files), "pages": total_pages, "chunks": 0}

    print(f"[*] Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"[*] Generating embeddings for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    # Upsert into ChromaDB in batches
    batch_size = 128
    print(f"[*] Storing chunks in ChromaDB at {chroma_dir.resolve()}...")
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batch_emb = embeddings[i : i + batch_size]
        
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=batch_emb,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"[OK] Successfully ingested {len(pdf_files)} PDFs ({total_pages} pages, {len(all_chunks)} chunks).")
    return {
        "files": len(pdf_files),
        "pages": total_pages,
        "chunks": len(all_chunks),
        "collection_total": collection.count(),
    }


def ingest_single_pdf(
    pdf_path: Path,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    model: Optional[SentenceTransformer] = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict:
    """
    Ingests or updates a single PDF into ChromaDB without wiping existing documents.
    Accepts an optional pre-loaded SentenceTransformer model instance to avoid redundant initialization.
    """
    pdf_path = Path(pdf_path)
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chunks = process_pdf_file(pdf_path, chunk_size, chunk_overlap)
    if not chunks:
        return {"filename": pdf_path.name, "pages": 0, "chunks": 0}

    unique_pages = len(set(c["metadata"]["page"] for c in chunks))

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if model is None:
        model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_emb = embeddings[i : i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=batch_emb,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    return {
        "filename": pdf_path.name,
        "pages": unique_pages,
        "chunks": len(chunks),
        "collection_total": collection.count(),
    }

