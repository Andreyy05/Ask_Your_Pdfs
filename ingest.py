#!/usr/bin/env python3
"""
Ingestion script for Ask-Your-PDFs (RAG).
Reads PDF documents, splits pages into chunks with overlap, computes sentence-transformer embeddings,
and stores them in ChromaDB with page-level metadata: [filename, p. N].

Usage:
    python ingest.py
    python ingest.py --data-dir ./my_pdfs
    python ingest.py --reset
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_DATA_DIR, DEFAULT_CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest import ingest_folder


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a directory of PDF documents into the ChromaDB vector database."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to the folder containing PDF files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=str(DEFAULT_CHROMA_DIR),
        help=f"Path to store ChromaDB persistent database (default: {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe and recreate the vector database collection before ingesting.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Target character size per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Character overlap between consecutive chunks (default: {CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    chroma_path = Path(args.chroma_dir)

    print("=" * 60)
    print(" [*] Ask-Your-PDFs: Ingestion Pipeline")
    print("=" * 60)
    print(f" Source Folder : {data_path.resolve()}")
    print(f" Vector Store  : {chroma_path.resolve()}")
    print(f" Chunk Size    : {args.chunk_size} chars (overlap: {args.chunk_overlap})")
    print(f" Reset DB      : {args.reset}")
    print("=" * 60)

    stats = ingest_folder(
        data_dir=data_path,
        chroma_dir=chroma_path,
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("\n" + "=" * 60)
    print(" Ingestion Summary")
    print("=" * 60)
    print(f" PDF files processed : {stats.get('files', 0)}")
    print(f" Pages indexed       : {stats.get('pages', 0)}")
    print(f" Total chunks stored : {stats.get('chunks', 0)}")
    print(f" Collection size     : {stats.get('collection_total', 0)}")
    print("=" * 60)
    print("\n[OK] Ingestion complete. You can now query using: python query.py \"your question\"")


if __name__ == "__main__":
    main()
