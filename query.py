#!/usr/bin/env python3
"""
CLI Query runner for Ask-Your-PDFs (RAG).
Retrieves relevant PDF chunks from ChromaDB and generates answers with page-level citations: [filename, p. N].

Usage:
    python query.py "What is the formula for Scaled Dot-Product Attention?"
    python query.py --interactive
    python query.py -q "How does LoRA work?" --show-sources --top-k 5
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_CHROMA_DIR, DEFAULT_TOP_K
from src.rag import RAGService


def print_banner():
    banner = """
======================================================================
           Ask-Your-PDFs: Semantic Question Answering (RAG)
======================================================================
"""
    print(banner.strip())


def run_single_query(rag: RAGService, query: str, top_k: int, provider: str, show_sources: bool):
    print(f"\n[?] Question: {query}")
    print("-" * 70)

    start_time = time.time()
    result = rag.answer_question(query=query, top_k=top_k, provider=provider)
    elapsed = time.time() - start_time

    answer = result.get("answer", "")
    citations = result.get("citations", [])
    chunks = result.get("chunks", [])
    provider_used = result.get("provider", "Unknown")

    print(f"[*] Provider: {provider_used}  (took {elapsed:.2f}s)")
    print("\nAnswer:")
    print(answer)

    if citations:
        print("\nCitations:")
        for c in citations:
            print(f"  • {c}")

    if show_sources and chunks:
        print("\n" + "=" * 70)
        print("Retrieved Source Chunks:")
        print("=" * 70)
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            dist = chunk.get("distance", 0.0)
            fname = meta.get("filename", "unknown.pdf")
            page = meta.get("page", "?")
            c_idx = meta.get("chunk_index", 0)
            text = chunk.get("text", "").strip()

            print(f"\n[Source #{i}] [{fname}, p. {page}] (chunk {c_idx}, cosine distance: {dist:.4f})")
            print("-" * 50)
            # Indent text slightly for readability
            indented = "\n".join("  " + line for line in text.split("\n"))
            print(indented)

    print("\n" + "=" * 70)


def interactive_mode(rag: RAGService, top_k: int, provider: str, show_sources: bool):
    print_banner()
    docs = rag.get_indexed_documents()
    print(f"[*] Total indexed chunks: {rag.count()}")
    if docs:
        print(f"[*] Available documents ({len(docs)}):")
        for d in docs:
            print(f"    - {d}")
    else:
        print("[!] No documents found in vector store! Run 'python ingest.py' first.")

    print("\nType your question and press Enter. (Commands: 'exit' or 'quit' to leave, ':sources' to toggle sources)")
    print("-" * 70)

    sources_flag = show_sources
    while True:
        try:
            prompt_str = f"Ask ({'sources ON' if sources_flag else 'sources OFF'})> "
            query = input(prompt_str).strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            if query.lower() == ":sources":
                sources_flag = not sources_flag
                print(f"[*] Show sources toggled to: {sources_flag}")
                continue

            run_single_query(
                rag=rag,
                query=query,
                top_k=top_k,
                provider=provider,
                show_sources=sources_flag,
            )

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Query the Ask-Your-PDFs RAG system from the command line."
    )
    parser.add_argument(
        "query_pos",
        nargs="?",
        default=None,
        help="Question to ask (positional argument)",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Question to ask (optional named flag)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive REPL session.",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        default="auto",
        choices=["auto", "gemini", "openai", "ollama", "fallback"],
        help="LLM provider to use for generation (default: auto)",
    )
    parser.add_argument(
        "-s", "--show-sources",
        action="store_true",
        help="Print the retrieved text chunks and page metadata.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=str(DEFAULT_CHROMA_DIR),
        help=f"Path to ChromaDB database (default: {DEFAULT_CHROMA_DIR})",
    )

    args = parser.parse_args()

    # Determine question from positional or --query flag
    question = args.query or args.query_pos

    chroma_path = Path(args.chroma_dir)
    rag = RAGService(chroma_dir=chroma_path)

    if args.interactive or not question:
        interactive_mode(rag, args.top_k, args.provider, args.show_sources)
    else:
        run_single_query(
            rag=rag,
            query=question,
            top_k=args.top_k,
            provider=args.provider,
            show_sources=args.show_sources,
        )


if __name__ == "__main__":
    main()

