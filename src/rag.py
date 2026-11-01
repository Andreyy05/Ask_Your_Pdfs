import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    DEFAULT_CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    DEFAULT_TOP_K,
)
from src.llm import get_llm_provider, LLMProvider


class RAGService:
    """
    Core RAG Pipeline:
    - Encodes queries with SentenceTransformer
    - Performs top-k semantic retrieval in ChromaDB
    - Formats context and metadata
    - Generates answers with page-level citations: [filename, p. N]
    """

    def __init__(
        self,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.model_name = model_name

        # Lazy-loaded attributes
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def count(self) -> int:
        """Returns total number of chunks stored in vector database."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_indexed_documents(self) -> List[str]:
        """Returns unique filenames currently indexed in ChromaDB."""
        try:
            results = self.collection.get(
                include=["metadatas"],
                limit=10_000,
            )
            metas = results.get("metadatas", [])
            filenames = sorted(list(set(m.get("filename") for m in metas if m and "filename" in m)))
            return filenames
        except Exception:
            return []

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Performs semantic search over the corpus for the query.
        Returns a list of chunk dicts:
        [{ 'id': ..., 'text': ..., 'metadata': {'filename': ..., 'page': ...}, 'distance': float }]
        """
        if self.count() == 0:
            return []

        # Encode query
        query_embedding = self.embedder.encode([query]).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            dist = float(distances[i]) if i < len(distances) else 0.0
            chunks.append({
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dist,
            })

        return chunks

    def answer_question(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        provider: str = "auto",
        distance_threshold: float = 0.72,
    ) -> Dict:
        """
        Main query pipeline:
        1. Retrieves top-k chunks.
        2. Evaluates semantic relevance.
        3. Calls LLM with strict citation instructions.
        4. Extracts citations and returns structured output.
        """
        # Guard against empty database
        if self.count() == 0:
            return {
                "query": query,
                "answer": "Knowledge base is empty. Please run ingestion first using: python ingest.py",
                "citations": [],
                "chunks": [],
                "provider": "None",
            }

        # 1. Retrieve top-k chunks
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return {
                "query": query,
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "chunks": [],
                "provider": "None",
            }

        # Check top chunk distance
        top_dist = chunks[0].get("distance", 0.0)
        # Cosine distance > threshold indicates poor relevance
        if top_dist > distance_threshold and top_dist > 0.0:
            return {
                "query": query,
                "answer": "I don't know based on the provided documents.",
                "citations": [],
                "chunks": chunks,
                "provider": "Semantic Filter (Low Similarity)",
            }

        # 2. Get LLM provider and generate response
        llm_instance, provider_name = get_llm_provider(preferred=provider)
        try:
            raw_answer = llm_instance.generate(query, chunks, distance_threshold=distance_threshold)
        except Exception as e:
            # Fallback if cloud API call fails
            print(f"[!] Primary LLM generation error ({provider_name}): {e}. Falling back to Extractive RAG.")
            from src.llm import FallbackExtractiveProvider
            fallback = FallbackExtractiveProvider()
            raw_answer = fallback.generate(query, chunks, distance_threshold=distance_threshold)
            provider_name = "Built-in Extractive Fallback"

        # 3. Extract all [filename, p. N] citations from the answer
        citation_pattern = r"\[([^,\[\]]+?\.pdf),\s*(?:p\.|page)\s*(\d+)\]"
        matches = re.findall(citation_pattern, raw_answer, flags=re.IGNORECASE)
        citations = []
        for filename, page in matches:
            citation_str = f"[{filename}, p. {page}]"
            if citation_str not in citations:
                citations.append(citation_str)

        # If answer didn't explicitly write citations but answered factually, add the top retrieved sources
        if not citations and "I don't know" not in raw_answer:
            for c in chunks[:2]:
                m = c.get("metadata", {})
                citations.append(f"[{m.get('filename')}, p. {m.get('page')}]")

        return {
            "query": query,
            "answer": raw_answer,
            "citations": citations,
            "chunks": chunks,
            "provider": provider_name,
        }

