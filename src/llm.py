from abc import ABC, abstractmethod
import os
import re
from typing import List, Dict, Optional, Tuple
import urllib.request
import json

from src.config import GEMINI_API_KEY, OPENAI_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """You are a precise, scholarly research assistant answering questions using a corpus of PDF documents.
Answer the user's question STRICTLY and SOLELY based on the provided context excerpts.

Strict Rules:
1. Every factual statement or claim MUST include a citation indicating its exact source in the format: [filename, p. N] (for example: [attention_is_all_you_need.pdf, p. 3]).
2. If the context does not contain enough information to answer the question, or if the answer is absent from the provided documents, you MUST respond with:
"I don't know based on the provided documents."
3. Do not speculate, invent citations, or use outside knowledge not found in the provided excerpts.
4. Keep answers concise, factual, and informative.
"""


def format_context_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Builds the user prompt containing formatted context chunks and the question.
    """
    if not chunks:
        return f"Context:\n(No relevant documents found)\n\nQuestion: {question}"

    context_parts = []
    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        filename = meta.get("filename", "unknown.pdf")
        page = meta.get("page", 1)
        text = c.get("text", "").strip()
        context_parts.append(
            f"--- Context Chunk {idx} [{filename}, p. {page}] ---\n{text}"
        )

    full_context = "\n\n".join(context_parts)
    return f"""Context from ingested documents:

{full_context}

-----------------------
User Question: {question}

Provide your answer strictly citing sources as [filename, p. N]. If the answer is not found in the excerpts above, state: "I don't know based on the provided documents."
"""


class LLMProvider(ABC):
    """Base interface for LLM generation."""
    @abstractmethod
    def generate(self, question: str, chunks: List[Dict], distance_threshold: float = 0.72) -> str:
        """Generate an answer for the given question using the provided context chunks."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini API Provider."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, question: str, chunks: List[Dict], distance_threshold: float = 0.72) -> str:
        prompt = format_context_prompt(question, chunks)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.1,
            }
        )
        return response.text.strip()


class OpenAIProvider(LLMProvider):
    """OpenAI API Provider."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, question: str, chunks: List[Dict], distance_threshold: float = 0.72) -> str:
        prompt = format_context_prompt(question, chunks)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Local Ollama Provider."""
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, question: str, chunks: List[Dict], distance_threshold: float = 0.72) -> str:
        prompt = format_context_prompt(question, chunks)
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()


class FallbackExtractiveProvider(LLMProvider):
    """
    Built-in extractive provider that operates 100% offline without any API key or external service.
    Extracts the most relevant sentences from top chunks and attaches exact [filename, p. N] citations.
    """
    def generate(self, question: str, chunks: List[Dict], distance_threshold: float = 0.72) -> str:
        if not chunks:
            return "I don't know based on the provided documents."

        # Check distance if available (cosine distance: lower is closer)
        top_chunk = chunks[0]
        top_distance = top_chunk.get("distance", 0.0)
        
        # If the closest chunk has low similarity (cosine distance > distance_threshold), answer unknown
        if top_distance > distance_threshold and top_distance != 0.0:
            return "I don't know based on the provided documents."

        # Tokenize query for lexical overlap (removing stop words)
        stop_words = {
            "what", "is", "the", "how", "why", "when", "where", "are", "and", "in", "of", "to",
            "a", "an", "does", "do", "can", "with", "for", "on", "at", "by", "from", "which",
            "explain", "describe", "tell", "me", "about"
        }
        query_words = set(re.findall(r"\w+", question.lower())) - stop_words

        if not query_words:
            query_words = set(re.findall(r"\w+", question.lower()))

        # Determine minimum required word matches
        min_required_overlap = 2 if len(query_words) >= 2 else 1

        matched_statements = []
        seen_texts = set()

        for c in chunks:
            text = c.get("text", "")
            meta = c.get("metadata", {})
            filename = meta.get("filename", "unknown.pdf")
            page = meta.get("page", 1)
            citation = f"[{filename}, p. {page}]"

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 25 or s_clean in seen_texts:
                    continue
                s_words = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(query_words & s_words)
                if overlap >= min_required_overlap:
                    matched_statements.append((overlap, f"{s_clean} {citation}"))
                    seen_texts.add(s_clean)

        if not matched_statements:
            return "I don't know based on the provided documents."

        # Sort by overlap score and pick the top 3-4 distinct informative sentences
        matched_statements.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [item[1] for item in matched_statements[:4]]
        return "\n\n".join(top_sentences)


def get_llm_provider(preferred: str = "auto") -> Tuple[LLMProvider, str]:
    """
    Factory function that detects and returns the best available LLM provider.
    Returns: (provider_instance, provider_name)
    """
    pref = preferred.lower()

    if pref == "gemini":
        try:
            return GeminiProvider(), "Google Gemini (gemini-2.5-flash)"
        except Exception as e:
            print(f"[!] Gemini requested but failed to init: {e}")

    if pref == "openai":
        try:
            return OpenAIProvider(), "OpenAI (gpt-4o-mini)"
        except Exception as e:
            print(f"[!] OpenAI requested but failed to init: {e}")

    if pref == "ollama":
        try:
            return OllamaProvider(), f"Ollama ({OLLAMA_MODEL})"
        except Exception as e:
            print(f"[!] Ollama requested but failed to init: {e}")

    # Auto mode: check keys in priority
    if os.getenv("GEMINI_API_KEY"):
        try:
            return GeminiProvider(), "Google Gemini (gemini-2.5-flash)"
        except Exception:
            pass

    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIProvider(), "OpenAI (gpt-4o-mini)"
        except Exception:
            pass

    # Check if local Ollama server is responding
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return OllamaProvider(), f"Local Ollama ({OLLAMA_MODEL})"
    except Exception:
        pass

    # Default fallback: 100% offline extractive RAG
    return FallbackExtractiveProvider(), "Built-in Extractive RAG (Offline)"

