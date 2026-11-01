"""
Unit tests for Ask-Your-PDFs ingestion and chunking logic.
"""

import unittest
import re
from src.ingest import chunk_text


class TestChunking(unittest.TestCase):
    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text("", chunk_size=800), [])
        self.assertEqual(chunk_text("   \n\n  ", chunk_size=800), [])

    def test_chunk_text_short_text(self):
        text = "This is a single short sentence that fits well within a single chunk."
        chunks = chunk_text(text, chunk_size=800, chunk_overlap=150)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_respects_overlap(self):
        para1 = "Transformers revolutionized natural language processing by replacing recurrent layers with self-attention. " * 5
        para2 = "Low-rank adaptation reduces the number of trainable parameters for downstream task fine-tuning. " * 5
        text = f"{para1}\n\n{para2}"
        
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
        self.assertGreaterEqual(len(chunks), 2)
        # Verify that consecutive chunks share text
        for i in range(len(chunks) - 1):
            tail = chunks[i][-40:].strip()
            # Some substring of the tail should be in the next chunk or close to start
            head = chunks[i + 1][:80].strip()
            overlap_found = any(word in head for word in tail.split() if len(word) > 5)
            self.assertTrue(overlap_found, f"Expected lexical overlap between chunk {i} and {i+1}")

    def test_chunk_text_no_infinite_loop_on_long_unbroken_text(self):
        long_string = "a" * 2500
        chunks = chunk_text(long_string, chunk_size=500, chunk_overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks), 10)

    def test_citation_pattern_robustness(self):
        pattern = r"\[([^,\[\]]+?\.pdf),\s*(?:p\.|page)\s*(\d+)\]"
        sample_text = (
            "According to [attention_is_all_you_need.pdf, p. 4] and "
            "[my custom report v2.pdf, page 12], attention is all you need."
        )
        matches = re.findall(pattern, sample_text)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0], ("attention_is_all_you_need.pdf", "4"))
        self.assertEqual(matches[1], ("my custom report v2.pdf", "12"))


if __name__ == "__main__":
    unittest.main()

