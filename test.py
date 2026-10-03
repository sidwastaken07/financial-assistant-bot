"""Offline tests for the Financial Assistant Bot.

These run without an OpenAI key or network access: a small bag-of-words
embedding stands in for OpenAI embeddings, and a fake LLM echoes its prompt.

Run with:
    python test.py          # or: python -m pytest test.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import main

SAMPLE_DOCS = Path(__file__).parent / "data"


class KeywordEmbeddings(Embeddings):
    """Deterministic hashed bag-of-words embeddings (offline stand-in)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.model = f"keyword-{dim}"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def echo_llm():
    """A fake chat model that returns the full prompt it was given."""
    return RunnableLambda(lambda prompt: AIMessage(content=prompt.to_string()))


class LoadingTests(unittest.TestCase):
    def test_finds_sample_documents(self):
        files = main.find_document_files(SAMPLE_DOCS)
        self.assertTrue(any(f.name == "sample_annual_report.txt" for f in files))

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            main.find_document_files("does/not/exist")

    def test_unsupported_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "data.xlsx"
            bad.write_text("x")
            with self.assertRaises(ValueError):
                main.find_document_files(bad)

    def test_empty_folder_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                main.load_documents(tmp)

    def test_loads_pdf(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with pdf.open("wb") as fh:
                writer.write(fh)
            docs = main.load_documents(pdf)
            self.assertEqual(len(docs), 1)
            self.assertEqual(main.describe_source(docs[0]), "blank.pdf, page 1")

    def test_split_respects_chunk_size(self):
        docs = main.load_documents(SAMPLE_DOCS)
        chunks = main.split_documents(docs, chunk_size=400, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.page_content) <= 400 for c in chunks))
        self.assertTrue(all("source" in c.metadata for c in chunks))


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.tmp.name) / "index"
        self.embeddings = KeywordEmbeddings()

    def tearDown(self):
        self.tmp.cleanup()

    def test_builds_then_reuses_cached_index(self):
        main.load_or_build_vectorstore(
            SAMPLE_DOCS, self.embeddings, self.index_dir, verbose=False
        )
        manifest = json.loads((self.index_dir / main.MANIFEST_FILE).read_text())
        self.assertGreater(manifest["chunks"], 0)

        mtime = (self.index_dir / "index.faiss").stat().st_mtime_ns
        main.load_or_build_vectorstore(
            SAMPLE_DOCS, self.embeddings, self.index_dir, verbose=False
        )
        self.assertEqual(mtime, (self.index_dir / "index.faiss").stat().st_mtime_ns)

    def test_rebuilds_when_documents_change(self):
        docs_dir = Path(self.tmp.name) / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("Revenue was $10 million.")
        main.load_or_build_vectorstore(docs_dir, self.embeddings, self.index_dir, verbose=False)
        first = json.loads((self.index_dir / main.MANIFEST_FILE).read_text())

        (docs_dir / "b.txt").write_text("Net income was $2 million.")
        store = main.load_or_build_vectorstore(
            docs_dir, self.embeddings, self.index_dir, verbose=False
        )
        second = json.loads((self.index_dir / main.MANIFEST_FILE).read_text())
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(len(second["files"]), 2)
        top = store.similarity_search("net income", k=1)[0]
        self.assertIn("Net income", top.page_content)


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = main.split_documents(main.load_documents(SAMPLE_DOCS))
        cls.vectorstore = main.build_vectorstore(chunks, KeywordEmbeddings())

    def test_retrieves_relevant_chunk(self):
        assistant = main.FinancialAssistant(self.vectorstore, k=2)
        docs = assistant.retrieve("What was free cash flow and capital expenditures?")
        self.assertTrue(any("Free cash flow" in d.page_content for d in docs))

    def test_prompt_contains_question_and_context(self):
        assistant = main.FinancialAssistant(self.vectorstore, llm=echo_llm(), k=2)
        result = assistant.ask("What are the main risk factors?")
        self.assertIn("What are the main risk factors?", result["answer"])
        self.assertIn("RISK FACTORS", result["answer"])
        self.assertIn("using ONLY the context", result["answer"])
        self.assertEqual(result["sources"], ["sample_annual_report.txt"])
        self.assertEqual(len(result["documents"]), 2)

    def test_no_llm_mode_returns_passages(self):
        assistant = main.FinancialAssistant(self.vectorstore, llm=None, k=1)
        result = assistant.ask("dividends paid per share")
        self.assertTrue(result["answer"].startswith("Retrieved passages"))
        self.assertIn("[1] (source: sample_annual_report.txt)", result["answer"])

    def test_empty_question_rejected(self):
        assistant = main.FinancialAssistant(self.vectorstore, llm=echo_llm())
        with self.assertRaises(ValueError):
            assistant.ask("   ")


class CliTests(unittest.TestCase):
    def test_missing_api_key_exits_cleanly(self):
        import contextlib
        import io
        import os
        from unittest import mock

        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            main, "load_dotenv"
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(main.main(["-q", "hello"]), 1)
        self.assertIn("OPENAI_API_KEY is not set", stderr.getvalue())

    def test_parse_args_defaults(self):
        args = main.parse_args([])
        self.assertEqual(args.k, 4)
        self.assertFalse(args.rebuild)
        self.assertFalse(args.no_llm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
