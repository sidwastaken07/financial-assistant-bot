"""LangChain-based Financial Assistant Bot.

Reads financial documents (.txt, .md, .pdf), splits them into chunks, embeds the
chunks into a FAISS vector store and answers questions about them with
Retrieval-Augmented Generation (RAG) using an OpenAI chat model.

Usage:
    python main.py                       # interactive chat over ./data
    python main.py -q "What was revenue?" # ask a single question
    python main.py --docs report.pdf     # use a specific file or folder
    python main.py --rebuild             # force re-indexing of documents
    python main.py --embeddings local    # sentence-transformers instead of OpenAI
    python main.py --no-llm              # only show the retrieved passages
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# langchain-community still provides the FAISS store and document loaders we
# need; silence its "being sunset" notice so it doesn't clutter the chat.
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_DOCS_PATH = "data"
DEFAULT_INDEX_DIR = "faiss_index"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MANIFEST_FILE = "manifest.json"

SYSTEM_PROMPT = """You are a careful financial assistant. Answer the user's question \
using ONLY the context excerpts from their financial documents below.

Rules:
- If the answer is not in the context, say you could not find it in the documents.
- Quote figures exactly as they appear (currency, units, periods).
- When you calculate something (growth rates, ratios, totals), show the calculation.
- Be concise. Do not give personalised investment advice.

Context:
{context}"""


# --------------------------------------------------------------------------- #
# Document loading and chunking
# --------------------------------------------------------------------------- #
def find_document_files(path: str | Path) -> list[Path]:
    """Return the supported document files at ``path`` (a file or a folder)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document path not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(path: str | Path) -> list[Document]:
    """Load every supported document at ``path`` into LangChain ``Document``s."""
    files = find_document_files(path)
    if not files:
        raise ValueError(
            f"No documents found in '{path}'. Add .txt, .md or .pdf files and try again."
        )

    documents: list[Document] = []
    for file in files:
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        else:
            loader = TextLoader(str(file), encoding="utf-8")
        documents.extend(loader.load())
    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --------------------------------------------------------------------------- #
# Embeddings and vector store
# --------------------------------------------------------------------------- #
def get_embeddings(provider: str = "openai") -> Embeddings:
    """Create the embedding model: ``openai`` (default) or ``local``."""
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        )
    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Local embeddings need extra packages. Install them with:\n"
                "    pip install langchain-huggingface sentence-transformers"
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
        )
    raise ValueError(f"Unknown embeddings provider '{provider}'. Use 'openai' or 'local'.")


def compute_fingerprint(files: list[Path], embeddings_id: str) -> str:
    """Fingerprint the document set + embedding model so stale indexes are rebuilt."""
    digest = hashlib.sha256(embeddings_id.encode())
    for file in files:
        digest.update(str(file.resolve()).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def embeddings_identifier(embeddings: Embeddings) -> str:
    model = getattr(embeddings, "model", None) or getattr(embeddings, "model_name", None)
    return f"{type(embeddings).__name__}:{model}"


def build_vectorstore(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build a vector store from zero chunks.")
    return FAISS.from_documents(chunks, embeddings)


def load_or_build_vectorstore(
    docs_path: str | Path,
    embeddings: Embeddings,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
    verbose: bool = True,
) -> FAISS:
    """Load a cached FAISS index, or (re)build it when documents have changed."""
    index_dir = Path(index_dir)
    manifest_path = index_dir / MANIFEST_FILE
    files = find_document_files(docs_path)
    fingerprint = compute_fingerprint(files, embeddings_identifier(embeddings))

    if not rebuild and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("fingerprint") == fingerprint:
            if verbose:
                print(f"Loading cached index from '{index_dir}'...")
            # The index was written by this program, so deserialising it is safe.
            return FAISS.load_local(
                str(index_dir), embeddings, allow_dangerous_deserialization=True
            )

    if verbose:
        print(f"Indexing {len(files)} document(s) from '{docs_path}'...")
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "files": [str(f) for f in files],
                "chunks": len(chunks),
            },
            indent=2,
        )
    )
    if verbose:
        print(f"Indexed {len(chunks)} chunk(s); saved to '{index_dir}'.")
    return vectorstore


# --------------------------------------------------------------------------- #
# RAG chain
# --------------------------------------------------------------------------- #
def get_chat_model(temperature: float = 0.0) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL), temperature=temperature
    )


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks as numbered, source-tagged context for the prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[{i}] (source: {describe_source(doc)})\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def describe_source(doc: Document) -> str:
    source = Path(doc.metadata.get("source", "unknown")).name
    page = doc.metadata.get("page")
    return f"{source}, page {page + 1}" if isinstance(page, int) else source


def build_rag_chain(llm: BaseChatModel | Runnable) -> Runnable:
    """Prompt -> LLM -> string. Expects ``{"question": ..., "context": ...}``."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()


class FinancialAssistant:
    """Ties a retriever and an (optional) LLM together to answer questions."""

    def __init__(
        self,
        vectorstore: FAISS,
        llm: BaseChatModel | Runnable | None = None,
        k: int = 4,
    ):
        self.retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        self.chain = build_rag_chain(llm) if llm is not None else None

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def ask(self, question: str) -> dict:
        """Answer ``question``. Returns ``{"answer", "sources", "documents"}``."""
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        docs = self.retrieve(question)
        sources = list(dict.fromkeys(describe_source(d) for d in docs))

        if self.chain is None:
            answer = "Retrieved passages (LLM disabled):\n\n" + format_docs(docs)
        elif not docs:
            answer = "I could not find any relevant information in the documents."
        else:
            answer = self.chain.invoke({"question": question, "context": format_docs(docs)})
        return {"answer": answer, "sources": sources, "documents": docs}


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def explain_error(exc: Exception) -> str | None:
    """Turn common OpenAI / network failures into actionable messages."""
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
    if isinstance(exc, openai.RateLimitError):
        if "insufficient_quota" in str(exc):
            return (
                "Your OpenAI account has run out of quota. Add credit, or run with "
                "'--embeddings local --no-llm' to search documents without OpenAI."
            )
        return "OpenAI rate limit hit. Wait a moment and try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach the OpenAI API. Check your internet connection."
    if isinstance(exc, OSError):  # includes requests' errors (e.g. tiktoken downloads)
        return f"Network error: {exc}"
    return None


def print_result(result: dict) -> None:
    print(f"\nAssistant: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {'; '.join(result['sources'])}")
    print()


def chat_loop(assistant: FinancialAssistant) -> None:
    print("\nFinancial Assistant ready. Ask a question about your documents.")
    print("Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            return
        try:
            print_result(assistant.ask(question))
        except Exception as exc:  # keep the chat alive on API hiccups
            print(f"\nError: {explain_error(exc) or exc}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about your financial documents (RAG with LangChain)."
    )
    parser.add_argument(
        "--docs",
        default=os.getenv("DOCS_PATH", DEFAULT_DOCS_PATH),
        help="File or folder of .txt/.md/.pdf documents (default: %(default)s)",
    )
    parser.add_argument(
        "--index-dir",
        default=os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR),
        help="Where to cache the FAISS index (default: %(default)s)",
    )
    parser.add_argument(
        "--embeddings",
        choices=["openai", "local"],
        default=os.getenv("EMBEDDINGS_PROVIDER", "openai"),
        help="Embedding backend (default: %(default)s)",
    )
    parser.add_argument("-k", type=int, default=4, help="Chunks to retrieve (default: 4)")
    parser.add_argument("-q", "--question", help="Ask one question and exit")
    parser.add_argument("--rebuild", action="store_true", help="Force re-indexing")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the chat model and just print the most relevant passages",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    needs_openai = args.embeddings == "openai" or not args.no_llm
    if needs_openai and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key,\n"
            "or run fully offline with: python main.py --embeddings local --no-llm",
            file=sys.stderr,
        )
        return 1

    try:
        embeddings = get_embeddings(args.embeddings)
        vectorstore = load_or_build_vectorstore(
            args.docs, embeddings, index_dir=args.index_dir, rebuild=args.rebuild
        )
        llm = None if args.no_llm else get_chat_model()
        assistant = FinancialAssistant(vectorstore, llm=llm, k=args.k)

        if args.question:
            print_result(assistant.ask(args.question))
        else:
            chat_loop(assistant)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        message = explain_error(exc)
        if message is None:
            raise
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
