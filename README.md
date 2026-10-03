# finance-bot
# LangChain-based Financial Assistant Bot

This project is a simple financial assistant chatbot that reads financial documents (e.g., text or PDF) and answers user queries using Retrieval-Augmented Generation (RAG) powered by LangChain and OpenAI.

---

## Features

- Reads and processes financial documents (`.txt`, `.md`, `.pdf`)
- Converts document chunks into vector embeddings
- Retrieves relevant information to answer user queries, citing the source file/page
- Uses OpenAI API and FAISS for document search
- Caches the FAISS index on disk and rebuilds it automatically when documents change
- Includes a simple command-line chat interface and an offline test suite

---

## File Structure
```
├── main.py                 # Main script (entry point)
├── test.py                 # Offline tests (no API key needed)
├── requirements.txt        # Python dependencies
├── data/                   # Put your financial documents here
│   └── sample_annual_report.txt   # Fictional sample report
├── .env                    # Contains your API keys (create from .env.example)
├── .env.example            # Sample env file for setup
├── .gitignore              # Ignore venv, .env, etc.
├── faiss_index/            # Cached vector index (generated, ignored)
└── venv/                   # Python virtual environment (ignored)
```

---

## Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/sidwastaken07/financial-assistant-bot.git
   cd financial-assistant-bot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your OpenAI API key**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   OPENAI_API_KEY=your-key-here
   ```

5. **Run the bot**
   ```bash
   python main.py
   ```

---

## Usage

Drop your documents into `data/` (or point `--docs` at any file or folder) and start chatting:

```text
$ python main.py
Indexing 1 document(s) from 'data'...
Indexed 5 chunk(s); saved to 'faiss_index'.

Financial Assistant ready. Ask a question about your documents.
Type 'exit' or 'quit' to leave.

You: What was net income and how much did it grow?
Assistant: Net income was $41.7 million in FY2024, up from $31.2 million in FY2023 ...
Sources: sample_annual_report.txt
```

| Command | What it does |
| --- | --- |
| `python main.py` | Interactive chat over the documents in `data/` |
| `python main.py -q "What was free cash flow?"` | Ask a single question and exit |
| `python main.py --docs reports/10-K.pdf` | Use a specific file or folder |
| `python main.py --rebuild` | Force the vector index to be rebuilt |
| `python main.py -k 6` | Retrieve more chunks per question (default 4) |
| `python main.py --embeddings local` | Use local sentence-transformers embeddings |
| `python main.py --no-llm` | Skip the chat model; print the most relevant passages |

Settings such as the chat model (`OPENAI_MODEL`, default `gpt-4o-mini`) and embedding model can be overridden in `.env` — see `.env.example`.

---

## Testing

The tests use a fake embedding model and a fake LLM, so they run without an API key or network access:

```bash
python test.py
# or
python -m pytest test.py
```

---

## NOTES
- If you encounter OpenAI quota errors, consider switching to local models (e.g., sentence-transformers):
  ```bash
  pip install langchain-huggingface sentence-transformers
  python main.py --embeddings local --no-llm   # fully offline document search
  ```
  `--embeddings local` on its own keeps OpenAI for answering but embeds documents locally.
- The FAISS index is tied to the embedding model; switching models triggers an automatic rebuild.
- The project is modular and can be extended to support PDF upload, Streamlit interface, etc. —
  `FinancialAssistant` in `main.py` can be reused directly from any UI.
- Answers are generated from your documents only and are not financial advice.
