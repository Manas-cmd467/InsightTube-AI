# InsightTube AI

InsightTube AI is a Streamlit application that lets you chat with any YouTube video using transcript-based Retrieval-Augmented Generation (RAG).
It extracts a video transcript, builds a vector index with FAISS, retrieves relevant context for each question, and answers using Google Gemini.

---

## Features

- ✅ Chat with YouTube videos using transcript context
- ✅ RAG pipeline with semantic retrieval (FAISS + sentence-transformers)
- ✅ Gemini-powered responses constrained to transcript content
- ✅ Streamlit chat UI with session-based conversation history
- ✅ Secure API key loading from `.env` or sidebar input

---

## Tech Stack

- **Frontend/UI:** Streamlit
- **LLM Orchestration:** LangChain
- **LLM:** Google Gemini (`gemini-1.5-flash`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS
- **Transcript Source:** `youtube-transcript-api`

---

## Project Structure

```text
InsightTube-AI/
├── src/
│   └── app.py              # Main Streamlit app
├── main.py                 # Simple project entry script
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # pip-compatible dependency list
└── README.md
```

---

## Prerequisites

- Python 3.10+
- A Google Gemini API key

---

## Setup

### Option A: Using `uv` (recommended)

```bash
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
uv pip install -r requirements.txt
```

### Option B: Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the repository root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

You can also paste the key directly in the app sidebar.

---

## Run the App

From the repository root:

```bash
streamlit run src/app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## How to Use

1. Paste a YouTube URL in the sidebar.
2. Provide your Gemini API key (or use `.env`).
3. Click **Process Video**.
4. Ask questions in the chat box.

The assistant answers based only on transcript context. If the transcript does not contain an answer, it will say so.

---

## Troubleshooting

- **“Could not extract video ID”**
  - Verify the YouTube link format and retry with a standard watch/share URL.

- **Transcript fetch errors**
  - Some videos may not have transcripts enabled, may be private, or region-restricted.

- **API key/authentication errors**
  - Confirm `GOOGLE_API_KEY` is valid and has Gemini access enabled.

- **Dependency/import errors**
  - Recreate the virtual environment and reinstall:
    ```bash
    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

---

## Notes

- This project is transcript-driven; answer quality depends on transcript availability and quality.
- The retriever uses MMR for diverse context selection.


