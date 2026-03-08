# ✨ Prompt Enhancer (HackBU 2026)

An intelligent Chrome Extension that uses **RAG (Retrieval-Augmented Generation)** to transform basic user requests into expert-level prompts.

## 🚀 The Problem

Non-technical users often struggle to get high-quality results from AI because they don't know "Prompt Engineering." This leads to frustration and wasted time/tokens.

## 🛠️ Tech Stack

- **Frontend:** Chrome Extension API (JS/HTML/CSS)
- **Backend:** FastAPI (Python)
- **AI Models:** Gemini 1.5 Flash (via LangChain)
- **Vector DB:** ChromaDB (for storing prompt engineering research)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`

## 📊 Quantifiable Impact

Our extension tracks:

- **Tokens Saved:** By getting it right the first time.
- **Prompts Avoided:** No more "try again" cycles.
- **Time Saved:** Estimated minutes reclaimed per session.

## 🛠️ Local Setup (Backend)

1. **Clone the repo:**

   ```bash
    git clone [https://github.com/your-username/prompt-enhancer-ext.git](https://github.com/your-username/prompt-enhancer-ext.git)
    cd prompt-enhancer-ext/backend

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

    create a .env file in /backend dir and insert
    GOOGLE_API_KEY=your_gemini_api_key_here

    then run the server!

    python main.py
   ```
