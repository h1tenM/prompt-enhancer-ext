import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from doc_store import DocStore
from prompt_builder import build_meta_prompt

load_dotenv()

app = FastAPI(title="Prompt Enhancer API")

# 1. ALLOW THE EXTENSION TO TALK TO FASTAPI
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. PRO LEVEL!!! SETUP RAG
# I use free HuggingFace embeddings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
MANIFEST_PATH = os.path.join(DB_DIR, "docs.json")
SEED_DIR = os.path.join(BASE_DIR, "research_docs")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Chunks written before doc management existed have no `source` metadata, so the
# enabled-docs filter can never match them. Clear them out once, then reseed from
# research_docs/ so the corpus and the manifest agree.
if not os.path.exists(MANIFEST_PATH):
    legacy = vectorstore.get(include=[])
    if legacy["ids"]:
        vectorstore.delete(ids=legacy["ids"])

doc_store = DocStore(vectorstore, MANIFEST_PATH, seed_dir=SEED_DIR)

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables!")

# 3. SETUP LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=google_key,
    temperature=0.3,
)

# 4. METRICS
# These come out of the offline eval harness (backend/eval/) rather than being
# guessed at runtime. If the eval has never been run the dashboard stays hidden.
METRICS_PATH = os.path.join(BASE_DIR, "eval", "results", "metrics_config.json")


def load_eval_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def unwrap(content):
    """Gemini returns either a string or a list of content blocks."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return content


class PromptRequest(BaseModel):
    prompt: str
    persona: str = "Expert"
    reasoning: str = "Think step-by-step"
    format: str = "Markdown"


@app.post("/enhance")
async def enhance_prompt(request: PromptRequest):
    # A. Search for the best prompt engineering techniques in our RAG DB
    docs = doc_store.search(request.prompt, k=2)
    context = "\n".join(doc.page_content for doc in docs)

    final_query = build_meta_prompt(
        request.prompt,
        request.persona,
        request.reasoning,
        request.format,
        context,
    )

    # B. Generate, yes!!!
    response = llm.invoke(final_query)
    enhanced = unwrap(response.content)

    # C. Report real token usage for this call; quality numbers come from the eval.
    usage = getattr(response, "usage_metadata", None) or {}
    metrics = {
        "enhancement_tokens": usage.get("total_tokens")
        or (len(final_query) + len(enhanced)) // 4,
        "sources_used": [doc.metadata.get("name") for doc in docs],
    }

    eval_metrics = load_eval_metrics()
    if eval_metrics:
        metrics["eval"] = eval_metrics

    return {"enhanced_prompt": enhanced, "metrics": metrics}


# --- research document management -------------------------------------


class DocToggle(BaseModel):
    enabled: bool


@app.get("/sources")
async def list_docs():
    return {"docs": doc_store.list()}


@app.post("/sources")
async def add_doc(
    name: str = Form(...),
    text: str = Form(None),
    file: UploadFile = File(None),
):
    if file is not None:
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(400, "Only .txt files are supported.")
        raw = await file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be UTF-8 encoded text.")

    if not text or not text.strip():
        raise HTTPException(400, "Provide either pasted text or a .txt file.")

    try:
        return doc_store.add(name=name.strip() or "Untitled", text=text)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/sources/{doc_id}")
async def toggle_doc(doc_id: str, body: DocToggle):
    doc = doc_store.set_enabled(doc_id, body.enabled)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return {"id": doc_id, "enabled": doc["enabled"]}


@app.delete("/sources/{doc_id}")
async def delete_doc(doc_id: str):
    if not doc_store.delete(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"deleted": doc_id}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
