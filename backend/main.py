import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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
DB_DIR = os.path.join(os.getcwd(), "db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# For the Hackathon, we'll initialize a local DB in a folder called 'db'
if not os.path.exists(DB_DIR):
    # Create some initial "Research" data on how to write good prompts
    initial_data = [
        "Use delimiters like triple backticks to clearly separate parts of the prompt.",
        "Assign a persona: Tell the AI 'You are a Senior Software Engineer' or 'You are a Creative Writer'.",
        "Specify the output format: Ask for JSON, Markdown, or a bulleted list.",
        "Few-shot prompting: Provide 1-2 examples of the desired input and output.",
        "Ask the LLM to 'think step-by-step' to improve reasoning quality."
    ]
    vectorstore = Chroma.from_texts(initial_data, embeddings, persist_directory=DB_DIR)
else:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

google_key = os.getenv('GOOGLE_API_KEY')
if not google_key:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables!")
# 3. SETUP LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    google_api_key=google_key,
    temperature=0.3
)

class PromptRequest(BaseModel):
    prompt: str
    persona: str = "Expert"
    reasoning: str = "Think step-by-step"
    format: str = "Markdown"

@app.post("/enhance")
async def enhance_prompt(request: PromptRequest):
    # A. Search for the best prompt engineering techniques in our RAG DB
    docs = vectorstore.similarity_search(request.prompt, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    # B. Update the Meta Prompt to use the user's specific choices
    template = """
    You are a Master Prompt Engineer. 
    
    USER CONFIGURATION:
    - ASSIGNED PERSONA: {persona}
    - REASONING MODE: {reasoning}
    - REQUIRED FORMAT: {format}
    
    ENGINEERING TECHNIQUES TO APPLY:
    {context}

    TASK:
    Restructure the following user request into a high-quality prompt that adopts the 
    assigned persona and enforces the reasoning and format rules above.
    
    USER REQUEST: "{user_prompt}"

    Return ONLY the final optimized prompt.
    """
    
    final_query = template.format(
        persona=request.persona,
        reasoning=request.reasoning,
        format=request.format,
        context=context, 
        user_prompt=request.prompt
    )
    
    # C. Generate, yes!!!
    response = llm.invoke(final_query)
    
    # D. Calculatee Metric
    content_text = response.content[0]['text'] if isinstance(response.content, list) else response.content

    estimated_tokens_saved = (len(content_text) // 4) * 2
    # print(content_text)
    
    return {
        "enhanced_prompt": response.content,
        "metrics": {
            "tokens_saved": estimated_tokens_saved,
            "prompts_saved": 2,
            "time_saved_minutes": 5
        }
    }
    


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
    