import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize Embeddings (Same as main.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_db():
    # 2. Load documents from the research folder
    # This handles both .txt and .pdf files
    print("Reading research documents...")
    loader = DirectoryLoader('./research_docs', glob="./*.txt", loader_cls=TextLoader)
    # Uncomment below to process pdf files
    # loader = DirectoryLoader('./research_docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    
    docs = loader.load()

    # 3. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 4. Create and save the database
    print(f"Creating vector database with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./db"
    )
    print("Database saved to ./db folder!")

if __name__ == "__main__":
    build_vector_db()