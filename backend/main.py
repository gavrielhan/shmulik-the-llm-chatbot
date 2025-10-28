from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shmulik.rag_system.langgraph_rag import LangGraphRAGSystem
from src.shmulik.vectorstore.chroma_store import create_chroma_store
from src.shmulik.document_processing.pdf_processor import PDFProcessor
from config.config import get_settings

app = FastAPI(title="Shmulik Chat API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_system = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    response: str

@app.on_event("startup")
async def startup_event():
    global rag_system
    print("Initializing Shmulik RAG system...")
    
    try:
        # Initialize RAG system
        settings = get_settings()
        
        # Process PDF documents
        pdf_processor = PDFProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        documents = pdf_processor.process_pdf(settings.pdf_path)
        
        # Initialize vector store
        vectorstore = create_chroma_store(
            persist_directory=settings.vector_store_path,
            embedding_model=settings.embedding_model
        )
        
        # Try to load existing vectorstore first
        existing_store = vectorstore.load_existing_vectorstore()
        if existing_store is None:
            # Create new vectorstore
            vectorstore.create_vectorstore(documents)
        else:
            # Use existing vectorstore
            vectorstore = existing_store
        
        rag_system = LangGraphRAGSystem(
            vectorstore_retriever=vectorstore.get_retriever(),
            llm_model=settings.llm_model,
            llm_api_key=settings.openai_api_key,
            llm_base_url=settings.openai_api_base
        )
        print("✅ Shmulik RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            user_query=request.message,
            conversation_history=[]
        )
        
        if result["success"]:
            return ChatResponse(success=True, response=result["response"])
        else:
            return ChatResponse(success=False, response="Sorry, I encountered an error.")
            
    except Exception as e:
        return ChatResponse(success=False, response=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_system": rag_system is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
