# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # OpenRouter Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-is2asbkL8SkqGIv4HKxryrC1xeAXa7r70dTqnnb1wrBzpb55")
    OPENROUTER_BASE_URL = "https://api.gapgpt.app/v1"
    
    # Model Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    
    # App Configuration
    DATA_DIR = os.getenv("DATA_DIR", "./data")