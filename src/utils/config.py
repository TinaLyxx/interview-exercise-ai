"""
Configuration management for the Knowledge Assistant.
"""
import os
from typing import Optional



class Config:
    """Configuration class for the Knowledge Assistant."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    
    # Document Configuration
    DOCS_PATH: str = os.getenv("DOCS_PATH", "./data/docs")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # RAG Configuration
    MAX_RELEVANT_CHUNKS: int = int(os.getenv("MAX_RELEVANT_CHUNKS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    

# Global config instance
config = Config()
