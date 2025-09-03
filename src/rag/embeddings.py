"""
Embedding generation utilities for the RAG system.
"""
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.utils.config import config


class EmbeddingGenerator:
    """Generates embeddings for documents and queries."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
    
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of document texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.
        
        Args:
            query: Query text to encode
            
        Returns:
            Numpy array representing the query embedding
        """
        embedding = self.model.encode([query])
        return embedding[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.
        
        Returns:
            The embedding dimension
        """
        # Get dimension by encoding a sample text
        sample_embedding = self.encode_query("sample text")
        return len(sample_embedding)
