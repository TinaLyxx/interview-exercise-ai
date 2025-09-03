"""
Vector database implementation using FAISS for document storage and retrieval.
"""
import os
import pickle
import numpy as np
from typing import List, Tuple, Optional
import faiss
from src.models.schemas import DocumentChunk
from src.rag.embeddings import EmbeddingGenerator
from src.utils.config import config


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, embedding_dimension: int = None, index_path: str = None):
        """Initialize the FAISS vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings
            index_path: Path to save/load the FAISS index
        """
        self.embedding_generator = EmbeddingGenerator()
        self.embedding_dimension = embedding_dimension or self.embedding_generator.get_embedding_dimension()
        self.index_path = index_path or config.VECTOR_DB_PATH
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.documents: List[DocumentChunk] = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def add_documents(self, documents: List[DocumentChunk]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document chunks to add
        """
        if not documents:
            return
        
        # Generate embeddings for all documents
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_generator.encode_documents(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store document metadata
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar documents.
        
        Args:
            query: The search query
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_query(query)
        query_embedding = np.array([query_embedding])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Filter results by threshold and return documents with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, path: str = None) -> None:
        """Save the FAISS index and document metadata to disk.
        
        Args:
            path: Path to save the index (optional, uses default if not provided)
        """
        save_path = path or self.index_path
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.faiss")
        
        # Save document metadata
        with open(f"{save_path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Saved vector store to {save_path}")
    
    def load_index(self, path: str = None) -> bool:
        """Load the FAISS index and document metadata from disk.
        
        Args:
            path: Path to load the index from (optional, uses default if not provided)
            
        Returns:
            True if successfully loaded, False otherwise
        """
        load_path = path or self.index_path
        
        try:
            # Load FAISS index
            if os.path.exists(f"{load_path}.faiss"):
                self.index = faiss.read_index(f"{load_path}.faiss")
            else:
                return False
            
            # Load document metadata
            if os.path.exists(f"{load_path}.docs"):
                with open(f"{load_path}.docs", 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                return False
            
            print(f"Loaded vector store from {load_path}")
            return True
        
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False