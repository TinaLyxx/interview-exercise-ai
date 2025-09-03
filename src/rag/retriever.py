"""
Document retrieval system that combines vector search with relevance filtering.
"""
from typing import List, Tuple
from src.models.schemas import DocumentChunk
from src.rag.vector_store import FAISSVectorStore
from src.rag.document_processor import DocumentProcessor
from src.utils.config import config

class DocumentRetriever:
    """High-level document retrieval system."""
    
    def __init__(self, docs_path: str = None, vector_store_path: str = None):
        """Initialize the document retriever.
        
        Args:
            docs_path: Path to the documents directory
            vector_store_path: Path to the vector store index
        """
        self.docs_path = docs_path or config.DOCS_PATH
        self.vector_store_path = vector_store_path or config.VECTOR_DB_PATH
        
        self.document_processor = DocumentProcessor(self.docs_path)
        self.vector_store = FAISSVectorStore(index_path=self.vector_store_path)
        
        self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """Initialize the vector store by loading existing index or creating new one."""
        # Try to load existing index
        if self.vector_store.load_index():
            print("Loaded existing vector store")
        else:
            print("Creating new vector store...")
            self._build_vector_store()
    
    def _build_vector_store(self) -> None:
        """Build the vector store from documents."""
        # Load and process documents
        documents = self.document_processor.load_documents()
        
        if not documents:
            raise ValueError("No documents found to build vector store")
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        
        # Save the index
        self.vector_store.save_index()
        
        print(f"Built vector store with {len(documents)} document chunks")
    
    def retrieve_relevant_context(self, query: str, max_chunks: int = None, 
                                 threshold: float = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant document chunks for a query.
        
        Args:
            query: The user query
            max_chunks: Maximum number of chunks to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (document_chunk, similarity_score) tuples
        """
        max_chunks = max_chunks or config.MAX_RELEVANT_CHUNKS
        threshold = threshold or config.SIMILARITY_THRESHOLD
        
        # Search for relevant documents
        results = self.vector_store.search(query, k=max_chunks, threshold=threshold)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_context_string(self, query: str, max_chunks: int = None) -> str:
        """Get formatted context string for LLM prompt.
        
        Args:
            query: The user query
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted context string
        """
        relevant_docs = self.retrieve_relevant_context(query, max_chunks)
        
        if not relevant_docs:
            return "No relevant documentation found."
        
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            context_parts.append(f"Source {i}: {doc.source}\n{doc.content}\n")
        
        return "\n".join(context_parts)
    
    
    
    