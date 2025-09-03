"""
Main Knowledge Assistant service that orchestrates RAG pipeline and LLM generation.
"""
from typing import Optional
from src.models.schemas import TicketRequest, TicketResponse
from src.rag.retriever import DocumentRetriever
from src.rag.llm_client import LLMClient
from src.utils.config import config


class KnowledgeAssistant:
    """Main service for processing support tickets using RAG + LLM."""
    
    def __init__(self, docs_path: str = None, vector_store_path: str = None):
        """Initialize the Knowledge Assistant.
        
        Args:
            docs_path: Path to documentation directory
            vector_store_path: Path to vector store index
        """
        self.retriever = DocumentRetriever(docs_path, vector_store_path)
        self.llm_client = LLMClient()
        
        print("Knowledge Assistant initialized successfully")
    
    def resolve_ticket(self, ticket_request: TicketRequest) -> TicketResponse:
        """Process a support ticket and generate a structured response.
        
        Args:
            ticket_request: The incoming support ticket
            
        Returns:
            Structured response with answer, references, and action required
        """
        ticket_text = ticket_request.ticket_text
        
        # Step 1: Retrieve relevant context using RAG
        context = self.retriever.get_context_string(ticket_text)
        references = self.retriever.get_references(ticket_text)
        
        # Step 2: Generate response using LLM with retrieved context
        response = self.llm_client.generate_response(
            ticket_text=ticket_text,
            context=context,
            references=references
        )
        
        return response

    def get_system_stats(self) -> dict:
        """Get system statistics and health information.
        
        Returns:
            Dictionary with system statistics
        """
        retriever_stats = self.retriever.get_stats()
        
        return {
            "status": "healthy",
            "components": {
                "retriever": retriever_stats,
                "llm_model": self.llm_client.model,
                "config": {
                    "max_chunks": config.MAX_RELEVANT_CHUNKS,
                    "similarity_threshold": config.SIMILARITY_THRESHOLD
                }
            }
        }

    