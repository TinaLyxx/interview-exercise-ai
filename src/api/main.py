"""
FastAPI main application for the Knowledge Assistant.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from src.models.schemas import TicketRequest, TicketResponse
from src.rag.knowledge_assistant import KnowledgeAssistant
from src.utils.config import config



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Knowledge Assistant instance
knowledge_assistant: KnowledgeAssistant = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global knowledge_assistant
    
    # Startup
    logger.info("Starting Knowledge Assistant...")
    try:
        # Validate configuration
        config.validate()
        
        # Initialize Knowledge Assistant
        knowledge_assistant = KnowledgeAssistant()
        logger.info("Knowledge Assistant initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Knowledge Assistant: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Knowledge Assistant...")


# Create FastAPI app
app = FastAPI(
    title="Knowledge Assistant API",
    description="AI-powered support ticket resolution system using RAG and LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_knowledge_assistant() -> KnowledgeAssistant:
    """Dependency to get the Knowledge Assistant instance."""
    if knowledge_assistant is None:
        raise HTTPException(status_code=500, detail="Knowledge Assistant not initialized")
    return knowledge_assistant


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Knowledge Assistant API",
        "version": "1.0.0",
        "description": "AI-powered support ticket resolution system",
        "endpoints": {
            "resolve_ticket": "POST /resolve-ticket",
            "rebuild_knowledge_base": "POST /rebuild-knowledge-base",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


@app.post("/resolve-ticket", response_model=TicketResponse)
async def resolve_ticket(
    request: TicketRequest,
    assistant: KnowledgeAssistant = Depends(get_knowledge_assistant)
) -> TicketResponse:
    """
    Resolve a customer support ticket using RAG and LLM.
    
    Args:
        request: Ticket request containing the customer's message
        
    Returns:
        Structured response with answer, references, and action required
    """
    try:
        logger.info(f"Processing ticket: {request.ticket_text[:100]}...")
        
        # Process the ticket
        response = assistant.resolve_ticket(request)
        
        logger.info(f"Generated response with action: {response.action_required}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing ticket: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process support ticket: {str(e)}"
        )


@app.get("/health")
async def health_check(
    assistant: KnowledgeAssistant = Depends(get_knowledge_assistant)
):
    """Health check endpoint."""
    try:
        stats = assistant.get_system_stats()
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp in production
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/stats")
async def get_stats(
    assistant: KnowledgeAssistant = Depends(get_knowledge_assistant)
):
    """Get system statistics."""
    try:
        stats = assistant.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system statistics: {str(e)}"
        )


@app.post("/rebuild-knowledge-base")
async def rebuild_knowledge_base(
    assistant: KnowledgeAssistant = Depends(get_knowledge_assistant)
):
    """Rebuild the knowledge base from source documents."""
    try:
        logger.info("Starting knowledge base rebuild...")
        result = assistant.rebuild_knowledge_base()
        logger.info(f"Knowledge base rebuild completed: {result['status']}")
        return result
    except Exception as e:
        logger.error(f"Failed to rebuild knowledge base: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild knowledge base: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )