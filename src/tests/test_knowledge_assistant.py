"""
Unit tests for KnowledgeAssistant.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.rag.knowledge_assistant import KnowledgeAssistant
from src.models.schemas import TicketRequest, TicketResponse


class TestKnowledgeAssistant:
    """Test cases for KnowledgeAssistant."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create mock DocumentRetriever."""
        mock_retriever = Mock()
        mock_retriever.get_context_string.return_value = "Test context from documentation"
        mock_retriever.get_references.return_value = ["Policy: Test Policy, Section 1"]
        mock_retriever.get_stats.return_value = {
            "total_documents": 10,
            "index_size": 10,
            "embedding_dimension": 384
        }
        mock_retriever.rebuild_index.return_value = None
        return mock_retriever
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLMClient."""
        mock_llm = Mock()
        mock_llm.model = "gpt-3.5-turbo"
        mock_llm.generate_response.return_value = TicketResponse(
            answer="Test response from LLM",
            references=["Policy: Test Policy, Section 1"],
            action_required="no_action_required"
        )
        return mock_llm
    
    @pytest.fixture
    def sample_ticket_request(self):
        """Create sample ticket request."""
        return TicketRequest(
            ticket_text="My domain was suspended and I need help reactivating it."
        )
    
    def test_init_default_paths(self, mock_retriever, mock_llm_client):
        """Test initialization with default paths."""
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            
            assert assistant.retriever == mock_retriever
            assert assistant.llm_client == mock_llm_client
    
    def test_init_custom_paths(self, mock_retriever, mock_llm_client):
        """Test initialization with custom paths."""
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            docs_path = "/custom/docs"
            vector_store_path = "/custom/vector"
            
            assistant = KnowledgeAssistant(docs_path, vector_store_path)
            
            # Verify DocumentRetriever was called with custom paths
            from src.rag.knowledge_assistant import DocumentRetriever
            DocumentRetriever.assert_called_once_with(docs_path, vector_store_path)
    
    def test_resolve_ticket_success(self, mock_retriever, mock_llm_client, sample_ticket_request):
        """Test successful ticket resolution."""
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            response = assistant.resolve_ticket(sample_ticket_request)
            
            # Verify retriever methods were called
            mock_retriever.get_context_string.assert_called_once_with(sample_ticket_request.ticket_text)
            mock_retriever.get_references.assert_called_once_with(sample_ticket_request.ticket_text)
            
            # Verify LLM client was called with correct parameters
            mock_llm_client.generate_response.assert_called_once_with(
                ticket_text=sample_ticket_request.ticket_text,
                context="Test context from documentation",
                references=["Policy: Test Policy, Section 1"]
            )
            
            # Verify response
            assert isinstance(response, TicketResponse)
            assert response.answer == "Test response from LLM"
            assert response.references == ["Policy: Test Policy, Section 1"]
            assert response.action_required == "no_action_required"
    
    def test_resolve_ticket_with_empty_context(self, sample_ticket_request):
        """Test ticket resolution when no context is found."""
        mock_retriever = Mock()
        mock_retriever.get_context_string.return_value = ""
        mock_retriever.get_references.return_value = []
        
        mock_llm_client = Mock()
        mock_llm_client.generate_response.return_value = TicketResponse(
            answer="No relevant documentation found",
            references=[],
            action_required="escalate_to_technical_team"
        )
        
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            response = assistant.resolve_ticket(sample_ticket_request)
            
            # Verify LLM was called with empty context
            mock_llm_client.generate_response.assert_called_once_with(
                ticket_text=sample_ticket_request.ticket_text,
                context="",
                references=[]
            )
    
    def test_get_system_stats_success(self, mock_retriever, mock_llm_client):
        """Test getting system statistics."""
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client), \
             patch('src.rag.knowledge_assistant.config') as mock_config:
            
            mock_config.MAX_RELEVANT_CHUNKS = 5
            mock_config.SIMILARITY_THRESHOLD = 0.7
            
            assistant = KnowledgeAssistant()
            stats = assistant.get_system_stats()
            
            # Verify retriever stats were called
            mock_retriever.get_stats.assert_called_once()
            
            # Verify response structure
            assert stats["status"] == "healthy"
            assert "components" in stats
            assert "retriever" in stats["components"]
            assert "llm_model" in stats["components"]
            assert "config" in stats["components"]
            
            # Verify config values
            assert stats["components"]["config"]["max_chunks"] == 5
            assert stats["components"]["config"]["similarity_threshold"] == 0.7
            assert stats["components"]["llm_model"] == "gpt-3.5-turbo"
    
    def test_rebuild_knowledge_base_success(self, mock_retriever, mock_llm_client):
        """Test successful knowledge base rebuild."""
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            result = assistant.rebuild_knowledge_base()
            
            # Verify retriever methods were called
            mock_retriever.rebuild_index.assert_called_once()
            mock_retriever.get_stats.assert_called_once()
            
            # Verify response
            assert result["status"] == "success"
            assert "message" in result
            assert "stats" in result
            assert result["message"] == "Knowledge base rebuilt successfully"
    
    def test_rebuild_knowledge_base_failure(self, mock_retriever, mock_llm_client):
        """Test knowledge base rebuild with failure."""
        mock_retriever.rebuild_index.side_effect = Exception("Rebuild failed")
        
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            result = assistant.rebuild_knowledge_base()
            
            # Verify retriever was called
            mock_retriever.rebuild_index.assert_called_once()
            
            # Verify error response
            assert result["status"] == "error"
            assert "message" in result
            assert "Failed to rebuild knowledge base" in result["message"]
            assert "Rebuild failed" in result["message"]
    
    def test_resolve_ticket_with_retriever_exception(self, sample_ticket_request):
        """Test ticket resolution when retriever raises exception."""
        mock_retriever = Mock()
        mock_retriever.get_context_string.side_effect = Exception("Retriever error")
        
        mock_llm_client = Mock()
        
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            
            # Should propagate the exception
            with pytest.raises(Exception, match="Retriever error"):
                assistant.resolve_ticket(sample_ticket_request)
    
    def test_resolve_ticket_with_llm_exception(self, sample_ticket_request):
        """Test ticket resolution when LLM client raises exception."""
        mock_retriever = Mock()
        mock_retriever.get_context_string.return_value = "Test context"
        mock_retriever.get_references.return_value = ["Test ref"]
        
        mock_llm_client = Mock()
        mock_llm_client.generate_response.side_effect = Exception("LLM error")
        
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            
            # Should propagate the exception
            with pytest.raises(Exception, match="LLM error"):
                assistant.resolve_ticket(sample_ticket_request)
    
    def test_get_system_stats_with_retriever_exception(self):
        """Test system stats when retriever raises exception."""
        mock_retriever = Mock()
        mock_retriever.get_stats.side_effect = Exception("Stats error")
        
        mock_llm_client = Mock()
        mock_llm_client.model = "gpt-3.5-turbo"
        
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client):
            
            assistant = KnowledgeAssistant()
            
            # Should propagate the exception
            with pytest.raises(Exception, match="Stats error"):
                assistant.get_system_stats()
    
    def test_initialization_print_statement(self, mock_retriever, mock_llm_client):
        """Test that initialization prints success message."""
        with patch('src.rag.knowledge_assistant.DocumentRetriever', return_value=mock_retriever), \
             patch('src.rag.knowledge_assistant.LLMClient', return_value=mock_llm_client), \
             patch('builtins.print') as mock_print:
            
            KnowledgeAssistant()
            
            # Verify print was called with success message
            mock_print.assert_called_once_with("Knowledge Assistant initialized successfully")
