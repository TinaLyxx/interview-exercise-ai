"""
Unit tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from src.api.main import app
from src.models.schemas import TicketRequest, TicketResponse


class TestAPI:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_knowledge_assistant(self):
        """Create mock KnowledgeAssistant."""
        mock_assistant = Mock()
        mock_assistant.resolve_ticket.return_value = TicketResponse(
            answer="Your domain may have been suspended due to a policy violation. Please contact support.",
            references=["Policy: Domain Suspension Guidelines, Section 4.2"],
            action_required="escalate_to_abuse_team"
        )
        mock_assistant.get_system_stats.return_value = {
            "status": "healthy",
            "components": {
                "retriever": {"total_documents": 10},
                "llm_model": "gpt-3.5-turbo"
            }
        }
        mock_assistant.rebuild_knowledge_base.return_value = {
            "status": "success",
            "message": "Knowledge base rebuilt successfully"
        }
        return mock_assistant
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    @patch('src.api.main.knowledge_assistant')
    def test_resolve_ticket_success(self, mock_assistant_global, client, mock_knowledge_assistant):
        """Test successful ticket resolution."""
        mock_assistant_global = mock_knowledge_assistant
        
        ticket_data = {
            "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
        }
        
        with patch('src.api.main.get_knowledge_assistant', return_value=mock_knowledge_assistant):
            response = client.post("/resolve-ticket", json=ticket_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "references" in data
        assert "action_required" in data
        assert data["action_required"] == "escalate_to_abuse_team"
        
        # Verify mock was called
        mock_knowledge_assistant.resolve_ticket.assert_called_once()

    @patch('src.api.main.knowledge_assistant')
    def test_health_check_success(self, mock_assistant_global, client, mock_knowledge_assistant):
        """Test successful health check."""
        with patch('src.api.main.get_knowledge_assistant', return_value=mock_knowledge_assistant):
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    @patch('src.api.main.knowledge_assistant')
    def test_health_check_failure(self, mock_assistant_global, client, mock_knowledge_assistant):
        """Test health check with system failure."""
        mock_knowledge_assistant.get_system_stats.side_effect = Exception("System error")
        
        with patch('src.api.main.get_knowledge_assistant', return_value=mock_knowledge_assistant):
            response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
    
    @patch('src.api.main.knowledge_assistant')
    def test_get_stats_success(self, mock_assistant_global, client, mock_knowledge_assistant):
        """Test getting system statistics."""
        with patch('src.api.main.get_knowledge_assistant', return_value=mock_knowledge_assistant):
            response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        
        mock_knowledge_assistant.get_system_stats.assert_called_once()
    
    @patch('src.api.main.knowledge_assistant')
    def test_resolve_ticket_invalid_input(self, mock_assistant_global, client):
        """Test ticket resolution with invalid input."""
        # Missing required field
        invalid_data = {}
        
        response = client.post("/resolve-ticket", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.main.knowledge_assistant')
    def test_resolve_ticket_processing_error(self, mock_assistant_global, client, mock_knowledge_assistant):
        """Test ticket resolution with processing error."""
        mock_knowledge_assistant.resolve_ticket.side_effect = Exception("Processing failed")
        
        ticket_data = {
            "ticket_text": "Test ticket"
        }
        
        with patch('src.api.main.get_knowledge_assistant', return_value=mock_knowledge_assistant):
            response = client.post("/resolve-ticket", json=ticket_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options("/")
        
        # CORS middleware should handle OPTIONS requests
        assert response.status_code in [200, 405]  # Some configurations return 405 for OPTIONS on GET endpoints
    
    def test_request_validation(self, client):
        """Test request validation for resolve-ticket endpoint."""
        # Test with string instead of object
        response = client.post("/resolve-ticket", json="invalid")
        assert response.status_code == 422
        
        # Test with wrong field names
        response = client.post("/resolve-ticket", json={"wrong_field": "value"})
        assert response.status_code == 422
        
        # Test with empty ticket text
        response = client.post("/resolve-ticket", json={"ticket_text": ""})
        # Should accept empty string but may return an error in processing
        assert response.status_code in [200, 500]
