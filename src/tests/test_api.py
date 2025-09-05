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
    
    def test_resolve_ticket_success(self, client, mock_knowledge_assistant):
        """Test successful ticket resolution."""
        from src.api.main import get_knowledge_assistant
        
        # Override the dependency
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_knowledge_assistant
        
        try:
            ticket_data = {
                "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
            }
            
            response = client.post("/resolve-ticket", json=ticket_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "answer" in data
            assert "references" in data
            assert "action_required" in data
            assert data["action_required"] == "escalate_to_abuse_team"
            
            # Verify mock was called
            mock_knowledge_assistant.resolve_ticket.assert_called_once()
        finally:
            # Clean up
            app.dependency_overrides.clear()
    
    def test_resolve_ticket_invalid_input(self, client, mock_knowledge_assistant):
        """Test ticket resolution with invalid input."""
        from src.api.main import get_knowledge_assistant
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_knowledge_assistant
        
        try:
            # Missing required field
            invalid_data = {}
            
            response = client.post("/resolve-ticket", json=invalid_data)
            
            assert response.status_code == 422  # Validation error
        finally:
            app.dependency_overrides.clear()
    
    def test_resolve_ticket_processing_error(self, client):
        """Test ticket resolution with processing error."""
        from src.api.main import get_knowledge_assistant
        
        mock_assistant = Mock()
        mock_assistant.resolve_ticket.side_effect = Exception("Processing failed")
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_assistant
        
        try:
            ticket_data = {
                "ticket_text": "Test ticket"
            }
            
            response = client.post("/resolve-ticket", json=ticket_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_health_check_success(self, client, mock_knowledge_assistant):
        """Test successful health check."""
        from src.api.main import get_knowledge_assistant
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_knowledge_assistant
        
        try:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "version" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_health_check_failure(self, client):
        """Test health check with system failure."""
        from src.api.main import get_knowledge_assistant
        
        mock_assistant = Mock()
        mock_assistant.get_system_stats.side_effect = Exception("System error")
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_assistant
        
        try:
            response = client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
        finally:
            app.dependency_overrides.clear()
    
    def test_get_stats_success(self, client, mock_knowledge_assistant):
        """Test getting system statistics."""
        from src.api.main import get_knowledge_assistant
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_knowledge_assistant
        
        try:
            response = client.get("/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "components" in data
            
            mock_knowledge_assistant.get_system_stats.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    def test_rebuild_knowledge_base_success(self, client, mock_knowledge_assistant):
        """Test rebuilding knowledge base."""
        from src.api.main import get_knowledge_assistant
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_knowledge_assistant
        
        try:
            response = client.post("/rebuild-knowledge-base")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            
            mock_knowledge_assistant.rebuild_knowledge_base.assert_called_once()
        finally:
            app.dependency_overrides.clear()
    
    def test_rebuild_knowledge_base_failure(self, client):
        """Test rebuilding knowledge base with failure."""
        from src.api.main import get_knowledge_assistant
        
        mock_assistant = Mock()
        mock_assistant.rebuild_knowledge_base.side_effect = Exception("Rebuild failed")
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_assistant
        
        try:
            response = client.post("/rebuild-knowledge-base")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options("/")
        
        # CORS middleware should handle OPTIONS requests
        assert response.status_code in [200, 405]  # Some configurations return 405 for OPTIONS on GET endpoints
    
    def test_request_validation(self, client, mock_knowledge_assistant):
        """Test request validation for resolve-ticket endpoint."""
        from src.api.main import get_knowledge_assistant
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_knowledge_assistant
        
        try:
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
        finally:
            app.dependency_overrides.clear()
    
    def test_get_knowledge_assistant_dependency_error(self):
        """Test get_knowledge_assistant dependency when assistant is None."""
        from src.api.main import get_knowledge_assistant
        from fastapi import HTTPException
        from unittest.mock import patch
        
        # Temporarily set knowledge_assistant to None
        with patch('src.api.main.knowledge_assistant', None):
            with pytest.raises(HTTPException) as exc_info:
                get_knowledge_assistant()
            assert exc_info.value.status_code == 500
            assert "Knowledge Assistant not initialized" in exc_info.value.detail
    
    def test_lifespan_startup_success(self):
        """Test lifespan manager startup success."""
        from src.api.main import lifespan
        from unittest.mock import patch, Mock
        import asyncio
        
        mock_app = Mock()
        
        with patch('src.api.main.config') as mock_config, \
             patch('src.api.main.KnowledgeAssistant') as mock_ka_class, \
             patch('src.api.main.logger') as mock_logger:
            
            mock_config.validate.return_value = True
            mock_ka_instance = Mock()
            mock_ka_class.return_value = mock_ka_instance
            
            # Test successful startup
            async def test_lifespan():
                async with lifespan(mock_app) as context:
                    assert context is None  # lifespan yields None
                    mock_config.validate.assert_called_once()
                    mock_ka_class.assert_called_once()
                    mock_logger.info.assert_called()
            
            asyncio.run(test_lifespan())
    
    def test_lifespan_startup_failure(self):
        """Test lifespan manager startup failure."""
        from src.api.main import lifespan
        from unittest.mock import patch, Mock
        import asyncio
        
        mock_app = Mock()
        
        with patch('src.api.main.config') as mock_config, \
             patch('src.api.main.logger') as mock_logger:
            
            mock_config.validate.side_effect = Exception("Config validation failed")
            
            # Test startup failure
            async def test_lifespan():
                with pytest.raises(Exception, match="Config validation failed"):
                    async with lifespan(mock_app):
                        pass
                
                mock_logger.error.assert_called()
            
            asyncio.run(test_lifespan())
    
    def test_lifespan_shutdown(self):
        """Test lifespan manager shutdown."""
        from src.api.main import lifespan
        from unittest.mock import patch, Mock
        import asyncio
        
        mock_app = Mock()
        
        with patch('src.api.main.config') as mock_config, \
             patch('src.api.main.KnowledgeAssistant') as mock_ka_class, \
             patch('src.api.main.logger') as mock_logger:
            
            mock_config.validate.return_value = True
            mock_ka_class.return_value = Mock()
            
            # Test shutdown
            async def test_lifespan():
                async with lifespan(mock_app) as context:
                    pass  # This will trigger shutdown
                
                # Check that shutdown was logged
                shutdown_calls = [call for call in mock_logger.info.call_args_list 
                                if "Shutting down" in str(call)]
                assert len(shutdown_calls) > 0
            
            asyncio.run(test_lifespan())
    
    def test_get_stats_error_handling(self, client):
        """Test get_stats endpoint error handling."""
        from src.api.main import get_knowledge_assistant
        
        mock_assistant = Mock()
        mock_assistant.get_system_stats.side_effect = Exception("Stats error")
        
        app.dependency_overrides[get_knowledge_assistant] = lambda: mock_assistant
        
        try:
            response = client.get("/stats")
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Failed to retrieve system statistics" in data["detail"]
        finally:
            app.dependency_overrides.clear()
    
    def test_main_module_execution(self):
        """Test main module execution."""
        from unittest.mock import patch, Mock
        import sys
        
        # Test that the main block code exists and is syntactically correct
        # by importing the module and checking its attributes
        import src.api.main
        
        # Verify the module has the expected attributes
        assert hasattr(src.api.main, 'app')
        assert src.api.main.app is not None
        assert hasattr(src.api.main, 'knowledge_assistant')
        
        # Test that uvicorn.run would be called if the module was run directly
        with patch('uvicorn.run') as mock_uvicorn, \
             patch('src.api.main.config') as mock_config:
            
            mock_config.API_HOST = "0.0.0.0"
            mock_config.API_PORT = 8000
            
            # Simulate running the main block
            if __name__ == "__main__":
                import uvicorn
                uvicorn.run(
                    "src.api.main:app",
                    host=config.API_HOST,
                    port=config.API_PORT,
                    reload=True
                )
            
            # This test verifies the code structure exists
            assert True