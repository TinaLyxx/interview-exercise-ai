"""
Unit tests for LLMClient.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import openai
from src.rag.llm_client import LLMClient
from src.models.schemas import TicketResponse


class TestLLMClient:
    """Test cases for LLMClient."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "answer": "Test response from LLM",
            "references": ["Policy: Test Policy, Section 1"],
            "action_required": "no_action_required"
        })
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return "Test context from documentation about domain policies."
    
    @pytest.fixture
    def sample_references(self):
        """Create sample references for testing."""
        return ["Policy: Domain Suspension, Section 1", "FAQ: General Support"]
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        with patch('src.rag.llm_client.config') as mock_config, \
             patch('src.rag.llm_client.openai.OpenAI') as mock_openai:
            
            mock_config.OPENAI_API_KEY = "test-api-key"
            mock_config.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_openai.return_value = Mock()
            
            client = LLMClient()
            
            assert client.api_key == "test-api-key"
            assert client.model == "gpt-3.5-turbo"
            mock_openai.assert_called_once_with(
                api_key="test-api-key",
                timeout=30,
                max_retries=3
            )
    
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('src.rag.llm_client.openai.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            
            client = LLMClient(api_key="custom-key", model="gpt-4")
            
            assert client.api_key == "custom-key"
            assert client.model == "gpt-4"
            mock_openai.assert_called_once_with(
                api_key="custom-key",
                timeout=30,
                max_retries=3
            )
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with patch('src.rag.llm_client.config') as mock_config:
            mock_config.OPENAI_API_KEY = None
            
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                LLMClient()
    
    def test_generate_response_success(self, mock_openai_client, sample_context, sample_references):
        """Test successful response generation."""
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_openai_client):
            client = LLMClient(api_key="test-key")
            
            ticket_text = "My domain was suspended"
            response = client.generate_response(ticket_text, sample_context, sample_references)
            
            # Verify OpenAI was called correctly
            mock_openai_client.chat.completions.create.assert_called_once()
            call_args = mock_openai_client.chat.completions.create.call_args
            
            assert call_args[1]["model"] == "gpt-4o-mini"
            assert call_args[1]["temperature"] == 0.1
            assert call_args[1]["max_tokens"] == 1000
            assert call_args[1]["response_format"] == {"type": "json_object"}
            
            # Verify messages structure
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert ticket_text in messages[1]["content"]
            assert sample_context in messages[1]["content"]
            
            # Verify response
            assert isinstance(response, TicketResponse)
            assert response.answer == "Test response from LLM"
            assert response.references == ["Policy: Test Policy, Section 1"]
            assert response.action_required == "no_action_required"
    
    def test_generate_response_json_decode_error(self, sample_context, sample_references):
        """Test response generation with JSON decode error."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_client):
            client = LLMClient(api_key="test-key")
            
            response = client.generate_response("test ticket", sample_context, sample_references)
            
            # Should return fallback response
            assert isinstance(response, TicketResponse)
            assert "I apologize, but I encountered an error" in response.answer
            assert response.references == sample_references
            assert response.action_required == "escalate_to_technical_team"
    
    def test_generate_response_rate_limit_error(self, sample_context, sample_references):
        """Test response generation with rate limit error."""
        mock_client = Mock()
        # Create a proper mock response for RateLimitError
        mock_response = Mock()
        mock_response.request = Mock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_client), \
             patch('builtins.print') as mock_print:
            
            client = LLMClient(api_key="test-key")
            
            response = client.generate_response("test ticket", sample_context, sample_references)
            
            # Should return rate limit response
            assert isinstance(response, TicketResponse)
            assert "I'm currently experiencing high demand" in response.answer
            assert response.references == sample_references
            assert response.action_required == "escalate_to_technical_team"
            
            # Verify error was printed (after retries)
            assert mock_print.call_count >= 1
            # Check that the final error message was printed
            final_calls = [call for call in mock_print.call_args_list if "Rate limit exceeded:" in str(call)]
            assert len(final_calls) >= 1
    
    def test_generate_response_general_exception(self, sample_context, sample_references):
        """Test response generation with general exception."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_client), \
             patch('builtins.print') as mock_print:
            
            client = LLMClient(api_key="test-key")
            
            response = client.generate_response("test ticket", sample_context, sample_references)
            
            # Should return general error response
            assert isinstance(response, TicketResponse)
            assert "I'm experiencing technical difficulties" in response.answer
            assert response.references == []
            assert response.action_required == "escalate_to_technical_team"
            
            # Verify error was printed
            mock_print.assert_called_once_with("OpenAI API error: API error")
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            prompt = client._get_system_prompt()
            
            assert isinstance(prompt, str)
            assert "customer support assistant" in prompt
            assert "domain registration" in prompt
            assert "JSON" in prompt
            assert "escalate_to_technical_team" in prompt
            assert "escalate_to_abuse_team" in prompt
            assert "escalate_to_billing_team" in prompt
            assert "escalate_to_management" in prompt
            assert "escalate_to_legal_team" in prompt
            assert "contact_customer_directly" in prompt
            assert "no_action_required" in prompt
    
    def test_build_mcp_prompt(self, sample_context, sample_references):
        """Test MCP prompt building."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            ticket_text = "My domain was suspended"
            prompt = client._build_mcp_prompt(ticket_text, sample_context, sample_references)
            
            assert isinstance(prompt, str)
            assert "TASK: Analyze the customer support ticket" in prompt
            assert "CONTEXT (Company Documentation):" in prompt
            assert sample_context in prompt
            assert "CUSTOMER TICKET:" in prompt
            assert ticket_text in prompt
            assert "INSTRUCTIONS:" in prompt
            assert "OUTPUT SCHEMA:" in prompt
            assert "answer" in prompt
            assert "references" in prompt
            assert "action_required" in prompt
    
    def test_determine_action_required_abuse(self):
        """Test action determination for abuse-related tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test various abuse keywords
            test_cases = [
                "My domain was suspended",
                "Policy violation occurred",
                "Abuse complaint filed"
            ]
            
            for ticket_text in test_cases:
                action = client._determine_action_required(ticket_text, "context")
                assert action == "escalate_to_abuse_team"
    
    def test_determine_action_required_billing(self):
        """Test action determination for billing-related tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test various billing keywords
            test_cases = [
                "I have a billing question",
                "Payment was charged incorrectly",
                "I need a refund for my invoice"
            ]
            
            for ticket_text in test_cases:
                action = client._determine_action_required(ticket_text, "context")
                assert action == "escalate_to_billing_team"
    
    def test_determine_action_required_technical(self):
        """Test action determination for technical-related tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test various technical keywords
            test_cases = [
                "DNS is not working",
                "Nameserver configuration error",
                "Technical issue with my domain"
            ]
            
            for ticket_text in test_cases:
                action = client._determine_action_required(ticket_text, "context")
                assert action == "escalate_to_technical_team"
    
    def test_determine_action_required_legal(self):
        """Test action determination for legal-related tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test various legal keywords
            test_cases = [
                "Legal action against domain",
                "DMCA takedown request",
                "Court order received"
            ]
            
            for ticket_text in test_cases:
                action = client._determine_action_required(ticket_text, "context")
                assert action == "escalate_to_legal_team"
    
    def test_determine_action_required_management(self):
        """Test action determination for management-related tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test various management keywords
            test_cases = [
                "I want to speak to a manager",
                "Customer complaint about service",
                "I am unsatisfied with support"
            ]
            
            for ticket_text in test_cases:
                action = client._determine_action_required(ticket_text, "context")
                assert action == "escalate_to_management"
    
    def test_determine_action_required_urgent(self):
        """Test action determination for urgent tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test various urgent keywords
            test_cases = [
                "This is urgent",
                "Emergency situation",
                "Need help ASAP"
            ]
            
            for ticket_text in test_cases:
                action = client._determine_action_required(ticket_text, "context")
                assert action == "contact_customer_directly"
    
    def test_determine_action_required_no_action(self):
        """Test action determination for general tickets."""
        with patch('src.rag.llm_client.openai.OpenAI'):
            client = LLMClient(api_key="test-key")
            
            # Test general ticket
            ticket_text = "I have a general question about my domain"
            action = client._determine_action_required(ticket_text, "context")
            assert action == "no_action_required"
    
    def test_call_openai_with_backoff_success(self, mock_openai_client):
        """Test successful OpenAI call with backoff."""
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_openai_client):
            client = LLMClient(api_key="test-key")
            
            result = client._call_openai_with_backoff(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}]
            )
            
            assert result == mock_openai_client.chat.completions.create.return_value
            mock_openai_client.chat.completions.create.assert_called_once()
    
    def test_call_openai_with_backoff_rate_limit_retry(self):
        """Test OpenAI call with rate limit retry."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"answer": "test"}'
        
        # Create proper mock response for RateLimitError
        mock_error_response = Mock()
        mock_error_response.request = Mock()
        
        # First call fails with rate limit, second succeeds
        mock_client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit", response=mock_error_response, body=None),
            mock_response
        ]
        
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_client), \
             patch('time.sleep') as mock_sleep:
            
            client = LLMClient(api_key="test-key")
            
            result = client._call_openai_with_backoff(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}]
            )
            
            # Should have been called twice
            assert mock_client.chat.completions.create.call_count == 2
            
            # Should have slept once with exponential backoff
            mock_sleep.assert_called_once_with(1)  # base_delay * (2 ** 0)
            
            assert result == mock_response
    
    def test_call_openai_with_backoff_max_retries_exceeded(self):
        """Test OpenAI call when max retries are exceeded."""
        mock_client = Mock()
        # Create proper mock response for RateLimitError
        mock_error_response = Mock()
        mock_error_response.request = Mock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError("Rate limit", response=mock_error_response, body=None)
        
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_client), \
             patch('time.sleep') as mock_sleep, \
             patch('builtins.print') as mock_print:
            
            client = LLMClient(api_key="test-key")
            
            # Should raise the rate limit error after max retries
            with pytest.raises(openai.RateLimitError):
                client._call_openai_with_backoff(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}]
                )
            
            # Should have been called 5 times (max_retries)
            assert mock_client.chat.completions.create.call_count == 5
            
            # Should have slept 4 times (0, 1, 2, 3)
            assert mock_sleep.call_count == 4
    
    def test_call_openai_with_backoff_other_exception(self):
        """Test OpenAI call with non-rate-limit exception."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Other error")
        
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_client):
            client = LLMClient(api_key="test-key")
            
            # Should raise the exception immediately (no retry)
            with pytest.raises(Exception, match="Other error"):
                client._call_openai_with_backoff(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}]
                )
            
            # Should have been called only once
            assert mock_client.chat.completions.create.call_count == 1
    
    def test_generate_response_with_empty_references(self, mock_openai_client, sample_context):
        """Test response generation with empty references."""
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_openai_client):
            client = LLMClient(api_key="test-key")
            
            response = client.generate_response("test ticket", sample_context, [])
            
            # Should still work with empty references
            assert isinstance(response, TicketResponse)
    
    def test_generate_response_with_none_references(self, mock_openai_client, sample_context):
        """Test response generation with None references."""
        with patch('src.rag.llm_client.openai.OpenAI', return_value=mock_openai_client):
            client = LLMClient(api_key="test-key")
            
            response = client.generate_response("test ticket", sample_context, None)
            
            # Should still work with None references
            assert isinstance(response, TicketResponse)
