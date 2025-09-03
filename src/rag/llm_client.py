"""
LLM client for generating responses using OpenAI GPT with MCP-compliant prompting.
"""
import json
import openai
from typing import Dict, Any, Optional
from src.utils.config import config
from src.models.schemas import TicketResponse

class LLMClient:
    """OpenAI GPT client for generating support responses."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the LLM client.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, ticket_text: str, context: str, references: list) -> TicketResponse:
        """Generate a support response using MCP-compliant prompting.
        
        Args:
            ticket_text: The customer support ticket text
            context: Retrieved context from RAG system
            references: List of reference sources
            
        Returns:
            Structured response following MCP format
        """
        # Construct MCP-compliant prompt
        prompt = self._build_mcp_prompt(ticket_text, context, references)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent responses
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            response_data = json.loads(response_text)
            
            # Validate and return structured response
            return TicketResponse(**response_data)
        
        except json.JSONDecodeError as e:
            # Fallback response if JSON parsing fails
            return TicketResponse(
                answer="I apologize, but I encountered an error processing your request. Please contact our support team directly for assistance.",
                references=references or [],
                action_required="escalate_to_technical_team"
            )
        
        except Exception as e:
            # General error handling
            return TicketResponse(
                answer="I'm experiencing technical difficulties. Please try again or contact our support team for immediate assistance.",
                references=[],
                action_required="escalate_to_technical_team"
            )
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines the AI's role and behavior.
        
        Returns:
            System prompt string
        """
        pass
    
    def _build_mcp_prompt(self, ticket_text: str, context: str, references: list) -> str:
        """Build MCP-compliant prompt with clear structure.
        
        Args:
            ticket_text: Customer ticket text
            context: Retrieved documentation context
            references: List of reference sources
            
        Returns:
            Formatted prompt string
        """
        pass
    
    