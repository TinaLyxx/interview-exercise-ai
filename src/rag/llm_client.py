"""
LLM client for generating responses using OpenAI GPT with MCP-compliant prompting.
"""
import json
import time
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
        
        # Initialize OpenAI client with timeout and retry settings
        self.client = openai.OpenAI(
            api_key=self.api_key,
            timeout=30,
            max_retries=3
        )
    
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
            # Call OpenAI API with exponential backoff for rate limiting
            response = self._call_openai_with_backoff(
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
        
        except openai.RateLimitError as e:
            # Specific handling for rate limit errors
            print(f"Rate limit exceeded: {e}")
            return TicketResponse(
                answer="I'm currently experiencing high demand. Please try again in a few minutes or contact our support team for immediate assistance.",
                references=references or [],
                action_required="escalate_to_technical_team"
            )
        
        except Exception as e:
            # General error handling
            print(f"OpenAI API error: {e}")
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
        return """You are a knowledgeable customer support assistant for a domain registration and hosting company. Your role is to analyze customer support tickets and provide helpful, accurate responses based on company documentation and policies.

                CRITICAL INSTRUCTIONS:
                1. Always respond with valid JSON in the exact format specified
                2. Base your answers on the provided documentation context
                3. Be professional, empathetic, and solution-oriented
                4. Recommend appropriate escalation actions when needed
                5. Cite specific policy sections when relevant

                Your response must be a valid JSON object with exactly these three fields:
                - "answer": A helpful response to the customer's question or issue
                - "references": An array of strings citing relevant documentation sources
                - "action_required": One of the predefined action types

                Valid action_required values:
                - "escalate_to_technical_team"
                - "escalate_to_abuse_team" 
                - "escalate_to_billing_team"
                - "escalate_to_management"
                - "escalate_to_legal_team"
                - "contact_customer_directly"
                - "no_action_required"

                Always respond with valid JSON only, no additional text or formatting."""
    
    def _build_mcp_prompt(self, ticket_text: str, context: str, references: list) -> str:
        """Build MCP-compliant prompt with clear structure.
        
        Args:
            ticket_text: Customer ticket text
            context: Retrieved documentation context
            references: List of reference sources
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""TASK: Analyze the customer support ticket and provide a structured response.

                CONTEXT (Company Documentation):
                {context}

                CUSTOMER TICKET:
                {ticket_text}

                INSTRUCTIONS:
                1. Analyze the customer's issue based on the provided documentation
                2. Provide a helpful and professional response
                3. Include relevant policy references
                4. Determine the appropriate action required
                5. Return response as valid JSON only

                OUTPUT SCHEMA:
                {{
                    "answer": "Your helpful response to the customer",
                    "references": ["List of relevant documentation sections"],
                    "action_required": "appropriate_action_type"
                }}

                Generate the JSON response now:"""
        
        return prompt
    
    def _determine_action_required(self, ticket_text: str, context: str) -> str:
        """Determine the appropriate action based on ticket content.
        
        Args:
            ticket_text: Customer ticket text
            context: Retrieved context
            
        Returns:
            Action required string
        """
        ticket_lower = ticket_text.lower()
        
        # Check for various escalation triggers
        if any(keyword in ticket_lower for keyword in ["suspended", "suspension", "abuse", "policy violation"]):
            return "escalate_to_abuse_team"
        
        elif any(keyword in ticket_lower for keyword in ["billing", "payment", "charge", "refund", "invoice"]):
            return "escalate_to_billing_team"
        
        elif any(keyword in ticket_lower for keyword in ["dns", "nameserver", "technical", "not working", "error"]):
            return "escalate_to_technical_team"
        
        elif any(keyword in ticket_lower for keyword in ["legal", "lawsuit", "court", "dmca"]):
            return "escalate_to_legal_team"
        
        elif any(keyword in ticket_lower for keyword in ["manager", "complaint", "unsatisfied"]):
            return "escalate_to_management"
        
        elif any(keyword in ticket_lower for keyword in ["urgent", "emergency", "asap"]):
            return "contact_customer_directly"
        
        else:
            return "no_action_required"
    
    def _call_openai_with_backoff(self, **kwargs):
        """Call OpenAI API with exponential backoff for rate limiting.
        
        Args:
            **kwargs: Arguments to pass to the OpenAI API call
            
        Returns:
            OpenAI API response
        """
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            
            except openai.RateLimitError as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise the exception
                    raise e
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            except Exception as e:
                # For other exceptions, don't retry
                raise e
