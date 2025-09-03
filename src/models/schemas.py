"""
Pydantic schemas for API request/response models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class TicketRequest(BaseModel):
    """Request model for support ticket resolution."""
    ticket_text: str = Field(..., description="The customer support ticket text")


class TicketResponse(BaseModel):
    """Response model for support ticket resolution following MCP format."""
    answer: str = Field(..., description="The generated response to the support ticket")
    references: List[str] = Field(..., description="List of referenced documentation sections")
    action_required: str = Field(..., description="Required action for the support team")