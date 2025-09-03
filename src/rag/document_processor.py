"""
Document processing utilities for the RAG system.
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path
from src.models.schemas import DocumentChunk


class DocumentProcessor:
    """Processes documents for the RAG system."""
    
    def __init__(self, docs_path: str):
        """Initialize the document processor.
        
        Args:
            docs_path: Path to the directory containing documents
        """
        self.docs_path = Path(docs_path)
    
    def load_documents(self) -> List[DocumentChunk]:
        """Load and process all documents from the docs directory.
        
        Returns:
            List of document chunks ready for embedding
        """
        chunks = []
        
        if not self.docs_path.exists():
            raise ValueError(f"Documents path does not exist: {self.docs_path}")
        
        for file_path in self.docs_path.glob("*.md"):
            file_chunks = self._process_file(file_path)
            chunks.extend(file_chunks)
        
        return chunks
    def _process_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single file into chunks.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of document chunks from the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by sections (headings)
        sections = self._split_by_headings(content)
        
        chunks = []
        for section_title, section_content in sections:
            if section_content.strip():
                stripped_content = section_content.strip()
                chunk = DocumentChunk(
                    content=stripped_content,
                    source=f"{file_path.name}: {section_title}",
                    metadata={
                        "file": str(file_path.name),
                        "section": section_title,
                        "length": len(stripped_content)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_headings(self, content: str) -> List[tuple]:
        """Split content by markdown headings.
        
        Args:
            content: The file content to split
            
        Returns:
            List of (heading, content) tuples
        """
        # Split by headings (# ## ### etc.)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        sections = []
        current_section = ""
        current_heading = "Introduction"
        
        for line in lines:
            heading_match = re.match(heading_pattern, line)
            if heading_match:
                # Save previous section if it has content
                if current_section.strip():
                    sections.append((current_heading, current_section))
                
                # Start new section
                current_heading = heading_match.group(2)
                current_section = ""
            else:
                current_section += line + "\n"
        
        # Add the last section
        if current_section.strip():
            sections.append((current_heading, current_section))
        
        return sections
    
    def chunk_text(self, text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            max_length: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + max_length - 100, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks

    
   