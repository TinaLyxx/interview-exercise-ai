"""
Unit tests for DocumentProcessor.
"""
import pytest
import tempfile
import os
from pathlib import Path
from src.rag.document_processor import DocumentProcessor
from src.models.schemas import DocumentChunk


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    @pytest.fixture
    def temp_docs_dir(self):
        """Create a temporary directory with test documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)
            
            # Create test markdown file
            test_doc = docs_dir / "test_policy.md"
            test_content = """# Test Policy

## Section 1: General Rules
This is the first section with general rules.
It contains multiple lines of content.

## Section 2: Specific Guidelines
This section covers specific guidelines.
Each section should be processed separately.

### Subsection 2.1: Details
This is a subsection with more details.
"""
            test_doc.write_text(test_content)
            
            yield str(docs_dir)
    
    def test_init(self, temp_docs_dir):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(temp_docs_dir)
        assert processor.docs_path == Path(temp_docs_dir)
    
    def test_load_documents(self, temp_docs_dir):
        """Test loading documents from directory."""
        processor = DocumentProcessor(temp_docs_dir)
        chunks = processor.load_documents()
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
        # Check that we have the expected sections
        sources = [chunk.source for chunk in chunks]
        assert any("General Rules" in source for source in sources)
        assert any("Specific Guidelines" in source for source in sources)
    
    def test_load_documents_empty_dir(self):
        """Test loading documents from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = DocumentProcessor(temp_dir)
            chunks = processor.load_documents()
            assert chunks == []
    
    def test_load_documents_nonexistent_dir(self):
        """Test loading documents from non-existent directory."""
        processor = DocumentProcessor("/nonexistent/path")
        with pytest.raises(ValueError):
            processor.load_documents()
    
    def test_split_by_headings(self, temp_docs_dir):
        """Test splitting content by markdown headings."""
        processor = DocumentProcessor(temp_docs_dir)
        
        content = """# Main Title

Some intro content.

## Section A
Content for section A.

## Section B
Content for section B.
More content here.

### Subsection B.1
Subsection content.
"""
        
        sections = processor._split_by_headings(content)
        
        assert len(sections) >= 3
        
        # Check section titles
        titles = [title for title, _ in sections]
        assert "Section A" in titles
        assert "Section B" in titles
        assert "Subsection B.1" in titles
    
    def test_chunk_text_short(self, temp_docs_dir):
        """Test chunking short text."""
        processor = DocumentProcessor(temp_docs_dir)
        
        short_text = "This is a short text."
        chunks = processor.chunk_text(short_text, max_length=500)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_chunk_text_long(self, temp_docs_dir):
        """Test chunking long text."""
        processor = DocumentProcessor(temp_docs_dir)
        
        # Create text longer than max_length
        long_text = "This is a sentence. " * 50  # Should be > 100 chars
        chunks = processor.chunk_text(long_text, max_length=100, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 110 for chunk in chunks)  # Allow for overlap
    
    def test_process_file_metadata(self, temp_docs_dir):
        """Test that processed files include proper metadata."""
        processor = DocumentProcessor(temp_docs_dir)
        chunks = processor.load_documents()
        
        for chunk in chunks:
            assert chunk.source is not None
            assert chunk.metadata is not None
            assert "file" in chunk.metadata
            assert "section" in chunk.metadata
            assert "length" in chunk.metadata
            assert chunk.metadata["length"] == len(chunk.content)
