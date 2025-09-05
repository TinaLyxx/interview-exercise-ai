"""
Unit tests for DocumentRetriever.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.rag.retriever import DocumentRetriever
from src.models.schemas import DocumentChunk


class TestDocumentRetriever:
    """Test cases for DocumentRetriever."""
    
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
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary path for index storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield os.path.join(temp_dir, "test_index")
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            DocumentChunk(
                content="Domain suspension occurs when policies are violated.",
                source="Policy: Domain Suspension, Section 1",
                metadata={"type": "policy", "section": "1"}
            ),
            DocumentChunk(
                content="WHOIS information must be accurate and up-to-date.",
                source="Policy: WHOIS Requirements, Section 2",
                metadata={"type": "requirement", "section": "2"}
            ),
            DocumentChunk(
                content="Contact support for assistance with domain issues.",
                source="FAQ: General Support",
                metadata={"type": "faq", "section": "general"}
            )
        ]
    
    @pytest.fixture
    def mock_document_processor(self):
        """Create mock DocumentProcessor."""
        mock_processor = Mock()
        mock_processor.load_documents.return_value = [
            DocumentChunk(
                content="Test document content",
                source="test_doc.md",
                metadata={"file": "test_doc.md", "section": "Test Section"}
            )
        ]
        return mock_processor
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock FAISSVectorStore."""
        mock_store = Mock()
        mock_store.load_index.return_value = True
        mock_store.add_documents.return_value = None
        mock_store.save_index.return_value = None
        mock_store.search.return_value = [
            (DocumentChunk(
                content="Test search result",
                source="test_source",
                metadata={}
            ), 0.95)
        ]
        mock_store.get_stats.return_value = {
            "total_documents": 1,
            "index_size": 1,
            "embedding_dimension": 384
        }
        return mock_store
    
    def test_init_with_default_paths(self, mock_document_processor, mock_vector_store):
        """Test initialization with default paths."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store), \
             patch('src.rag.retriever.config') as mock_config:
            
            mock_config.DOCS_PATH = "/default/docs"
            mock_config.VECTOR_DB_PATH = "/default/vector"
            
            retriever = DocumentRetriever()
            
            assert retriever.docs_path == "/default/docs"
            assert retriever.vector_store_path == "/default/vector"
            assert retriever.document_processor == mock_document_processor
            assert retriever.vector_store == mock_vector_store
    
    def test_init_with_custom_paths(self, mock_document_processor, mock_vector_store):
        """Test initialization with custom paths."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            docs_path = "/custom/docs"
            vector_store_path = "/custom/vector"
            
            retriever = DocumentRetriever(docs_path, vector_store_path)
            
            assert retriever.docs_path == docs_path
            assert retriever.vector_store_path == vector_store_path
            
            # Verify DocumentProcessor was called with custom path
            from src.rag.retriever import DocumentProcessor
            DocumentProcessor.assert_called_once_with(docs_path)
            
            # Verify FAISSVectorStore was called with custom path
            from src.rag.retriever import FAISSVectorStore
            FAISSVectorStore.assert_called_once_with(index_path=vector_store_path)
    
    def test_initialize_vector_store_load_existing(self, mock_document_processor, mock_vector_store):
        """Test vector store initialization with existing index."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store), \
             patch('builtins.print') as mock_print:
            
            retriever = DocumentRetriever()
            
            # Verify load_index was called
            mock_vector_store.load_index.assert_called_once()
            
            # Verify print message
            mock_print.assert_called_once_with("Loaded existing vector store")
    
    def test_initialize_vector_store_create_new(self, mock_document_processor, mock_vector_store):
        """Test vector store initialization creating new index."""
        mock_vector_store.load_index.return_value = False
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store), \
             patch('builtins.print') as mock_print:
            
            retriever = DocumentRetriever()
            
            # Verify load_index was called
            mock_vector_store.load_index.assert_called_once()
            
            # Verify build process was called
            mock_document_processor.load_documents.assert_called_once()
            mock_vector_store.add_documents.assert_called_once()
            mock_vector_store.save_index.assert_called_once()
            
            # Verify print messages
            assert mock_print.call_count == 2
            assert "Creating new vector store..." in mock_print.call_args_list[0][0][0]
            assert "Built vector store with 1 document chunks" in mock_print.call_args_list[1][0][0]
    
    def test_build_vector_store_no_documents(self, mock_document_processor, mock_vector_store):
        """Test building vector store with no documents."""
        mock_document_processor.load_documents.return_value = []
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            
            # Should raise ValueError when no documents found
            with pytest.raises(ValueError, match="No documents found to build vector store"):
                retriever._build_vector_store()
    
    def test_retrieve_relevant_context_success(self, mock_document_processor, mock_vector_store):
        """Test successful context retrieval."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store), \
             patch('src.rag.retriever.config') as mock_config:
            
            mock_config.MAX_RELEVANT_CHUNKS = 5
            mock_config.SIMILARITY_THRESHOLD = 0.7
            
            retriever = DocumentRetriever()
            results = retriever.retrieve_relevant_context("test query")
            
            # Verify vector store search was called
            mock_vector_store.search.assert_called_once_with("test query", k=5, threshold=0.7)
            
            # Verify results
            assert len(results) == 1
            assert isinstance(results[0], tuple)
            assert isinstance(results[0][0], DocumentChunk)
            assert isinstance(results[0][1], float)
    
    def test_retrieve_relevant_context_custom_params(self, mock_document_processor, mock_vector_store):
        """Test context retrieval with custom parameters."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            results = retriever.retrieve_relevant_context(
                "test query", 
                max_chunks=10, 
                threshold=0.8
            )
            
            # Verify vector store search was called with custom params
            mock_vector_store.search.assert_called_once_with("test query", k=10, threshold=0.8)
    
    def test_get_context_string_success(self, mock_document_processor, mock_vector_store):
        """Test getting formatted context string."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            context = retriever.get_context_string("test query")
            
            # Verify context format
            assert "Source 1:" in context
            assert "Test search result" in context
            assert "test_source" in context
    
    def test_get_context_string_no_results(self, mock_document_processor, mock_vector_store):
        """Test getting context string when no results found."""
        mock_vector_store.search.return_value = []
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            context = retriever.get_context_string("test query")
            
            assert context == "No relevant documentation found."
    
    def test_get_references_success(self, mock_document_processor, mock_vector_store):
        """Test getting reference sources."""
        # Mock multiple results with same source
        mock_vector_store.search.return_value = [
            (DocumentChunk(content="content1", source="source1", metadata={}), 0.9),
            (DocumentChunk(content="content2", source="source2", metadata={}), 0.8),
            (DocumentChunk(content="content3", source="source1", metadata={}), 0.7)
        ]
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            references = retriever.get_references("test query")
            
            # Should return unique sources
            assert len(references) == 2
            assert "source1" in references
            assert "source2" in references
    
    def test_get_references_no_results(self, mock_document_processor, mock_vector_store):
        """Test getting references when no results found."""
        mock_vector_store.search.return_value = []
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            references = retriever.get_references("test query")
            
            assert references == []
    
    def test_rebuild_index(self, mock_document_processor, mock_vector_store):
        """Test rebuilding the index."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store), \
             patch('builtins.print') as mock_print:
            
            retriever = DocumentRetriever()
            retriever.rebuild_index()
            
            # Verify build process was called
            mock_document_processor.load_documents.assert_called()
            mock_vector_store.add_documents.assert_called()
            mock_vector_store.save_index.assert_called()
            
            # Verify print message was called (among others)
            assert mock_print.call_count >= 1
            # Check that the rebuild message was printed
            rebuild_calls = [call for call in mock_print.call_args_list if "Rebuilding vector store index..." in str(call)]
            assert len(rebuild_calls) >= 1
    
    def test_get_stats(self, mock_document_processor, mock_vector_store):
        """Test getting retrieval system statistics."""
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            stats = retriever.get_stats()
            
            # Verify vector store stats were called
            mock_vector_store.get_stats.assert_called_once()
            
            # Verify response structure
            assert "docs_path" in stats
            assert "vector_store_path" in stats
            assert "total_documents" in stats
            assert "index_size" in stats
            assert "embedding_dimension" in stats
    
    def test_retrieve_relevant_context_sorting(self, mock_document_processor, mock_vector_store):
        """Test that results are sorted by similarity score."""
        # Mock results with different scores
        mock_vector_store.search.return_value = [
            (DocumentChunk(content="content1", source="source1", metadata={}), 0.5),
            (DocumentChunk(content="content2", source="source2", metadata={}), 0.9),
            (DocumentChunk(content="content3", source="source3", metadata={}), 0.7)
        ]
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            results = retriever.retrieve_relevant_context("test query")
            
            # Verify results are sorted by score (descending)
            assert len(results) == 3
            assert results[0][1] == 0.9  # Highest score first
            assert results[1][1] == 0.7
            assert results[2][1] == 0.5  # Lowest score last
    
    def test_get_context_string_multiple_sources(self, mock_document_processor, mock_vector_store):
        """Test context string with multiple sources."""
        # Mock multiple results
        mock_vector_store.search.return_value = [
            (DocumentChunk(content="First content", source="source1", metadata={}), 0.9),
            (DocumentChunk(content="Second content", source="source2", metadata={}), 0.8)
        ]
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            context = retriever.get_context_string("test query")
            
            # Verify both sources are included
            assert "Source 1:" in context
            assert "Source 2:" in context
            assert "source1" in context
            assert "source2" in context
            assert "First content" in context
            assert "Second content" in context
    
    def test_initialization_with_real_documents(self, temp_docs_dir, temp_index_path):
        """Test initialization with real documents (integration test)."""
        with patch('src.rag.retriever.config') as mock_config:
            mock_config.DOCS_PATH = temp_docs_dir
            mock_config.VECTOR_DB_PATH = temp_index_path
            mock_config.MAX_RELEVANT_CHUNKS = 5
            mock_config.SIMILARITY_THRESHOLD = 0.7
            
            # This should work with real documents
            retriever = DocumentRetriever()
            
            # Verify it was initialized
            assert retriever.docs_path == temp_docs_dir
            assert retriever.vector_store_path == temp_index_path
            assert retriever.document_processor is not None
            assert retriever.vector_store is not None
    
    def test_error_handling_in_build_vector_store(self, mock_document_processor, mock_vector_store):
        """Test error handling during vector store building."""
        mock_document_processor.load_documents.side_effect = Exception("Document loading failed")
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            retriever = DocumentRetriever()
            
            # Should propagate the exception
            with pytest.raises(Exception, match="Document loading failed"):
                retriever._build_vector_store()
    
    def test_vector_store_save_error(self, mock_document_processor, mock_vector_store):
        """Test error handling when vector store save fails."""
        mock_vector_store.load_index.return_value = False
        mock_vector_store.save_index.side_effect = Exception("Save failed")
        
        with patch('src.rag.retriever.DocumentProcessor', return_value=mock_document_processor), \
             patch('src.rag.retriever.FAISSVectorStore', return_value=mock_vector_store):
            
            # Should propagate the exception during initialization
            with pytest.raises(Exception, match="Save failed"):
                DocumentRetriever()
