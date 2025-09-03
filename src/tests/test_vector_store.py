"""
Unit tests for FAISSVectorStore.
"""
import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from src.rag.vector_store import FAISSVectorStore
from src.models.schemas import DocumentChunk


class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore."""
    
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
                metadata={"type": "policy"}
            ),
            DocumentChunk(
                content="WHOIS information must be accurate and up-to-date.",
                source="Policy: WHOIS Requirements, Section 2",
                metadata={"type": "requirement"}
            ),
            DocumentChunk(
                content="Contact support for assistance with domain issues.",
                source="FAQ: General Support",
                metadata={"type": "faq"}
            )
        ]
    
    def test_init(self, temp_index_path):
        """Test FAISSVectorStore initialization."""
        vector_store = FAISSVectorStore(
            embedding_dimension=384,
            index_path=temp_index_path
        )
        
        assert vector_store.embedding_dimension == 384
        assert vector_store.index_path == temp_index_path
        assert vector_store.index.ntotal == 0
        assert len(vector_store.documents) == 0
    
    def test_add_documents(self, temp_index_path, sample_documents):
        """Test adding documents to vector store."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        
        # Add documents
        vector_store.add_documents(sample_documents)
        
        assert vector_store.index.ntotal == len(sample_documents)
        assert len(vector_store.documents) == len(sample_documents)
    
    def test_add_empty_documents(self, temp_index_path):
        """Test adding empty list of documents."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        
        vector_store.add_documents([])
        
        assert vector_store.index.ntotal == 0
        assert len(vector_store.documents) == 0
    
    def test_search_no_documents(self, temp_index_path):
        """Test searching with no documents in store."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        
        results = vector_store.search("test query")
        
        assert results == []
    
    def test_search_with_documents(self, temp_index_path, sample_documents):
        """Test searching with documents in store."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        vector_store.add_documents(sample_documents)
        
        # Search for domain-related content
        results = vector_store.search("domain suspension", k=2, threshold=0.0)
        
        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        # Check that results contain DocumentChunk and float score
        for doc, score in results:
            assert isinstance(doc, DocumentChunk)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_search_relevance(self, temp_index_path, sample_documents):
        """Test that search returns relevant results."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        vector_store.add_documents(sample_documents)
        
        # Search for WHOIS-related content
        results = vector_store.search("WHOIS information requirements", k=3, threshold=0.0)
        
        # Should return results in order of relevance
        assert len(results) > 0
        
        # First result should be most relevant (highest score)
        if len(results) > 1:
            assert results[0][1] >= results[1][1]
    
    def test_save_and_load_index(self, temp_index_path, sample_documents):
        """Test saving and loading index."""
        # Create and populate vector store
        vector_store1 = FAISSVectorStore(index_path=temp_index_path)
        vector_store1.add_documents(sample_documents)
        
        # Save index
        vector_store1.save_index()
        
        # Create new vector store and load index
        vector_store2 = FAISSVectorStore(index_path=temp_index_path)
        loaded = vector_store2.load_index()
        
        assert loaded is True
        assert vector_store2.index.ntotal == len(sample_documents)
        assert len(vector_store2.documents) == len(sample_documents)
        
        # Test that search works after loading
        results = vector_store2.search("domain", k=1, threshold=0.0)
        assert len(results) > 0
    
    def test_load_nonexistent_index(self, temp_index_path):
        """Test loading non-existent index."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        loaded = vector_store.load_index()
        
        assert loaded is False
    
    
    def test_threshold_filtering(self, temp_index_path, sample_documents):
        """Test that threshold filtering works correctly."""
        vector_store = FAISSVectorStore(index_path=temp_index_path)
        vector_store.add_documents(sample_documents)
        
        # Search with high threshold - should return fewer results
        results_high = vector_store.search("unrelated query about cooking", k=10, threshold=0.8)
        
        # Search with low threshold - should return more results
        results_low = vector_store.search("unrelated query about cooking", k=10, threshold=0.1)
        
        assert len(results_high) <= len(results_low)
