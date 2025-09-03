"""
Unit tests for EmbeddingGenerator.
"""
import pytest
import numpy as np
from src.rag.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""
    
    @pytest.fixture
    def embedding_generator(self):
        """Create an EmbeddingGenerator instance for testing."""
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    def test_init_default_model(self):
        """Test initialization with default model."""
        generator = EmbeddingGenerator()
        assert generator.model_name == "all-MiniLM-L6-v2"  # Default from config
        assert generator.model is not None
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        model_name = "all-MiniLM-L6-v2"
        generator = EmbeddingGenerator(model_name=model_name)
        assert generator.model_name == model_name
    
    def test_encode_query(self, embedding_generator):
        """Test encoding a single query."""
        query = "What is the domain suspension policy?"
        embedding = embedding_generator.encode_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) > 0
    
    def test_encode_documents(self, embedding_generator):
        """Test encoding multiple documents."""
        texts = [
            "Domain suspension occurs when policies are violated.",
            "WHOIS information must be accurate and up-to-date.",
            "Contact support for assistance with domain issues."
        ]
        
        embeddings = embedding_generator.encode_documents(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
        assert embeddings.ndim == 2
    
    def test_encode_documents_empty_list(self, embedding_generator):
        """Test encoding empty list of documents."""
        embeddings = embedding_generator.encode_documents([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.size == 0
    
    def test_get_embedding_dimension(self, embedding_generator):
        """Test getting embedding dimension."""
        dimension = embedding_generator.get_embedding_dimension()
        
        assert isinstance(dimension, int)
        assert dimension > 0
        
        # Verify dimension matches actual embeddings
        test_embedding = embedding_generator.encode_query("test")
        assert len(test_embedding) == dimension
    
    def test_consistent_embeddings(self, embedding_generator):
        """Test that same text produces consistent embeddings."""
        text = "Domain registration requires accurate WHOIS information."
        
        embedding1 = embedding_generator.encode_query(text)
        embedding2 = embedding_generator.encode_query(text)
        
        # Should be identical (or very close due to floating point precision)
        np.testing.assert_allclose(embedding1, embedding2, rtol=1e-6)
    
    def test_different_texts_different_embeddings(self, embedding_generator):
        """Test that different texts produce different embeddings."""
        text1 = "Domain suspension policy"
        text2 = "Billing and payment information"
        
        embedding1 = embedding_generator.encode_query(text1)
        embedding2 = embedding_generator.encode_query(text2)
        
        # Should be different
        assert not np.allclose(embedding1, embedding2, rtol=0.1)
    
    def test_similar_texts_similar_embeddings(self, embedding_generator):
        """Test that similar texts produce similar embeddings."""
        text1 = "Domain was suspended due to policy violation"
        text2 = "Domain suspension occurred because of policy breach"
        
        embedding1 = embedding_generator.encode_query(text1)
        embedding2 = embedding_generator.encode_query(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Similar texts should have high similarity
        assert similarity > 0.7
