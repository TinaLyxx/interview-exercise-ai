
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY env.example .env

# Create directory for vector database with proper permissions
RUN mkdir -p /app/data/vector_db && chmod 755 /app/data/vector_db

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OPENAI_MODEL=gpt-3.5-turbo
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV VECTOR_DB_PATH=/app/data/vector_db
ENV DOCS_PATH=/app/data/docs
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MAX_RELEVANT_CHUNKS=5
ENV SIMILARITY_THRESHOLD=0.7


# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
