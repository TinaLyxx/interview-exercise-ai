# 🧠 Knowledge Assistant for Support Team

A production-ready **LLM-powered RAG system** that helps support teams respond to customer tickets efficiently using relevant documentation. Built with FastAPI, OpenAI GPT, and FAISS vector database.

## 🎯 Overview

The Knowledge Assistant analyzes customer support queries and returns structured, relevant responses using a Retrieval-Augmented Generation (RAG) pipeline. It follows the Model Context Protocol (MCP) to produce consistent, actionable responses.

### Example Input/Output

**Input (Support Ticket):**
```json
{
  "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
}
```

**Output (MCP-compliant JSON):**
```json
{
  "answer": "I understand that your domain has been suspended and you did not receive any notice. To reactivate y...",
  "references": ["'support_faqs.md', 'escalation_procedures.md', 'domain_policies.md'"],
  "action_required": "escalate_to_abuse_team"
}
```


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))
- Docker/Podman (for containerized deployment)


### 1. Clone and Setup

```bash
git clone <repo-url>
cd interview-exercise-ai
```

### 2. Environment Configuration

```bash
cp env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_openai_api_key_here
```
**⚠️ Important:** Replace the placeholder API key with your actual OpenAI API key to avoid rate limiting issues.



### 3. Running the Application
#### Option A: Using Podman (Recommended)

```bash
# Start Podman machine (if on macOS)
podman machine start

# Build and run with Podman
podman build -t knowledge-assistant .
podman run -d --replace --name knowledge-assistant -p 8000:8000 --env-file .env knowledge-assistant

# The API will be available at http://localhost:8000

```

#### Option B: Using Docker

```bash
# Build and run with Docker
docker build -t knowledge-assistant .
docker rm -f knowledge-assistant 2>/dev/null || true
docker run -d --name knowledge-assistant -p 8000:8000 --env-file .env knowledge-assistant

# The API will be available at http://localhost:8000
```

#### Option C: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py

```

### 4. Testing the API

Run the comprehensive test suite:
```bash
python3 example_usage.py
```

This will test:
- ✅ Health check endpoint
- ✅ 5 sample support tickets
- ✅ System statistics
- ✅ Knowledge base rebuild

Or you can test the API with a simple support ticket:
```bash
curl -X POST "http://localhost:8000/resolve-ticket" \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "Test message"}'| jq
```



## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and available endpoints |
| `POST` | `/resolve-ticket` | Main endpoint for ticket resolution |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | System statistics |
| `POST` | `/rebuild-knowledge-base` | Rebuild the knowledge base |

### Health Checks

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed system statistics
curl http://localhost:8000/stats
```

### Tickets Resolutions

Call `/resolve-ticket` to resolve a customer support ticket using RAG and LLM.

**Request:**
```json
{
  "ticket_text": "Customer support query text"
}
```

**Response:**
```json
{
  "answer": "Generated response",
  "references": ["Source references"],
  "action_required": "escalation_action"
}
```

### Adding New Documents

1. Add markdown files to `data/docs/`
2. Restart the application or call `/rebuild-knowledge-base`
```bash
curl http://localhost:8000/rebuild-knowledge-base
```
3. The system will automatically process and index new content




## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `VECTOR_DB_PATH` | `./data/vector_db` | Path to store vector database |
| `DOCS_PATH` | `./data/docs` | Path to documentation files |
| `MAX_RELEVANT_CHUNKS` | `5` | Maximum chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity score (0.0-1.0) |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

### Tuning Recommendations

- **Similarity Threshold**: Lower values (0.3-0.5) return more results but may include less relevant content
- **Max Relevant Chunks**: 3-7 chunks usually provide good context without overwhelming the LLM
- **Embedding Model**: `all-MiniLM-L6-v2` is fast and effective; `all-mpnet-base-v2` is more accurate but slower

### 🔑 Getting Your OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Click "Create new secret key"
5. Copy the key and paste it in your `.env` file




## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest src/tests/test_api.py -v
```




## 🔧 Technical Details

### 🏗️ Architecture

The Knowledge Assistant uses a modular RAG architecture:

```
📦 Knowledge Assistant
├── 🔍 Document Processing (Markdown → Chunks)
├── 🧠 Embedding Generation (Sentence Transformers)
├── 📊 Vector Storage (FAISS)
├── 🔎 Retrieval System (Semantic Search)
├── 🤖 LLM Integration (OpenAI GPT)
├── 🌐 API Layer (FastAPI)
└── 🧪 Testing Suite (Pytest)
```

### 📁 Project Structure

```
interview-exercise-ai/
├── src/
│   ├── api/                    # FastAPI application
│   │   └── main.py            # Main API endpoints
│   ├── models/                 # Pydantic models
│   │   └── schemas.py         # Request/response schemas
│   ├── rag/                   # RAG pipeline components
│   │   ├── document_processor.py  # Document processing
│   │   ├── embeddings.py      # Embedding generation
│   │   ├── knowledge_assistant.py # Main orchestrator
│   │   ├── llm_client.py      # OpenAI integration
│   │   ├── retriever.py       # Document retrieval
│   │   └── vector_store.py    # FAISS vector database
│   ├── tests/                 # Unit tests
│   │   ├── test_api.py
│   │   ├── test_document_processor.py
│   │   ├── test_embeddings.py
│   │   └── test_vector_store.py
│   └── utils/                 # Utilities
│       └── config.py          # Configuration management
├── data/
│   ├── docs/                  # Documentation files
│   │   ├── domain_policies.md
│   │   ├── escalation_procedures.md
│   │   └── support_faqs.md
│   └── vector_db/             # Vector database storage
├── example_usage.py           # Comprehensive test script
├── run.py                     # Startup script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── env.example               # Environment variables template
└── README.md                 # This file
```

### RAG Pipeline Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│ DocumentProcessor│───▶│  DocumentChunks │
│   (Markdown)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│EmbeddingGenerator│───▶│   Embeddings    │
│                 │    │ (Sentence Trans.)│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  LLM Response   │◀───│   LLM Client     │◀───│  Retrieved Docs │
│  (Structured)   │    │   (OpenAI GPT)   │    │   (FAISS)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **Document Processor**: Parses and chunks documentation
2. **Embedding Generator**: Creates vector embeddings using Sentence Transformers
3. **Vector Store**: FAISS-based similarity search
4. **Document Retriever**: Combines search with relevance filtering
5. **LLM Client**: OpenAI GPT integration with MCP prompting
6. **Knowledge Assistant**: Main orchestration service
7. **API**: FastAPI REST endpoints


### Model Context Protocol (MCP)

The system uses structured prompting with:
- **Clear Role Definition**: AI assistant for domain support
- **Context Injection**: Retrieved documentation snippets
- **Task Specification**: Analyze ticket and generate structured response
- **Output Schema**: JSON with answer, references, and action_required

### Action Types

The system can recommend the following actions:
- `escalate_to_technical_team`: Technical issues requiring specialist help
- `escalate_to_abuse_team`: Policy violations, suspensions, security issues
- `escalate_to_billing_team`: Payment, billing, and account issues
- `escalate_to_management`: Customer complaints, policy exceptions
- `escalate_to_legal_team`: Legal disputes, compliance issues
- `contact_customer_directly`: Urgent issues requiring immediate contact
- `no_action_required`: Issue resolved with provided information




## 📄 License


This project is licensed under the [MIT License] © 2025 TinaLxx.

---
