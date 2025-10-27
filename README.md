# 🤖 Shmulik - Samuel Neaman Institute RAG Chatbot

Shmulik is an intelligent RAG (Retrieval-Augmented Generation) chatbot designed for the Samuel Neaman Institute for National Policy Research. It specializes in providing information about digital health literacy research using advanced AI techniques.

## 🌟 Features

- **📚 PDF Document Processing**: Automatically processes and chunks PDF documents using LangChain
- **🔍 Vector Search**: Uses ChromaDB and sentence transformers for semantic document retrieval
- **🧠 LangGraph RAG System**: Advanced retrieval-augmented generation using LangGraph workflows
- **💬 Interactive Interfaces**: Both command-line and web-based (Streamlit) interfaces
- **🔧 Configurable**: Easily configurable through environment variables and settings
- **🏥 Domain-Specific**: Specialized for digital health literacy and policy research

## 🛠️ Architecture

The system consists of several key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Document  │───▶│  Document        │───▶│  Vector Store   │
│   Processing    │    │  Chunking        │    │  (ChromaDB)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   User Query    │───▶│  LangGraph       │◄────────────┘
└─────────────────┘    │  RAG System      │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │  LLM Response    │
                       │  Generation      │
                       └──────────────────┘
```

### Components:

1. **Document Processing (`src/shmulik/document_processing/`)**: PDF loading and text chunking
2. **Vector Store (`src/shmulik/vectorstore/`)**: ChromaDB integration with embeddings
3. **RAG System (`src/shmulik/rag_system/`)**: LangGraph-based retrieval and generation
4. **Interface (`src/shmulik/interface/`)**: Streamlit web interface
5. **Configuration (`config/`)**: Environment and application settings

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended)
- API access to https://litellm.sph-prod.ethz.ch/

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n shmulik-rag python=3.11 -y
conda activate shmulik-rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://litellm.sph-prod.ethz.ch/v1
LLM_MODEL=openai/gpt-4.1-mini

# Vector Store Configuration  
VECTOR_STORE_PATH=./data/vectorstore
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Application Configuration
APP_NAME=Shmulik RAG Chatbot
LOG_LEVEL=INFO
MAX_TOKENS=2048
TEMPERATURE=0.7

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### 3. Run the System

#### Option A: Interactive Command-Line Interface

```bash
# Initialize and run interactive mode
python main.py --mode interactive

# Force recreate vector store if needed
python main.py --mode interactive --recreate-vectorstore
```

#### Option B: Web Interface (Streamlit)

```bash
# Launch Streamlit web app
python main.py --mode web

# Or directly with streamlit
streamlit run src/shmulik/interface/streamlit_app.py
```

#### Option C: Setup Only

```bash
# Just setup the system without running
python main.py --setup-only
```

## 📖 Usage

### Interactive Mode

```bash
🚀 Initializing Shmulik RAG Chatbot System...
✅ System initialization complete!
Type 'quit', 'exit', or 'bye' to end the conversation
Type 'stats' to see system statistics
============================================================

🤔 You: What are the key findings about digital health literacy?

🤖 Shmulik: Based on the research report, the key findings about digital health literacy include...

📚 Referenced 3 document(s)
```

### Web Interface

1. Open your browser to `http://localhost:8501`
2. Click "🚀 Initialize System" in the sidebar
3. Start chatting with Shmulik!

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for LLM access | Required |
| `OPENAI_API_BASE` | Base URL for API | `https://litellm.sph-prod.ethz.ch/v1` |
| `LLM_MODEL` | Model name to use | `openai/gpt-4.1-mini` |
| `VECTOR_STORE_PATH` | Path to store vectors | `./data/vectorstore` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `MAX_TOKENS` | Max response tokens | `2048` |
| `TEMPERATURE` | Response randomness | `0.7` |

### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2` by default for creating embeddings. You can change this in the configuration.

## 📁 Project Structure

```
shmulik/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── Report_Digital-Health-Literacy-among-Students_v1.pdf  # Source document
├── config/
│   ├── __init__.py
│   └── config.py                   # Configuration settings
├── src/
│   └── shmulik/
│       ├── __init__.py
│       ├── document_processing/    # PDF processing
│       │   ├── __init__.py
│       │   └── pdf_processor.py
│       ├── vectorstore/           # Vector storage
│       │   ├── __init__.py
│       │   └── chroma_store.py
│       ├── rag_system/            # RAG implementation
│       │   ├── __init__.py
│       │   └── langgraph_rag.py
│       └── interface/             # User interfaces
│           ├── __init__.py
│           └── streamlit_app.py
├── data/
│   └── vectorstore/               # Vector database storage
└── logs/                          # Application logs
```

## 🔬 Technical Details

### Document Processing

- Uses LangChain's `PyPDFLoader` for PDF processing
- Implements `RecursiveCharacterTextSplitter` for intelligent text chunking
- Preserves document metadata including page numbers and source information

### Vector Storage

- ChromaDB for persistent vector storage
- Sentence Transformers for embedding generation
- Configurable similarity search with metadata filtering

### RAG System

- LangGraph for workflow orchestration
- Two-node workflow: retrieval → generation
- Custom prompts optimized for the Samuel Neaman Institute domain
- Error handling and conversation history management

### LLM Integration

- Supports custom API endpoints (configured for litellm.sph-prod.ethz.ch)
- Compatible with OpenAI API format
- Configurable model parameters (temperature, max_tokens, etc.)

## 🛠️ Development

### Adding New Features

1. **New Document Types**: Extend `pdf_processor.py` or create new processors
2. **Custom Retrievers**: Implement new retrieval strategies in the RAG system
3. **Interface Enhancements**: Modify the Streamlit app or add new interfaces
4. **Advanced Workflows**: Extend the LangGraph workflow with additional nodes

### Testing

```bash
# Run tests (when available)
pytest tests/

# Check code formatting
black src/ config/
isort src/ config/
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your API key is correctly set in the `.env` file
   - Verify access to the litellm endpoint

2. **Vector Store Issues**
   - Delete the `data/vectorstore` directory to recreate
   - Use `--recreate-vectorstore` flag when running

3. **PDF Processing Issues**
   - Ensure the PDF file exists in the project root
   - Check file permissions

4. **Memory Issues**
   - Reduce `CHUNK_SIZE` for large documents
   - Lower the number of retrieved documents (`k` parameter)

### Logs

Check the application logs for detailed error information:
- Interactive mode: Logs printed to console
- Streamlit mode: Check the Streamlit console output

## 📄 License

This project is developed for the Samuel Neaman Institute for National Policy Research.

## 🤝 Contributing

For contributions or issues, please contact the development team.

## 📞 Support

For technical support or questions about the system, please refer to the documentation or contact the development team.

---

**Shmulik** - Your intelligent assistant for digital health literacy research! 🤖📚
