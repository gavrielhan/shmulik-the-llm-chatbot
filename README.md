# Shmulik - AI RAG Chatbot

A clean, modern AI chatbot for the Samuel Neaman Institute that answers questions about the Digital Health Literacy research report.

## 🚀 **Architecture**

- **Frontend**: Pure HTML/CSS/JavaScript (no frameworks)
- **Backend**: FastAPI with Python
- **RAG System**: LangGraph + ChromaDB + multilingual embeddings
- **LLM**: OpenAI GPT-4.1-mini via LiteLLM

## 📁 **Project Structure**

```
shmulik/
├── frontend/                 # Clean HTML/CSS/JS frontend
│   ├── index.html           # Main chat interface
│   ├── style.css            # Facebook-style chat styling
│   ├── chat.js              # Chat functionality
│   └── assets/              # Images (background, avatars)
├── backend/                 # FastAPI backend
│   └── main.py              # API server with RAG endpoints
├── src/shmulik/             # Core RAG system
│   ├── document_processing/ # PDF processing
│   ├── vectorstore/         # ChromaDB integration
│   └── rag_system/          # LangGraph RAG workflow
└── config/                  # Configuration settings
```

## 🛠️ **Quick Start**

### 1. **Setup Environment**
```bash
conda create -n shmulik-rag python=3.11
conda activate shmulik-rag
pip install -r requirements.txt
```

### 2. **Configure API Key**
Create `.env` file:
```
API_KEY=your_openai_api_key_here
```

### 3. **Start Services**

**Terminal 1 - Backend:**
```bash
cd /path/to/shmulik
conda activate shmulik-rag
python backend/main.py
```

**Terminal 2 - Frontend:**
```bash
cd /path/to/shmulik/frontend
python -m http.server 3000
```

### 4. **Access Chat**
Open browser: **http://localhost:3000**

## ✨ **Features**

- ✅ **Clean UI** - Facebook-style chat widget
- ✅ **Multilingual** - Handles English and Hebrew
- ✅ **RAG-powered** - Answers from PDF content
- ✅ **Fast & Responsive** - No page reloads
- ✅ **Easy to Deploy** - Simple HTML + API

## 🔧 **API Endpoints**

- `GET /health` - Health check
- `POST /chat` - Send message to Shmulik

## 📊 **Technical Details**

- **Embedding Model**: `intfloat/multilingual-e5-base`
- **Vector Store**: ChromaDB
- **RAG Framework**: LangGraph
- **PDF Processing**: PyMuPDF + LangChain
- **Frontend**: Pure HTML/CSS/JavaScript
- **Backend**: FastAPI + Uvicorn

## 🎯 **Usage**

1. Ask questions about the Digital Health Literacy research
2. Shmulik responds with relevant information from the PDF
3. Supports both English and Hebrew questions
4. Maintains conversation context

## 📝 **Development**

The system is designed to be:
- **Simple** - No complex frameworks
- **Fast** - Direct API communication
- **Reliable** - Clean separation of concerns
- **Maintainable** - Easy to modify and extend