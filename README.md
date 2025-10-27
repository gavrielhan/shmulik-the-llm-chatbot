# 🤖 Shmulik - The LLM Chatbot

> **Multilingual RAG Chatbot for Samuel Neaman Institute**  
> Advanced AI assistant with Hebrew/English support and proper RTL formatting

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

### 🌍 **Multilingual Excellence**
- **Hebrew & English** support with proper RTL formatting
- **Dynamic language detection** with automatic text alignment
- **Cross-language retrieval** - English queries find Hebrew sources and vice versa

### 🧠 **Advanced AI Architecture**
- **Superior Embeddings**: `intfloat/multilingual-e5-base` for best-in-class multilingual understanding
- **Enhanced PDF Processing**: PyMuPDF with text cleaning and validation
- **Smart Chunking**: RecursiveCharacterTextSplitter with 1000/200 overlap
- **LangGraph RAG**: Sophisticated retrieval and generation workflow

### 🎨 **Professional UI/UX**
- **Custom Shmulik Branding**: Face avatar in header, half-body in chat
- **Auto-Initialization**: Ready to use immediately - no setup required
- **Responsive Design**: Modern Streamlit interface with custom CSS
- **Hebrew RTL Support**: JavaScript-powered language detection and formatting

### 🔧 **Enterprise Features**
- **Quality Validation**: PDF extraction metrics and chunk validation
- **Error Handling**: Graceful fallbacks and comprehensive logging
- **Configuration Management**: Environment-based settings with Pydantic
- **Modular Architecture**: Clean separation of concerns

## 🚀 Quick Start

### 1. **Clone & Setup**
```bash
git clone https://github.com/gavrielhan/shmulik-the-llm-chatbot.git
cd shmulik-the-llm-chatbot
chmod +x setup.sh
./setup.sh
```

### 2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
nano .env
```

### 3. **Launch Shmulik**
```bash
# Web interface (recommended)
python main.py --mode web

# Or CLI interface
python main.py --mode cli
```

### 4. **Start Chatting**
Open http://localhost:8501 and ask Shmulik anything!

**English**: *"What are the main findings of this research?"*  
**Hebrew**: *"מה הממצאים העיקריים של המחקר?"*

## 🏗️ Architecture

```
shmulik/
├── 📁 src/shmulik/
│   ├── 📁 document_processing/    # PDF extraction & cleaning
│   ├── 📁 vectorstore/           # ChromaDB vector storage
│   ├── 📁 rag_system/            # LangGraph RAG workflow
│   └── 📁 interface/             # Streamlit web UI
├── 📁 config/                    # Configuration management
├── 📁 assets/                    # Shmulik avatars
└── 📄 main.py                    # Application entry point
```

## ⚙️ Configuration

### **Environment Variables** (`.env`)
```bash
# LLM Configuration
API_KEY=your_api_key_here
OPENAI_API_BASE=your_api_base_url_here
LLM_MODEL=openai/gpt-4.1-mini

# Embedding Model
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# PDF Processing
PDF_PATH=./Report_Digital-Health-Literacy-among-Students_v1.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### **Customization** (`config/config.py`)
- LLM model and API settings
- Vector store parameters  
- PDF processing options
- UI preferences and styling

## 📊 Performance

- **PDF Processing**: 31 pages → 71 chunks in ~2 seconds
- **Vector Store**: 8,525 words indexed with validation
- **Response Time**: ~2-3 seconds per query
- **Memory Usage**: ~500MB with full model loaded

## 🛠️ Development

### **Requirements**
- Python 3.11+
- Conda environment
- 4GB+ RAM recommended
- API key for LLM service

### **Key Dependencies**
- `langchain` & `langgraph` - RAG framework
- `sentence-transformers` - Multilingual embeddings
- `chromadb` - Vector storage
- `streamlit` - Web interface
- `pymupdf` - Enhanced PDF processing

### **Installation**
```bash
# Create conda environment
conda create -n shmulik-rag python=3.11
conda activate shmulik-rag

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Use Cases

- **Research Assistant**: Query academic papers and reports
- **Multilingual Support**: Hebrew and English content
- **Policy Analysis**: Samuel Neaman Institute research
- **Educational Tool**: Digital health literacy insights

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**Built with ❤️ for the Samuel Neaman Institute for National Policy Research**