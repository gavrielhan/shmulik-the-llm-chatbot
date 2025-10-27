#!/bin/bash

# Shmulik RAG Chatbot Setup Script
echo "🤖 Setting up Shmulik RAG Chatbot..."
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for conda
if ! command_exists conda; then
    echo "❌ Error: conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Found conda installation"

# Check if environment already exists
if conda env list | grep -q "shmulik-rag"; then
    echo "⚠️  Environment 'shmulik-rag' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n shmulik-rag -y
    else
        echo "📦 Using existing environment..."
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "shmulik-rag"; then
    echo "📦 Creating conda environment 'shmulik-rag' with Python 3.11..."
    conda create -n shmulik-rag python=3.11 -y
fi

echo "🔧 Activating environment and installing dependencies..."

# Activate environment and install packages
eval "$(conda shell.bash hook)"
conda activate shmulik-rag

# Install dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/vectorstore
mkdir -p logs

# Check for PDF file
if [[ -f "Report_Digital-Health-Literacy-among-Students_v1.pdf" ]]; then
    echo "✅ Found PDF document"
else
    echo "⚠️  PDF document not found: Report_Digital-Health-Literacy-among-Students_v1.pdf"
    echo "   Please ensure the PDF file is in the project root directory"
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo "📝 Creating .env template file..."
    cat > .env << 'EOF'
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

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EOF
    echo "⚠️  Please edit .env file with your actual API key!"
else
    echo "✅ Found existing .env file"
fi

echo ""
echo "🎉 Setup complete! Next steps:"
echo "=================================="
echo "1. Edit .env file with your API key:"
echo "   nano .env"
echo ""
echo "2. Activate the environment:"
echo "   conda activate shmulik-rag"
echo ""
echo "3. Run the chatbot:"
echo "   # Interactive mode:"
echo "   python main.py --mode interactive"
echo ""
echo "   # Web interface:"
echo "   python main.py --mode web"
echo ""
echo "   # Setup only (no run):"
echo "   python main.py --setup-only"
echo ""
echo "4. For help:"
echo "   python main.py --help"
echo ""
echo "📚 Read README.md for more detailed instructions!"
echo "🤖 Shmulik is ready to help with digital health literacy research!"
