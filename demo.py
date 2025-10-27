#!/usr/bin/env python3
"""
Demo script to test Shmulik RAG Chatbot imports and basic functionality
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all main components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from config.config import get_settings, load_env_file
        print("✅ Configuration module imported successfully")
        
        from src.shmulik.document_processing.pdf_processor import create_pdf_processor
        print("✅ PDF processor module imported successfully")
        
        from src.shmulik.vectorstore.chroma_store import create_chroma_store
        print("✅ Vector store module imported successfully")
        
        from src.shmulik.rag_system.langgraph_rag import create_rag_system
        print("✅ RAG system module imported successfully")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config.config import get_settings
        settings = get_settings()
        
        print(f"✅ App Name: {settings.app_name}")
        print(f"✅ LLM Model: {settings.llm_model}")
        print(f"✅ Embedding Model: {settings.embedding_model}")
        print(f"✅ Chunk Size: {settings.chunk_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

def test_pdf_exists():
    """Check if PDF file exists"""
    print("\n📄 Testing PDF file...")
    
    pdf_path = project_root / "Report_Digital-Health-Literacy-among-Students_v1.pdf"
    if pdf_path.exists():
        print(f"✅ PDF file found: {pdf_path.name}")
        print(f"   Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        return False

def show_project_structure():
    """Show the project structure"""
    print("\n📁 Project Structure:")
    print("=" * 50)
    
    def print_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        items = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                print_tree(item, prefix + extension, max_depth, current_depth + 1)
    
    print_tree(project_root, max_depth=4)

def main():
    """Run all tests"""
    print("🤖 Shmulik RAG Chatbot - System Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        return False
    
    # Test configuration
    if not test_configuration():
        return False
    
    # Test PDF file
    pdf_exists = test_pdf_exists()
    
    # Show project structure
    show_project_structure()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("✅ All Python modules can be imported")
    print("✅ Configuration system working")
    if pdf_exists:
        print("✅ PDF document ready for processing")
    else:
        print("⚠️  PDF document missing (needs to be added)")
    
    print("\n🚀 Next Steps:")
    print("1. Add your API key to the .env file")
    print("2. Ensure the PDF file is in the project root")
    print("3. Run: python main.py --setup-only")
    print("4. Run: python main.py --mode interactive")
    print("   or: python main.py --mode web")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
