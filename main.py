#!/usr/bin/env python3
"""
Main entry point for the Shmulik RAG Chatbot
"""
import os
import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.config import get_settings, load_env_file
from src.shmulik.document_processing.pdf_processor import create_pdf_processor
from src.shmulik.vectorstore.chroma_store import create_chroma_store
from src.shmulik.rag_system.langgraph_rag import create_rag_system


class ShmulkChatbot:
    """Main class for the Shmulik RAG Chatbot"""
    
    def __init__(self):
        """Initialize the chatbot system"""
        # Load environment variables and settings
        load_env_file()
        self.settings = get_settings()
        
        # Setup logging
        logger.remove()
        logger.add(
            sys.stderr,
            level=self.settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Initialize components
        self.pdf_processor = None
        self.vectorstore = None
        self.rag_system = None
        
        logger.info("Shmulik Chatbot initialized")
    
    def setup_document_processing(self):
        """Setup document processing pipeline"""
        logger.info("Setting up document processing...")
        
        self.pdf_processor = create_pdf_processor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        
        logger.info("Document processing setup complete")
    
    def setup_vectorstore(self, force_recreate: bool = False):
        """Setup vector store with embeddings"""
        logger.info("Setting up vector store...")
        
        # Ensure vector store directory exists
        vector_store_path = Path(self.settings.vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Create vector store instance
        self.vectorstore = create_chroma_store(
            persist_directory=str(vector_store_path),
            embedding_model=self.settings.embedding_model
        )
        
        # Check if we should load existing or create new
        if force_recreate:
            logger.info("Force recreating vector store...")
            # Delete existing collection if it exists
            try:
                self.vectorstore.delete_collection()
            except:
                pass
            self._create_new_vectorstore()
        else:
            # Try to load existing vectorstore
            existing_store = self.vectorstore.load_existing_vectorstore()
            if existing_store is None:
                logger.info("No existing vector store found, creating new one...")
                self._create_new_vectorstore()
            else:
                logger.info("Loaded existing vector store")
        
        logger.info("Vector store setup complete")
    
    def _create_new_vectorstore(self):
        """Create new vector store from PDF documents"""
        if self.pdf_processor is None:
            self.setup_document_processing()
        
        # Process the PDF document
        pdf_path = Path(self.settings.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        documents = self.pdf_processor.process_pdf(str(pdf_path))
        
        # Create vector store
        logger.info("Creating vector embeddings...")
        self.vectorstore.create_vectorstore(documents)
        
        # Print statistics
        stats = self.pdf_processor.get_document_stats(documents)
        logger.info(f"Processed {stats['total_documents']} document chunks")
        logger.info(f"Total characters: {stats['total_characters']:,}")
        logger.info(f"Total words: {stats['total_words']:,}")
    
    def setup_rag_system(self):
        """Setup the RAG system with LangGraph"""
        logger.info("Setting up RAG system...")
        
        if self.vectorstore is None:
            raise ValueError("Vector store must be initialized first")
        
        # Get retriever from vector store
        retriever = self.vectorstore.get_retriever(search_kwargs={"k": 4})
        
        # Create RAG system
        self.rag_system = create_rag_system(
            vectorstore_retriever=retriever,
            llm_api_key=self.settings.openai_api_key,
            llm_base_url=self.settings.openai_api_base,
            llm_model=self.settings.llm_model,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature
        )
        
        logger.info("RAG system setup complete")
    
    def initialize_system(self, force_recreate_vectorstore: bool = False):
        """Initialize the complete system"""
        logger.info("🚀 Initializing Shmulik RAG Chatbot System...")
        
        try:
            # Setup components in order
            self.setup_document_processing()
            self.setup_vectorstore(force_recreate=force_recreate_vectorstore)
            self.setup_rag_system()
            
            logger.info("✅ System initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            raise
    
    def run_interactive_mode(self):
        """Run the chatbot in interactive command-line mode"""
        logger.info("Starting interactive mode...")
        logger.info("Type 'quit', 'exit', or 'bye' to end the conversation")
        logger.info("Type 'stats' to see system statistics")
        logger.info("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                # Check for stats command
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the query
                print("🤖 Shmulik: ", end="", flush=True)
                result = self.rag_system.query(
                    user_query=user_input,
                    conversation_history=conversation_history
                )
                
                if result["success"]:
                    print(result["response"])
                    
                    # Show retrieved documents info
                    if result["retrieved_docs"]:
                        print(f"\n📚 Referenced {len(result['retrieved_docs'])} document(s)")
                    
                    # Update conversation history
                    conversation_history = result["messages"]
                else:
                    print(f"❌ Error: {result.get('response', 'Unknown error')}")
                
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
    
    def _show_stats(self):
        """Show system statistics"""
        print("\n📊 System Statistics:")
        print("=" * 40)
        
        try:
            # Vector store stats
            if self.vectorstore:
                stats = self.vectorstore.get_collection_stats()
                print(f"Documents in vector store: {stats.get('document_count', 'Unknown')}")
                print(f"Embedding model: {stats.get('embedding_model', 'Unknown')}")
            
            # Settings
            print(f"LLM model: {self.settings.llm_model}")
            print(f"Chunk size: {self.settings.chunk_size}")
            print(f"Chunk overlap: {self.settings.chunk_overlap}")
            
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
        
        print("=" * 40)
    
    def run_streamlit_app(self):
        """Run the Streamlit web interface"""
        logger.info("Starting Streamlit web interface...")
        
        try:
            import streamlit.web.cli as stcli
            import sys
            
            # Path to the Streamlit app
            app_path = project_root / "src" / "shmulik" / "interface" / "streamlit_app.py"
            
            # Configure Streamlit arguments
            sys.argv = [
                "streamlit",
                "run",
                str(app_path),
                "--server.port", str(self.settings.streamlit_server_port),
                "--server.address", self.settings.streamlit_server_address,
                "--server.headless", "false"
            ]
            
            stcli.main()
            
        except ImportError:
            logger.error("Streamlit not installed. Please install with: pip install streamlit")
        except Exception as e:
            logger.error(f"Error starting Streamlit app: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Shmulik RAG Chatbot")
    parser.add_argument(
        "--mode",
        choices=["interactive", "web"],
        default="interactive",
        help="Run mode: interactive CLI or web interface"
    )
    parser.add_argument(
        "--recreate-vectorstore",
        action="store_true",
        help="Force recreation of vector store from PDF"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup the system without running"
    )
    
    args = parser.parse_args()
    
    try:
        # Create chatbot instance
        chatbot = ShmulkChatbot()
        
        # Initialize system
        chatbot.initialize_system(force_recreate_vectorstore=args.recreate_vectorstore)
        
        if args.setup_only:
            logger.info("✅ Setup complete. Exiting.")
            return
        
        # Run based on mode
        if args.mode == "web":
            chatbot.run_streamlit_app()
        else:
            chatbot.run_interactive_mode()
            
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
