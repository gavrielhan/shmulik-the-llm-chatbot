"""
Streamlit Interface for Shmulik RAG Chatbot
"""
import streamlit as st
import os
import sys
from typing import List, Dict, Any
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

from config.config import get_settings, load_env_file
from src.shmulik.document_processing.pdf_processor import create_pdf_processor
from src.shmulik.vectorstore.chroma_store import create_chroma_store
from src.shmulik.rag_system.langgraph_rag import create_rag_system
from langchain_core.messages import HumanMessage, AIMessage


class ShmulkStreamlitApp:
    """Streamlit application for Shmulik RAG Chatbot"""
    
    def __init__(self):
        # Load environment variables first
        load_env_file()
        self.settings = get_settings()
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        # Use custom avatar as page icon if available
        avatar_path = os.path.join(project_root, "assets", "shmulik.png")
        page_icon = avatar_path if os.path.exists(avatar_path) else "🤖"
        
        st.set_page_config(
            page_title="Shmulik - Samuel Neaman Institute AI Assistant",
            page_icon=page_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for avatars and proper Hebrew RTL support
        st.markdown("""
        <style>
        /* Make chat message avatars bigger for half-body image */
        .stChatMessage > div:first-child img {
            width: 80px !important;
            height: 80px !important;
            border-radius: 15% !important;
            border: 2px solid #f0f2f6 !important;
            object-fit: cover !important;
        }
        
        /* Better spacing for chat messages */
        .stChatMessage {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Enhance avatar visibility with shadow */
        .stChatMessage [data-testid="chatAvatarIcon-assistant"] img {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        
        /* Special styling for assistant messages */
        .stChatMessage[data-testid="chat-message-assistant"] {
            background-color: #f8f9fa !important;
            border-left: 4px solid #4CAF50 !important;
        }
        
        /* Hebrew RTL Support */
        .stChatMessage .stMarkdown {
            direction: auto !important;
            text-align: start !important;
        }
        
        /* Detect Hebrew text and apply RTL */
        .stChatMessage .stMarkdown p:has-text([\u0590-\u05FF]) {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif !important;
        }
        
        /* General RTL support for any content with Hebrew characters */
        .stChatMessage .stMarkdown *:lang(he) {
            direction: rtl !important;
            text-align: right !important;
        }
        
        /* Numbered lists in Hebrew should be RTL */
        .stChatMessage .stMarkdown ol, 
        .stChatMessage .stMarkdown ul {
            direction: auto !important;
            text-align: start !important;
        }
        
        /* Mixed content handling */
        .stChatMessage .stMarkdown {
            unicode-bidi: plaintext !important;
        }
        
        /* Force RTL for Hebrew-heavy content */
        .hebrew-rtl {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif !important;
        }
        
        /* Force LTR for English-heavy content */
        .english-ltr {
            direction: ltr !important;
            text-align: left !important;
        }
        </style>
        
        <script>
        // Hebrew RTL detection and formatting
        function applyRTLFormatting() {
            // Hebrew Unicode range: \u0590-\u05FF
            const hebrewRegex = /[\u0590-\u05FF]/g;
            
            setTimeout(() => {
                const chatMessages = document.querySelectorAll('.stChatMessage .stMarkdown');
                
                chatMessages.forEach(message => {
                    const text = message.innerText || message.textContent || '';
                    const hebrewMatches = text.match(hebrewRegex);
                    const hebrewChars = hebrewMatches ? hebrewMatches.length : 0;
                    const totalChars = text.replace(/\s/g, '').length;
                    
                    // If more than 30% Hebrew characters, apply RTL
                    if (totalChars > 0 && (hebrewChars / totalChars) > 0.3) {
                        message.classList.add('hebrew-rtl');
                        message.style.direction = 'rtl';
                        message.style.textAlign = 'right';
                    } else if (totalChars > 0) {
                        message.classList.add('english-ltr');
                        message.style.direction = 'ltr';
                        message.style.textAlign = 'left';
                    }
                });
            }, 100);
        }
        
        // Apply formatting on page load and when new messages appear
        document.addEventListener('DOMContentLoaded', applyRTLFormatting);
        
        // Observer for dynamically added content
        const observer = new MutationObserver(applyRTLFormatting);
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        # Auto-initialize system on first run
        if not st.session_state.system_initialized:
            self.auto_initialize_system()
    
    def render_sidebar(self):
        """Render the sidebar with system controls and information"""
        with st.sidebar:
            # Sidebar title with custom avatar
            avatar_path = os.path.join(project_root, "assets", "shmulik.png")
            if os.path.exists(avatar_path):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(avatar_path, width=40)
                with col2:
                    st.title("Shmulik Settings")
            else:
                st.title("🤖 Shmulik Settings")
            st.markdown("---")
            
            # System initialization section
            st.subheader("📚 Document Processing")
            
            if not st.session_state.system_initialized:
                if st.button("🚀 Initialize System", key="init_btn"):
                    with st.spinner("Initializing Shmulik RAG system..."):
                        self.initialize_rag_system()
            else:
                st.success("✅ System initialized!")
                if st.button("🔄 Reinitialize System", key="reinit_btn"):
                    with st.spinner("Reinitializing system..."):
                        self.reinitialize_system()
            
            st.markdown("---")
            
            # System information
            st.subheader("ℹ️ System Information")
            st.info(f"**Model**: {self.settings.llm_model}")
            st.info(f"**Embedding Model**: {self.settings.embedding_model}")
            st.info(f"**Chunk Size**: {self.settings.chunk_size}")
            
            if st.session_state.vectorstore:
                try:
                    stats = st.session_state.vectorstore.get_collection_stats()
                    st.info(f"**Documents**: {stats.get('document_count', 0)}")
                except:
                    pass
            
            st.markdown("---")
            
            # Conversation controls
            st.subheader("💬 Conversation")
            if st.button("🗑️ Clear History", key="clear_btn"):
                st.session_state.conversation_history = []
            
            # Export conversation
            if st.session_state.conversation_history:
                if st.button("📥 Export Chat", key="export_btn"):
                    self.export_conversation()
    
    def auto_initialize_system(self):
        """Auto-initialize the system on startup"""
        with st.spinner("🚀 Initializing Shmulik automatically..."):
            try:
                self.initialize_rag_system()
            except Exception as e:
                st.error(f"Auto-initialization failed: {str(e)}")
    
    def initialize_rag_system(self):
        """Initialize the RAG system components"""
        try:
            # Load environment variables
            load_env_file()
            
            # Step 1: Process PDF document
            st.info("📄 Processing PDF document...")
            pdf_processor = create_pdf_processor(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            
            pdf_path = os.path.join(project_root, self.settings.pdf_path)
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found: {pdf_path}")
                return
            
            documents = pdf_processor.process_pdf(pdf_path)
            
            # Step 2: Initialize vector store
            st.info("🔍 Creating vector embeddings...")
            vectorstore = create_chroma_store(
                persist_directory=os.path.join(project_root, self.settings.vector_store_path),
                embedding_model=self.settings.embedding_model
            )
            
            # Try to load existing vectorstore first
            existing_store = vectorstore.load_existing_vectorstore()
            if existing_store is None:
                # Create new vectorstore
                vectorstore.create_vectorstore(documents)
            
            st.session_state.vectorstore = vectorstore
            
            # Step 3: Initialize RAG system
            st.info("🧠 Initializing RAG system...")
            retriever = vectorstore.get_retriever(search_kwargs={"k": 4})
            
            rag_system = create_rag_system(
                vectorstore_retriever=retriever,
                llm_api_key=self.settings.openai_api_key,
                llm_base_url=self.settings.openai_api_base,
                llm_model=self.settings.llm_model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature
            )
            
            st.session_state.rag_system = rag_system
            st.session_state.system_initialized = True
            st.session_state.documents_loaded = True
            
            st.success("🎉 Shmulik is ready to help!")
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            st.exception(e)
    
    def reinitialize_system(self):
        """Reinitialize the system (clear cache and reload)"""
        st.session_state.rag_system = None
        st.session_state.vectorstore = None
        st.session_state.system_initialized = False
        st.session_state.documents_loaded = False
        st.session_state.conversation_history = []
        self.initialize_rag_system()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Custom title with Shmulik face avatar
        avatar_path = os.path.join(project_root, "assets", "shmulik.png")
        if os.path.exists(avatar_path):
            col1, col2 = st.columns([1, 10])
            with col1:
                st.image(avatar_path, width=80)
            with col2:
                st.title("Shmulik - Your Guide to My Website")
        else:
            st.title("🤖 Shmulik - Your Guide to My Website")
        st.markdown("*Ask me anything about the Digital Health Literacy research report!*")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ Please initialize the system using the sidebar controls.")
            st.info("👈 Click 'Initialize System' in the sidebar to get started.")
            return
        
        # Display conversation history
        self.display_conversation_history()
        
        # Chat input
        user_input = st.chat_input("Ask Shmulik a question about digital health literacy...")
        
        if user_input:
            self.process_user_input(user_input)
    
    def display_conversation_history(self):
        """Display the conversation history"""
        # Path to custom avatar for chat responses
        avatar_path = os.path.join(project_root, "assets", "half body shmulik.png")
        
        for message in st.session_state.conversation_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                # Use custom avatar if it exists, otherwise fallback to emoji
                if os.path.exists(avatar_path):
                    with st.chat_message("assistant", avatar=avatar_path):
                        st.write(message.content)
                else:
                    with st.chat_message("assistant"):
                        st.write(message.content)
    
    def process_user_input(self, user_input: str):
        """Process user input and generate response"""
        try:
            # Add user message to history
            user_message = HumanMessage(content=user_input)
            st.session_state.conversation_history.append(user_message)
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response with half-body Shmulik avatar
            avatar_path = os.path.join(project_root, "assets", "half body shmulik.png")
            
            # Use custom avatar if it exists, otherwise fallback to default
            if os.path.exists(avatar_path):
                with st.chat_message("assistant", avatar=avatar_path):
                    with st.spinner("Shmulik is thinking..."):
                        # Query the RAG system
                        result = st.session_state.rag_system.query(
                            user_query=user_input,
                            conversation_history=st.session_state.conversation_history[:-1]  # Exclude current message
                        )
                        
                        if result["success"]:
                            response = result["response"]
                            st.write(response)
                            
                            # Show retrieved documents in expander
                            if result["retrieved_docs"]:
                                with st.expander(f"📚 Referenced Documents ({len(result['retrieved_docs'])})"):
                                    for i, doc in enumerate(result["retrieved_docs"]):
                                        st.markdown(f"**Document {i+1}** (Page {doc.metadata.get('page', 'Unknown')})")
                                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                        st.markdown("---")
                            
                            # Add AI message to history
                            ai_message = AIMessage(content=response)
                            st.session_state.conversation_history.append(ai_message)
                            
                        else:
                            error_msg = result.get("response", "Sorry, I encountered an error.")
                            st.error(error_msg)
                            
                            # Add error message to history
                            ai_message = AIMessage(content=error_msg)
                            st.session_state.conversation_history.append(ai_message)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Shmulik is thinking..."):
                        # Query the RAG system
                        result = st.session_state.rag_system.query(
                            user_query=user_input,
                            conversation_history=st.session_state.conversation_history[:-1]  # Exclude current message
                        )
                        
                        if result["success"]:
                            response = result["response"]
                            st.write(response)
                            
                            # Show retrieved documents in expander
                            if result["retrieved_docs"]:
                                with st.expander(f"📚 Referenced Documents ({len(result['retrieved_docs'])})"):
                                    for i, doc in enumerate(result["retrieved_docs"]):
                                        st.markdown(f"**Document {i+1}** (Page {doc.metadata.get('page', 'Unknown')})")
                                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                        st.markdown("---")
                            
                            # Add AI message to history
                            ai_message = AIMessage(content=response)
                            st.session_state.conversation_history.append(ai_message)
                            
                        else:
                            error_msg = result.get("response", "Sorry, I encountered an error.")
                            st.error(error_msg)
                            
                            # Add error message to history
                            ai_message = AIMessage(content=error_msg)
                            st.session_state.conversation_history.append(ai_message)
            
            # Display updates automatically - no rerun needed
            
        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            st.exception(e)
    
    def export_conversation(self):
        """Export conversation history to JSON"""
        try:
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": [
                    {
                        "type": "human" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()
                    }
                    for msg in st.session_state.conversation_history
                ]
            }
            
            st.download_button(
                label="📥 Download Conversation",
                data=json.dumps(conversation_data, indent=2),
                file_name=f"shmulik_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error exporting conversation: {str(e)}")
    
    def run(self):
        """Run the Streamlit application"""
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main function to run the Streamlit app"""
    app = ShmulkStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
