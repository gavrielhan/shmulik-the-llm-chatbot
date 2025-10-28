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
            initial_sidebar_state="collapsed"
        )
        
        # Apply CSS
        self.apply_css()
    
    def apply_css(self):
        """Apply Facebook-style chat CSS"""
        background_base64 = self.get_avatar_base64("background.png")
        # Create CSS with background image
        css_content = f"""
        <style>
        /* Background image */
        .stApp {{
            background-image: url('data:image/png;base64,{background_base64}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }}
        </style>
        """
        
        # Add the rest of the CSS as a regular string
        css_content += """
        <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container styling */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
            margin: 0 !important;
        }
        
        /* Ensure main content doesn't interfere with chat */
        .main {
            padding-right: 400px !important;
        }
        
        /* Facebook-style chat container */
        .facebook-chat-container {
            position: fixed !important;
            right: 20px !important;
            bottom: 20px !important;
            width: 350px !important;
            height: 500px !important;
            background: white !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
            display: flex !important;
            flex-direction: column !important;
            z-index: 9999 !important;
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Chat header */
        .chat-header {
            background: #1877f2;
            color: white;
            padding: 12px 16px;
            border-radius: 12px 12px 0 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .chat-header img {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
        }
        
        .chat-header-info h3 {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }
        
        .chat-header-info p {
            margin: 0;
            font-size: 12px;
            opacity: 0.9;
        }
        
        /* Chat messages area */
        .chat-messages {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            background: #f8f9fa;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        /* Individual message styling */
        .message {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            max-width: 80%;
        }
        
        .message.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }
        
        .message.assistant {
            align-self: flex-start;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            object-fit: cover;
            flex-shrink: 0;
        }
        
        .message-content {
            background: white;
            padding: 8px 12px;
            border-radius: 18px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: #1877f2;
            color: white;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
        }
        
        /* Chat input area */
        .chat-input {
            padding: 16px;
            background: white;
            border-radius: 0 0 12px 12px;
            border-top: 1px solid #e4e6ea;
        }
        
        .input-container {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f0f2f5;
            border-radius: 20px;
            padding: 8px 12px;
        }
        
        .input-container input {
            flex: 1;
            border: none;
            background: transparent;
            outline: none;
            font-size: 14px;
            padding: 4px 8px;
        }
        
        .input-container input::placeholder {
            color: #8a8d91;
        }
        
        .mic-button, .send-button {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            width: 32px;
            height: 32px;
        }
        
        .mic-button:hover, .send-button:hover {
            background: #e4e6ea;
        }
        
        .send-button {
            background: #1877f2;
            color: white;
        }
        
        .send-button:hover {
            background: #166fe5;
        }
        
        /* Hebrew RTL Support */
        .message-content {
            direction: auto !important;
            text-align: start !important;
        }
        
        .hebrew-rtl {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif !important;
        }
        
        .english-ltr {
            direction: ltr !important;
            text-align: left !important;
        }
        
        /* Scrollbar styling */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        </style>
        """
        st.markdown(css_content, unsafe_allow_html=True)
    
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
    
    def render_facebook_chat(self):
        """Render Facebook-style chat interface"""
        # Auto-initialize system on first run
        if not st.session_state.system_initialized:
            self.auto_initialize_system()
        
        # Create the Facebook-style chat container - SIMPLE VERSION
        avatar_base64 = self.get_avatar_base64("half body shmulik.png")
        
        st.markdown(f"""
        <div class="facebook-chat-container">
            <div class="chat-header">
                <img src="data:image/png;base64,{avatar_base64}" alt="Shmulik">
                <div class="chat-header-info">
                    <h3>Shmulik</h3>
                </div>
            </div>
            <div class="chat-messages" id="chat-messages">
            </div>
            <div class="chat-input">
                <div class="input-container">
                    <button class="mic-button" onclick="startVoiceInput()" title="Voice Input">🎤</button>
                    <input type="text" id="user-input" placeholder="Type a message..." onkeypress="handleKeyPress(event)">
                    <button class="send-button" onclick="sendMessage()" title="Send Message">➤</button>
                </div>
            </div>
        </div>
        
        <script>
        function handleKeyPress(event) {{
            if (event.key === 'Enter') {{
                sendMessage();
            }}
        }}
        
        function sendMessage() {{
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (message) {{
                // Add user message to chat
                addMessageToChat(message, 'user');
                input.value = '';
                
                // Send to hidden Streamlit input
                const hiddenInput = document.querySelector('input[key="hidden_chat_input"]');
                if (hiddenInput) {{
                    hiddenInput.value = message;
                    hiddenInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    hiddenInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            }}
        }}
        
        function addMessageToChat(message, sender) {{
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{sender}}`;
            
            const avatar = sender === 'user' ? '👤' : '🤖';
            const avatarBase64 = '{avatar_base64}';
            messageDiv.innerHTML = `
                <img src="${{sender === 'assistant' ? 'data:image/png;base64,' + avatarBase64 : ''}}" class="message-avatar" alt="${{sender}}">
                <div class="message-content">${{message}}</div>
            `;
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }}
        
        function startVoiceInput() {{
            // Voice input functionality would go here
            alert('Voice input feature coming soon!');
        }}
        
        </script>
        """, unsafe_allow_html=True)
        
        # Render conversation history separately
        self.display_conversation_history()
        
        # Handle user input - using URL parameters
        self.process_user_input_simple()
    
    def get_avatar_base64(self, filename):
        """Get base64 encoded avatar image"""
        try:
            avatar_path = os.path.join(project_root, "assets", filename)
            if os.path.exists(avatar_path):
                import base64
                with open(avatar_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode()
        except:
            pass
        return ""
    
    def display_conversation_history(self):
        """Display conversation history in Facebook-style chat"""
        if st.session_state.conversation_history:
            for message in st.session_state.conversation_history:
                if isinstance(message, HumanMessage):
                    st.markdown(f"""
                    <div class="message user">
                        <div class="message-content">{message.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif isinstance(message, AIMessage):
                    avatar_base64 = self.get_avatar_base64("half body shmulik.png")
                    st.markdown(f"""
                    <div class="message assistant">
                        <img src="data:image/png;base64,{avatar_base64}" class="message-avatar" alt="Shmulik">
                        <div class="message-content">{message.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def process_user_input_simple(self):
        """Process user input using completely hidden input - NO visible inputs"""
        # Create a completely invisible input that JavaScript can interact with
        st.markdown("""
        <style>
        .stTextInput > div > div > input {
            display: none !important;
        }
        .stTextInput > div > div > label {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Completely hidden input for JavaScript communication
        user_input = st.text_input("", key="hidden_chat_input", label_visibility="collapsed")
        
        if user_input and st.session_state.rag_system:
            # Add user message to history
            user_message = HumanMessage(content=user_input)
            st.session_state.conversation_history.append(user_message)
            
            try:
                # Query the RAG system
                result = st.session_state.rag_system.query(
                    user_query=user_input,
                    conversation_history=st.session_state.conversation_history[:-1]
                )
                
                if result["success"]:
                    response = result["response"]
                    
                    # Add AI message to history
                    ai_message = AIMessage(content=response)
                    st.session_state.conversation_history.append(ai_message)
                    
                    # Rerun to show the new message
                    st.rerun()
                else:
                    error_msg = result.get("response", "Sorry, I encountered an error.")
                    ai_message = AIMessage(content=error_msg)
                    st.session_state.conversation_history.append(ai_message)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
    
    def run(self):
        """Run the Streamlit application"""
        self.render_facebook_chat()


def main():
    """Main function to run the Streamlit app"""
    app = ShmulkStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
