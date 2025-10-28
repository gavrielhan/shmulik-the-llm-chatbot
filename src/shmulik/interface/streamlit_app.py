"""
Streamlit Interface for Shmulik RAG Chatbot
"""
import streamlit as st
import streamlit.components.v1 as components
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
        if 'last_processed_message_id' not in st.session_state:
            st.session_state.last_processed_message_id = None
        
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

            if not st.session_state.conversation_history:
                welcome_message = (
                    "Shalom! I'm Shmulik from the Samuel Neaman Institute. "
                    "Ask me anything about the Digital Health Literacy research report."
                )
                st.session_state.conversation_history.append(AIMessage(content=welcome_message))
            
            
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
        """Render Facebook-style chat interface entirely within the floating widget"""
        if not st.session_state.system_initialized:
            self.auto_initialize_system()

        conversation_data = [
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
            }
            for message in st.session_state.conversation_history
            if isinstance(message, (HumanMessage, AIMessage))
        ]

        if not conversation_data:
            conversation_data.append(
                {
                    "role": "assistant",
                    "content": "Shalom! I'm Shmulik. Ask me anything about the Digital Health Literacy research report.",
                }
            )

        avatar_base64 = self.get_avatar_base64("half body shmulik.png")
        conversation_json = json.dumps(conversation_data).replace("</", "<\/")

        html_content = f"""
        <div class="facebook-chat-container">
            <div class="chat-header">
                <img src="data:image/png;base64,{avatar_base64}" alt="Shmulik">
                <div class="chat-header-info">
                    <h3>Shmulik</h3>
                    <p>Samuel Neaman Institute AI Assistant</p>
                </div>
            </div>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input">
                <div class="input-container">
                    <input type="text" id="user-input" placeholder="Type a message..." autocomplete="off" />
                    <button class="send-button" id="send-button" title="Send Message">➤</button>
                </div>
            </div>
        </div>
        <script>
        const conversation = {conversation_json};
        const assistantAvatar = "{avatar_base64}";

        function renderConversation() {{
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML = '';

            conversation.forEach((message) => {{
                const wrapper = document.createElement('div');
                wrapper.className = 'message ' + (message.role === 'user' ? 'user' : 'assistant');

                if (message.role === 'assistant' && assistantAvatar) {{
                    const avatar = document.createElement('img');
                    avatar.src = 'data:image/png;base64,' + assistantAvatar;
                    avatar.className = 'message-avatar';
                    avatar.alt = 'Shmulik';
                    wrapper.appendChild(avatar);
                }}

                const content = document.createElement('div');
                content.className = 'message-content';
                content.innerText = message.content;
                wrapper.appendChild(content);

                chatMessages.appendChild(wrapper);
            }});

            chatMessages.scrollTop = chatMessages.scrollHeight;
        }}

        function sendPayload(payload) {{
            const streamlitMessage = {{
                isStreamlitMessage: true,
                type: 'streamlit:setComponentValue',
                value: JSON.stringify(payload)
            }};
            window.parent.postMessage(streamlitMessage, '*');
        }}

        function handleSubmit() {{
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) {{
                return;
            }}

            const payload = {{
                id: Date.now(),
                content: message
            }};

            conversation.push({{ role: 'user', content: message }});
            renderConversation();
            input.value = '';
            sendPayload(payload);
        }}

        document.getElementById('send-button').addEventListener('click', handleSubmit);
        document.getElementById('user-input').addEventListener('keydown', (event) => {{
            if (event.key === 'Enter') {{
                event.preventDefault();
                handleSubmit();
            }}
        }});

        renderConversation();
        window.parent.postMessage({{ isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: 520 }}, '*');
        document.getElementById('user-input').focus();
        </script>
        """

        component_value = components.html(html_content, height=520, scrolling=False, key="shmulik_facebook_chat")

        if component_value:
            self.process_component_message(component_value)

    def process_component_message(self, payload: str):
        """Handle messages sent from the embedded Facebook-style chat component."""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return

        message_id = data.get("id")
        message_content = data.get("content", "").strip()

        if not message_content:
            return

        if message_id and st.session_state.last_processed_message_id == message_id:
            return

        st.session_state.last_processed_message_id = message_id

        self.process_user_message(message_content)

    def process_user_message(self, user_input: str):
        """Process a message received from the chat widget and update the conversation."""
        user_message = HumanMessage(content=user_input)
        st.session_state.conversation_history.append(user_message)

        if not st.session_state.rag_system:
            st.session_state.conversation_history.append(
                AIMessage(content="I'm still initializing. Please try again in a moment.")
            )
            st.experimental_rerun()
            return

        try:
            result = st.session_state.rag_system.query(
                user_query=user_input,
                conversation_history=st.session_state.conversation_history[:-1]
            )

            if result["success"]:
                response = result["response"]
            else:
                response = result.get("response", "Sorry, I encountered an error.")

        except Exception as exc:
            response = f"I'm sorry, but I ran into an issue: {str(exc)}"

        st.session_state.conversation_history.append(AIMessage(content=response))

        st.experimental_rerun()

    def get_avatar_base64(self, filename):
        """Get base64 encoded avatar image"""
        try:
            avatar_path = os.path.join(project_root, "assets", filename)
            if os.path.exists(avatar_path):
                import base64
                with open(avatar_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode()
        except Exception:
            pass
        return ""

    def run(self):
        """Run the Streamlit application"""
        self.render_facebook_chat()


def main():
    """Main function to run the Streamlit app"""
    app = ShmulkStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
