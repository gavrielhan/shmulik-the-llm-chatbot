"""
Configuration settings for the Shmulik RAG Chatbot
"""
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # LLM Configuration
    openai_api_key: str = os.getenv('API_KEY')
    openai_api_base: str = "https://litellm.sph-prod.ethz.ch/v1"
    llm_model: str = "openai/gpt-4.1-mini"
    
    # Vector Store Configuration
    vector_store_path: str = "./data/vectorstore"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Document Processing
    pdf_path: str = "./Report_Digital-Health-Literacy-among-Students_v1.pdf"
    
    # Application Configuration
    app_name: str = "Shmulik RAG Chatbot"
    log_level: str = "INFO"
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Streamlit Configuration
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "localhost"
    
    # Embedding Configuration - Superior multilingual model for Hebrew + English
    embedding_model: str = "intfloat/multilingual-e5-base"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)
        # Reload settings to pick up new env vars
        global settings
        settings = Settings()
