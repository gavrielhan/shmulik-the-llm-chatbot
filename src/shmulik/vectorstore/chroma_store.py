"""
ChromaDB Vector Store Module for document embeddings
"""
import os
from typing import List, Optional, Dict, Any
import time
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from loguru import logger


class ChromaVectorStore:
    """ChromaDB vector store for document embeddings and retrieval"""
    
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = None
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist the vector store
            embedding_model: HuggingFace embedding model name
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_model = embedding_model
        # Create unique collection name if none provided
        if collection_name is None:
            timestamp = str(int(time.time()))
            self.collection_name = f"shmulik_docs_{timestamp}"
            # Also use unique directory to avoid ChromaDB instance conflicts
            self.persist_directory = os.path.join(persist_directory, f"instance_{timestamp}")
        else:
            self.collection_name = collection_name
            self.persist_directory = persist_directory
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings with proper handling for meta tensors
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True,
                    'use_auth_token': False
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
        except NotImplementedError as e:
            if "meta tensor" in str(e).lower():
                logger.warning(f"Meta tensor error with {embedding_model}, trying alternative initialization...")
                # Alternative initialization for meta tensor issue
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={
                        'device': 'cpu',
                        'trust_remote_code': True,
                        'use_auth_token': False
                    }
                )
            else:
                raise e
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.vectorstore = None
        logger.info(f"ChromaVectorStore initialized with model: {embedding_model}")
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create vector store from documents
        
        Args:
            documents: List of Document objects to vectorize
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        try:
            # Try to delete existing collection first
            try:
                self.delete_collection()
                logger.info("Deleted existing collection")
            except Exception as cleanup_error:
                logger.warning(f"Could not delete existing collection: {cleanup_error}")
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            logger.info("Vector store created successfully")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk
        
        Returns:
            Chroma vector store instance if exists, None otherwise
        """
        try:
            if self._vectorstore_exists():
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                logger.info("Existing vector store loaded successfully")
                return self.vectorstore
            else:
                logger.info("No existing vector store found")
                return None
                
        except Exception as e:
            logger.error(f"Error loading existing vector store: {str(e)}")
            return None
    
    def _vectorstore_exists(self) -> bool:
        """Check if vector store exists on disk"""
        try:
            collections = self.chroma_client.list_collections()
            return any(col.name == self.collection_name for col in collections)
        except:
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of Document objects to add
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_existing_vectorstore first.")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        try:
            self.vectorstore.add_documents(documents)
            logger.info("Documents added successfully")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of documents to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Search query
            k: Number of documents to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Similarity search with scores returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {str(e)}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever interface for the vector store
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Vector store retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.vectorstore = None
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise


def create_chroma_store(
    persist_directory: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    collection_name: str = None
) -> ChromaVectorStore:
    """
    Factory function to create ChromaVectorStore instance
    
    Args:
        persist_directory: Directory to persist the vector store
        embedding_model: HuggingFace embedding model name
        collection_name: Name of the ChromaDB collection
        
    Returns:
        ChromaVectorStore instance
    """
    return ChromaVectorStore(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        collection_name=collection_name
    )
