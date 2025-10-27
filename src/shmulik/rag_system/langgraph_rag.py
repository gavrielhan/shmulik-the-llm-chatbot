"""
LangGraph-based RAG System for Shmulik Chatbot
"""
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from loguru import logger
import operator


class RAGState(TypedDict):
    """State definition for the RAG workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    retrieved_docs: List[Document]
    context: str
    response: str
    metadata: Dict[str, Any]


class LangGraphRAGSystem:
    """LangGraph-based RAG system using retrieval and generation nodes"""
    
    def __init__(
        self,
        vectorstore_retriever,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str = "openai/gpt-4.1-mini",
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        """
        Initialize the LangGraph RAG system
        
        Args:
            vectorstore_retriever: Vector store retriever instance
            llm_api_key: OpenAI API key
            llm_base_url: Base URL for the LLM API
            llm_model: Model name
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
        """
        self.retriever = vectorstore_retriever
        
        # Initialize LLM with custom base URL
        self.llm = ChatOpenAI(
            openai_api_key=llm_api_key,
            openai_api_base=llm_base_url,
            model_name=llm_model.replace("openai/", ""),  # Remove prefix if present
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Create the RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template("""
You are Shmulik, an expert AI assistant for the Samuel Neaman Institute for National Policy Research. 
You are highly knowledgeable about digital health literacy research and provide authoritative policy insights.

Context Information:
{context}

User Question: {query}

RESPONSE LANGUAGE: Look at the user question above. If it's written in English, your entire response must be in English. If it's written in Hebrew, your entire response must be in Hebrew.

Instructions:
1. LANGUAGE RULE - ABSOLUTELY MANDATORY: 
   - If the user asks in ENGLISH → You MUST respond in ENGLISH ONLY
   - If the user asks in HEBREW → You MUST respond in HEBREW ONLY  
   - NEVER mix languages in your response
   - Translate information from Hebrew sources to English when answering English questions
2. The context contains information in both Hebrew and English - use ALL relevant information regardless of source language.
3. When translating from Hebrew sources to English responses, maintain accuracy of the original meaning.
4. Provide confident, definitive answers based on the research findings in the context.
5. Be specific and cite concrete data, statistics, and findings when available.
6. Structure your response clearly with key points and evidence.
7. Maintain an authoritative but accessible tone - you are the expert on this research.
8. Be concise and to teh point of the user's question, unless the users askd to expand on the topic.
Answer:""")
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        logger.info("LangGraph RAG system initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def retrieve_documents(state: RAGState) -> RAGState:
            """Node: Retrieve relevant documents from vector store"""
            query = state["query"]
            logger.info(f"Retrieving documents for query: {query}")
            
            try:
                # Retrieve documents (using invoke instead of deprecated get_relevant_documents)
                docs = self.retriever.invoke(query)
                
                # Create context from retrieved documents
                context = "\n\n".join([
                    f"Document {i+1} (Page {doc.metadata.get('page', 'Unknown')}):\n{doc.page_content}"
                    for i, doc in enumerate(docs)
                ])
                
                logger.info(f"Retrieved {len(docs)} documents")
                
                return {
                    **state,
                    "retrieved_docs": docs,
                    "context": context,
                    "metadata": {
                        **state.get("metadata", {}),
                        "num_retrieved_docs": len(docs),
                        "retrieval_successful": True
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in document retrieval: {str(e)}")
                return {
                    **state,
                    "retrieved_docs": [],
                    "context": "No relevant documents found.",
                    "metadata": {
                        **state.get("metadata", {}),
                        "num_retrieved_docs": 0,
                        "retrieval_successful": False,
                        "error": str(e)
                    }
                }
        
        def generate_response(state: RAGState) -> RAGState:
            """Node: Generate response using LLM"""
            query = state["query"]
            context = state["context"]
            
            logger.info("Generating response with LLM")
            
            try:
                # Format the prompt
                prompt = self.rag_prompt.format(query=query, context=context)
                
                # Generate response
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_text = response.content
                
                # Create AI message for conversation history
                ai_message = AIMessage(content=response_text)
                
                logger.info("Response generated successfully")
                
                return {
                    **state,
                    "response": response_text,
                    "messages": state["messages"] + [ai_message],
                    "metadata": {
                        **state.get("metadata", {}),
                        "generation_successful": True
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in response generation: {str(e)}")
                error_response = f"I apologize, but I encountered an error while generating a response: {str(e)}"
                
                return {
                    **state,
                    "response": error_response,
                    "messages": state["messages"] + [AIMessage(content=error_response)],
                    "metadata": {
                        **state.get("metadata", {}),
                        "generation_successful": False,
                        "error": str(e)
                    }
                }
        
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_response)
        
        # Define the flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def query(self, user_query: str, conversation_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline
        
        Args:
            user_query: User's question
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing query: {user_query}")
        
        # Prepare initial state
        messages = conversation_history or []
        messages.append(HumanMessage(content=user_query))
        
        initial_state = RAGState(
            messages=messages,
            query=user_query,
            retrieved_docs=[],
            context="",
            response="",
            metadata={}
        )
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            logger.info("Query processed successfully")
            
            return {
                "response": final_state["response"],
                "retrieved_docs": final_state["retrieved_docs"],
                "context": final_state["context"],
                "messages": final_state["messages"],
                "metadata": final_state["metadata"],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "retrieved_docs": [],
                "context": "",
                "messages": messages + [AIMessage(content=f"Error: {str(e)}")],
                "metadata": {"error": str(e)},
                "success": False
            }
    
    def get_conversation_summary(self, messages: List[BaseMessage], max_length: int = 500) -> str:
        """
        Generate a summary of the conversation
        
        Args:
            messages: List of conversation messages
            max_length: Maximum length of summary
            
        Returns:
            Conversation summary
        """
        if not messages:
            return "No conversation history"
        
        try:
            # Create summary prompt
            conversation_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages[-10:]  # Last 10 messages
            ])
            
            summary_prompt = f"""
            Please provide a brief summary of this conversation:
            
            {conversation_text}
            
            Summary (max {max_length} characters):
            """
            
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            return summary_response.content[:max_length]
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return f"Conversation with {len(messages)} messages"


def create_rag_system(
    vectorstore_retriever,
    llm_api_key: str,
    llm_base_url: str,
    llm_model: str = "openai/gpt-4.1-mini",
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> LangGraphRAGSystem:
    """
    Factory function to create LangGraphRAGSystem instance
    
    Args:
        vectorstore_retriever: Vector store retriever instance
        llm_api_key: OpenAI API key
        llm_base_url: Base URL for the LLM API
        llm_model: Model name
        max_tokens: Maximum tokens for response
        temperature: Temperature for response generation
        
    Returns:
        LangGraphRAGSystem instance
    """
    return LangGraphRAGSystem(
        vectorstore_retriever=vectorstore_retriever,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        max_tokens=max_tokens,
        temperature=temperature
    )
