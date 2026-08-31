# rag_chain.py
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document

from config import Config
from vector_store import VectorStoreManager


class RAGChain:
    """RAG Chain implementation using LangChain"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.config = Config()
        self.vector_store_manager = vector_store_manager
        self.llm = self._initialize_llm()
        self.rag_prompt = self._create_rag_prompt()
        self.condense_prompt = self._create_condense_prompt()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize LLM with OpenRouter"""
        return ChatOpenAI(
            model=self.config.LLM_MODEL,
            openai_api_key=self.config.OPENROUTER_API_KEY,
            openai_api_base=self.config.OPENROUTER_BASE_URL,
            temperature=0.7,
            max_tokens=2048,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "RAG Application"
            }
        )
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template"""
        template = """You are a helpful AI assistant. Use the following context to answer the user's question. 
If you cannot find the answer in the context, say so clearly and provide what information you can.

Context:
{context}

Question: {question}

Instructions:
1. Answer based primarily on the provided context
2. If the context doesn't contain enough information, acknowledge this
3. Be concise but comprehensive
4. If relevant, cite which part of the context your answer comes from

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_condense_prompt(self) -> ChatPromptTemplate:
        """Create prompt for condensing chat history into standalone question"""
        template = """Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question that captures the full context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[Document {i} - Source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        return self.vector_store_manager.similarity_search(query)
    
    def get_relevant_documents_with_scores(self, query: str) -> List[tuple]:
        """Retrieve relevant documents with relevance scores"""
        return self.vector_store_manager.similarity_search_with_score(query)
    
    def invoke(self, question: str, chat_history: Optional[List] = None) -> dict:
        """Invoke the RAG chain"""
        # Get relevant documents
        docs = self.get_relevant_documents(question)
        
        # Format context
        context = self._format_docs(docs)
        
        # Create and invoke chain
        chain = self.rag_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": response,
            "source_documents": docs,
            "context": context
        }
    
    def stream(self, question: str, chat_history: Optional[List] = None):
        """Stream the RAG chain response"""
        # Get relevant documents
        docs = self.get_relevant_documents(question)
        
        # Format context
        context = self._format_docs(docs)
        
        # Create chain
        chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # Stream response
        for chunk in chain.stream({
            "context": context,
            "question": question
        }):
            yield chunk
        
        # Yield source documents at the end
        yield {"source_documents": docs}