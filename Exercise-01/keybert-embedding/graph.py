# graph.py
from typing import TypedDict, Annotated, List, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

from config import Config
from vector_store import VectorStoreManager
from rag_chain import RAGChain


class GraphState(TypedDict):
    """State for the RAG graph"""
    question: str
    chat_history: List[BaseMessage]
    context: str
    source_documents: List
    answer: str
    relevance_scores: List[float]
    needs_retrieval: bool
    iteration: int


class RAGGraph:
    """LangGraph-based RAG workflow"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.config = Config()
        self.vector_store_manager = vector_store_manager
        self.rag_chain = RAGChain(vector_store_manager)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the RAG workflow graph"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("transform_query", self._transform_query)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._should_grade,
            {
                "grade": "grade_documents",
                "generate": "generate"
            }
        )
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform": "transform_query",
                "generate": "generate"
            }
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: GraphState) -> GraphState:
        """Analyze the incoming query"""
        question = state["question"]
        chat_history = state.get("chat_history", [])
        
        # If there's chat history, we might need to reformulate the question
        if chat_history:
            # Use LLM to create standalone question
            standalone_question = self._create_standalone_question(question, chat_history)
            state["question"] = standalone_question
        
        state["iteration"] = state.get("iteration", 0)
        state["needs_retrieval"] = True
        
        return state
    
    def _create_standalone_question(self, question: str, chat_history: List[BaseMessage]) -> str:
        """Create a standalone question from chat history"""
        history_text = "\n".join([
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in chat_history[-4:]  # Last 4 messages
        ])
        
        prompt = f"""Given this conversation history:
{history_text}

And this follow-up question: {question}

Rewrite it as a standalone question:"""
        
        response = self.rag_chain.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _retrieve(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents"""
        question = state["question"]
        
        # Get documents with scores
        docs_with_scores = self.rag_chain.get_relevant_documents_with_scores(question)
        
        documents = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        
        # Format context
        context = self.rag_chain._format_docs(documents)
        
        state["source_documents"] = documents
        state["relevance_scores"] = scores
        state["context"] = context
        
        return state
    
    def _should_grade(self, state: GraphState) -> str:
        """Decide whether to grade documents or go straight to generation"""
        # If we have good documents, grade them
        if state.get("source_documents"):
            return "grade"
        return "generate"
    
    def _grade_documents(self, state: GraphState) -> GraphState:
        """Grade retrieved documents for relevance"""
        question = state["question"]
        documents = state["source_documents"]
        
        # Simple relevance grading based on scores
        # In production, you might use an LLM to grade relevance
        relevance_threshold = 1.5  # Adjust based on your embedding model
        
        relevant_docs = []
        for doc, score in zip(documents, state["relevance_scores"]):
            if score < relevance_threshold:  # Lower score = more similar
                relevant_docs.append(doc)
        
        state["source_documents"] = relevant_docs if relevant_docs else documents[:2]
        state["context"] = self.rag_chain._format_docs(state["source_documents"])
        
        return state
    
    def _decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate or transform query"""
        iteration = state.get("iteration", 0)
        documents = state.get("source_documents", [])
        
        # If we have relevant documents or have tried too many times, generate
        if documents or iteration >= 2:
            return "generate"
        
        return "transform"
    
    def _transform_query(self, state: GraphState) -> GraphState:
        """Transform the query to improve retrieval"""
        question = state["question"]
        
        # Use LLM to rewrite query
        prompt = f"""The following question didn't retrieve good results. 
Rewrite it to be more specific and searchable:

Original question: {question}

Rewritten question:"""
        
        response = self.rag_chain.llm.invoke(prompt)
        transformed = response.content if hasattr(response, 'content') else str(response)
        
        state["question"] = transformed
        state["iteration"] = state.get("iteration", 0) + 1
        
        return state
    
    def _generate(self, state: GraphState) -> GraphState:
        """Generate the final answer"""
        question = state["question"]
        context = state.get("context", "No relevant context found.")
        
        # Generate answer
        result = self.rag_chain.invoke(question)
        
        state["answer"] = result["answer"]
        
        return state
    
    def invoke(self, question: str, chat_history: Optional[List[BaseMessage]] = None) -> dict:
        """Invoke the RAG graph"""
        initial_state: GraphState = {
            "question": question,
            "chat_history": chat_history or [],
            "context": "",
            "source_documents": [],
            "answer": "",
            "relevance_scores": [],
            "needs_retrieval": True,
            "iteration": 0
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "context": result["context"]
        }
    
    def stream(self, question: str, chat_history: Optional[List[BaseMessage]] = None):
        """Stream the RAG graph execution"""
        initial_state: GraphState = {
            "question": question,
            "chat_history": chat_history or [],
            "context": "",
            "source_documents": [],
            "answer": "",
            "relevance_scores": [],
            "needs_retrieval": True,
            "iteration": 0
        }
        
        for event in self.graph.stream(initial_state):
            yield event