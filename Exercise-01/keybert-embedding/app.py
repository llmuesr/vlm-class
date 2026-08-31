# app.py
import streamlit as st
import os
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from vector_store import VectorStoreManager
from rag_chain import RAGChain
from graph import RAGGraph


# Page configuration
st.set_page_config(
    page_title="RAG Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .body {
        direction: rtl
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-document {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .stats-card {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    """Initialize RAG components (cached)"""
    vector_store_manager = VectorStoreManager()
    rag_chain = RAGChain(vector_store_manager)
    rag_graph = RAGGraph(vector_store_manager)
    return vector_store_manager, rag_chain, rag_graph


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


def display_chat_message(role: str, content: str, sources: List = None):
    """Display a chat message with optional sources"""
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander("📄 View Sources", expanded=False):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get('source', 'Unknown')
                    st.markdown(f"""
                    <div class="source-document">
                        <strong>Source {i}:</strong> {source_name}<br>
                        <em>{doc.page_content[:300]}...</em>
                    </div>
                    """, unsafe_allow_html=True)


def sidebar_content(vector_store_manager: VectorStoreManager):
    """Render sidebar content"""
    st.sidebar.title("KEYBERT نرم‌افزار رگ")
    st.sidebar.markdown("---")
    
    # File Upload Section
    st.sidebar.header("📁 آپلود فایل")
    
    uploaded_files = st.sidebar.file_uploader(
        "فایل متنی",
        type=["txt"],
        accept_multiple_files=True,
        help="یک یا چند فایل متنی آپلود کنید"
    )
    
    if uploaded_files:
        if st.sidebar.button("📤 پردازش فایل متنی", type="primary"):
            with st.sidebar.status("در حال پردازش...") as status:
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    content = uploaded_file.read().decode('utf-8')
                    documents = vector_store_manager.load_txt_from_content(
                        content, 
                        uploaded_file.name
                    )
                    chunks = vector_store_manager.add_documents(documents)
                    total_chunks += chunks
                    st.sidebar.write(f"✅ {uploaded_file.name}: {chunks} chunks")
                
                status.update(label=f"پردازش شد {len(uploaded_files)} فایل ({total_chunks} تکه)", state="complete")
                st.sidebar.success(f"اضافه شد {total_chunks} تکه به پایگاه داده!")
    
    st.sidebar.markdown("---")
    
    # Load from Directory
    st.sidebar.header("📂 آپلود یک پوشه کامل")
    
    data_dir = st.sidebar.text_input(
        "پوشه کامل",
        value=Config.DATA_DIR,
        help="پوشه کامل را آپلود کنید"
    )
    
    if st.sidebar.button("📥 آپلود"):
        with st.spinner("در حال آپلود پوشه..."):
            chunks = vector_store_manager.load_directory(data_dir)
            if chunks > 0:
                st.sidebar.success(f"تعداد {chunks} تکه پردازش شد")
            else:
                st.sidebar.warning("هیچ فایلی موجود نبود")
    
    st.sidebar.markdown("---")
    
    # Collection Statistics
    st.sidebar.header("📊 آمار")
    
    stats = vector_store_manager.get_collection_stats()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Documents", stats.get("total_documents", 0))
    with col2:
        st.metric("Collection", stats.get("collection_name", "N/A")[:10])
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    
    use_langgraph = st.sidebar.toggle(
        "Use LangGraph Workflow",
        value=True,
        help="Enable advanced RAG workflow with document grading"
    )
    
    show_sources = st.sidebar.toggle(
        "Show Source Documents",
        value=True,
        help="Display source documents with answers"
    )
    
    st.sidebar.markdown("---")
    
    # Actions
    st.sidebar.header("🔧 Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear KB"):
            vector_store_manager.clear_collection()
            st.sidebar.success("Knowledge base cleared!")
            st.rerun()
    
    return use_langgraph, show_sources


def main():
    """Main application"""
    # Initialize components
    initialize_session_state()
    
    try:
        vector_store_manager, rag_chain, rag_graph = initialize_components()
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.info("Please check your configuration and API keys.")
        st.stop()
    
    # Render sidebar
    use_langgraph, show_sources = sidebar_content(vector_store_manager)
    
    # Main content
    st.title("نرم افزار رگ با استفاده از KEYBERT")
    
    # Check if knowledge base has documents
    stats = vector_store_manager.get_collection_stats()
    if stats.get("total_documents", 0) == 0:
        st.warning("No documents in knowledge base. Please upload some TXT files using the sidebar.")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("sources") if show_sources else None
        )
    
    # Chat input
    if prompt := st.chat_input("سوال مورد نظر خود را بپرسید..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if use_langgraph:
                        # Use LangGraph workflow
                        result = rag_graph.invoke(
                            prompt,
                            st.session_state.chat_history
                        )
                    else:
                        # Use simple RAG chain
                        result = rag_chain.invoke(
                            prompt,
                            st.session_state.chat_history
                        )
                    
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if show_sources and sources:
                        with st.expander("📄 View Sources", expanded=False):
                            for i, doc in enumerate(sources, 1):
                                source_name = doc.metadata.get('source', 'Unknown')
                                st.markdown(f"""
                                <div class="source-document">
                                    <strong>Source {i}:</strong> {source_name}<br>
                                    <em>{doc.page_content[:300]}...</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Update session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Update chat history for context
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    # Keep only last 10 messages in history
                    if len(st.session_state.chat_history) > 10:
                        st.session_state.chat_history = st.session_state.chat_history[-10:]
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Please check your OpenRouter API key and try again.")


if __name__ == "__main__":
    main()