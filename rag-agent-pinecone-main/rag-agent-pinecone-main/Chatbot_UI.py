import streamlit as st
import requests
import time
import os

# Page config
st.set_page_config(
    page_title="Islamic Knowledge RAG Chatbot",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .metadata-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .source-box {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 10px;
        margin: 5px 0;
        border-radius: 3px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🕌 Islamic Knowledge RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about Islamic knowledge in Arabic, Urdu, or English</p>', unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL configuration
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="URL of your RAG API server"
    )
    
    # Knowledge base info
    st.header("📚 Knowledge Base")
    st.info("""
    **Available Categories:**
    - 📖 Quran & Tafsir
    - ⚖️ Islamic Jurisprudence (Fiqh)
    - 🕌 Prophetic Traditions (Hadith)
    - 🕯️ Islamic Theology
    - 🙏 Spiritual Practices
    - 🤲 Supplications (Duas)
    - 🕐 Prayer Guidelines
    - 💧 Purification Rules
    - 🌙 Fasting & Ramadan
    - 💍 Marriage & Family
    - ⚰️ Funeral Procedures
    - 📅 Special Religious Days
    - ✋ Declarations of Faith
    - 👤 Religious Personalities
    - 📿 Daily Spiritual Practices
    - ❓ Religious Q&A
    """)
    
    # Test connection
    if st.button("🔗 Test Connection"):
        try:
            response = requests.get(f"{api_url}/knowledge-base/info")
            if response.ok:
                st.success("✅ Connected to API successfully!")
                info = response.json()
                st.json(info)
            else:
                st.error(f"❌ Connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection error: {e}")

# Main chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display existing chat messages
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.chat_message('user').write(msg['content'])
    else:
        st.chat_message('assistant').write(msg['content'])
        
        # Display metadata if available
        if 'metadata' in msg and msg['metadata']:
            metadata = msg['metadata']
            
            # Navigation info
            if 'navigation' in metadata:
                nav = metadata['navigation']
                with st.expander("🗂️ Navigation Information"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Category:** {nav.get('category_name', 'N/A')}")
                        st.write(f"**Content Type:** {nav.get('content_type', 'N/A')}")
                        st.write(f"**Priority:** {nav.get('priority', 'N/A')}")
                    with col2:
                        st.write(f"**Icon:** {nav.get('category_icon', 'N/A')}")
                        st.write(f"**Breadcrumb:** {nav.get('breadcrumb', 'N/A')}")
            
            # Sources
            if 'sources' in metadata and metadata['sources']:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(metadata['sources'][:3], 1):
                        st.markdown(f"""
                        **Source {i}:** {source.get('source', 'Unknown')}
                        - Category: {source.get('category', 'Unknown')}
                        - Relevance: {source.get('relevance_score', 0):.3f}
                        """)
            
            # Related content
            if 'related_content' in metadata and metadata['related_content']:
                with st.expander("🔗 Related Content"):
                    for content in metadata['related_content'][:3]:
                        st.write(f"**{content.get('title', 'N/A')}**")
                        st.write(f"*{content.get('description', 'N/A')}*")

# Question input
st.markdown("### 💬 Ask Your Question")
question_input = st.text_input(
    "Type your question in Arabic, Urdu, or English:",
    placeholder="e.g., How to perform wudu? / کیسے وضو کریں؟ / كيف تتوضأ؟",
    key="question_input"
)

# Send button
if st.button("🚀 Send Question", type="primary"):
    if not question_input.strip():
        st.warning("Please enter a question.")
        st.stop()
    
    # Add user message to chat
    st.session_state['messages'].append({'role': 'user', 'content': question_input})
    st.chat_message('user').write(question_input)
    
    # Get answer from API
    with st.spinner("🤔 Thinking..."):
        try:
            response = requests.post(
                f"{api_url}/ask/",
                json={"question": question_input},
                timeout=60
            )
            
            if response.ok:
                data = response.json()
                answer = data.get("answer", "")
                metadata = data.get("metadata", {})
                
                # Add assistant message to chat
                st.session_state['messages'].append({
                    'role': 'assistant', 
                    'content': answer,
                    'metadata': metadata
                })
                
                # Display answer
                st.chat_message('assistant').write(answer)
                
                # Display metadata
                if metadata:
                    st.markdown("---")
                    st.markdown("### 📊 Answer Details")
                    
                    # Navigation info
                    if 'navigation' in metadata:
                        nav = metadata['navigation']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Category", nav.get('category_name', 'N/A'))
                        with col2:
                            st.metric("Content Type", nav.get('content_type', 'N/A'))
                        with col3:
                            st.metric("Priority", nav.get('priority', 'N/A'))
                        
                        st.info(f"📍 **Navigation:** {nav.get('breadcrumb', 'N/A')}")
                    
                    # Sources
                    if 'sources' in metadata and metadata['sources']:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(metadata['sources'][:3], 1):
                            with st.expander(f"Source {i}: {source.get('source', 'Unknown')}"):
                                st.write(f"**Category:** {source.get('category', 'Unknown')}")
                                st.write(f"**Content Type:** {source.get('content_type', 'Unknown')}")
                                st.write(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                    
                    # Related content suggestions
                    if 'related_content' in metadata and metadata['related_content']:
                        st.markdown("### 🔗 Related Content")
                        cols = st.columns(len(metadata['related_content'][:3]))
                        for i, content in enumerate(metadata['related_content'][:3]):
                            with cols[i]:
                                st.info(f"**{content.get('title', 'N/A')}**\n\n{content.get('description', 'N/A')}")
                
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                st.error(error_msg)
                st.session_state['messages'].append({'role': 'assistant', 'content': error_msg})
                
        except Exception as e:
            error_msg = f"Connection error: {e}"
            st.error(error_msg)
            st.session_state['messages'].append({'role': 'assistant', 'content': error_msg})

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state['messages'] = []
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🕌 Islamic Knowledge RAG Chatbot | Powered by OpenAI & Pinecone</p>
        <p>Ask questions in Arabic, Urdu, or English about Islamic knowledge</p>
    </div>
    """,
    unsafe_allow_html=True
)
