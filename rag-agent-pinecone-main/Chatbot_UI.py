import streamlit as st
import requests
import time
import os
import asyncio

# Backend-free mode: call retrieval and LLM directly
from topic_based_chatbot import process_question_with_topic
from topic_based_retriever import get_available_topics_from_index

# Pinecone client (support both new and legacy)
def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]  # type: ignore[index]
    except Exception:
        return os.getenv(name, default)

@st.cache_resource(show_spinner=False)
def get_pinecone_index():
    """Create and cache Pinecone index client for direct use in Streamlit Cloud."""
    api_key = _get_secret("PINECONE_API_KEY")
    index_name = _get_secret("PINECONE_INDEX_NAME", "islamic-knowledge-topics-v2")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in Streamlit secrets or environment")
    try:
        # New SDK style
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        return pc.Index(index_name)
    except Exception:
        # Legacy SDK fallback
        import pinecone
        pinecone.init(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"))
        return pinecone.Index(index_name)

# Page config
st.set_page_config(
    page_title="Noorbakshia365 AI Bot",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with high-contrast, visible colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --bg: #0b1410; /* deep green-black */
        --bg-soft: #0f1d17;
        --card: #0f2018;
        --text: #e6fff2;
        --text-soft: #bfe8d3;
        --accent: #21c38a; /* emerald */
        --accent-strong: #16a374;
        --border: #1e3d30;
        --shadow: rgba(33, 195, 138, 0.12);
    }
    @media (prefers-color-scheme: light) {
        :root {
            --bg: #f8fffb;
            --bg-soft: #eefaf3;
            --card: #ffffff;
            --text: #0d1b16;
            --text-soft: #174a39;
            --accent: #16a374;
            --accent-strong: #0e7a57;
            --border: #cfe9dc;
            --shadow: rgba(14, 122, 87, 0.12);
        }
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--bg);
        color: var(--text);
    }
    
    .main-header {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        margin-bottom: 0.5rem;
        color: var(--text);
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-soft);
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    .topic-selector {
        background: var(--card);
        border-radius: 16px;
        padding: 20px;
        margin: 18px 0;
        border: 1px solid var(--border);
        box-shadow: 0 6px 22px var(--shadow);
    }
    
    .topic-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .chat-container {
        background: var(--card);
        border-radius: 14px;
        padding: 22px;
        margin: 20px 0;
        border: 1px solid var(--border);
        box-shadow: 0 10px 26px var(--shadow);
        color: var(--text);
    }
    
    /* Main page question input */
    .stTextInput > div > div > input {
        background: var(--bg) !important;
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-size: 1.1rem !important;
        color: var(--text) !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(33, 195, 138, 0.20) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin-top: 12px !important;
        box-shadow: 0 10px 24px var(--shadow) !important;
    }
    
    .stButton > button:hover {
        filter: brightness(0.95);
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 28px var(--shadow) !important;
    }
    
    .answer-container {
        background: linear-gradient(135deg, rgba(33, 195, 138, 0.08) 0%, rgba(14, 122, 87, 0.08) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        color: var(--text);
        box-shadow: 0 6px 22px var(--shadow);
    }
    
    .topic-info {
        background: linear-gradient(135deg, rgba(33,195,138,0.14) 0%, rgba(14,122,87,0.14) 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        margin: 12px 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
    }
    
    /* Sidebar styling */
    .css-1d391kg { background: var(--bg-soft) !important; }
    [data-testid="stSidebar"] { background: var(--bg-soft) !important; border-right: 1px solid var(--border) !important; }
    [data-testid="stSidebar"] * { color: var(--text) !important; font-size: 1.05rem !important; }
    .stMarkdown, .stText, .stCaption, .stHeader { color: var(--text) !important; }
    
    .stSelectbox > div > div {
        background: var(--bg) !important;
        border: 2px solid var(--border) !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        color: var(--text) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(33, 195, 138, 0.20) !important;
    }
    
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        color: #111827 !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #dc2626 !important;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
    }
    
    .knowledge-category { color: var(--text-soft); }
    
    .connection-success { color: var(--text); border: 1px solid var(--border); }
    
    .connection-error { color: #ffdcdc; border: 1px solid #7a1c1c; }
    
    .stSelectbox > div > div { background: var(--bg); border: 2px solid var(--border); border-radius: 12px; }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .clear-chat-btn { background: linear-gradient(135deg, #1a3b2d 0%, #153026 100%) !important; margin-top: 24px !important; }
    
    .clear-chat-btn:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'available_topics' not in st.session_state:
    st.session_state['available_topics'] = []
if 'selected_topic' not in st.session_state:
    st.session_state['selected_topic'] = 'all'
if 'topics_loaded' not in st.session_state:
    st.session_state['topics_loaded'] = False

# Configuration
# API_BASE_URL is optional now (only used if you prefer hitting the FastAPI backend).
API_BASE_URL = _get_secret("API_BASE_URL", "")

@st.cache_data(show_spinner=False)
def load_topics_direct(pinecone_index):
    try:
        topics = get_available_topics_from_index(pinecone_index)
        return topics, True
    except Exception:
        return DEFAULT_TOPICS, False

# Default topics as fallback (using original names)
DEFAULT_TOPICS = [
    {"folder_name": "all", "display_name": "All Topics", "description": "Search across all Islamic knowledge categories"},
    {"folder_name": "03_Hadith_Mawdat_ul_Qurba", "display_name": "Hadith Mawdat ul Qurba", "description": "Prophetic traditions and sayings"},
    {"folder_name": "04_Kitab_ul_Etiqadia", "display_name": "Kitab ul Etiqadia", "description": "Islamic beliefs and theology"},
    {"folder_name": "05_Awrad_Prayers", "display_name": "Awrad Prayers", "description": "Daily spiritual recitations"},
    {"folder_name": "06_Dua_Collection", "display_name": "Dua Collection", "description": "Collection of Islamic supplications"},
    {"folder_name": "07_Namaz_Prayers", "display_name": "Namaz Prayers", "description": "Islamic prayer guidelines"},
    {"folder_name": "08_Taharat_Cleanliness", "display_name": "Taharat Cleanliness", "description": "Purification and cleanliness rules"},
    {"folder_name": "09_Zakat_Khums", "display_name": "Zakat Khums", "description": "Islamic charity and financial obligations"},
    {"folder_name": "10_Ramzan_Fasting", "display_name": "Ramzan Fasting", "description": "Ramadan and fasting guidelines"},
    {"folder_name": "11_Nikah_Marriage", "display_name": "Nikah Marriage", "description": "Islamic marriage laws and procedures"},
    {"folder_name": "12_Mayat_Death_Rites", "display_name": "Mayat Death Rites", "description": "Islamic funeral and burial procedures"},
    {"folder_name": "13_Ayam_Special_Days", "display_name": "Ayam Special Days", "description": "Important Islamic dates and occasions"},
    {"folder_name": "14_Kalmay", "display_name": "Kalmay", "description": "Islamic declarations of faith"},
    {"folder_name": "15_Buzurgan_e_Deen", "display_name": "Buzurgan e Deen", "description": "Religious personalities and scholars"},
    {"folder_name": "16_Daily_Wazaif", "display_name": "Daily Wazaif", "description": "Daily spiritual practices and recitations"},
    {"folder_name": "17_Question_Answer", "display_name": "Question Answer", "description": "Religious questions and answers"},
    {"folder_name": "18_Additional_Content", "display_name": "Additional Content", "description": "Additional Islamic knowledge and resources"}
]

def load_topics_from_api(api_url):
    """Load topics from API with fallback to default topics."""
    try:
        response = requests.get(f"{api_url}/topics", timeout=2)  # Shorter timeout
        if response.ok:
            topics_data = response.json()
            return topics_data['topics'], True
        else:
            return DEFAULT_TOPICS, False
    except Exception:
        # Silently fall back to default topics on connection error
        return DEFAULT_TOPICS, False

# Auto-load topics (prefer direct Pinecone; fallback to API if provided)
if not st.session_state['topics_loaded']:
    topics = []
    loaded = False
    try:
        pinecone_index = get_pinecone_index()
        topics, loaded = load_topics_direct(pinecone_index)
    except Exception:
        pass
    if (not loaded) and API_BASE_URL:
        topics, loaded = load_topics_from_api(API_BASE_URL)
    st.session_state['available_topics'] = topics
    st.session_state['topics_loaded'] = True

# App title and description
st.markdown('<h1 class="main-header">🕌 Noorbakshia365 AI Bot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about Islamic knowledge (Arabic • Urdu • English)</p>', unsafe_allow_html=True)

# Clean Sidebar - Only Topic Selection
with st.sidebar:
    st.header("📂 Select Topic")
    
    # Ensure we always have topics (use defaults if needed)
    if not st.session_state['available_topics']:
        st.session_state['available_topics'] = DEFAULT_TOPICS

    topic_options = [(topic['display_name'], topic['folder_name']) for topic in st.session_state['available_topics']]
    topic_names = [option[0] for option in topic_options]
    topic_folders = [option[1] for option in topic_options]

    # Find current selection index
    try:
        current_index = topic_folders.index(st.session_state['selected_topic'])
    except ValueError:
        current_index = 0  # Default to first option (All Topics)

    selected_display_name = st.selectbox(
        "Choose topic:",
        topic_names,
        index=current_index,
        help="Select a topic to focus your search",
        key="sidebar_topic_selector"
    )

    # Update selected topic folder
    selected_index = topic_names.index(selected_display_name)
    st.session_state['selected_topic'] = topic_folders[selected_index]

    # Show topic description
    selected_topic_data = next((t for t in st.session_state['available_topics'] if t['folder_name'] == st.session_state['selected_topic']), None)
    if selected_topic_data and selected_topic_data['folder_name'] != 'all':
        st.success(f"📋 **{selected_topic_data['display_name']}**")
        st.info(selected_topic_data['description'])
    
    # Hidden debug mode (can be enabled if needed)
    debug_mode = False

# Main page simplified: only heading and Q/A

# Main chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display existing chat messages
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.chat_message('user').write(msg['content'])
    else:
        st.chat_message('assistant').write(msg['content'])
        
        # Show translations and topic info if available (debug mode)
        if debug_mode and 'translations' in msg and msg['translations']:
            with st.expander("🔄 Translations Used"):
                st.info(f"**Translations:** {msg['translations']}")
        
        # Show topic info if available
        if 'topic_info' in msg and msg['topic_info']:
            topic_info = msg['topic_info']
            if topic_info.get('topic_name'):
                st.caption(f"📂 Source Topic: {topic_info['topic_name']}")

# Question input
question_input = st.text_input(
    "💬 Ask your question:",
    placeholder="e.g., How to perform wudu? / کیسے وضو کریں؟ / كيف تتوضأ؟",
    key="question_input"
)

# Send button
if st.button("🚀 Ask AI Assistant", type="primary"):
    if not question_input.strip():
        st.warning("Please enter a question.")
        st.stop()
    
    # Add user message to chat
    st.session_state['messages'].append({'role': 'user', 'content': question_input})
    st.chat_message('user').write(question_input)
    # Get answer via direct backend-free call (preferred), fallback to API if configured
    with st.spinner("🤔 AI Assistant is thinking..."):
        try:
            try:
                pinecone_index = get_pinecone_index()
                result = asyncio.run(process_question_with_topic(
                    pinecone_index,
                    question_input,
                    st.session_state['selected_topic'] if st.session_state['selected_topic'] != 'all' else None
                ))
                answer = result.get("answer", "")
                topic_name = result.get("topic_name")
                metadata = result.get("metadata", {})
                topic_info = {
                    'topic_name': topic_name,
                    'topic_folder': st.session_state['selected_topic'],
                    'selected_topic': st.session_state['selected_topic']
                }
                st.session_state['messages'].append({
                    'role': 'assistant',
                    'content': answer,
                    'translations': metadata.get("translations", ""),
                    'topic_info': topic_info
                })
                st.chat_message('assistant').write(answer)
            except Exception as direct_err:
                if not API_BASE_URL:
                    raise
                endpoint = f"{API_BASE_URL}/ask/"
                payload = {
                    "question": question_input,
                    "topic_folder": st.session_state['selected_topic'] if st.session_state['selected_topic'] != 'all' else None
                }
                response = requests.post(endpoint, json=payload, timeout=60)
                data = response.json()
                answer = data.get("answer", "")
                topic_info = {
                    'topic_name': data.get('topic_name'),
                    'topic_folder': data.get('topic_folder'),
                    'selected_topic': st.session_state['selected_topic']
                }
                st.session_state['messages'].append({'role': 'assistant','content': answer,'translations': data.get("metadata", {}).get("translations", ""),'topic_info': topic_info})
                st.chat_message('assistant').write(answer)
                
        except Exception as e:
            error_msg = f"Connection error: {e}"
            st.error(error_msg)
            st.session_state['messages'].append({'role': 'assistant', 'content': error_msg})

# Clear chat button with custom styling
if st.button("🗑️ Clear Chat", key="clear_chat", help="Clear all chat messages"):
    st.session_state['messages'] = []
    st.rerun()

# Add custom CSS for the clear button
st.markdown("""
<style>
    div[data-testid="stButton"] button[kind="secondary"] {
        background: linear-gradient(135deg, #1a3b2d 0%, #153026 100%) !important;
        color: white !important;
        border: none !important;
        margin-top: 24px !important;
    }
    
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        filter: brightness(0.95);
        box-shadow: 0 8px 24px var(--shadow) !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🕌 Islamic Knowledge RAG Chatbot | Powered by FastAPI & Pinecone</p>
        <p>Ask questions in Arabic, Urdu, or English about Islamic knowledge</p>
    </div>
    """,
    unsafe_allow_html=True
)
