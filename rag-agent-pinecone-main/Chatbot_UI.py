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

# Custom CSS with high-contrast, visible colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }
    
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    
    .sub-header {
        text-align: center;
        color: #374151;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .topic-selector {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 24px 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
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
        background: #ffffff;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Main page question input */
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 1.1rem !important;
        color: #111827 !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #dc2626 !important;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
    }
    
    .stButton > button {
        background: #dc2626 !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin-top: 12px !important;
    }
    
    .stButton > button:hover {
        background: #b91c1c !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
    }
    
    .answer-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 4px 16px rgba(14, 165, 233, 0.1);
    }
    
    .topic-info {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #fbbf24;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        font-weight: 500;
        color: #92400e;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc !important;
    }
    
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        color: #111827 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #dc2626 !important;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
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
    
    .knowledge-category {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
        color: #475569;
        font-size: 0.9rem;
    }
    
    .connection-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 16px;
        color: #065f46;
        font-weight: 500;
    }
    
    .connection-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        color: #991b1b;
        font-weight: 500;
    }
    
    .stSelectbox > div > div {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        font-size: 1rem;
    }
    
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
    
    .clear-chat-btn {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important;
        margin-top: 24px !important;
    }
    
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
# Robustly read API_BASE_URL: env var first, then Streamlit secrets, else local default
def _get_api_base_url() -> str:
    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url
    try:
        # Accessing st.secrets may raise if no secrets.toml exists in local runs
        return st.secrets["API_BASE_URL"]  # type: ignore[index]
    except Exception:
        return "http://localhost:8002"

API_BASE_URL = _get_api_base_url()

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

# Auto-load topics on first run (but don't show messages on main page)
if not st.session_state['topics_loaded']:
    topics, loaded_from_api = load_topics_from_api(API_BASE_URL)
    st.session_state['available_topics'] = topics
    st.session_state['topics_loaded'] = True

# App title and description
st.markdown('<h1 class="main-header">🕌 Islamic Knowledge RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about Islamic knowledge in Arabic, Urdu, or English</p>', unsafe_allow_html=True)

# Clean Sidebar - Only Base URL and Topic Selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL configuration
    api_url = st.text_input(
        "🌐 API Base URL",
        value=API_BASE_URL,
        help="URL of your RAG API server"
    )
    
    st.markdown("---")
    
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

# Show current selected topic on main page
selected_topic_data = next((t for t in st.session_state['available_topics'] if t['folder_name'] == st.session_state['selected_topic']), None)
if selected_topic_data:
    st.info(f"🎯 **Current Topic**: {selected_topic_data['display_name']}")

st.markdown("---")

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
    
    # Get answer from API
    with st.spinner("🤔 AI Assistant is thinking..."):
        try:
            endpoint = f"{api_url}/ask/"
            payload = {
                "question": question_input,
                "topic_folder": st.session_state['selected_topic'] if st.session_state['selected_topic'] != 'all' else None
            }
            
            if debug_mode:
                st.info(f"🌐 **API Endpoint:** {endpoint}")
                st.info(f"📤 **Request Payload:** {payload}")
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=60
            )
            
            if response.ok:
                data = response.json()
                answer = data.get("answer", "")
                metadata = data.get("metadata", {})
                
                # Add assistant message to chat with topic info
                topic_info = {
                    'topic_name': data.get('topic_name'),
                    'topic_folder': data.get('topic_folder'),
                    'selected_topic': st.session_state['selected_topic']
                }
                
                st.session_state['messages'].append({
                    'role': 'assistant', 
                    'content': answer,
                    'translations': metadata.get("translations", ""),
                    'topic_info': topic_info
                })
                
                # Display answer
                st.chat_message('assistant').write(answer)
                
                # Display metadata in debug mode
                if debug_mode:
                    st.markdown("---")
                    st.markdown("### 🔍 API Response Details")
                    st.json(data)
                
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                st.error(error_msg)
                st.session_state['messages'].append({'role': 'assistant', 'content': error_msg})
                
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
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important;
        color: white !important;
        border: none !important;
        margin-top: 24px !important;
    }
    
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4) !important;
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
