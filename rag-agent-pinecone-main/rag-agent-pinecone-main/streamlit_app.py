import streamlit as st
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Shah Syed AI - Islamic Knowledge",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .answer-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .reference-box {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar-info {
        background-color: #e8f5e8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize Pinecone and OpenAI
@st.cache_resource
def initialize_ai_components():
    """Initialize Pinecone and OpenAI components."""
    try:
        # Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "n365-rag-agent-index")
        
        if not pinecone_api_key:
            st.error("❌ PINECONE_API_KEY not found in environment variables")
            return None, None, None
        
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        
        # OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("❌ OPENAI_API_KEY not found in environment variables")
            return None, None, None
        
        embedder = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            max_tokens=512,
            openai_api_key=openai_api_key
        )
        
        return index, embedder, llm
        
    except Exception as e:
        st.error(f"❌ Error initializing AI components: {e}")
        return None, None, None

# Create prompt template
def create_prompt():
    """Create the prompt template for Islamic knowledge Q&A."""
    template = """
    You are Shah Syed AI, a knowledgeable Islamic scholar and AI assistant specializing in Islamic knowledge. 
    Your task is to answer questions based on the provided context with accuracy, clarity, and cultural sensitivity.

    **IMPORTANT INSTRUCTIONS:**
    1. **Language Matching**: Always respond in the SAME LANGUAGE as the question asked
    2. **Cultural Sensitivity**: Be respectful of Islamic traditions and cultural context
    3. **Source Citation**: When possible, cite specific sources from the context
    4. **Accuracy**: Use ONLY information from the provided context
    5. **Clarity**: Provide clear, well-structured answers
    6. **Religious Respect**: Maintain appropriate reverence for religious content

    **For Arabic Questions**: Respond in Arabic with proper Arabic grammar and script
    **For Urdu Questions**: Respond in Urdu with proper Urdu grammar and script  
    **For English Questions**: Respond in English

    **Context Information:**
    {context}

    **Question (Language: {detected_language}):** {question}

    **Answer (in {detected_language}):**
    """
    
    return PromptTemplate(
        input_variables=["question", "context", "detected_language"],
        template=template
    )

# Language detection
def detect_language(text):
    """Simple language detection for Arabic, Urdu, and English."""
    try:
        # Check for Arabic/Urdu characters
        arabic_urdu_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F')
        if arabic_urdu_chars > len(text) * 0.3:
            return 'ar'  # Arabic/Urdu
        
        # Check for specific Urdu words
        urdu_words = ['کیسے', 'کیا', 'کہاں', 'کب', 'کیوں', 'وضو', 'نماز', 'روزہ']
        if any(word in text for word in urdu_words):
            return 'ur'  # Urdu
        
        # Check for specific Arabic words
        arabic_words = ['كيف', 'ما', 'أين', 'متى', 'لماذا', 'وضوء', 'صلاة', 'صوم']
        if any(word in text for word in arabic_words):
            return 'ar'  # Arabic
        
        return 'en'  # Default to English
        
    except:
        return 'en'

def get_language_name(lang_code):
    """Get human-readable language name."""
    return {
        'ar': 'Arabic/Urdu',
        'ur': 'Urdu',
        'en': 'English'
    }.get(lang_code, 'English')

# Main Q&A function
async def get_answer(question, index, embedder, llm):
    """Get answer from the Islamic knowledge base."""
    try:
        # Detect language
        detected_lang = detect_language(question)
        lang_name = get_language_name(detected_lang)
        
        # Create query embedding
        query_vector = embedder.embed_query(question)
        
        # Search Pinecone
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        
        if not results.matches:
            return "Sorry, I couldn't find an answer to this question in the available knowledge base.", []
        
        # Prepare context from top matches
        context_parts = []
        references = []
        
        for i, match in enumerate(results.matches[:3]):
            meta = match.metadata or {}
            text = meta.get("text", "")
            source = meta.get("source", "Unknown")
            category = meta.get("category", "General")
            
            if text.strip():
                context_parts.append(f"[Source: {source} | Category: {category}]\n{text}")
                
                # Add reference
                references.append({
                    "source": source,
                    "category": category,
                    "relevance": float(match.score) if hasattr(match, 'score') else 0.0
                })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt and get answer
        prompt = create_prompt()
        chain = prompt | llm | StrOutputParser()
        
        result = await chain.ainvoke({
            "question": question,
            "context": context,
            "detected_language": detected_lang
        })
        
        return result, references
        
    except Exception as e:
        return f"Error processing your question: {str(e)}", []

# Main app
def main():
    # App header
    st.markdown('<h1 class="main-header">🕌 Shah Syed AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Islamic Knowledge Assistant - Ask questions in Arabic, Urdu, or English</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment info
        st.markdown("""
        <div class="sidebar-info">
        <h4>🔑 Environment Variables</h4>
        <p>Make sure these are set in Streamlit Cloud:</p>
        <ul>
        <li>PINECONE_API_KEY</li>
        <li>PINECONE_INDEX_NAME</li>
        <li>OPENAI_API_KEY</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Knowledge base info
        st.header("📚 Knowledge Base")
        st.info("""
        **Available Categories:**
        - 📖 Quran & Tafsir
        - ⚖️ Islamic Jurisprudence
        - 🕌 Prophetic Traditions
        - 🕯️ Islamic Theology
        - 🙏 Spiritual Practices
        - 🤲 Supplications
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
            index, embedder, llm = initialize_ai_components()
            if index and embedder and llm:
                st.success("✅ All components connected successfully!")
            else:
                st.error("❌ Connection failed")
    
    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display existing chat
    for msg in st.session_state['messages']:
        if msg['role'] == 'user':
            st.chat_message('user').write(msg['content'])
        else:
            st.chat_message('assistant').write(msg['content'])
            
            # Show reference if available
            if 'references' in msg and msg['references']:
                with st.expander("📚 References"):
                    for i, ref in enumerate(msg['references'], 1):
                        st.markdown(f"""
                        **Reference {i}:**
                        - **Source:** {ref['source']}
                        - **Category:** {ref['category']}
                        - **Relevance:** {ref['relevance']:.3f}
                        """)
    
    # Question input
    st.markdown("### 💬 Ask Your Question")
    question_input = st.text_input(
        "Type your question in Arabic, Urdu, or English:",
        placeholder="e.g., How to perform wudu? / کیسے وضو کریں؟ / كيف تتوضأ؟",
        key="question_input"
    )
    
    # Send button
    if st.button("🚀 Ask Shah Syed AI", type="primary"):
        if not question_input.strip():
            st.warning("Please enter a question.")
            st.stop()
        
        # Add user message
        st.session_state['messages'].append({'role': 'user', 'content': question_input})
        st.chat_message('user').write(question_input)
        
        # Get answer
        with st.spinner("🤔 Shah Syed AI is thinking..."):
            try:
                # Initialize components
                index, embedder, llm = initialize_ai_components()
                
                if not all([index, embedder, llm]):
                    st.error("❌ AI components not initialized. Please check your environment variables.")
                    st.stop()
                
                # Get answer
                answer, references = asyncio.run(get_answer(question_input, index, embedder, llm))
                
                # Add assistant message
                st.session_state['messages'].append({
                    'role': 'assistant', 
                    'content': answer,
                    'references': references
                })
                
                # Display answer
                st.chat_message('assistant').write(answer)
                
                # Display references
                if references:
                    st.markdown("---")
                    st.markdown("### 📚 References")
                    for i, ref in enumerate(references, 1):
                        with st.expander(f"Reference {i}: {ref['source']}"):
                            st.markdown(f"""
                            **Source:** {ref['source']}
                            **Category:** {ref['category']}
                            **Relevance Score:** {ref['relevance']:.3f}
                            """)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
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
            <p>🕌 Shah Syed AI | Islamic Knowledge Assistant</p>
            <p>Powered by OpenAI & Pinecone | Deployed on Streamlit Cloud</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
