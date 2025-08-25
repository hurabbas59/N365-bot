import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import MULTILINGUAL_QA_PROMPT
from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain_openai import OpenAIEmbeddings
import re
from langdetect import detect, LangDetectException

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



prompt = PromptTemplate(
    input_variables=["question", "context", "detected_language"],
    template=MULTILINGUAL_QA_PROMPT
)

llm = ChatOpenAI(
    model="gpt-4", 
    temperature=0.3,
    max_tokens=512,  # Increased for multilingual responses
    openai_api_key=OPENAI_API_KEY
)
chain = prompt | llm | StrOutputParser()

# Shared embedder to avoid re-instantiation overhead
openai_api_key = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

def detect_question_language(question: str) -> str:
    """Detect the language of the question for appropriate response."""
    try:
        # Clean the question for language detection
        clean_question = re.sub(r'[^\w\s]', '', question)
        if len(clean_question.strip()) < 3:
            return 'en'  # Default to English for very short text
        
        lang = detect(clean_question)
        
        # Map language codes to our supported languages
        if lang in ['ar', 'fa', 'ur']:
            return 'ar'  # Arabic/Urdu/Persian
        elif lang == 'en':
            return 'en'
        else:
            return 'en'  # Default to English
            
    except LangDetectException:
        return 'en'

def is_arabic_urdu_text(text: str) -> bool:
    """Check if text contains Arabic or Urdu characters."""
    arabic_urdu_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text)
    return len(arabic_urdu_chars) > len(text) * 0.1

def get_language_name(lang_code: str) -> str:
    """Get human-readable language name."""
    language_map = {
        'ar': 'Arabic/Urdu',
        'en': 'English',
        'unknown': 'Unknown'
    }
    return language_map.get(lang_code, 'English')

def enhance_context_with_metadata(matches: list, question_lang: str) -> str:
    """Enhance context with metadata and language-specific formatting."""
    context_parts = []
    
    for i, match in enumerate(matches):
        meta = match.metadata or {}
        text = meta.get("text") or meta.get("chunk_text") or ""
        
        if not text.strip():
            continue
            
        # Add source information
        source = meta.get("source", "Unknown")
        category = meta.get("category", "General")
        
        # Format context based on language
        if question_lang == 'ar':
            header = f"[مصدر: {source} | فئة: {category}]"
        else:
            header = f"[Source: {source} | Category: {category}]"
        
        context_parts.append(f"{header}\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

def extract_answer_metadata(matches: list, question: str, question_lang: str) -> Dict[str, Any]:
    """Extract comprehensive metadata for the answer."""
    if not matches:
        return {}
    
    # Get the best match for primary metadata
    best_match = matches[0]
    meta = best_match.metadata or {}
    
    # Extract navigation information
    navigation_info = {
        "category_id": meta.get("category_id", "general"),
        "category_name": meta.get("category_name", "General"),
        "category_icon": meta.get("category_icon", "📖"),
        "breadcrumb": meta.get("breadcrumb", "Islamic Knowledge"),
        "subtopics": meta.get("subtopics", []),
        "content_type": meta.get("content_type", "text"),
        "priority": meta.get("priority", "medium")
    }
    
    # Extract source references
    sources = []
    for match in matches[:3]:  # Top 3 sources
        match_meta = match.metadata or {}
        source_info = {
            "source": match_meta.get("source", "Unknown"),
            "category": match_meta.get("category", "General"),
            "content_type": match_meta.get("content_type", "text"),
            "relevance_score": float(match.score) if hasattr(match, 'score') else 0.0
        }
        
        # Add specific reference information
        if match_meta.get("content_type") == "quranic_text":
            source_info["quran_reference"] = match_meta.get("quran_reference", "")
            source_info["surah_number"] = match_meta.get("surah_number", "")
            source_info["surah_name"] = match_meta.get("surah_name", "")
        elif match_meta.get("content_type") == "question_answer":
            source_info["qa_reference"] = match_meta.get("qa_reference", "")
            source_info["topic_number"] = match_meta.get("topic_number", "")
            source_info["topic_name"] = match_meta.get("topic_name", "")
        
        sources.append(source_info)
    
    # Generate mobile app navigation data
    mobile_navigation = {
        "current_category": navigation_info["category_id"],
        "category_name": navigation_info["category_name"],
        "breadcrumb": navigation_info["breadcrumb"],
        "icon": navigation_info["category_icon"],
        "subtopics": navigation_info["subtopics"],
        "related_categories": _get_related_categories(navigation_info["content_type"]),
        "search_tags": _generate_search_tags_from_question(question, navigation_info),
        "user_interaction": {
            "bookmarkable": True,
            "shareable": True,
            "offline_available": True
        }
    }
    
    # Generate related content suggestions
    related_content = _generate_related_content_suggestions(navigation_info, sources)
    
    return {
        "navigation": navigation_info,
        "sources": sources,
        "mobile_app": mobile_navigation,
        "related_content": related_content,
        "metadata": {
            "question_language": question_lang,
            "total_sources": len(sources),
            "primary_source": sources[0] if sources else None,
            "answer_timestamp": time.time(),
            "content_priority": navigation_info["priority"]
        }
    }

def _get_related_categories(content_type: str) -> List[Dict[str, Any]]:
    """Get related categories based on content type."""
    related_map = {
        "prayer": [
            {"id": "08_Taharat_Cleanliness", "name": "Purification Rules", "icon": "💧"},
            {"id": "06_Dua_Collection", "name": "Supplications", "icon": "🤲"}
        ],
        "fasting": [
            {"id": "16_Daily_Wazaif", "name": "Daily Practices", "icon": "📿"},
            {"id": "06_Dua_Collection", "name": "Supplications", "icon": "🤲"}
        ],
        "quranic_text": [
            {"id": "06_Dua_Collection", "name": "Quranic Duas", "icon": "🤲"},
            {"id": "04_Kitab_ul_Etiqadia", "name": "Theology", "icon": "🕯️"}
        ],
        "question_answer": [
            {"id": "02_Fiqh_Al_Ahwat", "name": "Islamic Law", "icon": "⚖️"},
            {"id": "03_Hadith_Mawdat_ul_Qurba", "name": "Prophetic Traditions", "icon": "🕌"}
        ]
    }
    
    return related_map.get(content_type, [])

def _generate_search_tags_from_question(question: str, navigation_info: Dict[str, Any]) -> List[str]:
    """Generate search tags based on the question and navigation context."""
    tags = []
    
    # Add category tags
    tags.append(navigation_info["category_name"])
    tags.extend(navigation_info["subtopics"])
    
    # Add content type tags
    tags.append(navigation_info["content_type"])
    
    # Add priority tags
    tags.append(f"priority_{navigation_info['priority']}")
    
    # Add language tags
    tags.extend(["Arabic", "Urdu", "English"])
    
    return list(set(tags))

def _generate_related_content_suggestions(navigation_info: Dict[str, Any], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate related content suggestions for mobile app."""
    suggestions = []
    
    # Suggest content from same category
    suggestions.append({
        "type": "same_category",
        "title": f"More {navigation_info['category_name']}",
        "description": f"Explore more content in {navigation_info['category_name']}",
        "category_id": navigation_info["category_id"],
        "icon": navigation_info["category_icon"]
    })
    
    # Suggest related categories
    related_cats = _get_related_categories(navigation_info["content_type"])
    for cat in related_cats[:2]:  # Limit to 2 related categories
        suggestions.append({
            "type": "related_category",
            "title": cat["name"],
            "description": f"Related to {navigation_info['category_name']}",
            "category_id": cat["id"],
            "icon": cat["icon"]
        })
    
    # Suggest based on content type
    if navigation_info["content_type"] == "prayer":
        suggestions.append({
            "type": "complementary",
            "title": "Daily Spiritual Practices",
            "description": "Enhance your prayer with daily spiritual practices",
            "category_id": "16_Daily_Wazaif",
            "icon": "📿"
        })
    
    return suggestions

async def answer_question(pinecone_index: Any, question: str, k: int = 5) -> Dict[str, Any]:
    """Enhanced multilingual question answering with comprehensive metadata."""
    print(f"Starting question processing: '{question[:50]}...' at {time.strftime('%H:%M:%S')}")
    total_start_time = time.time()
    
    # Detect question language
    question_lang = detect_question_language(question)
    lang_name = get_language_name(question_lang)
    print(f"[Chatbot] Detected question language: {lang_name} ({question_lang})")
    
    # Document retrieval timing
    print(f"Retrieving top {k} documents...")
    retrieval_start_time = time.time()
    
    # Compute query embedding and search via Pinecone
    try:
        query_vec = embedder.embed_query(question)
    except Exception as e:
        print(f"[Chatbot] Error creating query embedding: {e}")
        error_response = {
            "answer": f"Error: Unable to process question. Please try again.",
            "metadata": {
                "error": True,
                "error_message": "Embedding creation failed"
            }
        }
        return error_response
    
    # Enhanced search with metadata filtering
    try:
        res = pinecone_index.query(
            vector=query_vec, 
            top_k=k, 
            include_metadata=True,
            filter={}  # Can add category filters here if needed
        )
        matches = res.matches if hasattr(res, "matches") else res["matches"]
    except Exception as e:
        print(f"[Chatbot] Error querying Pinecone: {e}")
        error_response = {
            "answer": f"Error: Unable to search knowledge base. Please try again.",
            "metadata": {
                "error": True,
                "error_message": "Pinecone query failed"
            }
        }
        return error_response
    
    retrieval_time = time.time() - retrieval_start_time
    print(f"Document retrieval completed in {retrieval_time:.2f} seconds")
    print(f"Retrieved {len(matches)} documents")
    
    if not matches:
        if question_lang == 'ar':
            answer_text = "عذراً، لم أجد إجابة لهذا السؤال في قاعدة المعرفة المتاحة."
        else:
            answer_text = "Sorry, I couldn't find an answer to this question in the available knowledge base."
        
        return {
            "answer": answer_text,
            "metadata": {
                "error": False,
                "no_results": True,
                "sources": [],
                "navigation": {"category_name": "General", "breadcrumb": "Islamic Knowledge"}
            }
        }
    
    # Prepare enhanced context
    context_start_time = time.time()
    context = enhance_context_with_metadata(matches, question_lang)
    context_time = time.time() - context_start_time
    print(f"Context prepared in {context_time:.3f} seconds (length: {len(context)} chars)")
    
    # LLM inference timing
    print("Generating answer with LLM...")
    llm_start_time = time.time()
    
    try:
        result: str = await chain.ainvoke({
            "question": question, 
            "context": context,
            "detected_language": question_lang
        })
    except Exception as e:
        print(f"[Chatbot] Error generating LLM response: {e}")
        if question_lang == 'ar':
            answer_text = "عذراً، حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى."
        else:
            answer_text = "Sorry, an error occurred while processing your question. Please try again."
        
        return {
            "answer": answer_text,
            "metadata": {
                "error": True,
                "error_message": "LLM response generation failed"
            }
        }
    
    llm_time = time.time() - llm_start_time
    print(f"LLM response generated in {llm_time:.2f} seconds")
    print(f"Answer length: {len(result)} characters")
    
    # Extract comprehensive metadata
    answer_metadata = extract_answer_metadata(matches, question, question_lang)
    
    # Total timing
    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Question answered successfully at {time.strftime('%H:%M:%S')}")
    print("-" * 60)
    
    # Return comprehensive response with metadata
    return {
        "answer": result,
        "metadata": answer_metadata,
        "question": question,
        "language": question_lang,
        "processing_time": total_time
    }
