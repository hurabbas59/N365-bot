import os
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import ORJSONResponse
from typing import Any
from models import AskResponse
from chatbot import answer_question

from fastapi.middleware.cors import CORSMiddleware

# ─── App & Directories ────────────────────────────────────
app = FastAPI(default_response_class=ORJSONResponse)

# Add CORS middleware for cross-origin requests (when UI and API are separate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store Pinecone index
app.state.pinecone_index: Any | None = None

# ─── Dependency to fetch Pinecone index ─────────────────
def get_pinecone_index():
    """Get the Pinecone index from app state."""
    if app.state.pinecone_index is None:
        # Try to connect to existing index
        try:
            from pinecone import Pinecone
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME", "n365-rag-agent-index")
            
            if not pinecone_api_key:
                raise HTTPException(500, "PINECONE_API_KEY not configured")
            
            pc = Pinecone(api_key=pinecone_api_key)
            app.state.pinecone_index = pc.Index(index_name)
            print(f"✅ Connected to existing Pinecone index: {index_name}")
            
        except Exception as e:
            raise HTTPException(500, f"Failed to connect to Pinecone: {str(e)}")
    
    return app.state.pinecone_index

# ─── Ask Endpoint ────────────────────────────────────────
@app.post("/ask/", response_model=AskResponse)
async def ask(
    request: Request,
    pinecone_index: Any = Depends(get_pinecone_index),
):
    """Handle text question and return enhanced answer with metadata."""
    print(f"API request received at {time.strftime('%H:%M:%S')}")
    api_start_time = time.time()

    # Parse JSON body for {"question": ...}
    try:
        data = await request.json()
        question = data.get("question") if isinstance(data, dict) else None
    except Exception:
        question = None

    if not question or not question.strip():
        raise HTTPException(400, "Provide a 'question' in the request body")

    # Log final question
    print(f"[API] Question sent to Pinecone & LLM: {question}")

    # RAG answer with enhanced metadata
    response = await answer_question(pinecone_index, question)
    
    # Extract answer and metadata
    answer_text = response.get("answer", "")
    metadata = response.get("metadata", {})
    
    # Check for errors
    if metadata.get("error"):
        raise HTTPException(500, f"Processing error: {metadata.get('error_message', 'Unknown error')}")

    api_total_time = time.time() - api_start_time
    print(f"API response completed in {api_total_time:.2f} seconds")
    print("=" * 60)

    return AskResponse(
        question=question, 
        answer=answer_text,
        metadata=metadata
    )

# ─── Knowledge Base Info Endpoint ──────────────────────────
@app.get("/knowledge-base/info")
async def get_knowledge_base_info():
    """Get information about the knowledge base structure."""
    try:
        # Get index stats
        pinecone_index = get_pinecone_index()
        index_stats = pinecone_index.describe_index_stats()
        
        # Extract metadata from index
        total_vectors = index_stats.total_vector_count if hasattr(index_stats, 'total_vector_count') else 0
        
        return {
            "status": "active",
            "total_vectors": total_vectors,
            "index_name": os.getenv("PINECONE_INDEX_NAME", "n365-rag-agent-index"),
            "knowledge_hierarchy": {
                "01_Quran": "Holy Quran - Complete text with translations",
                "02_Fiqh_Al_Ahwat": "Islamic Jurisprudence",
                "03_Hadith_Mawdat_ul_Qurba": "Prophetic Traditions",
                "04_Kitab_ul_Etiqadia": "Islamic Theology",
                "05_Awrad_Prayers": "Spiritual Recitations",
                "06_Dua_Collection": "Islamic Supplications",
                "07_Namaz_Prayers": "Prayer Guidelines",
                "08_Taharat_Cleanliness": "Purification Rules",
                "09_Zakat_Khums": "Islamic Charity",
                "10_Ramzan_Fasting": "Fasting Guidelines",
                "11_Nikah_Marriage": "Marriage Laws",
                "12_Mayat_Death_Rites": "Funeral Procedures",
                "13_Ayam_Special_Days": "Special Religious Days",
                "14_Kalmay": "Declarations of Faith",
                "15_Buzurgan_e_Deen": "Religious Personalities",
                "16_Daily_Wazaif": "Daily Spiritual Practices",
                "17_Question_Answer": "Religious Q&A",
                "18_Additional_Content": "Additional Islamic Content"
            },
            "mobile_app_support": True,
            "languages_supported": ["Arabic", "Urdu", "English"],
            "last_updated": time.time()
        }
    except Exception as e:
        print(f"Error getting knowledge base info: {e}")
        raise HTTPException(500, "Error retrieving knowledge base information")

# ─── Search Endpoint ──────────────────────────────────────
@app.post("/search/")
async def search_knowledge_base(
    request: Request,
    pinecone_index: Any = Depends(get_pinecone_index),
):
    """Search the knowledge base with metadata filtering."""
    try:
        data = await request.json()
        query = data.get("query", "")
        category_filter = data.get("category", None)
        content_type_filter = data.get("content_type", None)
        top_k = data.get("top_k", 10)
        
        if not query:
            raise HTTPException(400, "Query is required")
        
        # Create query embedding
        from chatbot import embedder
        query_vector = embedder.embed_query(query)
        
        # Build filter
        filter_dict = {}
        if category_filter:
            filter_dict["category"] = category_filter
        if content_type_filter:
            filter_dict["content_type"] = content_type_filter
        
        # Search with filters
        results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else {}
        )
        
        # Format results
        search_results = []
        for match in results.matches:
            meta = match.metadata or {}
            search_results.append({
                "source": meta.get("source", "Unknown"),
                "category": meta.get("category", "General"),
                "content_type": meta.get("content_type", "text"),
                "relevance_score": float(match.score) if hasattr(match, 'score') else 0.0,
                "breadcrumb": meta.get("breadcrumb", "Islamic Knowledge"),
                "priority": meta.get("priority", "medium"),
                "preview": meta.get("text", "")[:200] + "..." if len(meta.get("text", "")) > 200 else meta.get("text", "")
            })
        
        return {
            "query": query,
            "results": search_results,
            "total_results": len(search_results),
            "filters_applied": filter_dict
        }
        
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

# ─── Health Check Endpoint ──────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try to connect to Pinecone
        pinecone_index = get_pinecone_index()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "pinecone_connection": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }






