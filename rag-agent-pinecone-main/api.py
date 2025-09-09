import os
import time
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import ORJSONResponse
from typing import Any
from models import AskResponse, TopicsResponse, TopicInfo
from fastapi.middleware.cors import CORSMiddleware
from topic_based_chatbot import process_question_with_topic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(default_response_class=ORJSONResponse)

print("🚀 Starting Islamic Knowledge RAG API...")
print(f"📝 PINECONE_API_KEY: {'SET' if os.getenv('PINECONE_API_KEY') else 'NOT SET'}")
print(f"📝 OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print(f"📝 PINECONE_INDEX_NAME: {os.getenv('PINECONE_INDEX_NAME', 'islamic-knowledge-topics-v2')}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.pinecone_index: Any | None = None

def get_pinecone_index():
    """Get the Pinecone index from app state."""
    if app.state.pinecone_index is None:
        try:
            try:
                from pinecone import Pinecone
            except ImportError:
                import pinecone
                Pinecone = pinecone.Pinecone
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME", "islamic-knowledge-topics-v2")
            
            if not pinecone_api_key:
                raise HTTPException(500, "PINECONE_API_KEY not configured")
            
            pc = Pinecone(api_key=pinecone_api_key)
            app.state.pinecone_index = pc.Index(index_name)
            print(f"✅ Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            raise HTTPException(500, f"Failed to connect to Pinecone: {str(e)}")
    
    return app.state.pinecone_index

@app.post("/ask/", response_model=AskResponse)
async def ask(
    request: Request,
    pinecone_index: Any = Depends(get_pinecone_index),
):
    """Enhanced endpoint: 2 LLM calls with topic filtering - Translation + Answer Generation."""
    print(f"\n🔥 NEW REQUEST RECEIVED: /ask/")
    try:
        data = await request.json()
        question = data.get("question") if isinstance(data, dict) else None
        topic_folder = data.get("topic_folder") if isinstance(data, dict) else None
        print(f"📝 Question: '{question}'")
        print(f"📂 Topic: '{topic_folder}'")
    except Exception as e:
        print(f"❌ Error parsing request: {e}")
        question = None
        topic_folder = None

    if not question or not question.strip():
        raise HTTPException(400, "Provide a 'question' in the request body")

    print(f"🔍 API received question: {question}")
    print(f"📂 Topic filter: {topic_folder or 'All Topics'}")
    
    try:
        # Call the main function from chatbot.py with topic filtering
        result = await process_question_with_topic(pinecone_index, question, topic_folder)
        
        # Return the response
        return AskResponse(
            question=question,
            answer=result["answer"],
            topic_folder=topic_folder,
            topic_name=result.get("topic_name"),
            metadata=result["metadata"]
        )
        
    except Exception as e:
        print(f"❌ API error: {e}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.get("/topics", response_model=TopicsResponse)
async def get_topics(pinecone_index: Any = Depends(get_pinecone_index)):
    """Get available topics for filtering."""
    try:
        from topic_based_retriever import get_available_topics_from_index
        
        topics_data = get_available_topics_from_index(pinecone_index)
        topics = [TopicInfo(**topic) for topic in topics_data]
        
        return TopicsResponse(
            topics=topics,
            total_topics=len(topics)
        )
        
    except Exception as e:
        print(f"❌ Error getting topics: {e}")
        raise HTTPException(500, f"Failed to get topics: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        pinecone_index = get_pinecone_index()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "pinecone_connection": "connected",
            "endpoints": ["/ask", "/topics"],
            "llm_calls_per_query": 2,
            "description": "Topic-based RAG API with translation and filtering"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }






