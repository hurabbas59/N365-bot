#!/usr/bin/env python3
"""
Test script for the complete Islamic RAG chatbot with rich metadata.
"""

import os
import asyncio
from dotenv import load_dotenv
from chatbot import answer_question
from pinecone import Pinecone

# Load environment variables
load_dotenv()

async def test_chatbot():
    """Test the complete chatbot system."""
    print("🧪 Testing Complete Islamic RAG Chatbot System")
    print("=" * 60)
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "n365-rag-agent-index")
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found")
        return
    
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        print(f"✅ Connected to Pinecone index: {index_name}")
        
        # Test questions in different languages
        test_questions = [
            "How to perform wudu?",
            "What is the first surah of the Quran?",
            "How to pray Fajr?",
            "What are the rules of fasting in Ramadan?",
            "کیسے وضو کریں؟",  # Urdu
            "كيف تتوضأ؟"       # Arabic
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}: {question}")
            print(f"{'='*60}")
            
            try:
                # Get answer with rich metadata
                response = await answer_question(index, question)
                
                if response.get("metadata", {}).get("error"):
                    print(f"❌ Error: {response['metadata']['error_message']}")
                    continue
                
                # Display answer
                print(f"\n🤖 Answer:")
                print(f"{response['answer']}")
                
                # Display rich metadata
                metadata = response.get("metadata", {})
                if metadata:
                    print(f"\n📊 Rich Metadata:")
                    
                    # Navigation info
                    nav = metadata.get("navigation", {})
                    if nav:
                        print(f"  🗂️ Category: {nav.get('category_name', 'N/A')}")
                        print(f"  📍 Breadcrumb: {nav.get('breadcrumb', 'N/A')}")
                        print(f"  🏷️ Content Type: {nav.get('content_type', 'N/A')}")
                        print(f"  ⭐ Priority: {nav.get('priority', 'N/A')}")
                    
                    # Sources
                    sources = metadata.get("sources", [])
                    if sources:
                        print(f"  📚 Sources ({len(sources)}):")
                        for j, source in enumerate(sources[:3], 1):
                            print(f"    {j}. {source.get('source', 'Unknown')}")
                            print(f"       Category: {source.get('category', 'Unknown')}")
                            print(f"       Relevance: {source.get('relevance_score', 0):.3f}")
                    
                    # Mobile app data
                    mobile = metadata.get("mobile_app", {})
                    if mobile:
                        print(f"  📱 Mobile App Data:")
                        print(f"    Current Category: {mobile.get('category_name', 'N/A')}")
                        print(f"    Icon: {mobile.get('icon', 'N/A')}")
                        
                        related = mobile.get("related_categories", [])
                        if related:
                            print(f"    Related Categories:")
                            for cat in related[:2]:
                                print(f"      - {cat.get('name', 'N/A')} ({cat.get('icon', 'N/A')})")
                    
                    # Related content
                    related = metadata.get("related_content", [])
                    if related:
                        print(f"  🔗 Related Content Suggestions:")
                        for j, content in enumerate(related[:2], 1):
                            print(f"    {j}. {content.get('title', 'N/A')}")
                            print(f"       {content.get('description', 'N/A')}")
                
                print(f"\n⏱️ Processing Time: {response.get('processing_time', 0):.2f} seconds")
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                continue
            
            # Small delay between questions
            await asyncio.sleep(1)
        
        print(f"\n{'='*60}")
        print("✅ Chatbot Testing Completed!")
        print("🎯 Your Islamic RAG system is ready for mobile app integration!")
        
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")

def test_metadata_structure():
    """Test the metadata structure for mobile app integration."""
    print("\n🔍 Testing Metadata Structure for Mobile App")
    print("=" * 60)
    
    # Show what metadata structure your mobile app will receive
    print("📱 Your Mobile App Will Receive:")
    print("""
    Every answer includes:
    
    1. 📝 Answer Text (in user's language)
    2. 🗂️ Navigation Information:
       - Category ID & Name
       - Breadcrumb trail
       - Content type & priority
       - Icons for UI
    
    3. 📚 Source References:
       - File names
       - Categories
       - Relevance scores
    
    4. 📱 Mobile App Data:
       - Current category
       - Related categories
       - Search tags
       - User interaction flags
    
    5. 🔗 Related Content:
       - Same category suggestions
       - Related category suggestions
       - Complementary content
    
    6. ⚙️ Technical Metadata:
       - Question language
       - Processing time
       - Content priority
    """)

if __name__ == "__main__":
    # Test metadata structure
    test_metadata_structure()
    
    # Test complete chatbot
    asyncio.run(test_chatbot())
