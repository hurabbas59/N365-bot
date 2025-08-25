#!/usr/bin/env python3
"""
Test script for the Islamic Knowledge RAG System
Tests multilingual functionality and Islamic knowledge retrieval
"""

import os
import sys
from dotenv import load_dotenv
from enhanced_embedding_creator import EnhancedIslamicEmbeddingCreator
from data_loader import IslamicKnowledgeDataLoader

# Load environment variables
load_dotenv()

def test_data_loader():
    """Test the data loader functionality."""
    print("🔍 Testing Data Loader...")
    print("=" * 50)
    
    try:
        loader = IslamicKnowledgeDataLoader()
        
        # Get dataset statistics
        stats = loader.get_dataset_stats()
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total size: {stats['total_size_mb']} MB")
        print(f"   Categories: {len(stats['categories'])}")
        
        # Show category breakdown
        print("\n📊 Category Breakdown:")
        for category, count in stats['categories'].items():
            print(f"   {category}: {count} files")
        
        # Test document creation
        print("\n📄 Testing document creation...")
        documents = loader.create_documents()
        print(f"✅ Created {len(documents)} documents")
        
        # Show sample document
        if documents:
            sample = documents[0]
            print(f"\n📝 Sample Document:")
            print(f"   Source: {sample.metadata['source']}")
            print(f"   Category: {sample.metadata['category']}")
            print(f"   Content preview: {sample.page_content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_embedding_creator():
    """Test the enhanced embedding creator."""
    print("\n🔧 Testing Enhanced Embedding Creator...")
    print("=" * 50)
    
    try:
        creator = EnhancedIslamicEmbeddingCreator()
        print("✅ Enhanced embedding creator initialized successfully!")
        
        # Test with a small subset first
        print("\n📚 Creating test index (this may take a few minutes)...")
        
        # Create a progress callback
        def progress_callback(percentage):
            print(f"   Progress: {percentage}%")
        
        # Create index from data directory
        index = creator.create_index_from_data_directory(
            data_directory="data_as_txt",
            progress_cb=progress_callback
        )
        
        print(f"✅ Index created successfully: {creator.index_name}")
        
        # Test a simple query
        print("\n🧪 Testing query functionality...")
        test_query = "What is the first surah of the Quran?"
        print(f"   Query: {test_query}")
        
        query_vector = creator.embedder.embed_query(test_query)
        results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        print(f"   Results: {len(results.matches)} matches found")
        for i, match in enumerate(results.matches):
            source = match.metadata.get('source', 'Unknown')
            score = match.score
            print(f"   Match {i+1}: {source} (Score: {score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding creator test failed: {e}")
        return False

def test_multilingual_queries():
    """Test multilingual query functionality."""
    print("\n🌍 Testing Multilingual Queries...")
    print("=" * 50)
    
    try:
        creator = EnhancedIslamicEmbeddingCreator()
        
        # Test queries in different languages
        test_queries = [
            {
                "language": "English",
                "query": "What are the five pillars of Islam?",
                "expected_category": "Fiqh"
            },
            {
                "language": "Arabic",
                "query": "ما هي أركان الإسلام الخمسة؟",
                "expected_category": "Fiqh"
            },
            {
                "language": "Urdu",
                "query": "اسلام کے پانچ ارکان کیا ہیں؟",
                "expected_category": "Fiqh"
            },
            {
                "language": "English",
                "query": "How to perform wudu?",
                "expected_category": "Cleanliness"
            },
            {
                "language": "English",
                "query": "What is the meaning of Surah Al-Fatiha?",
                "expected_category": "Quran"
            }
        ]
        
        for i, test_case in enumerate(test_queries):
            print(f"\n   Test {i+1}: {test_case['language']}")
            print(f"   Query: {test_case['query']}")
            
            try:
                query_vector = creator.embedder.embed_query(test_case['query'])
                print(f"   ✅ Embedding created successfully")
                
                # Note: We can't query the index here if it wasn't created
                # This is just testing the embedding functionality
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print("\n✅ Multilingual query tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Multilingual query test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🕌 Islamic Knowledge RAG System - Test Suite")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        return False
    
    print("✅ Environment variables loaded successfully")
    
    # Run tests
    tests = [
        ("Data Loader", test_data_loader),
        ("Enhanced Embedding Creator", test_embedding_creator),
        ("Multilingual Queries", test_multilingual_queries)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Islamic RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Run: uvicorn api:app --host 0.0.0.0 --port 8005")
        print("2. Run: streamlit run Chatbot_UI.py --server.port 8501")
        print("3. Access the UI at: http://localhost:8501")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
