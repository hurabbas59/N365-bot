import os
from langchain_openai import OpenAIEmbeddings
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

# Initialize embedder
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

async def search_documents_by_topic(
    pinecone_index: Any, 
    urdu_query: str, 
    topic_folder: str = None,
    top_k: int = 3
) -> List[Dict]:
    """Search documents using Urdu query with topic filtering."""
    try:
        print("\n" + "="*80)
        print("🔍 STARTING TOPIC-BASED DOCUMENT SEARCH")
        print("="*80)
        
        # Get index statistics first
        try:
            index_stats = pinecone_index.describe_index_stats()
            total_vectors = index_stats.total_vector_count
            print(f"📊 INDEX STATISTICS:")
            print(f"   📈 Total vectors in index: {total_vectors:,}")
            if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
                for namespace, stats in index_stats.namespaces.items():
                    print(f"   📁 Namespace '{namespace}': {stats.vector_count:,} vectors")
        except Exception as e:
            print(f"⚠️ Could not get index stats: {e}")
            total_vectors = "Unknown"
        
        print(f"\n🎯 SEARCH CONFIGURATION:")
        print(f"   📂 Topic filter: {topic_folder or 'All Topics'}")
        print(f"   📝 Urdu query: '{urdu_query}'")
        print(f"   🔢 Top-K: {top_k}")
        
        # Create single embedding for Urdu query
        print(f"\n🔄 CREATING URDU EMBEDDING:")
        try:
            print(f"   🔄 Embedding: '{urdu_query[:50]}{'...' if len(urdu_query) > 50 else ''}'")
            query_vector = embedder.embed_query(urdu_query)
            print(f"   ✅ Embedding created (dimension: 3072)")
        except Exception as e:
            print(f"   ❌ Error creating embedding: {e}")
            raise Exception("Failed to create query embedding")
        
        # Build topic filter and show filtering logic
        filter_dict = {}
        if topic_folder and topic_folder != "all":
            filter_dict["topic_folder"] = topic_folder
            print(f"\n🎯 TOPIC FILTERING APPLIED:")
            print(f"   📂 Filtering by topic_folder = '{topic_folder}'")
            print(f"   🔍 This will search ONLY chunks from this topic")
            print(f"   📊 Expected reduction in search space: ~90-95%")
        else:
            print(f"\n🌍 NO TOPIC FILTERING:")
            print(f"   📂 Searching across ALL topics")
            print(f"   📊 Will search all {total_vectors} vectors in index")
        
        # Simple single vector search
        print(f"\n🔍 EXECUTING SIMPLE SEARCH:")
        try:
            print(f"   ⚡ Single vector search with top_k={top_k}...")
            
            results = pinecone_index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            if results.matches:
                print(f"   ✅ Found {len(results.matches)} matches")
                print(f"   📊 Score range: {results.matches[0].score:.3f} to {results.matches[-1].score:.3f}")
                top_matches = results.matches
            else:
                print(f"   ⚠️ No matches found")
                top_matches = []
                
        except Exception as e:
            print(f"   ❌ Error in vector search: {e}")
            top_matches = []
        
        print(f"\n📊 SEARCH RESULTS:")
        print(f"   🏆 Selected {len(top_matches)} matches")
        
        # Show final topic distribution
        final_topics = {}
        for match in top_matches:
            topic = match.metadata.get('topic_name', 'Unknown')
            final_topics[topic] = final_topics.get(topic, 0) + 1
        
        print(f"\n📂 FINAL TOPIC DISTRIBUTION:")
        for topic, count in sorted(final_topics.items()):
            print(f"   - {topic}: {count} chunks")
        
        # Convert to list of dictionaries with enhanced metadata
        print(f"\n📄 FINAL SELECTED DOCUMENTS:")
        documents = []
        for i, match in enumerate(top_matches):
            meta = match.metadata or {}
            doc_info = {
                "text": meta.get("text", ""),
                "source": meta.get("source", "Unknown"),
                "source_url": meta.get("source_url", ""),
                "category": meta.get("category", "General"),
                "topic_name": meta.get("topic_name", "General"),
                "topic_folder": meta.get("topic_folder", ""),
                "content_type": meta.get("content_type", "text"),
                "priority": meta.get("priority", "medium"),
                "score": float(match.score) if hasattr(match, 'score') else 0.0
            }
            documents.append(doc_info)
            
            print(f"   📄 #{i+1}: {doc_info['topic_name']} | {doc_info['source']} | Score: {doc_info['score']:.3f}")
            print(f"      🔗 URL: {doc_info['source_url']}")
            print(f"      📝 Preview: {doc_info['text'][:100]}{'...' if len(doc_info['text']) > 100 else ''}")
            print()
        
        print("="*80)
        print(f"✅ SEARCH COMPLETED: {len(documents)} documents selected for context")
        print("="*80)
        
        return documents
        
    except Exception as e:
        print(f"\n❌ SEARCH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

def prepare_context_from_documents_with_attribution(documents: List[Dict]) -> str:
    """Prepare context string from retrieved documents with source attribution."""
    if not documents:
        return ""
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        if doc["text"].strip():
            # Enhanced attribution with topic and source URL
            attribution = f"[Source {i}: {doc['topic_name']} - {doc['source']}"
            if doc.get('source_url'):
                attribution += f" | URL: {doc['source_url']}"
            attribution += f" | Category: {doc['category']}]"
            
            context_part = f"{attribution}\n{doc['text']}"
            context_parts.append(context_part)
    
    return "\n\n---\n\n".join(context_parts)

async def get_relevant_documents_by_topic(
    pinecone_index: Any, 
    urdu_query: str, 
    topic_folder: str = None
) -> str:
    """Get relevant documents with topic filtering and return prepared context."""
    documents = await search_documents_by_topic(
        pinecone_index, 
        urdu_query, 
        topic_folder
    )
    return prepare_context_from_documents_with_attribution(documents)

def get_available_topics_from_index(pinecone_index: Any) -> List[Dict[str, str]]:
    """Get available topics from the index by querying unique topic_folder values."""
    try:
        print("🔍 Retrieving available topics from index...")
        
        # Try to get index stats first
        stats = pinecone_index.describe_index_stats()
        print(f"📊 Index has {stats.total_vector_count} total vectors")
        
        # Query multiple times with different dummy vectors to get more diverse results
        topics_set = set()
        
        # Try several different dummy vectors to capture more topics
        dummy_vectors = [
            [0.1] * 3072,  # Small positive values
            [-0.1] * 3072,  # Small negative values
            [0.01] * 3072,  # Very small values
        ]
        
        for i, dummy_vector in enumerate(dummy_vectors):
            try:
                results = pinecone_index.query(
                    vector=dummy_vector,
                    top_k=200,  # Get even more results
                    include_metadata=True
                )
                
                print(f"🔍 Query {i+1}: Found {len(results.matches)} matches")
                
                # Extract unique topics
                for match in results.matches:
                    meta = match.metadata or {}
                    topic_folder = meta.get("topic_folder", "")
                    topic_name = meta.get("topic_name", "")
                    if topic_folder and topic_name:
                        topics_set.add((topic_folder, topic_name))
                        
            except Exception as e:
                print(f"⚠️ Error in query {i+1}: {e}")
                continue
        
        print(f"📂 Found {len(topics_set)} unique topics from index")
        
        # Convert to list and sort
        topics = []
        
        # Add "All Topics" option first
        topics.append({
            "folder_name": "all",
            "display_name": "All Topics",
            "description": "Search across all Islamic knowledge categories"
        })
        
        # Add found topics
        for topic_folder, topic_name in sorted(topics_set):
            topics.append({
                "folder_name": topic_folder,
                "display_name": topic_name,
                "description": f"Content from {topic_name}"
            })
        
        # If we didn't find enough topics, add default ones
        if len(topics) < 5:
            print("⚠️ Found fewer topics than expected, adding defaults...")
            default_topics = [
                ("03_Hadith_Mawdat_ul_Qurba", "Hadith Mawdat ul Qurba"),
                ("04_Kitab_ul_Etiqadia", "Kitab ul Etiqadia"),
                ("05_Awrad_Prayers", "Awrad Prayers"),
                ("06_Dua_Collection", "Dua Collection"),
                ("07_Namaz_Prayers", "Namaz Prayers"),
                ("08_Taharat_Cleanliness", "Taharat Cleanliness"),
                ("09_Zakat_Khums", "Zakat Khums"),
                ("10_Ramzan_Fasting", "Ramzan Fasting"),
                ("11_Nikah_Marriage", "Nikah Marriage"),
                ("12_Mayat_Death_Rites", "Mayat Death Rites"),
                ("13_Ayam_Special_Days", "Ayam Special Days"),
                ("14_Kalmay", "Kalmay"),
                ("15_Buzurgan_e_Deen", "Buzurgan e Deen"),
                ("16_Daily_Wazaif", "Daily Wazaif"),
                ("17_Question_Answer", "Question Answer"),
                ("18_Additional_Content", "Additional Content")
            ]
            
            existing_folders = {t["folder_name"] for t in topics}
            for folder, name in default_topics:
                if folder not in existing_folders:
                    topics.append({
                        "folder_name": folder,
                        "display_name": name,
                        "description": f"Content from {name}"
                    })
        
        print(f"✅ Returning {len(topics)} topics total")
        return topics
        
    except Exception as e:
        print(f"⚠️ Error retrieving topics from index: {e}")
        # Return comprehensive default topics as fallback
        return [
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

# Backward compatibility function
async def get_relevant_documents(pinecone_index: Any, question: str, urdu_query: str = "", arabic_query: str = "") -> str:
    """Backward compatibility wrapper - searches all topics."""
    return await get_relevant_documents_by_topic(pinecone_index, question, None, urdu_query, arabic_query)
