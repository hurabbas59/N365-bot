import os
import time
from typing import List, Any, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    try:
        import pinecone
        Pinecone = pinecone.Pinecone
        ServerlessSpec = pinecone.ServerlessSpec
    except ImportError:
        raise ImportError("Could not import Pinecone. Please install with: pip install pinecone-client")
import concurrent.futures
from data_loader import IslamicKnowledgeDataLoader
import re
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TopicBasedIslamicEmbeddingCreator:
    """Topic-based embedding creator for Islamic knowledge dataset with filtering capability."""
    
    def __init__(self):
        # Load environment variables first
        load_dotenv()
        
        # Check OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required. Please check your .env file.")
        
        # Check Pinecone API key
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required. Please check your .env file.")
        
        # Use text-embedding-3-large for better multilingual support
        self.embedding_model = "text-embedding-3-large"
        self.embedder = OpenAIEmbeddings(
            model=self.embedding_model, 
            api_key=self.openai_api_key
        )
        
        # Optimized chunking parameters for Islamic content
        self.chunk_size = 800  # Smaller chunks for better precision
        self.chunk_overlap = 100  # Good overlap for context continuity
        self.max_chunks_per_batch = 50  # Smaller batches for better memory management
        
        # Pinecone configuration - NEW INDEX NAME
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "islamic-knowledge-topics-v2"  # New index name
        
        print(f"[TopicBasedEmbeddingCreator] Initialized with:")
        print(f"  - OpenAI Model: {self.embedding_model}")
        print(f"  - Pinecone Index: {self.index_name}")
        print(f"  - Chunk Size: {self.chunk_size}")
        print(f"  - Chunk Overlap: {self.chunk_overlap}")
    
    def clean_topic_name(self, folder_name: str) -> str:
        """Keep original folder names as topic names, just remove number prefix and replace underscores."""
        # Remove number prefix (e.g., "03_" -> "")
        cleaned = re.sub(r'^\d+_', '', folder_name)
        
        # Replace underscores with spaces
        cleaned = cleaned.replace('_', ' ')
        
        # Keep the original names as they are
        return cleaned
    
    def generate_source_url(self, file_path: str, topic_name: str) -> str:
        """Generate a source URL for answer attribution."""
        # Clean the file path
        filename = os.path.basename(file_path)
        directory = os.path.dirname(file_path)
        
        # Extract topic folder
        path_parts = directory.split(os.sep)
        topic_folder = None
        for part in path_parts:
            if re.match(r'^\d+_', part):  # Find numbered folder
                topic_folder = part
                break
        
        if not topic_folder:
            topic_folder = "general"
        
        # Create a clean URL-like path
        base_url = "islamic-knowledge"
        clean_topic = topic_folder.lower().replace('_', '-')
        clean_filename = filename.replace('.txt', '').replace('_', '-').lower()
        
        source_url = f"{base_url}/{clean_topic}/{clean_filename}"
        return source_url
    
    def extract_enhanced_metadata(self, filepath: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced metadata including topic_name and source_url."""
        # Get the directory path to determine topic
        directory = os.path.dirname(filepath)
        path_parts = directory.split(os.sep)
        
        # Find the main topic folder (numbered folder)
        topic_folder = None
        for part in path_parts:
            if re.match(r'^\d+_', part):  # Find numbered folder like "03_Hadith_Mawdat_ul_Qurba"
                topic_folder = part
                break
        
        if not topic_folder:
            topic_folder = "18_Additional_Content"  # Default
        
        # Clean topic name for user display
        topic_name = self.clean_topic_name(topic_folder)
        
        # Generate source URL
        source_url = self.generate_source_url(filepath, topic_name)
        
        # Add new metadata fields to existing metadata
        enhanced_metadata = existing_metadata.copy()
        enhanced_metadata.update({
            "topic_folder": topic_folder,  # Original folder name for filtering
            "topic_name": topic_name,      # Clean name for display
            "source_url": source_url,      # Attribution URL
            "index_version": "v2",         # Track index version
            "supports_topic_filtering": True
        })
        
        return enhanced_metadata
    
    def create_index_from_data_directory(self, data_directory: str = "data_as_txt", progress_cb = None) -> Any:
        """Create new topic-based Pinecone index from the data directory."""
        print(f"[TopicBasedEmbeddingCreator] Starting topic-based index creation from {data_directory}")
        start_time = time.time()
        
        # Initialize data loader
        data_loader = IslamicKnowledgeDataLoader(data_directory)
        
        # Get dataset statistics
        stats = data_loader.get_dataset_stats()
        print(f"[TopicBasedEmbeddingCreator] Dataset: {stats['total_files']} files, {stats['total_size_mb']} MB")
        
        if progress_cb:
            progress_cb(5)
        
        # Load all documents
        print("[TopicBasedEmbeddingCreator] Loading documents...")
        documents = data_loader.create_documents()
        
        if not documents:
            raise ValueError("No documents found to process")
        
        if progress_cb:
            progress_cb(15)
        
        # Enhance metadata with topic information
        print("[TopicBasedEmbeddingCreator] Enhancing metadata with topic information...")
        for doc in documents:
            doc.metadata = self.extract_enhanced_metadata(doc.metadata['file_path'], doc.metadata)
        
        if progress_cb:
            progress_cb(20)
        
        # Create chunks with enhanced splitting
        print("[TopicBasedEmbeddingCreator] Creating chunks...")
        chunks = self._create_enhanced_chunks(documents)
        
        print(f"[TopicBasedEmbeddingCreator] Created {len(chunks)} chunks")
        if progress_cb:
            progress_cb(30)
        
        # Get topic statistics
        topic_stats = self._get_topic_statistics(chunks)
        print(f"[TopicBasedEmbeddingCreator] Topic distribution:")
        for topic, count in topic_stats.items():
            print(f"  - {topic}: {count} chunks")
        
        if progress_cb:
            progress_cb(35)
        
        # Create or recreate Pinecone index
        print("[TopicBasedEmbeddingCreator] Setting up Pinecone index...")
        index = self._setup_pinecone_index()
        
        if progress_cb:
            progress_cb(40)
        
        # Process chunks in batches
        print("[TopicBasedEmbeddingCreator] Processing chunks...")
        self._process_chunks_in_batches(chunks, progress_cb)
        
        total_time = time.time() - start_time
        print(f"[TopicBasedEmbeddingCreator] Topic-based index creation completed in {total_time:.2f} seconds")
        
        if progress_cb:
            progress_cb(100)
        
        return index
    
    def _get_topic_statistics(self, chunks: List[Document]) -> Dict[str, int]:
        """Get statistics about topic distribution in chunks."""
        topic_counts = {}
        for chunk in chunks:
            topic_name = chunk.metadata.get('topic_name', 'Unknown')
            topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
        return dict(sorted(topic_counts.items()))
    
    def get_available_topics(self, data_directory: str = "data_as_txt") -> List[Dict[str, str]]:
        """Get list of available topics for frontend dropdown."""
        topics = []
        
        # Add "All Topics" option first
        topics.append({
            "folder_name": "all",
            "display_name": "All Topics",
            "description": "Search across all Islamic knowledge categories"
        })
        
        # Walk through data directory to find topic folders
        for item in os.listdir(data_directory):
            item_path = os.path.join(data_directory, item)
            if os.path.isdir(item_path) and re.match(r'^\d+_', item):
                topics.append({
                    "folder_name": item,
                    "display_name": self.clean_topic_name(item),
                    "description": f"Content from {self.clean_topic_name(item)}"
                })
        
        return topics
    
    def add_new_data_to_index(self, new_data_directory: str, progress_cb = None) -> Any:
        """Add new data to existing index without recreating everything."""
        print(f"[TopicBasedEmbeddingCreator] Adding new data from {new_data_directory}")
        
        # Check if index exists
        if not self.pinecone.has_index(self.index_name):
            raise ValueError(f"Index {self.index_name} does not exist. Create it first using create_index_from_data_directory()")
        
        # Load new documents
        data_loader = IslamicKnowledgeDataLoader(new_data_directory)
        new_documents = data_loader.create_documents()
        
        if not new_documents:
            print("No new documents found to add")
            return self.pinecone.Index(self.index_name)
        
        # Enhance metadata
        for doc in new_documents:
            doc.metadata = self.extract_enhanced_metadata(doc.metadata['file_path'], doc.metadata)
        
        # Create chunks
        new_chunks = self._create_enhanced_chunks(new_documents)
        print(f"[TopicBasedEmbeddingCreator] Created {len(new_chunks)} new chunks")
        
        # Get current index stats to determine starting ID
        index = self.pinecone.Index(self.index_name)
        stats = index.describe_index_stats()
        current_count = stats.total_vector_count
        
        # Process new chunks starting from current count
        self._process_chunks_in_batches(new_chunks, progress_cb, starting_id=current_count)
        
        print(f"[TopicBasedEmbeddingCreator] Successfully added {len(new_chunks)} new chunks to index")
        return index
    
    def _create_enhanced_chunks(self, documents: List[Document]) -> List[Document]:
        """Create enhanced chunks with language-aware splitting."""
        all_chunks = []
        
        for doc in documents:
            # Use language-specific splitting
            chunks = self._split_document_language_aware(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_document_language_aware(self, document: Document) -> List[Document]:
        """Split document with language-aware chunking."""
        content = document.page_content
        metadata = document.metadata.copy()
        
        # Detect if content contains Arabic/Urdu text
        if self._is_arabic_urdu_content(content):
            chunks = self._split_arabic_urdu_content(content)
        else:
            chunks = self._split_standard_content(content)
        
        # Create chunk documents with enhanced metadata
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_type": "arabic_urdu" if self._is_arabic_urdu_content(chunk) else "standard"
            })
            
            chunk_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            chunk_documents.append(chunk_doc)
        
        return chunk_documents
    
    def _is_arabic_urdu_content(self, text: str) -> bool:
        """Check if text contains significant Arabic/Urdu content."""
        arabic_urdu_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text)
        return len(arabic_urdu_chars) > len(text) * 0.15  # 15% threshold
    
    def _split_arabic_urdu_content(self, content: str) -> List[str]:
        """Split Arabic/Urdu content while preserving structure."""
        chunks = []
        
        # Split by major section breaks first
        sections = re.split(r'={20,}', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Split by verse markers (for Quran)
            if 'VERSE' in section:
                verses = re.split(r'VERSE \d+:', section)
                for verse in verses:
                    if len(verse.strip()) > 50:
                        chunks.append(verse.strip())
            else:
                # Split by natural paragraph breaks
                paragraphs = re.split(r'\n\n+', section)
                for para in paragraphs:
                    if len(para.strip()) > 50:
                        chunks.append(para.strip())
        
        # Merge small chunks and split large ones
        final_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= self.chunk_size:
                current_chunk += "\n\n" + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            final_chunks.append(current_chunk)
        
        return final_chunks
    
    def _split_standard_content(self, content: str) -> List[str]:
        """Split standard content using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        return splitter.split_text(content)
    
    def _setup_pinecone_index(self) -> Any:
        """Setup Pinecone index with optimal configuration."""
        # Delete existing index if it exists
        if self.pinecone.has_index(self.index_name):
            print(f"[TopicBasedEmbeddingCreator] Deleting existing index: {self.index_name}")
            self.pinecone.delete_index(self.index_name)
        
        # Create new index
        print(f"[TopicBasedEmbeddingCreator] Creating new index: {self.index_name}")
        
        # Get Pinecone configuration with defaults
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        
        print(f"[TopicBasedEmbeddingCreator] Using cloud: {cloud}, region: {region}")
        
        self.pinecone.create_index(
            name=self.index_name,
            dimension=3072,  # text-embedding-3-large dimension
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        
        # Wait for index to be ready
        print("[TopicBasedEmbeddingCreator] Waiting for index to be ready...")
        time.sleep(10)
        
        return self.pinecone.Index(self.index_name)
    
    def _process_chunks_in_batches(self, chunks: List[Document], progress_cb = None, starting_id: int = 0) -> None:
        """Process chunks in batches for better memory management."""
        total_chunks = len(chunks)
        processed = 0
        
        # Process in smaller batches
        for i in range(0, total_chunks, self.max_chunks_per_batch):
            batch = chunks[i:i + self.max_chunks_per_batch]
            batch_start = i + starting_id
            batch_end = min(i + self.max_chunks_per_batch, total_chunks) + starting_id
            
            print(f"[TopicBasedEmbeddingCreator] Processing batch {batch_start+1}-{batch_end} of {total_chunks + starting_id}")
            
            try:
                self._process_batch(batch, batch_start)
                processed += len(batch)
                
                # Update progress
                if progress_cb:
                    progress_pct = 40 + int(55 * processed / total_chunks)
                    progress_cb(progress_pct)
                
                print(f"[TopicBasedEmbeddingCreator] Processed {processed}/{total_chunks} chunks")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"[TopicBasedEmbeddingCreator] Error processing batch {batch_start+1}-{batch_end}: {e}")
                continue
    
    def _flatten_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten complex metadata to Pinecone-compatible format."""
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # Simple types - keep as is
                flattened[key] = value
            elif isinstance(value, list):
                # Lists - convert to strings if they contain complex objects
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    flattened[key] = value
                else:
                    flattened[key] = str(value)
            elif isinstance(value, dict):
                # Dictionaries - flatten or convert to string
                if key in ['mobile_navigation', 'related_content', 'navigation_path']:
                    # Convert complex nested objects to strings
                    flattened[key] = str(value)
                else:
                    # Flatten simple nested objects
                    for nested_key, nested_value in value.items():
                        flat_key = f"{key}_{nested_key}"
                        if isinstance(nested_value, (str, int, float, bool)):
                            flattened[flat_key] = nested_value
                        else:
                            flattened[flat_key] = str(nested_value)
            else:
                # Other types - convert to string
                flattened[key] = str(value)
        
        return flattened

    def _process_batch(self, batch: List[Document], batch_start: int) -> None:
        """Process a single batch of chunks."""
        texts = [doc.page_content for doc in batch]
        metadata_list = [doc.metadata for doc in batch]
        
        # Create embeddings
        embeddings = self.embedder.embed_documents(texts)
        
        # Prepare records for Pinecone
        records = []
        for j, (text, meta, embedding) in enumerate(zip(texts, metadata_list, embeddings)):
            # Flatten metadata for Pinecone compatibility
            flattened_meta = self._flatten_metadata_for_pinecone(meta)
            
            record = {
                "id": f"topic-chunk-{batch_start + j}",
                "values": embedding,
                "metadata": {
                    **flattened_meta,
                    "text": text,
                    "embedding_model": self.embedding_model
                }
            }
            records.append(record)
        
        # Get index and upsert
        index = self.pinecone.Index(self.index_name)
        index.upsert(vectors=records)
        
        print(f"[TopicBasedEmbeddingCreator] Upserted {len(records)} chunks to Pinecone")

    def test_topic_filtering(self, index, topic_folder: str = None, query: str = "What is Islam?", top_k: int = 3):
        """Test topic-based filtering functionality."""
        print(f"\n🧪 Testing topic filtering:")
        print(f"  Topic: {topic_folder or 'All Topics'}")
        print(f"  Query: {query}")
        
        query_vector = self.embedder.embed_query(query)
        
        # Build filter for topic
        filter_dict = {}
        if topic_folder and topic_folder != "all":
            filter_dict["topic_folder"] = topic_folder
        
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        print(f"📊 Query results: {len(results.matches)} matches found")
        for i, match in enumerate(results.matches):
            meta = match.metadata
            print(f"\nMatch {i+1}:")
            print(f"  Topic: {meta.get('topic_name', 'Unknown')}")
            print(f"  Source: {meta.get('source', 'Unknown')}")
            print(f"  Source URL: {meta.get('source_url', 'Unknown')}")
            print(f"  Category: {meta.get('category', 'Unknown')}")
            print(f"  Score: {match.score:.3f}")
            print(f"  Text: {meta.get('text', '')[:100]}...")


def main():
    """Test the topic-based embedding creator."""
    try:
        print("🚀 Starting Topic-Based Islamic Embedding Creator...")
        creator = TopicBasedIslamicEmbeddingCreator()
        
        # Show available topics
        print("\n📋 Available Topics:")
        topics = creator.get_available_topics()
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic['display_name']} ({topic['folder_name']})")
        
        # Create index from data directory
        print("\n🔨 Creating topic-based index...")
        index = creator.create_index_from_data_directory(
            data_directory="data_as_txt",
            progress_cb=lambda pct: print(f"Progress: {pct}%")
        )
        
        print(f"✅ Successfully created topic-based index: {creator.index_name}")
        
        # Test topic filtering
        print("\n🧪 Testing topic-based filtering...")
        
        # Test with all topics
        creator.test_topic_filtering(index, topic_folder=None, query="How to perform prayer?")
        
        # Test with specific topic
        creator.test_topic_filtering(index, topic_folder="07_Namaz_Prayers", query="How to perform prayer?")
        
        # Test with another specific topic
        creator.test_topic_filtering(index, topic_folder="04_Kitab_ul_Etiqadia", query="What is faith in Islam?")
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 Please check your .env file contains:")
        print("   - OPENAI_API_KEY")
        print("   - PINECONE_API_KEY")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
