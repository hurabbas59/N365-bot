import os
import time
from typing import List, Any, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import concurrent.futures
from data_loader import IslamicKnowledgeDataLoader
import re
from langdetect import detect, LangDetectException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EnhancedIslamicEmbeddingCreator:
    """Enhanced embedding creator specifically for Islamic knowledge dataset."""
    
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
        self.chunk_overlap = 150  # Good overlap for context continuity
        self.max_chunks_per_batch = 50  # Smaller batches for better memory management
        
        # Pinecone configuration
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "islamic-knowledge-index")
        
        print(f"[EnhancedEmbeddingCreator] Initialized with:")
        print(f"  - OpenAI Model: {self.embedding_model}")
        print(f"  - Pinecone Index: {self.index_name}")
        print(f"  - Chunk Size: {self.chunk_size}")
        print(f"  - Chunk Overlap: {self.chunk_overlap}")
    
    def create_index_from_data_directory(self, data_directory: str = "data_as_txt", progress_cb = None) -> Any:
        """Create Pinecone index from the data directory."""
        print(f"[EnhancedEmbeddingCreator] Starting index creation from {data_directory}")
        start_time = time.time()
        
        # Initialize data loader
        data_loader = IslamicKnowledgeDataLoader(data_directory)
        
        # Get dataset statistics
        stats = data_loader.get_dataset_stats()
        print(f"[EnhancedEmbeddingCreator] Dataset: {stats['total_files']} files, {stats['total_size_mb']} MB")
        
        if progress_cb:
            progress_cb(5)
        
        # Load all documents
        print("[EnhancedEmbeddingCreator] Loading documents...")
        documents = data_loader.create_documents()
        
        if not documents:
            raise ValueError("No documents found to process")
        
        if progress_cb:
            progress_cb(15)
        
        # Create chunks with enhanced splitting
        print("[EnhancedEmbeddingCreator] Creating chunks...")
        chunks = self._create_enhanced_chunks(documents)
        
        print(f"[EnhancedEmbeddingCreator] Created {len(chunks)} chunks")
        if progress_cb:
            progress_cb(25)
        
        # Create or recreate Pinecone index
        print("[EnhancedEmbeddingCreator] Setting up Pinecone index...")
        index = self._setup_pinecone_index()
        
        if progress_cb:
            progress_cb(35)
        
        # Process chunks in batches
        print("[EnhancedEmbeddingCreator] Processing chunks...")
        self._process_chunks_in_batches(chunks, progress_cb)
        
        total_time = time.time() - start_time
        print(f"[EnhancedEmbeddingCreator] Index creation completed in {total_time:.2f} seconds")
        
        if progress_cb:
            progress_cb(100)
        
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
            print(f"[EnhancedEmbeddingCreator] Deleting existing index: {self.index_name}")
            self.pinecone.delete_index(self.index_name)
        
        # Create new index
        print(f"[EnhancedEmbeddingCreator] Creating new index: {self.index_name}")
        
        # Get Pinecone configuration with defaults
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")
        
        print(f"[EnhancedEmbeddingCreator] Using cloud: {cloud}, region: {region}")
        
        self.pinecone.create_index(
            name=self.index_name,
            dimension=3072,  # text-embedding-3-large dimension
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        
        # Wait for index to be ready
        print("[EnhancedEmbeddingCreator] Waiting for index to be ready...")
        time.sleep(10)
        
        return self.pinecone.Index(self.index_name)
    
    def _process_chunks_in_batches(self, chunks: List[Document], progress_cb = None) -> None:
        """Process chunks in batches for better memory management."""
        total_chunks = len(chunks)
        processed = 0
        
        # Process in smaller batches
        for i in range(0, total_chunks, self.max_chunks_per_batch):
            batch = chunks[i:i + self.max_chunks_per_batch]
            batch_start = i
            batch_end = min(i + self.max_chunks_per_batch, total_chunks)
            
            print(f"[EnhancedEmbeddingCreator] Processing batch {batch_start+1}-{batch_end} of {total_chunks}")
            
            try:
                self._process_batch(batch, batch_start)
                processed += len(batch)
                
                # Update progress
                if progress_cb:
                    progress_pct = 35 + int(60 * processed / total_chunks)
                    progress_cb(progress_pct)
                
                print(f"[EnhancedEmbeddingCreator] Processed {processed}/{total_chunks} chunks")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"[EnhancedEmbeddingCreator] Error processing batch {batch_start+1}-{batch_end}: {e}")
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
                "id": f"chunk-{batch_start + j}",
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
        
        print(f"[EnhancedEmbeddingCreator] Upserted {len(records)} chunks to Pinecone")

    def _restore_rich_metadata(self, flattened_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Restore rich metadata structure from flattened Pinecone metadata."""
        restored = {}
        
        # Extract basic metadata
        for key, value in flattened_meta.items():
            if key in ['text', 'embedding_model', 'chunk_index', 'total_chunks', 'chunk_size', 'chunk_type']:
                restored[key] = value
            elif key.startswith('mobile_navigation_'):
                # Reconstruct mobile_navigation object
                if 'mobile_navigation' not in restored:
                    restored['mobile_navigation'] = {}
                nested_key = key.replace('mobile_navigation_', '')
                restored['mobile_navigation'][nested_key] = value
            elif key.startswith('related_content_'):
                # Reconstruct related_content object
                if 'related_content' not in restored:
                    restored['related_content'] = {}
                nested_key = key.replace('related_content_', '')
                restored['related_content'][nested_key] = value
            else:
                restored[key] = value
        
        return restored

    def test_query_with_metadata_restoration(self, index, query: str, top_k: int = 3):
        """Test query with metadata restoration."""
        print(f"\n🧪 Testing query: {query}")
        
        query_vector = self.embedder.embed_query(query)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"📊 Query results: {len(results.matches)} matches found")
        for i, match in enumerate(results.matches):
            # Restore rich metadata structure
            restored_meta = self._restore_rich_metadata(match.metadata)
            
            print(f"\nMatch {i+1}:")
            print(f"  Source: {restored_meta.get('source', 'Unknown')}")
            print(f"  Category: {restored_meta.get('category', 'Unknown')}")
            print(f"  Content Type: {restored_meta.get('content_type', 'Unknown')}")
            print(f"  Score: {match.score:.3f}")
            
            # Show some metadata fields
            if 'breadcrumb' in restored_meta:
                print(f"  Breadcrumb: {restored_meta['breadcrumb']}")
            if 'priority' in restored_meta:
                print(f"  Priority: {restored_meta['priority']}")
            if 'subtopics' in restored_meta:
                subtopics = restored_meta['subtopics']
                if isinstance(subtopics, str):
                    print(f"  Subtopics: {subtopics}")
                else:
                    print(f"  Subtopics: {', '.join(subtopics[:3])}...")

def test_environment():
    """Test if environment variables are loaded correctly."""
    print("🔍 Testing Environment Variables...")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OPENAI_API_KEY: {openai_key[:20]}...")
    else:
        print("❌ OPENAI_API_KEY: Not found")
    
    # Check Pinecone API key
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        print(f"✅ PINECONE_API_KEY: {pinecone_key[:20]}...")
    else:
        print("❌ PINECONE_API_KEY: Not found")
    
    # Check Pinecone index name
    index_name = os.getenv("PINECONE_INDEX_NAME", "islamic-knowledge-index")
    print(f"✅ PINECONE_INDEX_NAME: {index_name}")
    
    # Check Pinecone cloud and region
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    print(f"✅ PINECONE_CLOUD: {cloud}")
    print(f"✅ PINECONE_REGION: {region}")
    
    print("-" * 50)

def main():
    """Test the enhanced embedding creator."""
    try:
        # Test environment first
        test_environment()
        
        print("🚀 Starting Enhanced Islamic Embedding Creator...")
        creator = EnhancedIslamicEmbeddingCreator()
        
        # Create index from data directory
        index = creator.create_index_from_data_directory(
            data_directory="data_as_txt",
            progress_cb=lambda pct: print(f"Progress: {pct}%")
        )
        
        print(f"✅ Successfully created index: {creator.index_name}")
        
        # Test a simple query with metadata restoration
        test_query = "What is the first surah of the Quran?"
        creator.test_query_with_metadata_restoration(index, test_query)
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 Please check your .env file contains:")
        print("   - OPENAI_API_KEY")
        print("   - PINECONE_API_KEY")
        print("   - PINECONE_INDEX_NAME (optional)")
        print("   - PINECONE_CLOUD (optional, default: aws)")
        print("   - PINECONE_REGION (optional, default: us-east-1)")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
