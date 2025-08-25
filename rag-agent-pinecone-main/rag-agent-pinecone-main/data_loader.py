import os
import glob
from typing import List, Dict, Any
from langchain.docstore.document import Document
import re
from pathlib import Path

class IslamicKnowledgeDataLoader:
    """Specialized data loader for Islamic knowledge dataset with enhanced metadata."""
    
    def __init__(self, data_directory: str = "data_as_txt"):
        self.data_directory = data_directory
        self.supported_extensions = ['.txt']
        
        # Define the knowledge hierarchy structure
        self.knowledge_hierarchy = {
            "01_Quran": {
                "name": "Quran",
                "description": "Holy Quran - Complete text with translations",
                "icon": "📖",
                "subtopics": ["Surahs", "Verses", "Translations", "Tafsir"]
            },
            "02_Fiqh_Al_Ahwat": {
                "name": "Islamic Jurisprudence",
                "description": "Islamic legal rulings and religious law",
                "icon": "⚖️",
                "subtopics": ["Halal/Haram", "Religious Obligations", "Legal Rulings"]
            },
            "03_Hadith_Mawdat_ul_Qurba": {
                "name": "Prophetic Traditions",
                "description": "Sayings and actions of Prophet Muhammad (PBUH)",
                "icon": "🕌",
                "subtopics": ["Narrations", "Authenticity", "Context"]
            },
            "04_Kitab_ul_Etiqadia": {
                "name": "Islamic Theology",
                "description": "Core beliefs and theological concepts",
                "icon": "🕯️",
                "subtopics": ["Aqeedah", "Beliefs", "Faith Principles"]
            },
            "05_Awrad_Prayers": {
                "name": "Spiritual Recitations",
                "description": "Daily spiritual practices and supplications",
                "icon": "🙏",
                "subtopics": ["Morning/Evening", "Daily Wazaif", "Spiritual Practices"]
            },
            "06_Dua_Collection": {
                "name": "Islamic Supplications",
                "description": "Collection of prayers and supplications",
                "icon": "🤲",
                "subtopics": ["Daily Duas", "Special Occasions", "Personal Prayers"]
            },
            "07_Namaz_Prayers": {
                "name": "Prayer Guidelines",
                "description": "How to perform Islamic prayers",
                "icon": "🕐",
                "subtopics": ["Fard", "Sunnah", "Nawafil", "Prayer Times"]
            },
            "08_Taharat_Cleanliness": {
                "name": "Purification Rules",
                "description": "Islamic cleanliness and purification",
                "icon": "💧",
                "subtopics": ["Wudu", "Ghusl", "Tayammum", "Cleanliness"]
            },
            "09_Zakat_Khums": {
                "name": "Islamic Charity",
                "description": "Zakat, Khums and charitable giving",
                "icon": "💰",
                "subtopics": ["Zakat", "Khums", "Sadaqah", "Charity Rules"]
            },
            "10_Ramzan_Fasting": {
                "name": "Fasting Guidelines",
                "description": "Ramadan fasting and related practices",
                "icon": "🌙",
                "subtopics": ["Fasting Rules", "Suhoor", "Iftar", "Laylatul Qadr"]
            },
            "11_Nikah_Marriage": {
                "name": "Marriage Laws",
                "description": "Islamic marriage and family law",
                "icon": "💍",
                "subtopics": ["Marriage Contract", "Rights", "Obligations", "Divorce"]
            },
            "12_Mayat_Death_Rites": {
                "name": "Funeral Procedures",
                "description": "Islamic funeral and burial practices",
                "icon": "⚰️",
                "subtopics": ["Ghusl", "Kafan", "Burial", "Condolences"]
            },
            "13_Ayam_Special_Days": {
                "name": "Special Religious Days",
                "description": "Important Islamic dates and occasions",
                "icon": "📅",
                "subtopics": ["Eid", "Muharram", "Ramadan", "Hajj"]
            },
            "14_Kalmay": {
                "name": "Declarations of Faith",
                "description": "Core Islamic declarations and beliefs",
                "icon": "✋",
                "subtopics": ["Kalma", "Shahada", "Faith Statements"]
            },
            "15_Buzurgan_e_Deen": {
                "name": "Religious Personalities",
                "description": "Important figures in Islamic history",
                "icon": "👤",
                "subtopics": ["Prophets", "Companions", "Scholars", "Saints"]
            },
            "16_Daily_Wazaif": {
                "name": "Daily Spiritual Practices",
                "description": "Daily recitations and spiritual exercises",
                "icon": "📿",
                "subtopics": ["Morning", "Evening", "Night", "Weekly"]
            },
            "17_Question_Answer": {
                "name": "Religious Q&A",
                "description": "Common questions and answers about Islam",
                "icon": "❓",
                "subtopics": ["Fiqh Questions", "Aqeedah", "Daily Life", "Rituals"]
            },
            "18_Additional_Content": {
                "name": "Additional Islamic Content",
                "description": "Supplementary Islamic knowledge and resources",
                "icon": "📚",
                "subtopics": ["Quranic Duas", "Events", "Stories", "Lessons"]
            }
        }
        
    def get_all_text_files(self) -> List[str]:
        """Get all text files from the data directory recursively."""
        text_files = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if any(file.endswith(ext) for ext in self.supported_extensions):
                    file_path = os.path.join(root, file)
                    text_files.append(file_path)
        
        print(f"Found {len(text_files)} text files in {self.data_directory}")
        return text_files
    
    def extract_metadata_from_filename(self, filepath: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from filename and path."""
        filename = os.path.basename(filepath)
        directory = os.path.dirname(filepath)
        
        # Extract category and hierarchy information
        category_info = self._extract_category_from_path(directory)
        hierarchy_info = self._get_hierarchy_info(directory)
        
        # Base metadata
        metadata = {
            "source": filename,
            "file_path": filepath,
            "directory": directory,
            "category": category_info["category"],
            "category_id": category_info["category_id"],
            "category_name": category_info["category_name"],
            "category_description": category_info["category_description"],
            "category_icon": category_info["category_icon"],
            "subtopics": category_info["subtopics"],
            "navigation_path": hierarchy_info["navigation_path"],
            "breadcrumb": hierarchy_info["breadcrumb"],
            "mobile_app_support": True,
            "content_type": "text",
            "language_support": ["Arabic", "Urdu", "English"]
        }
        
        # Special handling for Quran files
        if 'Quran' in directory or 'Surah' in filename:
            surah_info = self._extract_surah_info(filename)
            metadata.update(surah_info)
            metadata["content_type"] = "quranic_text"
            metadata["priority"] = "high"  # Quran has highest priority
        
        # Special handling for Q&A files
        elif 'Question_Answer' in directory or 'QA' in filename:
            qa_info = self._extract_qa_info(filename)
            metadata.update(qa_info)
            metadata["content_type"] = "question_answer"
            metadata["priority"] = "high"  # Q&A is very important for users
        
        # Special handling for other categories
        elif 'Hadith' in directory:
            metadata["content_type"] = "hadith"
            metadata["priority"] = "high"
        elif 'Fiqh' in directory:
            metadata["content_type"] = "fiqh"
            metadata["priority"] = "high"
        elif 'Dua' in directory:
            metadata["content_type"] = "dua"
            metadata["priority"] = "medium"
        elif 'Namaz' in directory or 'Prayer' in directory:
            metadata["content_type"] = "prayer"
            metadata["priority"] = "high"
        elif 'Zakat' in directory or 'Khums' in directory:
            metadata["content_type"] = "charity"
            metadata["priority"] = "medium"
        elif 'Ramzan' in directory or 'Fasting' in directory:
            metadata["content_type"] = "fasting"
            metadata["priority"] = "medium"
        elif 'Nikah' in directory or 'Marriage' in directory:
            metadata["content_type"] = "marriage"
            metadata["priority"] = "medium"
        elif 'Mayat' in directory or 'Death' in directory:
            metadata["content_type"] = "funeral"
            metadata["priority"] = "medium"
        elif 'Ayam' in directory:
            metadata["content_type"] = "special_days"
            metadata["priority"] = "medium"
        elif 'Wazaif' in directory:
            metadata["content_type"] = "daily_practices"
            metadata["priority"] = "medium"
        elif 'Buzurgan' in directory:
            metadata["content_type"] = "personalities"
            metadata["priority"] = "low"
        elif 'Kalmay' in directory:
            metadata["content_type"] = "faith_declarations"
            metadata["priority"] = "high"
        
        # Add mobile app specific metadata
        metadata.update(self._get_mobile_app_metadata(metadata))
        
        return metadata
    
    def _extract_category_from_path(self, directory: str) -> Dict[str, Any]:
        """Extract detailed category information from directory path."""
        dir_parts = directory.split(os.sep)
        
        # Find the main category directory
        for part in dir_parts:
            if part in self.knowledge_hierarchy:
                hierarchy = self.knowledge_hierarchy[part]
                return {
                    "category": hierarchy["name"],
                    "category_id": part,
                    "category_name": hierarchy["name"],
                    "category_description": hierarchy["description"],
                    "category_icon": hierarchy["icon"],
                    "subtopics": hierarchy["subtopics"]
                }
        
        # Default category
        return {
            "category": "General",
            "category_id": "general",
            "category_name": "General Islamic Knowledge",
            "category_description": "General Islamic content and information",
            "category_icon": "📖",
            "subtopics": ["General", "Miscellaneous"]
        }
    
    def _get_hierarchy_info(self, directory: str) -> Dict[str, Any]:
        """Get navigation hierarchy information for mobile app."""
        dir_parts = directory.split(os.sep)
        navigation_path = []
        breadcrumb = []
        
        for part in dir_parts:
            if part in self.knowledge_hierarchy:
                hierarchy = self.knowledge_hierarchy[part]
                navigation_path.append({
                    "id": part,
                    "name": hierarchy["name"],
                    "icon": hierarchy["icon"],
                    "description": hierarchy["description"]
                })
                breadcrumb.append(hierarchy["name"])
        
        return {
            "navigation_path": navigation_path,
            "breadcrumb": " > ".join(breadcrumb) if breadcrumb else "Islamic Knowledge"
        }
    
    def _get_mobile_app_metadata(self, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mobile app specific metadata."""
        return {
            "mobile_navigation": {
                "category_id": base_metadata.get("category_id", "general"),
                "category_name": base_metadata.get("category_name", "General"),
                "subtopics": base_metadata.get("subtopics", []),
                "breadcrumb": base_metadata.get("breadcrumb", ""),
                "icon": base_metadata.get("category_icon", "📖")
            },
            "search_tags": self._generate_search_tags(base_metadata),
            "related_content": self._get_related_content_suggestions(base_metadata),
            "user_interaction": {
                "bookmarkable": True,
                "shareable": True,
                "offline_available": True,
                "last_updated": "2024-01-01"
            }
        }
    
    def _generate_search_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate search tags for better discoverability."""
        tags = []
        
        # Add category tags
        tags.append(metadata.get("category_name", "General"))
        tags.extend(metadata.get("subtopics", []))
        
        # Add content type tags
        content_type = metadata.get("content_type", "text")
        tags.append(content_type)
        
        # Add language tags
        tags.extend(metadata.get("language_support", []))
        
        # Add priority tags
        priority = metadata.get("priority", "medium")
        tags.append(f"priority_{priority}")
        
        return list(set(tags))  # Remove duplicates
    
    def _get_related_content_suggestions(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate related content suggestions for mobile app."""
        suggestions = []
        
        # Suggest content from same category
        suggestions.append({
            "type": "same_category",
            "description": f"More {metadata.get('category_name', 'content')}",
            "category_id": metadata.get("category_id", "general")
        })
        
        # Suggest related categories based on content type
        content_type = metadata.get("content_type", "text")
        if content_type == "prayer":
            suggestions.append({
                "type": "related_category",
                "description": "Purification rules for prayer",
                "category_id": "08_Taharat_Cleanliness"
            })
        elif content_type == "fasting":
            suggestions.append({
                "type": "related_category",
                "description": "Daily spiritual practices",
                "category_id": "16_Daily_Wazaif"
            })
        
        return suggestions
    
    def _extract_surah_info(self, filename: str) -> Dict[str, Any]:
        """Extract detailed Surah information from filename."""
        # Pattern: Surah_001_Al-Fatiha.txt
        surah_match = re.search(r'Surah_(\d+)_(.+)\.txt', filename)
        if surah_match:
            surah_number = int(surah_match.group(1))
            surah_name = surah_match.group(2)
            return {
                "surah_number": surah_number,
                "surah_name": surah_name,
                "type": "Quranic Text",
                "content_subtype": "surah",
                "quran_reference": f"Quran {surah_number}:{surah_name}",
                "mobile_navigation": {
                    "quran_section": True,
                    "surah_number": surah_number,
                    "surah_name": surah_name
                }
            }
        return {}
    
    def _extract_qa_info(self, filename: str) -> Dict[str, Any]:
        """Extract detailed Q&A information from filename."""
        # Pattern: QA_Topic_01_طھارت.txt
        qa_match = re.search(r'QA_Topic_(\d+)_(.+)\.txt', filename)
        if qa_match:
            topic_number = int(qa_match.group(1))
            topic_name = qa_match.group(2)
            return {
                "topic_number": topic_number,
                "topic_name": topic_name,
                "type": "Question & Answer",
                "content_subtype": "qa_topic",
                "qa_reference": f"Q&A Topic {topic_number}: {topic_name}",
                "mobile_navigation": {
                    "qa_section": True,
                    "topic_number": topic_number,
                    "topic_name": topic_name
                }
            }
        return {}
    
    def load_text_file(self, filepath: str) -> str:
        """Load text content from a file with proper encoding handling."""
        try:
            # Try UTF-8 first (most common for Arabic/Urdu)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            try:
                # Try Windows-1256 (Arabic encoding)
                with open(filepath, 'r', encoding='windows-1256') as f:
                    content = f.read()
                    return content
            except UnicodeDecodeError:
                try:
                    # Try ISO-8859-6 (Arabic encoding)
                    with open(filepath, 'r', encoding='iso-8859-6') as f:
                        content = f.read()
                        return content
                except UnicodeDecodeError:
                    # Last resort: try with error handling
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        print(f"Warning: Used error handling for {filepath}")
                        return content
    
    def create_documents(self) -> List[Document]:
        """Create LangChain documents from all text files with enhanced metadata."""
        text_files = self.get_all_text_files()
        documents = []
        
        print(f"Processing {len(text_files)} text files...")
        
        for filepath in text_files:
            try:
                # Load text content
                content = self.load_text_file(filepath)
                if not content.strip():
                    continue
                
                # Extract comprehensive metadata
                metadata = self.extract_metadata_from_filename(filepath)
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
                print(f"Loaded: {metadata['source']} ({metadata['category']}) - {metadata['content_type']}")
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        print(f"Successfully created {len(documents)} documents")
        return documents
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        text_files = self.get_all_text_files()
        categories = {}
        content_types = {}
        total_files = len(text_files)
        total_size = 0
        
        for filepath in text_files:
            try:
                # Get file size
                file_size = os.path.getsize(filepath)
                total_size += file_size
                
                # Get metadata
                metadata = self.extract_metadata_from_filename(filepath)
                category = metadata.get('category', 'Unknown')
                content_type = metadata.get('content_type', 'Unknown')
                
                categories[category] = categories.get(category, 0) + 1
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
            except Exception as e:
                print(f"Error getting stats for {filepath}: {e}")
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "categories": categories,
            "content_types": content_types,
            "knowledge_hierarchy": self.knowledge_hierarchy,
            "supported_extensions": self.supported_extensions,
            "mobile_app_support": True
        }

def main():
    """Test the enhanced data loader."""
    loader = IslamicKnowledgeDataLoader()
    
    # Get dataset statistics
    stats = loader.get_dataset_stats()
    print("\n=== ENHANCED DATASET STATISTICS ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']} MB")
    print(f"Categories: {len(stats['categories'])}")
    print(f"Content types: {len(stats['content_types'])}")
    
    print("\n📊 Category Breakdown:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count} files")
    
    print("\n📊 Content Type Breakdown:")
    for content_type, count in stats['content_types'].items():
        print(f"  {content_type}: {count} files")
    
    # Create documents
    print("\n=== CREATING ENHANCED DOCUMENTS ===")
    documents = loader.create_documents()
    
    print(f"\nSuccessfully processed {len(documents)} documents")
    
    # Show sample document with enhanced metadata
    if documents:
        print("\n=== SAMPLE ENHANCED DOCUMENT ===")
        sample_doc = documents[0]
        print(f"Source: {sample_doc.metadata['source']}")
        print(f"Category: {sample_doc.metadata['category']}")
        print(f"Content Type: {sample_doc.metadata['content_type']}")
        print(f"Priority: {sample_doc.metadata.get('priority', 'N/A')}")
        print(f"Navigation: {sample_doc.metadata['breadcrumb']}")
        print(f"Search Tags: {', '.join(sample_doc.metadata['search_tags'][:5])}...")
        print(f"Content preview: {sample_doc.page_content[:200]}...")

if __name__ == "__main__":
    main()
