from pydantic import BaseModel
from typing import Any, Dict, Optional

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    content_type: Optional[str] = None
    top_k: Optional[int] = 10

class SearchResult(BaseModel):
    source: str
    category: str
    content_type: str
    relevance_score: float
    breadcrumb: str
    priority: str
    preview: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int
    filters_applied: Dict[str, Any]

class KnowledgeBaseInfo(BaseModel):
    status: str
    total_vectors: int
    index_name: str
    knowledge_hierarchy: Dict[str, str]
    mobile_app_support: bool
    languages_supported: list[str]
    last_updated: float

class HealthCheck(BaseModel):
    status: str
    timestamp: float
    pinecone_connection: str
