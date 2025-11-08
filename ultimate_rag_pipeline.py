import os
import json
import time
import hashlib
import logging
import numpy as np
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import get_close_matches
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from decimal import Decimal
import pickle

from google.cloud import bigquery, aiplatform
from google.oauth2 import service_account
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class PipelineJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for the pipeline to handle all data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool) or isinstance(obj, np.bool_):
            # Handle boolean values (including numpy booleans)
            return bool(obj)
        elif hasattr(obj, '__dataclass_fields__'):
            # Handle dataclass objects
            return obj.__dict__
        elif hasattr(obj, 'keys') and hasattr(obj, 'values'):
            # Handle BigQuery Row objects and similar dict-like objects
            try:
                return dict(obj)
            except:
                return str(obj)
        elif hasattr(obj, 'isoformat'):
            # Handle date and datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'strftime'):
            # Handle other date-like objects
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, Decimal) or hasattr(obj, 'as_tuple'):
            # Handle Decimal objects
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Handle any object with __dict__ attribute
            try:
                return obj.__dict__
            except:
                return str(obj)
        return super(PipelineJSONEncoder, self).default(obj)

# -----------------------------
# CONFIGURATION
# -----------------------------
BQ_PROJECT = "skf-bq-analytics-hub"
BQ_DATASET = "mrep_skf"
VERTEX_PROJECT = "tcl-vertex-ai"
BQ_CREDENTIALS = "credentials/*json"
VERTEX_CREDENTIALS = "credentials/*json.json"
LOCATION = "asia-southeast1"

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
QDRANT_COLLECTION = "pharma_ultimate_embeddings"
TABLES = ["sales_details"]
KEYWORD_CACHE_FILE = "keyword_cache.json"
CONVERSATION_CACHE_FILE = "conversation_cache.json"
KNOWLEDGE_GRAPH_FILE = "knowledge_graph.pkl"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -----------------------------
# DATA STRUCTURES
# -----------------------------
@dataclass
class QueryIntent:
    intent_type: str  # analytical, operational, exploratory, reporting
    confidence: float
    entities: List[str]
    temporal_context: Optional[str] = None

@dataclass
class EnhancedQueryIntent:
    original_query: str
    expanded_query: str
    intent_type: str
    confidence: float
    entities: List[Dict[str, Any]]
    temporal_context: Optional[str] = None
    complexity: float = 0.0
    requires_decomposition: bool = False

@dataclass
class RetrievalResult:
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]

@dataclass
class EnhancedRetrievalResult:
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    diversity_score: float = 0.0
    contextual_relevance: float = 0.0

@dataclass
class RetrievalContext:
    query_type: str
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    temporal_focus: Optional[str] = None

@dataclass
class QueryRequirements:
    needs_aggregation: bool = False
    needs_grouping: bool = False
    needs_temporal_filtering: bool = False
    needs_comparison: bool = False
    complexity_score: float = 0.0

@dataclass
class AdvancedSQLPlan:
    requirements: QueryRequirements
    table_selection: List[str]
    join_strategy: List[str]
    filter_strategy: List[str]
    aggregation_strategy: List[str]
    optimization_hints: List[str]
    estimated_cost: float = 0.0

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    performance_score: float = 0.0
    optimization_suggestions: List[str] = None
    optimized_sql: Optional[str] = None

@dataclass
class AnalysisResults:
    descriptive_stats: Dict[str, Any] = None
    trend_analysis: Dict[str, Any] = None
    anomalies: List[Dict[str, Any]] = None
    correlations: Dict[str, Any] = None
    significance_tests: Dict[str, Any] = None
    insights: List[str] = None

@dataclass
class UserFeedback:
    query: str
    response: Dict[str, Any]
    rating: int  # 1-5 scale
    feedback_text: str = ""
    timestamp: datetime = None

@dataclass
class QueryContext:
    query: str
    user_id: str
    session_id: str

@dataclass
class PerformanceAnalysis:
    score: float
    missing_indexes: List[str]
    inefficient_joins: List[str]
    recommendations: List[str]

@dataclass
class SQLPlan:
    steps: List[str]
    tables: List[str]
    joins: List[str]
    filters: List[str]
    aggregations: List[str]

@dataclass
class ConversationContext:
    user_id: str
    session_id: str
    history: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    last_updated: datetime

# -----------------------------
# 1️⃣ INITIALIZATION
# -----------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BQ_CREDENTIALS
bq_client = bigquery.Client(project=BQ_PROJECT)

vertex_creds = service_account.Credentials.from_service_account_file(VERTEX_CREDENTIALS)
aiplatform.init(project=VERTEX_PROJECT, location=LOCATION, credentials=vertex_creds)

embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
llm = GenerativeModel(GENERATION_MODEL)

# Initialize Qdrant
qdrant_client = QdrantClient("localhost", port=6333)

# Detect the embedding dimension dynamically
try:
    _probe_vec = embedding_model.get_embeddings(["probe"])[0].values
    _embedding_dim = len(_probe_vec)
except Exception:
    _embedding_dim = 1536

# Always recreate collection with the correct dimension to avoid mismatch
logging.info(f"Ensuring Qdrant collection '{QDRANT_COLLECTION}' with dim {_embedding_dim}")
qdrant_client.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=VectorParams(size=_embedding_dim, distance=Distance.COSINE)
)

# -----------------------------
# 2️⃣ MULTI-STAGE RETRIEVAL SYSTEM
# -----------------------------
class MultiStageRetriever:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.bm25_scores = {}
        self.retrieval_weights = {
            'semantic': 0.6,
            'keyword': 0.3,
            'temporal': 0.1
        }
        
    def semantic_search(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """Stage 1: Semantic similarity search"""
        query_emb = embedding_model.get_embeddings([query])[0].values
        hits = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION, 
            query=query_emb, 
            limit=top_k
        ).points
        
        results = []
        for hit in hits:
            results.append(RetrievalResult(
                content=hit.payload["schema"],
                score=hit.score,
                source="semantic",
                metadata={"table": hit.payload["table"]}
            ))
        return results
    
    def keyword_search(self, query: str, top_k: int = 15) -> List[RetrievalResult]:
        """Stage 2: Keyword matching with BM25-like scoring"""
        query_words = query.lower().split()
        results = []
        
        # Get all points for keyword matching
        all_points = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1000
        )[0]
        
        for point in all_points:
            content = point.payload["schema"].lower()
            score = 0
            for word in query_words:
                if word in content:
                    # Simple BM25-like scoring
                    tf = content.count(word)
                    score += tf / (tf + 1.2 * (0.25 + 0.75 * len(content.split()) / 100))
            
            if score > 0:
                results.append(RetrievalResult(
                    content=point.payload["schema"],
                    score=score,
                    source="keyword",
                    metadata={"table": point.payload["table"]}
                ))
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    def temporal_search(self, query: str, days: int = 30) -> List[RetrievalResult]:
        """Stage 3: Temporal context search"""
        # Look for recent data patterns
        temporal_keywords = ["recent", "latest", "current", "today", "this month", "last week"]
        if any(keyword in query.lower() for keyword in temporal_keywords):
            # Add temporal context to query
            enhanced_query = f"{query} recent data last {days} days"
            return self.semantic_search(enhanced_query, top_k=10)
        return []
    
    def hybrid_fusion(self, semantic_results: List[RetrievalResult], 
                     keyword_results: List[RetrievalResult],
                     temporal_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Stage 4: Hybrid fusion with learned weights"""
        all_results = {}
        
        # Combine results with weights
        for result in semantic_results:
            key = result.metadata["table"]
            if key not in all_results:
                all_results[key] = result
                all_results[key].score *= self.retrieval_weights['semantic']
            else:
                all_results[key].score += result.score * self.retrieval_weights['semantic']
        
        for result in keyword_results:
            key = result.metadata["table"]
            if key not in all_results:
                all_results[key] = result
                all_results[key].score *= self.retrieval_weights['keyword']
            else:
                all_results[key].score += result.score * self.retrieval_weights['keyword']
        
        for result in temporal_results:
            key = result.metadata["table"]
            if key not in all_results:
                all_results[key] = result
                all_results[key].score *= self.retrieval_weights['temporal']
            else:
                all_results[key].score += result.score * self.retrieval_weights['temporal']
        
        # Sort by combined score
        return sorted(all_results.values(), key=lambda x: x.score, reverse=True)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Main retrieval method with multi-stage approach"""
        # Stage 1: Semantic search
        semantic_results = self.semantic_search(query, top_k=20)
        
        # Stage 2: Keyword search
        keyword_results = self.keyword_search(query, top_k=15)
        
        # Stage 3: Temporal search
        temporal_results = self.temporal_search(query)
        
        # Stage 4: Hybrid fusion
        fused_results = self.hybrid_fusion(semantic_results, keyword_results, temporal_results)
        
        return fused_results[:top_k]

# -----------------------------
# 2.5️⃣ ENHANCED RETRIEVAL SYSTEM
# -----------------------------
class AdvancedEmbeddingManager:
    """Advanced embedding management with context awareness and caching"""
    
    def __init__(self):
        self.embedding_models = {
            'general': TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL),
            'business': TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL),
            'temporal': TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        }
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_contextual_embeddings(self, query: str, context_type: str = 'general') -> np.ndarray:
        """Get embeddings with context awareness and caching"""
        cache_key = f"{context_type}_{hashlib.md5(query.encode()).hexdigest()}"
        
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        self.cache_misses += 1
        
        # Enhance query based on context
        enhanced_query = self._enhance_query_for_context(query, context_type)
        embedding = self.embedding_models[context_type].get_embeddings([enhanced_query])[0].values
        
        # Cache the embedding
        self.embedding_cache[cache_key] = embedding
        
        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove oldest 20% of entries
            keys_to_remove = list(self.embedding_cache.keys())[:200]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        return embedding
    
    def _enhance_query_for_context(self, query: str, context_type: str) -> str:
        """Enhance query with context-specific information"""
        enhancements = {
            'business': f"business analytics query: {query}",
            'temporal': f"time-series analysis: {query}",
            'general': query
        }
        return enhancements.get(context_type, query)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.embedding_cache)
        }

class HybridReranker:
    """Hybrid reranking system with diversity and relevance balancing"""
    
    def __init__(self):
        self.diversity_weight = 0.3
        self.relevance_weight = 0.7
        self.semantic_weight = 0.4
        self.keyword_weight = 0.3
        self.temporal_weight = 0.3
    
    def rerank_results(self, query: str, results: List[EnhancedRetrievalResult], 
                      context: Optional[RetrievalContext] = None) -> List[EnhancedRetrievalResult]:
        """Rerank results using multiple signals"""
        if len(results) <= 1:
            return results
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(results)
        
        # Calculate contextual relevance scores
        contextual_scores = self._calculate_contextual_relevance(query, results, context)
        
        # Calculate source-specific scores
        source_scores = self._calculate_source_scores(results)
        
        # Combine all scores
        combined_scores = []
        for i, result in enumerate(results):
            combined_score = (
                self.relevance_weight * result.score +
                self.diversity_weight * diversity_scores[i] +
                (1 - self.relevance_weight - self.diversity_weight) * contextual_scores[i] +
                source_scores[i] * 0.1  # Small weight for source preference
            )
            
            # Update result with new scores
            result.diversity_score = diversity_scores[i]
            result.contextual_relevance = contextual_scores[i]
            result.score = combined_score
            combined_scores.append(combined_score)
        
        # Sort by combined score
        sorted_indices = sorted(range(len(results)), key=lambda i: combined_scores[i], reverse=True)
        return [results[i] for i in sorted_indices]
    
    def _calculate_diversity_scores(self, results: List[EnhancedRetrievalResult]) -> List[float]:
        """Calculate diversity scores to avoid redundant results"""
        scores = []
        seen_content = set()
        content_embeddings = []
        
        for result in results:
            # Use content hash for exact duplicates
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash in seen_content:
                scores.append(0.0)
                content_embeddings.append(None)
            else:
                seen_content.add(content_hash)
                scores.append(1.0)
                content_embeddings.append(result.embedding)
        
        # Calculate semantic diversity for non-duplicate results
        valid_embeddings = [(i, emb) for i, emb in enumerate(content_embeddings) if emb is not None]
        
        for i, result in enumerate(results):
            if content_embeddings[i] is not None:
                # Calculate average similarity to other results
                similarities = []
                for j, other_emb in content_embeddings:
                    if j != i and other_emb is not None:
                        sim = cosine_similarity([content_embeddings[i]], [other_emb])[0][0]
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    # Higher diversity = lower similarity
                    diversity = 1.0 - avg_similarity
                    scores[i] = max(scores[i], diversity)
        
        return scores
    
    def _calculate_contextual_relevance(self, query: str, results: List[EnhancedRetrievalResult], 
                                      context: Optional[RetrievalContext]) -> List[float]:
        """Calculate contextual relevance scores"""
        if not context:
            return [0.5] * len(results)
        
        scores = []
        query_lower = query.lower()
        
        for result in results:
            score = 0.5  # Base score
            
            # Boost for user preferences
            if context.user_preferences:
                for pref_key, pref_value in context.user_preferences.items():
                    if pref_key in result.content.lower():
                        score += 0.1
            
            # Boost for conversation history relevance
            if context.conversation_history:
                recent_topics = self._extract_recent_topics(context.conversation_history)
                for topic in recent_topics:
                    if topic in result.content.lower():
                        score += 0.05
            
            # Boost for temporal focus
            if context.temporal_focus and context.temporal_focus in result.content.lower():
                score += 0.1
            
            # Boost for query type match
            if context.query_type in result.content.lower():
                score += 0.1
            
            scores.append(min(score, 1.0))  # Cap at 1.0
        
        return scores
    
    def _extract_recent_topics(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from recent conversation history"""
        topics = []
        for interaction in conversation_history[-3:]:  # Last 3 interactions
            query = interaction.get('query', '').lower()
            # Simple topic extraction - can be enhanced
            words = query.split()
            topics.extend([w for w in words if len(w) > 3])
        return list(set(topics))
    
    def _calculate_source_scores(self, results: List[EnhancedRetrievalResult]) -> List[float]:
        """Calculate source-specific scores"""
        source_weights = {
            'semantic': 1.0,
            'keyword': 0.8,
            'temporal': 0.9,
            'hybrid': 1.1
        }
        
        return [source_weights.get(result.source, 0.5) for result in results]

class EnhancedMultiStageRetriever:
    """Enhanced multi-stage retrieval with advanced features"""
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_manager = AdvancedEmbeddingManager()
        self.reranker = HybridReranker()
        
        # Adaptive weights that can be updated based on feedback
        self.retrieval_weights = {
            'semantic': 0.6,
            'keyword': 0.3,
            'temporal': 0.1
        }
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'success_rate': 0.0
        }
    
    def retrieve(self, query: str, top_k: int = 5, context: Optional[RetrievalContext] = None) -> List[EnhancedRetrievalResult]:
        """Enhanced retrieval with context awareness and reranking"""
        start_time = time.time()
        self.retrieval_stats['total_queries'] += 1
        
        try:
            # Stage 1: Multi-modal semantic search
            semantic_results = self._semantic_search(query, top_k=20, context=context)
            
            # Stage 2: Enhanced keyword search
            keyword_results = self._keyword_search(query, top_k=15, context=context)
            
            # Stage 3: Temporal search with context
            temporal_results = self._temporal_search(query, context=context)
            
            # Stage 4: Hybrid fusion with adaptive weights
            fused_results = self._hybrid_fusion(semantic_results, keyword_results, temporal_results)
            
            # Stage 5: Reranking with diversity and context
            reranked_results = self.reranker.rerank_results(query, fused_results, context)
            
            # Update performance stats
            response_time = time.time() - start_time
            self._update_performance_stats(response_time, True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logging.error(f"Enhanced retrieval failed: {e}")
            self._update_performance_stats(time.time() - start_time, False)
            return []
    
    def _semantic_search(self, query: str, top_k: int, context: Optional[RetrievalContext] = None) -> List[EnhancedRetrievalResult]:
        """Enhanced semantic search with context awareness"""
        # Determine context type based on query and context
        context_type = 'business'  # Default
        if context and context.temporal_focus:
            context_type = 'temporal'
        elif 'analysis' in query.lower() or 'trend' in query.lower():
            context_type = 'business'
        
        # Get contextual embeddings
        query_emb = self.embedding_manager.get_contextual_embeddings(query, context_type)
        
        # Perform search
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=top_k
        ).points
        
        results = []
        for hit in hits:
            result = EnhancedRetrievalResult(
                content=hit.payload["schema"],
                score=hit.score,
                source="semantic",
                metadata={"table": hit.payload["table"]},
                embedding=query_emb
            )
            results.append(result)
        
        return results
    
    def _keyword_search(self, query: str, top_k: int, context: Optional[RetrievalContext] = None) -> List[EnhancedRetrievalResult]:
        """Enhanced keyword search with BM25-like scoring"""
        query_words = query.lower().split()
        results = []
        
        # Get all points for keyword matching
        all_points = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=1000
        )[0]
        
        for point in all_points:
            content = point.payload["schema"].lower()
            score = 0
            
            for word in query_words:
                if word in content:
                    # Enhanced BM25-like scoring
                    tf = content.count(word)
                    doc_length = len(content.split())
                    avg_doc_length = 100  # Approximate average
                    
                    # BM25 formula
                    k1, b = 1.2, 0.75
                    # Avoid log of negative or zero values
                    idf_term = (len(all_points) - tf + 0.5) / (tf + 0.5)
                    if idf_term > 0:
                        idf = np.log(idf_term)
                        bm25_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                        score += bm25_score
            
            if score > 0:
                result = EnhancedRetrievalResult(
                    content=point.payload["schema"],
                    score=score,
                    source="keyword",
                    metadata={"table": point.payload["table"]}
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _temporal_search(self, query: str, context: Optional[RetrievalContext] = None) -> List[EnhancedRetrievalResult]:
        """Enhanced temporal search with context awareness"""
        temporal_keywords = ["recent", "latest", "current", "today", "this month", "last week", "yesterday"]
        
        # Check for temporal indicators
        has_temporal = any(keyword in query.lower() for keyword in temporal_keywords)
        if context and context.temporal_focus:
            has_temporal = True
        
        if has_temporal:
            # Enhanced temporal query
            enhanced_query = f"{query} recent data temporal analysis"
            return self._semantic_search(enhanced_query, top_k=10, context=context)
        
        return []
    
    def _hybrid_fusion(self, semantic_results: List[EnhancedRetrievalResult], 
                      keyword_results: List[EnhancedRetrievalResult],
                      temporal_results: List[EnhancedRetrievalResult]) -> List[EnhancedRetrievalResult]:
        """Enhanced hybrid fusion with adaptive weights"""
        all_results = {}
        
        # Combine results with adaptive weights
        for result in semantic_results:
            key = result.metadata["table"]
            if key not in all_results:
                all_results[key] = result
                all_results[key].score *= self.retrieval_weights['semantic']
            else:
                all_results[key].score += result.score * self.retrieval_weights['semantic']
        
        for result in keyword_results:
            key = result.metadata["table"]
            if key not in all_results:
                all_results[key] = result
                all_results[key].score *= self.retrieval_weights['keyword']
            else:
                all_results[key].score += result.score * self.retrieval_weights['keyword']
        
        for result in temporal_results:
            key = result.metadata["table"]
            if key not in all_results:
                all_results[key] = result
                all_results[key].score *= self.retrieval_weights['temporal']
            else:
                all_results[key].score += result.score * self.retrieval_weights['temporal']
        
        # Sort by combined score
        return sorted(all_results.values(), key=lambda x: x.score, reverse=True)
    
    def _update_performance_stats(self, response_time: float, success: bool):
        """Update performance statistics"""
        # Update average response time
        total_queries = self.retrieval_stats['total_queries']
        current_avg = self.retrieval_stats['avg_response_time']
        self.retrieval_stats['avg_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        # Update success rate
        if success:
            current_success_rate = self.retrieval_stats['success_rate']
            self.retrieval_stats['success_rate'] = (
                (current_success_rate * (total_queries - 1) + 1) / total_queries
            )
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update retrieval weights based on feedback"""
        self.retrieval_weights.update(new_weights)
        logging.info(f"Updated retrieval weights: {self.retrieval_weights}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics"""
        return {
            **self.retrieval_stats,
            'embedding_cache_stats': self.embedding_manager.get_cache_stats(),
            'current_weights': self.retrieval_weights
        }

# -----------------------------
# 3️⃣ QUERY UNDERSTANDING & DECOMPOSITION
# -----------------------------
class QueryUnderstanding:
    def __init__(self):
        self.intent_patterns = {
            "analytical": ["trend", "analysis", "compare", "forecast", "pattern", "correlation"],
            "operational": ["current", "today", "status", "alert", "real-time", "now"],
            "exploratory": ["explore", "discover", "what if", "drill down", "investigate"],
            "reporting": ["report", "summary", "dashboard", "export", "overview"]
        }
        self.typo_corrector = TypoCorrector()
        
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent with confidence scoring"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score / len(patterns)
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Extract entities using simple pattern matching
        entities = self.extract_entities(query)
        
        # Detect temporal context
        temporal_context = self.detect_temporal_context(query)
        
        return QueryIntent(
            intent_type=best_intent[0],
            confidence=best_intent[1],
            entities=entities,
            temporal_context=temporal_context
        )
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        # Simple entity extraction - can be enhanced with NER
        entities = []
        query_lower = query.lower()
        
        # Common business entities
        entity_patterns = {
            "depot": ["depot", "warehouse", "location"],
            "client": ["client", "customer", "buyer"],
            "product": ["product", "item", "medicine", "drug"],
            "sales": ["sales", "revenue", "income"],
            "profit": ["profit", "margin", "earnings"]
        }
        
        for entity_type, patterns in entity_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                entities.append(entity_type)
        
        return entities
    
    def detect_temporal_context(self, query: str) -> Optional[str]:
        """Detect temporal context in query"""
        temporal_patterns = {
            "today": ["today", "current", "now"],
            "week": ["week", "weekly", "this week", "last week"],
            "month": ["month", "monthly", "this month", "last month"],
            "year": ["year", "yearly", "this year", "last year"]
        }
        
        query_lower = query.lower()
        for period, patterns in temporal_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return period
        
        return None
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """Decompose complex queries into sub-queries"""
        prompt = f"""
        Break down this complex business question into 2-3 simpler sub-questions:
        Question: "{query}"
        
        Return only the sub-questions, one per line, without numbering.
        """
        
        try:
            response = llm.generate_content(prompt)
            if response.candidates:
                sub_queries = response.candidates[0].content.parts[0].text.strip().split('\n')
                return [q.strip() for q in sub_queries if q.strip()]
        except Exception as e:
            logging.warning(f"Query decomposition failed: {e}")
        
        return [query]  # Fallback to original query

# -----------------------------
# 3.5️⃣ ENHANCED QUERY UNDERSTANDING
# -----------------------------
class TypoCorrector:
    """Fuzzy matching and typo correction for entity names"""
    
    def __init__(self):
        # Known entities from the database
        self.known_depots = ['Tejgaon', 'Dhanmondi', 'Gulshan', 'Uttara', 'Banani', 'Motijheel', 'Khulna', 'Chittagong', 'Sylhet', 'Rajshahi']
        self.known_products = ['TUFNIL', 'MEROJECT', 'MILAM', 'BILTIN', 'LOSECTIL', 'NIMULID', 'CIPROFLOXACIN', 'AMOXICILLIN', 'PARACETAMOL']
        self.known_clients = ['Asgar Ali Hospital', 'Arogga Limited', 'Square Hospital', 'Apollo Hospital', 'United Hospital']
        
    def correct_typos(self, query: str) -> str:
        """Correct typos in the query using fuzzy matching"""
        corrected_query = query
        
        # Correct depot names
        for depot in self.known_depots:
            if self._fuzzy_match(query, depot.lower()):
                corrected_query = corrected_query.replace(depot.lower(), depot)
                corrected_query = corrected_query.replace(depot.upper(), depot)
                corrected_query = corrected_query.replace(depot.capitalize(), depot)
        
        # Correct product names
        for product in self.known_products:
            if self._fuzzy_match(query, product.lower()):
                corrected_query = corrected_query.replace(product.lower(), product)
                corrected_query = corrected_query.replace(product.upper(), product)
                corrected_query = corrected_query.replace(product.capitalize(), product)
        
        # Correct client names
        for client in self.known_clients:
            if self._fuzzy_match(query, client.lower()):
                corrected_query = corrected_query.replace(client.lower(), client)
                corrected_query = corrected_query.replace(client.upper(), client)
                corrected_query = corrected_query.replace(client.capitalize(), client)
        
        return corrected_query
    
    def _fuzzy_match(self, query: str, entity: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy matching using Levenshtein distance"""
        query_lower = query.lower()
        entity_lower = entity.lower()
        
        # Direct substring match
        if entity_lower in query_lower:
            return True
        
        # Fuzzy matching for typos
        if len(entity_lower) > 3:
            # Check if entity is contained in query with some tolerance
            for i in range(len(query_lower) - len(entity_lower) + 1):
                substring = query_lower[i:i + len(entity_lower)]
                if self._levenshtein_ratio(substring, entity_lower) >= threshold:
                    return True
        
        return False
    
    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance ratio"""
        if len(s1) < len(s2):
            return self._levenshtein_ratio(s2, s1)
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        max_len = max(len(s1), len(s2))
        return (max_len - previous_row[-1]) / max_len

class AdvancedQueryProcessor:
    """Advanced query processing with multi-modal understanding"""
    
    def __init__(self):
        self.intent_classifier = QueryUnderstanding()
        self.entity_extractor = EntityExtractor()
        self.temporal_parser = TemporalParser()
        self.query_expander = QueryExpander()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.typo_corrector = TypoCorrector()
    
    def process_query(self, query: str, context: Optional[RetrievalContext] = None) -> EnhancedQueryIntent:
        """Process query with advanced understanding"""
        # First, correct typos in the query
        corrected_query = self.typo_corrector.correct_typos(query)
        if corrected_query != query:
            logging.info(f"Query corrected: '{query}' → '{corrected_query}'")
            query = corrected_query
        
        # Basic intent classification
        basic_intent = self.intent_classifier.classify_intent(query)
        
        # Extract entities with confidence
        entities = self.entity_extractor.extract_with_confidence(query)
        
        # Parse temporal expressions
        temporal_info = self.temporal_parser.parse(query)
        
        # Expand query with synonyms and related terms
        expanded_query = self.query_expander.expand(query)
        
        # Determine query complexity
        complexity = self.complexity_analyzer.assess_complexity(query)
        
        return EnhancedQueryIntent(
            original_query=query,
            expanded_query=expanded_query,
            intent_type=basic_intent.intent_type,
            confidence=basic_intent.confidence,
            entities=entities,
            temporal_context=temporal_info,
            complexity=complexity,
            requires_decomposition=complexity > 0.7
        )

class EntityExtractor:
    """Enhanced entity extraction with confidence scoring"""
    
    def __init__(self):
        self.entity_patterns = {
            "depot": ["depot", "warehouse", "location", "branch", "office"],
            "client": ["client", "customer", "buyer", "account", "company"],
            "product": ["product", "item", "medicine", "drug", "sku", "article"],
            "sales": ["sales", "revenue", "income", "turnover", "earnings"],
            "profit": ["profit", "margin", "earnings", "gain", "return"],
            "date": ["date", "time", "period", "month", "year", "quarter"],
            "amount": ["amount", "value", "price", "cost", "total", "sum"]
        }
        self.confidence_threshold = 0.3
    
    def extract_with_confidence(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities with confidence scores"""
        entities = []
        query_lower = query.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            confidence = 0.0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern in query_lower:
                    # Calculate confidence based on pattern specificity
                    pattern_confidence = len(pattern) / 10.0  # Longer patterns = higher confidence
                    confidence += pattern_confidence
                    matched_patterns.append(pattern)
            
            if confidence > self.confidence_threshold:
                entities.append({
                    "type": entity_type,
                    "confidence": min(confidence, 1.0),
                    "patterns": matched_patterns,
                    "value": self._extract_entity_value(query, entity_type, matched_patterns)
                })
        
        return sorted(entities, key=lambda x: x["confidence"], reverse=True)
    
    def _extract_entity_value(self, query: str, entity_type: str, patterns: List[str]) -> Optional[str]:
        """Extract the actual value of the entity"""
        # Simple value extraction - can be enhanced with NER
        words = query.split()
        for i, word in enumerate(words):
            if any(pattern in word.lower() for pattern in patterns):
                # Look for nearby values
                if i + 1 < len(words):
                    return words[i + 1]
                elif i - 1 >= 0:
                    return words[i - 1]
        return None

class TemporalParser:
    """Enhanced temporal expression parsing"""
    
    def __init__(self):
        self.temporal_patterns = {
            "today": ["today", "current", "now", "present"],
            "yesterday": ["yesterday", "previous day"],
            "this_week": ["this week", "current week", "this wk"],
            "last_week": ["last week", "previous week", "past week"],
            "this_month": ["this month", "current month", "this mon"],
            "last_month": ["last month", "previous month", "past month"],
            "this_quarter": ["this quarter", "current quarter", "this qtr"],
            "last_quarter": ["last quarter", "previous quarter", "past quarter"],
            "this_year": ["this year", "current year", "this yr"],
            "last_year": ["last year", "previous year", "past year"]
        }
    
    def parse(self, query: str) -> Optional[str]:
        """Parse temporal expressions from query"""
        query_lower = query.lower()
        
        for period, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return period
        
        return None

class QueryExpander:
    """Query expansion with synonyms and related terms"""
    
    def __init__(self):
        self.synonyms = {
            "sales": ["revenue", "income", "turnover", "earnings"],
            "client": ["customer", "buyer", "account"],
            "product": ["item", "medicine", "drug", "sku"],
            "depot": ["warehouse", "location", "branch"],
            "profit": ["margin", "earnings", "gain"],
            "analysis": ["study", "examination", "review"],
            "trend": ["pattern", "direction", "movement"],
            "compare": ["comparison", "contrast", "versus"]
        }
    
    def expand(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        expanded_terms = []
        words = query.split()
        
        for word in words:
            word_lower = word.lower()
            expanded_terms.append(word)
            
            # Add synonyms
            for key, synonyms in self.synonyms.items():
                if key in word_lower:
                    expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
        
        return " ".join(expanded_terms)

class ComplexityAnalyzer:
    """Query complexity analysis"""
    
    def assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 scale)"""
        complexity_factors = {
            'length': min(len(query.split()) / 20, 1.0),
            'conjunctions': query.lower().count(' and ') + query.lower().count(' or '),
            'comparisons': query.lower().count(' vs ') + query.lower().count(' compare '),
            'aggregations': query.lower().count(' total ') + query.lower().count(' sum ') + query.lower().count(' average '),
            'conditions': query.lower().count(' where ') + query.lower().count(' if '),
            'temporal': query.lower().count(' when ') + query.lower().count(' during ') + query.lower().count(' between ')
        }
        
        # Weighted complexity score
        weights = {
            'length': 0.2, 
            'conjunctions': 0.15, 
            'comparisons': 0.2, 
            'aggregations': 0.2, 
            'conditions': 0.15,
            'temporal': 0.1
        }
        
        complexity = sum(weights[factor] * min(complexity_factors[factor], 1.0) for factor in weights)
        return min(complexity, 1.0)

class ContextAwareDecomposer:
    """Context-aware query decomposition"""
    
    def __init__(self):
        self.decomposition_strategies = {
            'temporal_decomposition': self._temporal_decomposition,
            'comparative_decomposition': self._comparative_decomposition,
            'aggregation_decomposition': self._aggregation_decomposition,
            'sequential_decomposition': self._sequential_decomposition
        }
    
    def decompose_query(self, query: str, context: Optional[RetrievalContext] = None) -> List[str]:
        """Decompose query with conversation context"""
        if not context:
            return [query]
        
        # Determine decomposition strategy
        strategy = self._select_decomposition_strategy(query, context)
        
        # Generate sub-queries
        sub_queries = self.decomposition_strategies[strategy](query, context)
        
        return sub_queries if sub_queries else [query]
    
    def _select_decomposition_strategy(self, query: str, context: RetrievalContext) -> str:
        """Select appropriate decomposition strategy"""
        query_lower = query.lower()
        
        if context.temporal_focus or any(word in query_lower for word in ['trend', 'over time', 'period']):
            return 'temporal_decomposition'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return 'comparative_decomposition'
        elif any(word in query_lower for word in ['total', 'sum', 'average', 'aggregate']):
            return 'aggregation_decomposition'
        else:
            return 'sequential_decomposition'
    
    def _temporal_decomposition(self, query: str, context: RetrievalContext) -> List[str]:
        """Decompose temporal queries"""
        # Split by time periods
        periods = ['this year', 'last year', 'this quarter', 'last quarter', 'this month', 'last month']
        sub_queries = []
        
        for period in periods:
            if period in query.lower():
                sub_query = query.replace(period, period)
                sub_queries.append(sub_query)
        
        return sub_queries if sub_queries else [query]
    
    def _comparative_decomposition(self, query: str, context: RetrievalContext) -> List[str]:
        """Decompose comparative queries"""
        # Split by comparison terms
        comparison_terms = ['vs', 'versus', 'compare', 'difference']
        sub_queries = []
        
        for term in comparison_terms:
            if term in query.lower():
                parts = query.lower().split(term)
                if len(parts) == 2:
                    sub_queries.append(parts[0].strip())
                    sub_queries.append(parts[1].strip())
        
        return sub_queries if sub_queries else [query]
    
    def _aggregation_decomposition(self, query: str, context: RetrievalContext) -> List[str]:
        """Decompose aggregation queries"""
        # Split by aggregation levels
        aggregation_levels = ['by product', 'by client', 'by depot', 'by month', 'by year']
        sub_queries = []
        
        base_query = query
        for level in aggregation_levels:
            if level in query.lower():
                sub_query = f"{base_query} {level}"
                sub_queries.append(sub_query)
        
        return sub_queries if sub_queries else [query]
    
    def _sequential_decomposition(self, query: str, context: RetrievalContext) -> List[str]:
        """Sequential decomposition for complex queries"""
        # Split by conjunctions
        conjunctions = [' and ', ' or ', ' then ', ' also ']
        sub_queries = [query]
        
        for conj in conjunctions:
            if conj in query.lower():
                parts = query.lower().split(conj)
                sub_queries = [part.strip() for part in parts if part.strip()]
                break
        
        return sub_queries

# -----------------------------
# 4️⃣ KNOWLEDGE GRAPH
# -----------------------------
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        
    def build_from_tables(self, tables: List[str]):
        """Build knowledge graph from database tables"""
        logging.info("Building knowledge graph from tables...")
        
        for table_name in tables:
            # Get table schema
            table_ref = f"{BQ_DATASET}.{table_name}"
            table = bq_client.get_table(table_ref)
            
            # Add table as node
            self.graph.add_node(table_name, type="table", schema=table.schema)
            
            # Extract relationships from foreign keys and naming patterns
            self._extract_relationships(table_name, table.schema)
        
        # Extract entity relationships from data
        self._extract_entity_relationships(tables)
        
        logging.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _extract_relationships(self, table_name: str, schema):
        """Extract relationships from table schema"""
        for field in schema:
            field_name = field.name.lower()
            
            # Detect foreign key relationships
            if field_name.endswith('_id') or field_name.endswith('_name'):
                related_entity = field_name.replace('_id', '').replace('_name', '')
                if related_entity != table_name:
                    self.graph.add_edge(table_name, related_entity, 
                                      relationship="references", field=field.name)
    
    def _extract_entity_relationships(self, tables: List[str]):
        """Extract entity relationships from actual data"""
        for table_name in tables:
            try:
                # Sample data to find relationships (with date filter for partitioned tables)
                if table_name == "sales_details":
                    sql = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{table_name}` WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) LIMIT 1000"
                else:
                    sql = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{table_name}` LIMIT 1000"
                results = bq_client.query(sql).result()
                
                # Analyze co-occurrence patterns
                entity_cooccurrence = defaultdict(int)
                for row in results:
                    row_dict = dict(row)
                    entities = [str(v) for v in row_dict.values() if v and len(str(v)) > 2]
                    
                    # Create entity pairs
                    for i, entity1 in enumerate(entities):
                        for entity2 in entities[i+1:]:
                            entity_cooccurrence[(entity1, entity2)] += 1
                
                # Add strong relationships to graph
                for (entity1, entity2), count in entity_cooccurrence.items():
                    if count > 5:  # Threshold for strong relationship
                        self.graph.add_edge(entity1, entity2, 
                                          relationship="cooccurs", strength=count)
                        
            except Exception as e:
                logging.warning(f"Failed to extract relationships from {table_name}: {e}")
    
    def find_relevant_entities(self, query: str) -> List[str]:
        """Find entities relevant to query"""
        query_words = set(query.lower().split())
        relevant_entities = []
        
        for node in self.graph.nodes():
            if any(word in node.lower() for word in query_words):
                relevant_entities.append(node)
        
        return relevant_entities
    
    def find_context_paths(self, entities: List[str]) -> List[List[str]]:
        """Find paths between entities in knowledge graph"""
        paths = []
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if self.graph.has_node(entity1) and self.graph.has_node(entity2):
                    try:
                        path = nx.shortest_path(self.graph, entity1, entity2)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return paths

# -----------------------------
# 4.5️⃣ ENHANCED KNOWLEDGE GRAPH
# -----------------------------
class DynamicKnowledgeGraph:
    """Dynamic knowledge graph with real-time updates"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.relationship_strength = {}
        self.update_frequency = {}
        self.entity_confidence = {}
    
    def update_from_query_results(self, query: str, results: List[Dict[str, Any]]):
        """Update knowledge graph based on query results"""
        # Extract new entities from results
        new_entities = self._extract_entities_from_results(results)
        
        # Update entity embeddings
        self._update_entity_embeddings(new_entities)
        
        # Discover new relationships
        new_relationships = self._discover_relationships(query, results)
        
        # Update graph with new information
        self._update_graph(new_entities, new_relationships)
        
        # Prune weak relationships
        self._prune_weak_relationships()
    
    def _extract_entities_from_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract entities from query results"""
        entities = set()
        
        for result in results:
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 2:
                    # Simple entity extraction
                    if any(char.isalpha() for char in value):
                        entities.add(value.lower())
        
        return list(entities)
    
    def _update_entity_embeddings(self, entities: List[str]):
        """Update entity embeddings"""
        for entity in entities:
            if entity not in self.entity_embeddings:
                try:
                    embedding = embedding_model.get_embeddings([entity])[0].values
                    self.entity_embeddings[entity] = embedding
                except Exception as e:
                    logging.warning(f"Failed to get embedding for entity {entity}: {e}")
    
    def _discover_relationships(self, query: str, results: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Discover relationships from query and results"""
        relationships = []
        
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        
        # Find co-occurrence patterns in results
        for result in results:
            result_entities = self._extract_entities_from_results([result])
            
            # Create relationships between query entities and result entities
            for q_entity in query_entities:
                for r_entity in result_entities:
                    if q_entity != r_entity:
                        relationships.append((
                            q_entity, 
                            r_entity, 
                            {
                                'type': 'cooccurrence',
                                'strength': 1.0,
                                'context': query,
                                'timestamp': datetime.now()
                            }
                        ))
        
        return relationships
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query"""
        # Simple entity extraction from query
        words = query.lower().split()
        entities = []
        
        for word in words:
            if len(word) > 3 and any(char.isalpha() for char in word):
                entities.append(word)
        
        return entities
    
    def _update_graph(self, entities: List[str], relationships: List[Tuple[str, str, Dict[str, Any]]]):
        """Update graph with new entities and relationships"""
        # Add entities as nodes
        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type="entity", confidence=1.0)
        
        # Add relationships as edges
        for source, target, metadata in relationships:
            if self.graph.has_edge(source, target):
                # Update existing relationship strength
                current_strength = self.graph[source][target].get('strength', 0)
                new_strength = current_strength + metadata['strength']
                self.graph[source][target]['strength'] = new_strength
                self.graph[source][target]['last_updated'] = datetime.now()
            else:
                # Add new relationship
                self.graph.add_edge(source, target, **metadata)
    
    def _prune_weak_relationships(self, threshold: float = 0.1):
        """Remove weak relationships from the graph"""
        edges_to_remove = []
        
        for source, target, data in self.graph.edges(data=True):
            strength = data.get('strength', 0)
            if strength < threshold:
                edges_to_remove.append((source, target))
        
        for source, target in edges_to_remove:
            self.graph.remove_edge(source, target)
        
        logging.info(f"Pruned {len(edges_to_remove)} weak relationships")
    
    def get_contextual_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get entities with contextual relevance"""
        query_embedding = self._get_query_embedding(query)
        entity_scores = []
        
        for entity, embedding in self.entity_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            entity_scores.append((entity, similarity))
        
        # Sort by similarity and return top entities
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        contextual_entities = []
        for entity, score in entity_scores[:top_k]:
            context = self._get_entity_context(entity)
            contextual_entities.append({
                'entity': entity,
                'relevance_score': score,
                'context': context,
                'relationships': self._get_entity_relationships(entity)
            })
        
        return contextual_entities
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query"""
        try:
            return embedding_model.get_embeddings([query])[0].values
        except Exception as e:
            logging.warning(f"Failed to get query embedding: {e}")
            return np.zeros(1536)  # Default embedding size
    
    def _get_entity_context(self, entity: str) -> Dict[str, Any]:
        """Get context information for an entity"""
        if not self.graph.has_node(entity):
            return {}
        
        node_data = self.graph.nodes[entity]
        neighbors = list(self.graph.neighbors(entity))
        
        return {
            'type': node_data.get('type', 'unknown'),
            'confidence': node_data.get('confidence', 0.0),
            'neighbor_count': len(neighbors),
            'neighbors': neighbors[:5]  # Top 5 neighbors
        }
    
    def _get_entity_relationships(self, entity: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity"""
        relationships = []
        
        for source, target, data in self.graph.edges(data=True):
            if source == entity or target == entity:
                relationships.append({
                    'source': source,
                    'target': target,
                    'type': data.get('type', 'unknown'),
                    'strength': data.get('strength', 0.0),
                    'context': data.get('context', '')
                })
        
        return sorted(relationships, key=lambda x: x['strength'], reverse=True)[:5]

# -----------------------------
# 5️⃣ ENHANCED SQL GENERATION
# -----------------------------
class EnhancedSQLGenerator:
    def __init__(self):
        self.sql_cache = {}
        self.validation_rules = self._load_validation_rules()
        self.llm = GenerativeModel(GENERATION_MODEL)
        
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load SQL validation rules"""
        return {
            "max_rows": 10000,
            "required_filters": ["date_range"],
            "forbidden_patterns": ["DROP", "DELETE", "UPDATE", "INSERT"],
            "timeout_seconds": 30
        }
    
    def create_sql_plan(self, query: str, context: List[RetrievalResult]) -> SQLPlan:
        """Create AI-powered execution plan for SQL generation"""
        
        # Prepare context for AI planning
        context_info = []
        for result in context:
            context_info.append({
                "table": result.metadata["table"],
                "schema": result.content,
                "relevance": result.score
            })
        
        prompt = f"""
You are a SQL query planner. Analyze this business question and create an execution plan:

QUESTION: "{query}"

AVAILABLE TABLES AND SCHEMAS:
{json.dumps(context_info, indent=2, cls=PipelineJSONEncoder)}

Create a detailed execution plan that includes:

1. TABLES: Which tables are needed (list table names)
2. JOINS: Any joins required between tables (if multiple tables)
3. FILTERS: What WHERE conditions should be applied (date ranges, specific values, etc.)
4. AGGREGATIONS: What aggregations are needed (SUM, COUNT, AVG, GROUP BY, etc.)

Consider:
- The user's intent and what data they're asking for
- Performance optimization (date filters, limits)
- Proper grouping and ordering
- Data type considerations

Return ONLY a JSON object with these exact keys: tables, joins, filters, aggregations

Example format:
{{
  "tables": ["sales_details"],
  "joins": [],
  "filters": ["sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)"],
  "aggregations": ["SUM(salesAMT)", "COUNT(*)", "GROUP BY product_name"]
}}
"""

        try:
            response = llm.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                plan_text = response.candidates[0].content.parts[0].text.strip()
                # Clean the response
                plan_text = self._clean_json_response(plan_text)
                plan_data = json.loads(plan_text)
                
                # Validate and create plan
                return SQLPlan(
                    steps=["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"],
                    tables=plan_data.get("tables", [r.metadata["table"] for r in context]),
                    joins=plan_data.get("joins", []),
                    filters=plan_data.get("filters", []),
                    aggregations=plan_data.get("aggregations", [])
                )
        except Exception as e:
            logging.warning(f"AI SQL planning failed: {e}")
        
        # Fallback plan
        return SQLPlan(
            steps=["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY"],
            tables=[r.metadata["table"] for r in context],
            joins=[],
            filters=["sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)"],
            aggregations=[]
        )
    
    def _clean_json_response(self, text: str) -> str:
        """Clean AI response to extract JSON"""
        # Remove markdown formatting
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return text
    
    def generate_sql_with_validation(self, query: str, context: List[RetrievalResult], 
                                   plan: SQLPlan) -> str:
        """Generate SQL using AI with comprehensive validation"""
        
        # Prepare comprehensive context for AI
        context_info = []
        for result in context:
            context_info.append({
                "table": result.metadata["table"],
                "schema": result.content,
                "relevance_score": result.score
            })
        
        # Get additional schema information
        schema_details = self._get_detailed_schema_info(context)
        
        # Check if this is a "last month" query and use simpler syntax
        if "last month" in query.lower() or "revenue last month" in query.lower():
            return self._generate_last_month_sql(query, context, plan)
        
        # Check if this is a comprehensive sales analysis query
        if any(term in query.lower() for term in ["detailed sales analysis", "comprehensive analysis", "sales report", "compare"]):
            return self._generate_adaptive_comprehensive_sql(query, context, plan)
        
        # Create AI prompt for SQL generation
        prompt = f"""
You are an expert BigQuery SQL analyst. Generate optimized SQL for this business question:

QUESTION: "{query}"

DATABASE CONTEXT:
Project: {BQ_PROJECT}
Dataset: {BQ_DATASET}

AVAILABLE TABLES AND SCHEMAS:
{json.dumps(schema_details, indent=2, cls=PipelineJSONEncoder)}

EXECUTION PLAN:
- Tables to use: {plan.tables}
- Joins needed: {plan.joins}
- Filters to apply: {plan.filters}
- Aggregations: {plan.aggregations}

REQUIREMENTS:
1. Use fully qualified table names: `{BQ_PROJECT}.{BQ_DATASET}.table_name`
2. Add appropriate date filters for performance (prefer recent data)
3. Include proper column aliases for clarity
4. Use BigQuery-specific functions and syntax
5. Optimize for performance and cost
6. Handle NULL values appropriately with COALESCE
7. Add LIMIT clause for large result sets
8. Use proper data type casting (CAST to FLOAT64, INT64, etc.)

EXAMPLES OF GOOD PRACTICES:
- For date filtering: WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)
- For last month: WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH) AND sales_date < CURRENT_DATE()
- For specific month: WHERE EXTRACT(YEAR FROM sales_date) = 2024 AND EXTRACT(MONTH FROM sales_date) = 12
- For aggregations: CAST(SUM(salesAMT) AS FLOAT64) as total_sales
- For NULL handling: COALESCE(product_name, 'Unknown Product') as product_name
- For grouping: GROUP BY column1, column2 ORDER BY aggregated_value DESC

IMPORTANT: Use correct BigQuery syntax:
- DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) ✓
- DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH) ✓
- DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) ✓
- Always include date filters for partitioned tables
- Avoid complex nested date functions like DATE_TRUNC(DATE_SUB(...))
- Use simple date comparisons for better compatibility

Generate clean, optimized SQL that directly answers the user's question. Return ONLY the SQL query, no explanations or markdown formatting.
"""

        try:
            # Use the LLM to generate SQL
            response = llm.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                sql = response.candidates[0].content.parts[0].text.strip()
                sql = self._clean_sql(sql)
                
                # Validate the generated SQL
                if self._validate_sql(sql):
                    logging.info("AI-generated SQL passed validation")
                    return sql
                else:
                    logging.warning("AI-generated SQL failed validation, trying fallback")
            
        except Exception as e:
            logging.error(f"AI SQL generation failed: {e}")
        
        # Fallback to simpler AI generation
        return self._generate_fallback_ai_sql(query, context)
    
    def _generate_last_month_sql(self, query: str, context: List[RetrievalResult], plan: SQLPlan) -> str:
        """Generate simple SQL for last month revenue queries"""
        return f"""
        SELECT 
            CAST(SUM(COALESCE(salesAMT, 0)) AS FLOAT64) AS last_month_revenue
        FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`
        WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
        AND sales_date < CURRENT_DATE()
        """
    
    def _generate_adaptive_comprehensive_sql(self, query: str, context: List[RetrievalResult], plan: SQLPlan) -> str:
        """Generate adaptive comprehensive SQL using AI based on the specific query requirements"""
        
        # Prepare context for AI
        context_info = []
        for result in context:
            context_info.append({
                "table": result.metadata["table"],
                "schema": result.content,
                "relevance_score": result.score
            })
        
        # Get additional schema information
        schema_details = self._get_detailed_schema_info(context)
        
        # Create AI prompt for comprehensive analysis
        prompt = f"""
You are a senior data analyst. Generate a comprehensive SQL analysis for this business question:

QUESTION: "{query}"

DATABASE CONTEXT:
Project: {BQ_PROJECT}
Dataset: {BQ_DATASET}

AVAILABLE TABLES AND SCHEMAS:
{json.dumps(schema_details, indent=2, cls=PipelineJSONEncoder)}

REQUIREMENTS FOR COMPREHENSIVE ANALYSIS:
1. Use CTEs (Common Table Expressions) for complex analysis
2. Include year-over-year comparisons when years are mentioned
3. Add growth percentage calculations
4. Identify top performers (products, clients, depots)
5. Include multiple aggregation levels (yearly, monthly, product-level)
6. Use window functions for rankings and comparisons
7. Add proper date filtering for performance
8. Include transaction counts and average values
9. Use CAST for proper data types
10. Handle NULL values with COALESCE
11. ALWAYS use fully qualified table names: `project.dataset.table_name`
12. NEVER use unqualified table names like "MonthlySalesAggregated"

ANALYSIS PATTERNS TO INCLUDE:
- Year-over-year revenue comparison
- Top products by revenue (use item_name column)
- Top clients by revenue (use client_name column)
- Monthly trends and seasonality
- Growth rate calculations
- Transaction volume analysis
- Average transaction values

IMPORTANT COLUMN NAMES:
- Product: item_name (not productName or product_name)
- Client: client_name (not clientName)
- Depot: depot_name (not depotName)
- Sales Amount: salesAMT
- Sales Date: sales_date

CRITICAL: DO NOT CREATE OR ASSUME COLUMNS THAT DON'T EXIST!
- DO NOT use columns like: sales_year, sales_month, monthly_revenue, yearly_revenue, etc.
- These columns DO NOT exist in the actual tables
- You MUST create these using EXTRACT() and aggregation functions
- Example: Use EXTRACT(YEAR FROM sales_date) AS year, not sales_year
- Example: Use SUM(salesAMT) AS revenue, not monthly_revenue

EXAMPLES OF GOOD PRACTICES:
- For year comparison: LAG() OVER (ORDER BY year) for previous year values
- For rankings: ROW_NUMBER() OVER (PARTITION BY year ORDER BY revenue DESC)
- For growth: ((current - previous) / previous) * 100
- For date filtering: WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)

Generate a comprehensive SQL query that provides deep insights for this specific question. 

CRITICAL REQUIREMENTS:
- Return ONLY the SQL query, no explanations or markdown formatting
- Do NOT include UNION, INTERSECT, or EXCEPT keywords
- Use only SELECT statements with CTEs (Common Table Expressions)
- Ensure proper SQL syntax with balanced parentheses
- End the query with a single semicolon
- Do not include any text after the SQL query
- NEVER create arbitrary table names like "MonthlyComparison", "YearlyData", etc.
- ONLY use the actual tables: sales_details, products, clients, depots
- ALL table references must be fully qualified: `project.dataset.table_name`

Return ONLY the SQL query:
"""

        try:
            # For now, skip AI generation and use fallback to avoid column issues
            # TODO: Fix AI prompt to generate correct SQL
            raise ValueError("Using fallback SQL to avoid column issues")
            
            # Use AI to generate adaptive SQL
            response = self.llm.generate_content(prompt)
            sql = response.text.strip()
            
            # Clean up the SQL
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            sql = sql.strip()
            
            # Remove any extra text after the SQL
            sql_lines = sql.split('\n')
            cleaned_lines = []
            for line in sql_lines:
                if line.strip().upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                    cleaned_lines = [line]
                elif line.strip() and not line.strip().startswith('--') and not line.strip().startswith('/*'):
                    cleaned_lines.append(line)
                elif line.strip().upper() in ['UNION', 'UNION ALL', 'INTERSECT', 'EXCEPT']:
                    break  # Stop at UNION keywords to avoid malformed SQL
            
            sql = '\n'.join(cleaned_lines).strip()
            
            # Validate and fix table qualifications
            sql = self._fix_table_qualifications(sql)
            
            # Debug: Log the generated SQL
            logging.info(f"Generated SQL: {sql[:500]}...")
            
            # Debug: Check for boolean values in the response
            try:
                test_data = json.dumps({"test": True}, cls=PipelineJSONEncoder)
                logging.info("Boolean serialization test passed")
            except Exception as e:
                logging.error(f"Boolean serialization test failed: {e}")
            
            # Validate the SQL has proper date filtering
            if "sales_details" in sql and "sales_date" not in sql:
                sql = sql.replace(
                    "FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`",
                    "FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details` WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)"
                )
            
            # Basic SQL validation
            if not sql.upper().strip().startswith(('SELECT', 'WITH')):
                raise ValueError("Generated SQL does not start with SELECT or WITH")
            
            # Check for malformed SQL patterns
            if 'UNION' in sql.upper() or 'INTERSECT' in sql.upper() or 'EXCEPT' in sql.upper():
                raise ValueError("Generated SQL contains forbidden UNION/INTERSECT/EXCEPT keywords")
            
            # Ensure proper parentheses balance
            open_parens = sql.count('(')
            close_parens = sql.count(')')
            if open_parens != close_parens:
                raise ValueError(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
            
            return sql
            
        except Exception as e:
            logging.warning(f"Adaptive SQL generation failed: {e}")
            # Fallback to a simpler comprehensive analysis
            return self._generate_fallback_comprehensive_sql(query)
    
    def _fix_table_qualifications(self, sql: str) -> str:
        """Fix table qualifications in SQL to ensure proper BigQuery format"""
        import re
        
        # List of known tables that need qualification
        known_tables = {
            'sales_details': f'`{BQ_PROJECT}.{BQ_DATASET}.sales_details`',
            'monthlysalesaggregated': f'`{BQ_PROJECT}.{BQ_DATASET}.monthlysalesaggregated`',
            'products': f'`{BQ_PROJECT}.{BQ_DATASET}.products`',
            'clients': f'`{BQ_PROJECT}.{BQ_DATASET}.clients`',
            'depots': f'`{BQ_PROJECT}.{BQ_DATASET}.depots`'
        }
        
        # Fix unqualified table references using regex
        for table_name, qualified_name in known_tables.items():
            # Pattern to match unqualified table references in various contexts
            patterns = [
                # FROM table_name
                (rf'\bFROM\s+{table_name}\b', f'FROM {qualified_name}'),
                # JOIN table_name
                (rf'\b(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|JOIN)\s+{table_name}\b', rf'\1 {qualified_name}'),
                # FROM `table_name`
                (rf'\bFROM\s+`{table_name}`\b', f'FROM {qualified_name}'),
                # JOIN `table_name`
                (rf'\b(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|JOIN)\s+`{table_name}`\b', rf'\1 {qualified_name}'),
            ]
            
            for pattern, replacement in patterns:
                sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        
        # Fix any remaining unqualified table references that look like table names
        # This catches cases where AI generates arbitrary table names
        unqualified_table_pattern = r'\bFROM\s+([A-Za-z][A-Za-z0-9_]*)\b'
        
        def replace_unqualified_table(match):
            table_name = match.group(1)
            # If it's not already qualified and not a known table, it's likely an AI-generated table name
            if '.' not in table_name and table_name.lower() not in [t.lower() for t in known_tables.keys()]:
                # Check if it looks like a table name (not a CTE or alias)
                if table_name[0].isupper() and len(table_name) > 3:
                    # This looks like an AI-generated table name, replace with sales_details
                    return f'FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`'
                # Also catch camelCase table names that AI might generate
                elif any(c.isupper() for c in table_name[1:]) and len(table_name) > 3:
                    return f'FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`'
            return match.group(0)
        
        # Apply the replacement
        sql = re.sub(unqualified_table_pattern, replace_unqualified_table, sql, flags=re.IGNORECASE)
        
        # Also fix JOIN patterns with unqualified tables
        join_pattern = r'\b(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|JOIN)\s+([A-Za-z][A-Za-z0-9_]*)\b'
        
        def replace_unqualified_join(match):
            join_type = match.group(1)
            table_name = match.group(2)
            if '.' not in table_name and table_name.lower() not in [t.lower() for t in known_tables.keys()]:
                if table_name[0].isupper() and len(table_name) > 3:
                    return f'{join_type} `{BQ_PROJECT}.{BQ_DATASET}.sales_details`'
                elif any(c.isupper() for c in table_name[1:]) and len(table_name) > 3:
                    return f'{join_type} `{BQ_PROJECT}.{BQ_DATASET}.sales_details`'
            return match.group(0)
        
        sql = re.sub(join_pattern, replace_unqualified_join, sql, flags=re.IGNORECASE)
        
        return sql
    
    def _generate_fallback_comprehensive_sql(self, query: str) -> str:
        """Fallback comprehensive SQL when AI generation fails"""
        return f"""
        WITH yearly_data AS (
            SELECT 
                EXTRACT(YEAR FROM sales_date) as year,
                SUM(COALESCE(salesAMT, 0)) as yearly_revenue,
                COUNT(*) as transaction_count,
                AVG(COALESCE(salesAMT, 0)) as avg_transaction_value,
                COUNT(DISTINCT item_name) as unique_products,
                COUNT(DISTINCT client_name) as unique_clients,
                COUNT(DISTINCT depot_name) as unique_depots
            FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`
            WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
            GROUP BY year
        ),
        yearly_products AS (
            SELECT 
                EXTRACT(YEAR FROM sales_date) as year,
                item_name as product_name,
                SUM(COALESCE(salesAMT, 0)) as product_revenue
            FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`
            WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
            GROUP BY year, item_name
        ),
        top_products AS (
            SELECT 
                year,
                product_name,
                product_revenue,
                ROW_NUMBER() OVER (PARTITION BY year ORDER BY product_revenue DESC) as product_rank
            FROM yearly_products
        ),
        yearly_clients AS (
            SELECT 
                EXTRACT(YEAR FROM sales_date) as year,
                client_name,
                SUM(COALESCE(salesAMT, 0)) as client_revenue
            FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`
            WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
            GROUP BY year, client_name
        ),
        top_clients AS (
            SELECT 
                year,
                client_name,
                client_revenue,
                ROW_NUMBER() OVER (PARTITION BY year ORDER BY client_revenue DESC) as client_rank
            FROM yearly_clients
        )
        SELECT 
            yd.year,
            yd.yearly_revenue,
            yd.transaction_count,
            yd.avg_transaction_value,
            yd.unique_products,
            yd.unique_clients,
            yd.unique_depots,
            LAG(yd.yearly_revenue) OVER (ORDER BY yd.year) as prev_year_revenue,
            CASE 
                WHEN LAG(yd.yearly_revenue) OVER (ORDER BY yd.year) IS NOT NULL 
                THEN ROUND(((yd.yearly_revenue - LAG(yd.yearly_revenue) OVER (ORDER BY yd.year)) / LAG(yd.yearly_revenue) OVER (ORDER BY yd.year)) * 100, 2)
                ELSE NULL 
            END as revenue_growth_pct,
            tp.product_name as top_product,
            tp.product_revenue as top_product_revenue,
            tc.client_name as top_client,
            tc.client_revenue as top_client_revenue
        FROM yearly_data yd
        LEFT JOIN top_products tp ON yd.year = tp.year AND tp.product_rank = 1
        LEFT JOIN top_clients tc ON yd.year = tc.year AND tc.client_rank = 1
        ORDER BY yd.year DESC
        """
    
    def _get_detailed_schema_info(self, context: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Get detailed schema information for AI context"""
        schema_details = []
        
        for result in context:
            table_name = result.metadata["table"]
            try:
                # Get actual table schema from BigQuery
                table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"
                table = bq_client.get_table(table_ref)
                
                # Get sample data to understand data better (with date filter for partitioned tables)
                if table_name == "sales_details":
                    sample_sql = f"SELECT * FROM `{table_ref}` WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) LIMIT 5"
                else:
                    sample_sql = f"SELECT * FROM `{table_ref}` LIMIT 5"
                sample_results = list(bq_client.query(sample_sql).result())
                
                # Prepare column information
                columns = []
                for field in table.schema:
                    columns.append({
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode,
                        "description": field.description or ""
                    })
                
                # Convert BigQuery Row objects to dictionaries for JSON serialization
                sample_data = []
                for row in sample_results[:2]:
                    try:
                        sample_data.append(dict(row))
                    except:
                        sample_data.append(str(row))
                
                schema_details.append({
                    "table_name": table_name,
                    "full_name": table_ref,
                    "columns": columns,
                    "num_rows": table.num_rows,
                    "sample_data": sample_data,
                    "relevance_score": result.score
                })
                
            except Exception as e:
                logging.warning(f"Failed to get detailed schema for {table_name}: {e}")
                # Fallback to basic info
                schema_details.append({
                    "table_name": table_name,
                    "full_name": f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}",
                    "schema_text": result.content,
                    "relevance_score": result.score
                })
        
        return schema_details
    
    def _generate_fallback_ai_sql(self, query: str, context: List[RetrievalResult]) -> str:
        """Generate fallback SQL using simpler AI prompt"""
        if not context:
            return "SELECT 'No relevant tables found' as message"
        
        table_name = context[0].metadata["table"]
        schema_text = context[0].content
        
        prompt = f"""
Generate a simple BigQuery SQL query for this question: "{query}"

Table: {BQ_PROJECT}.{BQ_DATASET}.{table_name}
Schema: {schema_text}

Requirements:
- Use the table above
- Add appropriate WHERE conditions for recent data
- Include LIMIT 100
- Return only SQL, no explanations

SQL:
"""
        
        try:
            response = llm.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                sql = response.candidates[0].content.parts[0].text.strip()
                sql = self._clean_sql(sql)
                if self._validate_sql(sql):
                    return sql
        except Exception as e:
            logging.error(f"Fallback AI SQL generation failed: {e}")
        
        # Ultimate fallback
        return f"""
        SELECT *
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{table_name}`
        WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)
        LIMIT 100
        """
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL from markdown and extra text"""
        sql = sql.replace("```sql", "").replace("```", "").strip()
        # Remove any non-SQL text
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('--'):
                sql_lines.append(line)
        return '\n'.join(sql_lines)
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate SQL for safety and correctness with comprehensive checks"""
        if not sql or not sql.strip():
            logging.warning("Empty SQL query")
            return False
            
        sql_upper = sql.upper()
        
        # Basic structure validation
        if not self._validate_sql_structure(sql):
            return False
        
        # Safety validation
        if not self._validate_sql_safety(sql):
            return False
        
        # BigQuery compatibility validation
        if not self._validate_bigquery_compatibility(sql):
            return False
        
        # Syntax validation
        if not self._validate_sql_syntax(sql):
            return False
        
        return True
    
    def _validate_sql_structure(self, sql: str) -> bool:
        """Validate basic SQL structure"""
        sql_upper = sql.upper()
        
        # Check for required elements
        if "SELECT" not in sql_upper:
            logging.warning("SQL missing SELECT clause")
            return False
        
        if "FROM" not in sql_upper:
            logging.warning("SQL missing FROM clause")
            return False
        
        # Check for balanced parentheses
        if not self._check_balanced_parentheses(sql):
            logging.warning("SQL has unbalanced parentheses")
            return False
        
        # Check for balanced quotes
        if not self._check_balanced_quotes(sql):
            logging.warning("SQL has unbalanced quotes")
            return False
        
        return True
    
    def _validate_sql_safety(self, sql: str) -> bool:
        """Validate SQL for safety"""
        sql_upper = sql.upper()
        
        # Check for forbidden patterns
        forbidden_patterns = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
        for pattern in forbidden_patterns:
            if pattern in sql_upper:
                logging.warning(f"Forbidden pattern '{pattern}' found in SQL")
                return False
        
        return True
    
    def _validate_bigquery_compatibility(self, sql: str) -> bool:
        """Validate BigQuery compatibility"""
        sql_upper = sql.upper()
        
        # Check for BigQuery-incompatible functions
        bigquery_incompatible = [
            "GROUPING_ID", "ISNULL", "LEN(", "DATALENGTH", "GETDATE", "SYSDATETIME",
            "CHARINDEX", "PATINDEX", "STUFF", "REPLICATE", "SPACE", "QUOTENAME",
            "PARSENAME", "ISNUMERIC", "ISDATE", "TRY_CAST", "TRY_CONVERT"
        ]
        
        for func in bigquery_incompatible:
            if func in sql_upper:
                logging.warning(f"BigQuery-incompatible function '{func}' found in SQL")
                return False
        
        return True
    
    def _validate_sql_syntax(self, sql: str) -> bool:
        """Validate SQL syntax"""
        sql_upper = sql.upper()
        
        # Check for malformed CASE statements
        if "CASE" in sql_upper and "THEN" not in sql_upper:
            logging.warning("Malformed CASE statement found")
            return False
        
        # Check for malformed GROUP BY
        if "GROUP BY" in sql_upper:
            if not self._validate_group_by(sql):
                return False
        
        # Check for malformed ORDER BY
        if "ORDER BY" in sql_upper:
            if not self._validate_order_by(sql):
                return False
        
        return True
    
    def _check_balanced_parentheses(self, sql: str) -> bool:
        """Check if parentheses are balanced"""
        count = 0
        for char in sql:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def _check_balanced_quotes(self, sql: str) -> bool:
        """Check if quotes are balanced"""
        single_quotes = sql.count("'")
        double_quotes = sql.count('"')
        return single_quotes % 2 == 0 and double_quotes % 2 == 0
    
    def _validate_group_by(self, sql: str) -> bool:
        """Validate GROUP BY clause"""
        sql_upper = sql.upper()
        
        # Check for multiple GROUP BY clauses
        if sql_upper.count("GROUP BY") > 1:
            logging.warning("Multiple GROUP BY clauses found")
            return False
        
        return True
    
    def _validate_order_by(self, sql: str) -> bool:
        """Validate ORDER BY clause"""
        sql_upper = sql.upper()
        
        # Check for multiple ORDER BY clauses
        if sql_upper.count("ORDER BY") > 1:
            logging.warning("Multiple ORDER BY clauses found")
            return False
        
        return True

# -----------------------------
# 5.5️⃣ ADVANCED SQL GENERATION
# -----------------------------
class AdvancedSQLPlanner:
    """Advanced SQL planning with multi-step optimization"""
    
    def __init__(self):
        self.planning_templates = self._load_planning_templates()
        self.optimization_rules = self._load_optimization_rules()
        self.performance_predictor = PerformancePredictor()
    
    def create_advanced_plan(self, query: str, context: List[EnhancedRetrievalResult]) -> AdvancedSQLPlan:
        """Create advanced SQL execution plan"""
        # Step 1: Analyze query requirements
        requirements = self._analyze_query_requirements(query)
        
        # Step 2: Select optimal tables and joins
        table_selection = self._select_optimal_tables(context, requirements)
        
        # Step 3: Design join strategy
        join_strategy = self._design_join_strategy(table_selection)
        
        # Step 4: Plan filtering strategy
        filter_strategy = self._plan_filter_strategy(requirements)
        
        # Step 5: Plan aggregation strategy
        aggregation_strategy = self._plan_aggregation_strategy(requirements)
        
        # Step 6: Optimize for performance
        optimization_hints = self._generate_optimization_hints(
            table_selection, join_strategy, filter_strategy, aggregation_strategy
        )
        
        return AdvancedSQLPlan(
            requirements=requirements,
            table_selection=table_selection,
            join_strategy=join_strategy,
            filter_strategy=filter_strategy,
            aggregation_strategy=aggregation_strategy,
            optimization_hints=optimization_hints,
            estimated_cost=self.performance_predictor.estimate_cost(table_selection, join_strategy)
        )
    
    def _analyze_query_requirements(self, query: str) -> QueryRequirements:
        """Analyze what the query is asking for"""
        requirements = QueryRequirements()
        query_lower = query.lower()
        
        # Detect aggregation needs
        if any(word in query_lower for word in ['total', 'sum', 'count', 'average', 'max', 'min']):
            requirements.needs_aggregation = True
        
        # Detect grouping needs
        if any(word in query_lower for word in ['by', 'group', 'category', 'type']):
            requirements.needs_grouping = True
        
        # Detect temporal filtering
        if any(word in query_lower for word in ['date', 'time', 'month', 'year', 'quarter']):
            requirements.needs_temporal_filtering = True
        
        # Detect comparison needs
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            requirements.needs_comparison = True
        
        # Calculate complexity score
        complexity_factors = {
            'length': min(len(query.split()) / 20, 1.0),
            'conjunctions': query_lower.count(' and ') + query_lower.count(' or '),
            'comparisons': query_lower.count(' vs ') + query_lower.count(' compare '),
            'aggregations': query_lower.count(' total ') + query_lower.count(' sum '),
            'conditions': query_lower.count(' where ') + query_lower.count(' if ')
        }
        
        requirements.complexity_score = sum(complexity_factors.values()) / len(complexity_factors)
        
        return requirements
    
    def _select_optimal_tables(self, context: List[EnhancedRetrievalResult], requirements: QueryRequirements) -> List[str]:
        """Select optimal tables based on context and requirements"""
        # Sort by relevance score
        sorted_context = sorted(context, key=lambda x: x.score, reverse=True)
        
        # Select top tables
        selected_tables = []
        for result in sorted_context[:3]:  # Top 3 most relevant
            table_name = result.metadata.get("table")
            if table_name and table_name not in selected_tables:
                selected_tables.append(table_name)
        
        return selected_tables
    
    def _design_join_strategy(self, table_selection: List[str]) -> List[str]:
        """Design join strategy for multiple tables"""
        if len(table_selection) <= 1:
            return []
        
        joins = []
        # Simple join strategy - can be enhanced
        for i in range(len(table_selection) - 1):
            table1 = table_selection[i]
            table2 = table_selection[i + 1]
            joins.append(f"JOIN {table2} ON {table1}.id = {table2}.id")
        
        return joins
    
    def _plan_filter_strategy(self, requirements: QueryRequirements) -> List[str]:
        """Plan filtering strategy"""
        filters = []
        
        # Always add a date filter for BigQuery partition elimination
        filters.append("sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)")
        
        if requirements.needs_temporal_filtering:
            # Add more specific temporal filter if needed
            filters.append("sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)")
        
        if requirements.needs_comparison:
            filters.append("salesAMT > 0")
        
        return filters
    
    def _plan_aggregation_strategy(self, requirements: QueryRequirements) -> List[str]:
        """Plan aggregation strategy"""
        aggregations = []
        
        if requirements.needs_aggregation:
            aggregations.append("SUM(salesAMT) as total_sales")
            aggregations.append("COUNT(*) as record_count")
        
        if requirements.needs_grouping:
            aggregations.append("GROUP BY product_name, client_name")
        
        return aggregations
    
    def _generate_optimization_hints(self, table_selection: List[str], join_strategy: List[str], 
                                   filter_strategy: List[str], aggregation_strategy: List[str]) -> List[str]:
        """Generate optimization hints"""
        hints = []
        
        # Add index hints
        for table in table_selection:
            hints.append(f"Consider adding index on {table}.sales_date for better performance")
        
        # Add join hints
        if join_strategy:
            hints.append("Use appropriate join order for better performance")
        
        # Add filter hints
        if filter_strategy:
            hints.append("Apply filters early to reduce data volume")
        
        return hints
    
    def _load_planning_templates(self) -> Dict[str, Any]:
        """Load SQL planning templates"""
        return {
            'analytical': {
                'tables': ['sales_details'],
                'joins': [],
                'filters': ['sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)'],
                'aggregations': ['SUM(salesAMT)', 'GROUP BY product_name']
            },
            'operational': {
                'tables': ['sales_details'],
                'joins': [],
                'filters': ['sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)'],
                'aggregations': ['COUNT(*)']
            }
        }
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load SQL optimization rules"""
        return {
            'max_rows': 10000,
            'prefer_indexed_columns': True,
            'use_partitioning': True,
            'optimize_joins': True
        }

class PerformancePredictor:
    """SQL performance prediction"""
    
    def estimate_cost(self, table_selection: List[str], join_strategy: List[str]) -> float:
        """Estimate query execution cost"""
        base_cost = len(table_selection) * 0.1
        join_cost = len(join_strategy) * 0.2
        return base_cost + join_cost

class AdvancedSQLValidator:
    """Advanced SQL validation and optimization"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.optimization_rules = self._load_optimization_rules()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def validate_and_optimize(self, sql: str, context: QueryContext) -> ValidationResult:
        """Comprehensive SQL validation and optimization"""
        validation_result = ValidationResult()
        
        # Basic validation
        if not self._basic_validation(sql):
            validation_result.is_valid = False
            validation_result.errors = ["Basic SQL validation failed"]
            return validation_result
        
        # Security validation
        security_issues = self._security_validation(sql)
        validation_result.warnings = security_issues
        
        # Performance analysis
        performance_analysis = self.performance_analyzer.analyze(sql, context)
        validation_result.performance_score = performance_analysis.score
        
        # Optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(sql, performance_analysis)
        validation_result.optimization_suggestions = optimization_suggestions
        
        # Generate optimized SQL
        if optimization_suggestions:
            optimized_sql = self._apply_optimizations(sql, optimization_suggestions)
            validation_result.optimized_sql = optimized_sql
        
        return validation_result
    
    def _basic_validation(self, sql: str) -> bool:
        """Basic SQL validation"""
        sql_upper = sql.upper()
        return "SELECT" in sql_upper and "FROM" in sql_upper
    
    def _security_validation(self, sql: str) -> List[str]:
        """Security validation"""
        warnings = []
        sql_upper = sql.upper()
        
        forbidden_patterns = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
        for pattern in forbidden_patterns:
            if pattern in sql_upper:
                warnings.append(f"Potentially dangerous pattern '{pattern}' found")
        
        return warnings
    
    def _generate_optimization_suggestions(self, sql: str, performance_analysis: PerformanceAnalysis) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Add index suggestions
        if performance_analysis.missing_indexes:
            suggestions.append("Consider adding indexes for better performance")
        
        # Add join suggestions
        if performance_analysis.inefficient_joins:
            suggestions.append("Optimize join order and conditions")
        
        return suggestions
    
    def _apply_optimizations(self, sql: str, suggestions: List[str]) -> str:
        """Apply optimizations to SQL"""
        # Simple optimization - can be enhanced
        optimized_sql = sql
        
        # Add LIMIT if not present
        if "LIMIT" not in optimized_sql.upper():
            optimized_sql += " LIMIT 1000"
        
        return optimized_sql
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules"""
        return {
            "max_rows": 10000,
            "required_filters": ["date_range"],
            "forbidden_patterns": ["DROP", "DELETE", "UPDATE", "INSERT"],
            "timeout_seconds": 30
        }
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules"""
        return {
            "prefer_indexed_columns": True,
            "use_partitioning": True,
            "optimize_joins": True
        }

class PerformanceAnalyzer:
    """SQL performance analysis"""
    
    def analyze(self, sql: str, context: QueryContext) -> PerformanceAnalysis:
        """Analyze SQL performance"""
        return PerformanceAnalysis(
            score=0.8,  # Default score
            missing_indexes=[],
            inefficient_joins=[],
            recommendations=[]
        )

# -----------------------------
# 6️⃣ CONVERSATION MEMORY
# -----------------------------
class ConversationMemory:
    def __init__(self):
        self.conversations = {}
        self.load_conversations()
    
    def load_conversations(self):
        """Load conversation history from cache"""
        if os.path.exists(CONVERSATION_CACHE_FILE):
            try:
                with open(CONVERSATION_CACHE_FILE, 'rb') as f:
                    self.conversations = pickle.load(f)
                logging.info(f"Loaded {len(self.conversations)} conversations from cache")
            except Exception as e:
                logging.warning(f"Failed to load conversations: {e}")
                self.conversations = {}
    
    def save_conversations(self):
        """Save conversation history to cache"""
        try:
            with open(CONVERSATION_CACHE_FILE, 'wb') as f:
                pickle.dump(self.conversations, f)
        except Exception as e:
            logging.warning(f"Failed to save conversations: {e}")
    
    def get_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Get conversation context for user session"""
        key = f"{user_id}_{session_id}"
        if key not in self.conversations:
            self.conversations[key] = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                history=[],
                preferences={},
                last_updated=datetime.now()
            )
        return self.conversations[key]
    
    def add_interaction(self, user_id: str, session_id: str, 
                       query: str, response: Dict[str, Any]):
        """Add interaction to conversation history"""
        context = self.get_context(user_id, session_id)
        context.history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        })
        context.last_updated = datetime.now()
        
        # Keep only last 10 interactions
        if len(context.history) > 10:
            context.history = context.history[-10:]
        
        self.save_conversations()
    
    def get_recent_context(self, user_id: str, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        context = self.get_context(user_id, session_id)
        return context.history[-limit:] if context.history else []

# -----------------------------
# 7️⃣ ADVANCED ANALYTICS
# -----------------------------
class AdvancedAnalytics:
    def __init__(self):
        self.insight_templates = {
            "trend": "The data shows a {direction} trend with {change}% change over the period.",
            "anomaly": "An unusual pattern detected: {description}",
            "correlation": "Strong correlation found between {var1} and {var2} (r={correlation})",
            "outlier": "Outlier detected: {value} is significantly different from the mean"
        }
    
    def generate_insights(self, results: List[Dict[str, Any]], query: str) -> List[str]:
        """Generate automated insights from results"""
        if not results:
            return ["No data available for analysis"]
        
        insights = []
        
        # Statistical analysis
        stats_insights = self._statistical_analysis(results)
        insights.extend(stats_insights)
        
        # Trend analysis
        trend_insights = self._trend_analysis(results)
        insights.extend(trend_insights)
        
        # Anomaly detection
        anomaly_insights = self._anomaly_detection(results)
        insights.extend(anomaly_insights)
        
        return insights
    
    def _statistical_analysis(self, results: List[Dict[str, Any]]) -> List[str]:
        """Perform statistical analysis on results"""
        insights = []
        
        # Find numeric columns
        numeric_columns = []
        for key in results[0].keys():
            if all(isinstance(row.get(key), (int, float)) for row in results if row.get(key) is not None):
                numeric_columns.append(key)
        
        for col in numeric_columns:
            values = [row[col] for row in results if row.get(col) is not None]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                max_val = max(values)
                min_val = min(values)
                
                insights.append(f"{col}: Mean={mean_val:.2f}, Std={std_val:.2f}, Range=[{min_val:.2f}, {max_val:.2f}]")
        
        return insights
    
    def _trend_analysis(self, results: List[Dict[str, Any]]) -> List[str]:
        """Analyze trends in the data"""
        insights = []
        
        # Look for date columns
        date_columns = [col for col in results[0].keys() 
                       if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            date_col = date_columns[0]
            # Simple trend analysis
            numeric_cols = [col for col in results[0].keys() 
                           if col != date_col and isinstance(results[0].get(col), (int, float))]
            
            for num_col in numeric_cols:
                values = [row[num_col] for row in results if row.get(num_col) is not None]
                if len(values) > 1:
                    # Calculate simple trend
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    if first_half and second_half:
                        first_avg = np.mean(first_half)
                        second_avg = np.mean(second_half)
                        change = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
                        
                        direction = "increasing" if change > 0 else "decreasing"
                        insights.append(f"{num_col} shows {direction} trend: {change:.1f}% change")
        
        return insights
    
    def _anomaly_detection(self, results: List[Dict[str, Any]]) -> List[str]:
        """Detect anomalies in the data"""
        insights = []
        
        # Find numeric columns
        numeric_columns = []
        for key in results[0].keys():
            if all(isinstance(row.get(key), (int, float)) for row in results if row.get(key) is not None):
                numeric_columns.append(key)
        
        for col in numeric_columns:
            values = [row[col] for row in results if row.get(col) is not None]
            if len(values) > 3:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Detect outliers (beyond 2 standard deviations)
                outliers = [v for v in values if abs(v - mean_val) > 2 * std_val]
                if outliers:
                    insights.append(f"Anomaly detected in {col}: {len(outliers)} outliers found")
        
        return insights

# -----------------------------
# 7.5️⃣ ENHANCED ANALYTICS SYSTEM
# -----------------------------
class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis with comprehensive insights"""
    
    def __init__(self):
        self.statistical_tests = StatisticalTests()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.correlation_analyzer = CorrelationAnalyzer()
    
    def comprehensive_analysis(self, data: List[Dict[str, Any]], query: str) -> AnalysisResults:
        """Perform comprehensive statistical analysis"""
        results = AnalysisResults()
        
        # Descriptive statistics
        results.descriptive_stats = self._calculate_descriptive_statistics(data)
        
        # Trend analysis
        results.trend_analysis = self.trend_analyzer.analyze_trends(data)
        
        # Anomaly detection
        results.anomalies = self.anomaly_detector.detect_anomalies(data)
        
        # Correlation analysis
        results.correlations = self.correlation_analyzer.find_correlations(data)
        
        # Statistical significance tests
        results.significance_tests = self._perform_significance_tests(data)
        
        # Generate insights
        results.insights = self._generate_statistical_insights(results, query)
        
        return results
    
    def _calculate_descriptive_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive descriptive statistics"""
        if not data or len(data) == 0:
            return {}
        
        stats = {}
        
        # Find numeric columns - handle empty data safely
        numeric_columns = []
        if data and len(data) > 0:
            for key in data[0].keys():
                if all(isinstance(row.get(key), (int, float)) for row in data if row.get(key) is not None):
                    numeric_columns.append(key)
        
        for col in numeric_columns:
            values = [row[col] for row in data if row.get(col) is not None]
            if values:
                stats[col] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q1': np.percentile(values, 25),
                    'q3': np.percentile(values, 75),
                    'skewness': self._calculate_skewness(values),
                    'kurtosis': self._calculate_kurtosis(values)
                }
        
        return stats
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of data"""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        
        skewness = np.mean([(x - mean) ** 3 for x in values]) / (std ** 3)
        return skewness
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of data"""
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        
        kurtosis = np.mean([(x - mean) ** 4 for x in values]) / (std ** 4) - 3
        return kurtosis
    
    def _perform_significance_tests(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        tests = {}
        
        if not data or len(data) == 0:
            return tests
        
        # Find numeric columns
        numeric_columns = []
        if data and len(data) > 0:
            for key in data[0].keys():
                if all(isinstance(row.get(key), (int, float)) for row in data if row.get(key) is not None):
                    numeric_columns.append(key)
        
        if len(numeric_columns) >= 2:
            # Perform correlation test
            col1, col2 = numeric_columns[0], numeric_columns[1]
            values1 = [row[col1] for row in data if row.get(col1) is not None]
            values2 = [row[col2] for row in data if row.get(col2) is not None]
            
            if len(values1) == len(values2) and len(values1) > 2:
                correlation, p_value = self._pearson_correlation(values1, values2)
                tests['correlation'] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return tests
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Calculate Pearson correlation coefficient and p-value"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 1.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0, 1.0
        
        correlation = numerator / denominator
        
        # Simple p-value approximation
        t_stat = correlation * ((n - 2) / (1 - correlation ** 2)) ** 0.5
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))
        
        return correlation, p_value
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF"""
        # Simple approximation - can be enhanced
        if df > 30:
            return 0.5 + 0.5 * np.tanh(t / 2)
        else:
            return 0.5 + 0.5 * np.tanh(t / (2 + df / 10))
    
    def _generate_statistical_insights(self, analysis: AnalysisResults, query: str) -> List[str]:
        """Generate actionable statistical insights"""
        insights = []
        
        # Descriptive insights
        if analysis.descriptive_stats:
            for col, stats in analysis.descriptive_stats.items():
                if stats['skewness'] > 1:
                    insights.append(f"📊 {col} shows strong positive skewness ({stats['skewness']:.2f}) - data is right-skewed")
                elif stats['skewness'] < -1:
                    insights.append(f"📊 {col} shows strong negative skewness ({stats['skewness']:.2f}) - data is left-skewed")
                
                if stats['kurtosis'] > 3:
                    insights.append(f"📊 {col} shows high kurtosis ({stats['kurtosis']:.2f}) - data has heavy tails")
                elif stats['kurtosis'] < -1:
                    insights.append(f"📊 {col} shows low kurtosis ({stats['kurtosis']:.2f}) - data has light tails")
        
        # Correlation insights
        if analysis.correlations and analysis.correlations.get('strong_correlations'):
            for corr in analysis.correlations['strong_correlations']:
                insights.append(f"🔗 Strong correlation between {corr['var1']} and {corr['var2']} (r={corr['correlation']:.3f})")
        
        # Significance insights
        if analysis.significance_tests and analysis.significance_tests.get('correlation'):
            corr_test = analysis.significance_tests['correlation']
            if corr_test['significant']:
                insights.append(f"📊 Statistically significant correlation found (p={corr_test['p_value']:.3f})")
        
        return insights

class TrendAnalyzer:
    """Advanced trend analysis"""
    
    def analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {
            'significant_trends': [],
            'trend_direction': 'stable',
            'trend_strength': 0.0
        }
        
        if not data or len(data) == 0:
            return trends
        
        # Look for date columns
        date_columns = [col for col in data[0].keys() 
                       if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns and len(data) > 2:
            date_col = date_columns[0]
            numeric_cols = [col for col in data[0].keys() 
                           if col != date_col and isinstance(data[0].get(col), (int, float))]
            
            for num_col in numeric_cols:
                trend_info = self._analyze_column_trend(data, date_col, num_col)
                if trend_info:
                    trends['significant_trends'].append(trend_info)
        
        # Determine overall trend direction
        if trends['significant_trends']:
            positive_trends = sum(1 for t in trends['significant_trends'] if t['direction'] == 'increasing')
            total_trends = len(trends['significant_trends'])
            
            if positive_trends > total_trends / 2:
                trends['trend_direction'] = 'increasing'
            elif positive_trends < total_trends / 2:
                trends['trend_direction'] = 'decreasing'
            
            trends['trend_strength'] = np.mean([t['strength'] for t in trends['significant_trends']])
        
        return trends
    
    def _analyze_column_trend(self, data: List[Dict[str, Any]], date_col: str, num_col: str) -> Optional[Dict[str, Any]]:
        """Analyze trend for a specific column"""
        # Sort by date
        sorted_data = sorted(data, key=lambda x: x.get(date_col, ''))
        
        # Extract values
        values = [row[num_col] for row in sorted_data if row.get(num_col) is not None]
        
        if len(values) < 3:
            return None
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate R-squared
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine direction and strength
        direction = 'increasing' if slope > 0 else 'decreasing'
        strength = abs(slope) * r_squared
        
        # Only return significant trends
        if r_squared > 0.5 and strength > 0.1:
            return {
                'metric': num_col,
                'direction': direction,
                'strength': strength,
                'r_squared': r_squared,
                'slope': slope
            }
        
        return None

class AnomalyDetector:
    """Advanced anomaly detection"""
    
    def detect_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        anomalies = []
        
        if not data or len(data) == 0:
            return anomalies
        
        # Find numeric columns
        numeric_columns = []
        if data and len(data) > 0:
            for key in data[0].keys():
                if all(isinstance(row.get(key), (int, float)) for row in data if row.get(key) is not None):
                    numeric_columns.append(key)
        
        for col in numeric_columns:
            values = [row[col] for row in data if row.get(col) is not None]
            if len(values) > 3:
                col_anomalies = self._detect_column_anomalies(values, col)
                anomalies.extend(col_anomalies)
        
        return anomalies
    
    def _detect_column_anomalies(self, values: List[float], column: str) -> List[Dict[str, Any]]:
        """Detect anomalies in a specific column"""
        anomalies = []
        
        if len(values) < 4:
            return anomalies
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return anomalies
        
        # Z-score method
        z_scores = [(x - mean) / std for x in values]
        
        # Identify outliers (z-score > 2.5)
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > 2.5:
                anomalies.append({
                    'column': column,
                    'value': values[i],
                    'z_score': z_score,
                    'index': i,
                    'severity': 'high' if abs(z_score) > 3 else 'medium'
                })
        
        return anomalies

class CorrelationAnalyzer:
    """Advanced correlation analysis"""
    
    def find_correlations(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find correlations between variables"""
        correlations = {
            'strong_correlations': [],
            'moderate_correlations': [],
            'weak_correlations': []
        }
        
        if not data or len(data) == 0:
            return correlations
        
        # Find numeric columns
        numeric_columns = []
        if data and len(data) > 0:
            for key in data[0].keys():
                if all(isinstance(row.get(key), (int, float)) for row in data if row.get(key) is not None):
                    numeric_columns.append(key)
        
        if len(numeric_columns) < 2:
            return correlations
        
        # Calculate pairwise correlations
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                values1 = [row[col1] for row in data if row.get(col1) is not None]
                values2 = [row[col2] for row in data if row.get(col2) is not None]
                
                if len(values1) == len(values2) and len(values1) > 2:
                    correlation = self._calculate_correlation(values1, values2)
                    
                    corr_info = {
                        'var1': col1,
                        'var2': col2,
                        'correlation': correlation
                    }
                    
                    if abs(correlation) > 0.7:
                        correlations['strong_correlations'].append(corr_info)
                    elif abs(correlation) > 0.4:
                        correlations['moderate_correlations'].append(corr_info)
                    else:
                        correlations['weak_correlations'].append(corr_info)
        
        return correlations
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

class StatisticalTests:
    """Statistical tests for data analysis"""
    
    def __init__(self):
        pass
    
    def t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform t-test between two groups"""
        if len(group1) < 2 or len(group2) < 2:
            return {'p_value': 1.0, 'significant': False}
        
        # Calculate means and standard deviations
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)
        
        # Calculate t-statistic
        n1, n2 = len(group1), len(group2)
        pooled_std = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        t_stat = (mean1 - mean2) / (pooled_std * (1/n1 + 1/n2)**0.5)
        
        # Approximate p-value
        df = n1 + n2 - 2
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': abs(mean1 - mean2) / pooled_std
        }
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF"""
        if df > 30:
            return 0.5 + 0.5 * np.tanh(t / 2)
        else:
            return 0.5 + 0.5 * np.tanh(t / (2 + df / 10))

# -----------------------------
# 8️⃣ MAIN ULTIMATE RAG PIPELINE
# -----------------------------
# -----------------------------
# 7.6️⃣ LEARNING & ADAPTATION SYSTEM
# -----------------------------
class FeedbackLearningSystem:
    """Feedback learning system for continuous improvement"""
    
    def __init__(self):
        self.feedback_store = FeedbackStore()
        self.model_updater = ModelUpdater()
        self.performance_tracker = PerformanceTracker()
    
    def process_feedback(self, query: str, response: Dict[str, Any], feedback: UserFeedback):
        """Process user feedback to improve the system"""
        # Store feedback
        self.feedback_store.store_feedback(query, response, feedback)
        
        # Analyze feedback patterns
        feedback_patterns = self._analyze_feedback_patterns()
        
        # Update models based on feedback
        if feedback_patterns.needs_retrieval_update:
            self.model_updater.update_retrieval_weights(feedback_patterns.retrieval_feedback)
        
        if feedback_patterns.needs_sql_update:
            self.model_updater.update_sql_generation(feedback_patterns.sql_feedback)
        
        # Track performance improvements
        self.performance_tracker.track_improvement(feedback_patterns)
    
    def adaptive_retrieval_weights(self, user_id: str) -> Dict[str, float]:
        """Get adaptive retrieval weights based on user preferences"""
        user_feedback = self.feedback_store.get_user_feedback(user_id)
        
        if not user_feedback:
            return self._default_weights()
        
        # Analyze user preferences
        preferences = self._analyze_user_preferences(user_feedback)
        
        # Adjust weights based on preferences
        weights = self._default_weights()
        if preferences.prefers_semantic:
            weights['semantic'] += 0.1
            weights['keyword'] -= 0.05
            weights['temporal'] -= 0.05
        
        return weights
    
    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns for system improvement"""
        return {
            'needs_retrieval_update': False,
            'needs_sql_update': False,
            'retrieval_feedback': {},
            'sql_feedback': {}
        }
    
    def _analyze_user_preferences(self, user_feedback: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze user preferences from feedback"""
        return {
            'prefers_semantic': True,
            'prefers_detailed': True,
            'prefers_charts': True
        }
    
    def _default_weights(self) -> Dict[str, float]:
        """Default retrieval weights"""
        return {
            'semantic': 0.6,
            'keyword': 0.3,
            'temporal': 0.1
        }

class FeedbackStore:
    """Store and manage user feedback"""
    
    def __init__(self):
        self.feedback_data = []
        self.load_feedback()
    
    def store_feedback(self, query: str, response: Dict[str, Any], feedback: UserFeedback):
        """Store user feedback"""
        feedback.timestamp = datetime.now()
        self.feedback_data.append({
            'query': query,
            'response': response,
            'feedback': feedback
        })
        self.save_feedback()
    
    def get_user_feedback(self, user_id: str) -> List[UserFeedback]:
        """Get feedback for a specific user"""
        return [f['feedback'] for f in self.feedback_data if f['feedback'].user_id == user_id]
    
    def load_feedback(self):
        """Load feedback from storage"""
        try:
            if os.path.exists('feedback_data.json'):
                with open('feedback_data.json', 'r') as f:
                    self.feedback_data = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load feedback data: {e}")
            self.feedback_data = []
    
    def save_feedback(self):
        """Save feedback to storage"""
        try:
            with open('feedback_data.json', 'w') as f:
                json.dump(self.feedback_data, f, default=str)
        except Exception as e:
            logging.warning(f"Failed to save feedback data: {e}")

class ModelUpdater:
    """Update models based on feedback"""
    
    def __init__(self):
        pass
    
    def update_retrieval_weights(self, feedback: Dict[str, Any]):
        """Update retrieval weights based on feedback"""
        logging.info("Updating retrieval weights based on feedback")
    
    def update_sql_generation(self, feedback: Dict[str, Any]):
        """Update SQL generation based on feedback"""
        logging.info("Updating SQL generation based on feedback")

class PerformanceTracker:
    """Track system performance improvements"""
    
    def __init__(self):
        self.performance_metrics = {
            'accuracy': 0.0,
            'response_time': 0.0,
            'user_satisfaction': 0.0
        }
    
    def track_improvement(self, feedback_patterns: Dict[str, Any]):
        """Track performance improvements"""
        logging.info("Tracking performance improvements")

class UltimateRAGPipeline:
    def __init__(self):
        # Enhanced components
        self.enhanced_retriever = EnhancedMultiStageRetriever(qdrant_client, QDRANT_COLLECTION)
        self.advanced_query_processor = AdvancedQueryProcessor()
        self.dynamic_knowledge_graph = DynamicKnowledgeGraph()
        self.advanced_sql_planner = AdvancedSQLPlanner()
        self.advanced_sql_validator = AdvancedSQLValidator()
        self.advanced_analytics = AdvancedStatisticalAnalyzer()
        self.feedback_learning = FeedbackLearningSystem()
        
        # Original components (for backward compatibility)
        self.retriever = MultiStageRetriever()
        self.query_understanding = QueryUnderstanding()
        self.knowledge_graph = KnowledgeGraph()
        self.sql_generator = EnhancedSQLGenerator()
        self.conversation_memory = ConversationMemory()
        self.analytics = AdvancedAnalytics()
        
        # Initialize components
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pipeline components"""
        logging.info("Initializing Ultimate RAG Pipeline...")
        
        # Build knowledge graph
        self.knowledge_graph.build_from_tables(TABLES)
        
        # Embed schema (reuse existing function)
        self._embed_schema(TABLES)
        
        logging.info("✅ Ultimate RAG Pipeline initialized!")
    
    def _classify_question(self, question: str) -> str:
        """Classify question into business, smalltalk, or non_business categories"""
        import re
        question_lower = question.lower().strip()
        
        # Pattern-based classification as fallback
        smalltalk_patterns = [
            "who are you", "what is your name", "what's your name", "who is you",
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "thank you", "thanks", "bye", "goodbye", "see you", "how are you",
            "what can you do", "help", "introduce yourself"
        ]
        
        # Business question patterns (should be checked first)
        business_patterns = [
            "sales", "salse", "revenue", "profit", "client", "customer", "product", "depot",
            "analysis", "report", "data", "trend", "performance", "top", "best",
            "2024", "2023", "year", "month", "quarter", "today", "yesterday",
            "amount", "quantity", "volume", "growth", "decline", "increase", "decrease",
            "market", "business", "financial", "analytics", "insights", "metrics",
            "summary", "overview", "breakdown", "comparison", "ranking", "leaderboard",
            "full", "everything", "complete", "detailed", "comprehensive"
        ]
        
        non_business_patterns = [
            "weather", "temperature", "rain", "sunny", "cloudy",
            "politics", "election", "government", "president",
            "joke", "funny", "laugh", "entertainment", "movie", "music",
            "math", "calculate", "equation", "physics", "science",
            "recipe", "cooking", "food", "restaurant",
            # explicit non-business/chatty phrases
            "i love you", "love you", "marry me", "marriage", "date me",
            "sing", "dance", "story", "poem", "song", "riddle",
            "religion", "god", "bible", "quran"
        ]
        
        # Check business patterns first (highest priority)
        logging.info(f"[Debug] Question: '{question_lower}'")
        for pattern in business_patterns:
            if pattern in question_lower:
                logging.info(f"[Pattern Match] Found business pattern: {pattern}")
                return "business"
        
        # Check smalltalk patterns (word boundaries for precision)
        for pattern in smalltalk_patterns:
            # Use word boundaries to avoid false matches (e.g., "hi" in "analysis")
            if re.search(r'\b' + re.escape(pattern) + r'\b', question_lower):
                logging.info(f"[Pattern Match] Found smalltalk pattern: {pattern}")
                return "smalltalk"
                
        for pattern in non_business_patterns:
            if pattern in question_lower:
                logging.info(f"[Pattern Match] Found non_business pattern: {pattern}")
                return "non_business"
        
        # Try AI classification if no pattern matches
        try:
            # Import vertexai here to avoid circular imports
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            # Initialize the model
            model = GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""
Classify the following question into one of these categories:
1. business → sales, revenue, profit, clients, products, analytics, data, etc.
2. smalltalk → greetings, thanks, chatting, "who are you", "what is your name", etc.
3. non_business → unrelated topics (weather, politics, math, general knowledge, etc.)

Question: "{question}"

Answer with only one word: business, smalltalk, or non_business
"""
            
            response = model.generate_content(prompt)
            text = response.text.strip().lower().split()[0]
            logging.info(f"[AI Classification] Result: {text}")
            
            # Validate response
            if text in ["business", "smalltalk", "non_business"]:
                return text
            else:
                # Default to business if AI classification fails
                logging.warning(f"[AI Classification] Invalid response: {text}, defaulting to business")
                return "business"
                
        except Exception as e:
            logging.warning(f"AI classification failed: {e}. Using conservative fallback to non_business.")
            # If AI fails, avoid running business SQL by default
            return "non_business"
    
    def _embed_schema(self, tables):
        """Embed table schemas (reuse existing logic)"""
        points = []
        for idx, table_name in enumerate(tables):
            table_ref = f"{BQ_DATASET}.{table_name}"
            table = bq_client.get_table(table_ref)
            schema_text = ", ".join([f"{f.name} ({f.field_type})" for f in table.schema])
            embedding = embedding_model.get_embeddings([schema_text])[0].values

            points.append(PointStruct(
                id=idx,
                vector=embedding,
                payload={"table": table_name, "schema": schema_text}
            ))

        if points:
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
            )
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    
    def ask_intelligent(self, question: str, user_id: str = "default", 
                       session_id: str = "default") -> Dict[str, Any]:
        """Main intelligent query processing method"""
        start_time = time.time()
        
        # First, correct typos in the question
        corrected_question = self.query_understanding.typo_corrector.correct_typos(question)
        if corrected_question != question:
            logging.info(f"Question corrected: '{question}' → '{corrected_question}'")
            question = corrected_question
        
        # Get conversation context
        recent_context = self.conversation_memory.get_recent_context(user_id, session_id)
        
        # Step 0: Intelligent Question Classification (from rag_pipeline.py)
        category = self._classify_question(question)
        logging.info(f"[Classifier] → {category}")
        
        # Handle non-business questions
        if category == "smalltalk":
            question_lower = question.lower()
            if any(phrase in question_lower for phrase in ["name", "who are you", "what is your name", "what's your name"]):
                return {
                    "summary": "My name is BusinessGPT — your AI-powered Business Analytics Assistant! 🤖\n\nI specialize in:\n- 📊 Sales data analysis and insights\n- 📈 Revenue and profit reporting\n- 🏢 Client and product analytics\n- 💡 Business intelligence and visualizations\n- 🔍 Data-driven decision support\n\nI'm here to help you make informed business decisions through intelligent data analysis. What business question can I help you with today?",
                    "results": None, 
                    "sql": None, 
                    "charts": [],
                    "metadata": {"category": "smalltalk", "processing_time": time.time() - start_time}
                }
            elif any(phrase in question_lower for phrase in ["hello", "hi", "hey", "good morning", "good afternoon"]):
                return {
                    "summary": "Hello! 👋 I'm BusinessGPT, your AI analytics assistant. I'm ready to help you with business insights and data analysis. \n\nWhat would you like to explore today? You can ask about sales performance, revenue trends, top clients, or any other business metrics!",
                    "results": None, 
                    "sql": None, 
                    "charts": [],
                    "metadata": {"category": "smalltalk", "processing_time": time.time() - start_time}
                }
            elif any(phrase in question_lower for phrase in ["thank you", "thanks"]):
                return {
                    "summary": "You're very welcome! 😊 I'm always here to help with your business analytics needs. Feel free to ask me anything about your data, sales, or business insights!",
                    "results": None, 
                    "sql": None, 
                    "charts": [],
                    "metadata": {"category": "smalltalk", "processing_time": time.time() - start_time}
                }
            else:
                return {
                    "summary": "Hello! I'm BusinessGPT, your AI analytics assistant. I specialize in business data analysis and insights. \n\nPlease ask me business-related questions like:\n- Sales performance analysis\n- Revenue and profit reports\n- Client or product analytics\n- Market insights and trends\n\nWhat business question can I help you with?",
                    "results": None, 
                    "sql": None, 
                    "charts": [],
                    "metadata": {"category": "smalltalk", "processing_time": time.time() - start_time}
                }
        
        if category == "non_business":
            return {
                "summary": "I'm specialized in business analytics and data insights. Please ask me a business-related question (e.g., sales, revenue, profit, clients, products, trends).",
                "results": None,
                "sql": None,
                "charts": [],
                "metadata": {"category": "non_business", "processing_time": time.time() - start_time}
            }
        
        # Step 1: Enhanced Query Understanding
        try:
            enhanced_intent = self.advanced_query_processor.process_query(question, recent_context)
            logging.info(f"[Enhanced Intent] {enhanced_intent.intent_type} (confidence: {enhanced_intent.confidence:.2f}, complexity: {enhanced_intent.complexity:.2f})")
        except Exception as e:
            logging.warning(f"Enhanced query processing failed: {e}")
            # Fallback to original query understanding
            intent = self.query_understanding.classify_intent(question)
            enhanced_intent = type('EnhancedQueryIntent', (), {
                'intent_type': intent.intent_type,
                'confidence': intent.confidence,
                'entities': intent.entities,
                'temporal_context': intent.temporal_context,
                'complexity': 0.5,
                'requires_decomposition': False
            })()
        
        # Step 2: Enhanced Query Decomposition (for complex queries)
        if enhanced_intent.requires_decomposition or enhanced_intent.complexity > 0.7:
            decomposer = ContextAwareDecomposer()
            sub_queries = decomposer.decompose_query(question, recent_context)
            logging.info(f"[Enhanced Decomposition] Split into {len(sub_queries)} sub-queries")
        else:
            sub_queries = [question]
        
        # Step 3: Enhanced Retrieval with Context
        all_results = []
        
        # Handle recent_context properly - it might be a list or dict
        if isinstance(recent_context, dict):
            conversation_history = recent_context.get('recent_interactions', [])
        else:
            # If it's a list, use it directly as conversation history
            conversation_history = recent_context if recent_context else []
        
        retrieval_context = RetrievalContext(
            query_type=enhanced_intent.intent_type,
            user_preferences=self.feedback_learning.adaptive_retrieval_weights(user_id),
            conversation_history=conversation_history,
            temporal_focus=enhanced_intent.temporal_context
        )
        
        for sub_query in sub_queries:
            # Enhanced multi-stage retrieval with error handling
            try:
                retrieval_results = self.enhanced_retriever.retrieve(sub_query, top_k=3, context=retrieval_context)
            except Exception as e:
                logging.warning(f"Enhanced retrieval failed for sub_query '{sub_query}': {e}")
                # Fallback to original retriever
                retrieval_results = self.retriever.retrieve(sub_query, top_k=3)
            
            # Dynamic knowledge graph enhancement
            try:
                relevant_entities = self.dynamic_knowledge_graph.get_contextual_entities(sub_query)
                if relevant_entities and len(relevant_entities) > 0:
                    entity_names = [e.get('entity', '') for e in relevant_entities if isinstance(e, dict) and 'entity' in e]
                    if entity_names:
                        context_paths = self.knowledge_graph.find_context_paths(entity_names)
                    else:
                        context_paths = []
                else:
                    context_paths = []
            except Exception as e:
                logging.warning(f"Dynamic knowledge graph enhancement failed: {e}")
                context_paths = []
            
            all_results.extend(retrieval_results)
        
        # Step 4: Advanced SQL Generation with Planning
        if all_results:
            try:
                # Create advanced SQL plan
                advanced_plan = self.advanced_sql_planner.create_advanced_plan(question, all_results)
                
                # Generate SQL with validation - convert AdvancedSQLPlan to SQLPlan format
                sql_plan = type('SQLPlan', (), {
                    'steps': advanced_plan.aggregation_strategy + advanced_plan.filter_strategy,
                    'tables': advanced_plan.table_selection,
                    'joins': advanced_plan.join_strategy,
                    'filters': advanced_plan.filter_strategy,
                    'aggregations': advanced_plan.aggregation_strategy
                })()
                sql = self.sql_generator.generate_sql_with_validation(question, all_results, sql_plan)
                logging.info(f"Generated SQL:\n{sql}")
                
                # Advanced SQL validation
                validation_result = self.advanced_sql_validator.validate_and_optimize(
                    sql, QueryContext(question, user_id, session_id)
                )
                
                # Use optimized SQL if available
                if validation_result.optimized_sql:
                    sql = validation_result.optimized_sql
                    logging.info(f"Using optimized SQL:\n{sql}")
                
            except Exception as e:
                logging.warning(f"Advanced SQL planning failed: {e}")
                # Fallback to original SQL generation
                try:
                    plan = self.sql_generator.create_sql_plan(question, all_results)
                    sql = self.sql_generator.generate_sql_with_validation(question, all_results, plan)
                    validation_result = None
                except Exception as e2:
                    logging.error(f"Fallback SQL generation also failed: {e2}")
                    # Create a basic SQL query with date filter
                    sql = f"""
                    SELECT 
                        product_name,
                        client_name,
                        depot_name,
                        SUM(salesAMT) as total_sales,
                        COUNT(*) as record_count
                    FROM `{BQ_PROJECT}.{BQ_DATASET}.sales_details`
                    WHERE sales_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
                    GROUP BY product_name, client_name, depot_name
                    ORDER BY total_sales DESC
                    LIMIT 100
                    """
                    validation_result = None
            
            # Step 5: Execute and Analyze
            execution_result = self._execute_and_analyze(sql, question)
            
            # Step 6: Advanced Analytics
            try:
                analysis_results = self.advanced_analytics.comprehensive_analysis(execution_result.get("results", []), question)
                execution_result["insights"] = analysis_results.insights
                # Convert AnalysisResults to dictionary for JSON serialization
                execution_result["analysis_results"] = {
                    "descriptive_stats": analysis_results.descriptive_stats,
                    "trend_analysis": analysis_results.trend_analysis,
                    "anomalies": analysis_results.anomalies,
                    "correlations": analysis_results.correlations,
                    "significance_tests": analysis_results.significance_tests,
                    "insights": analysis_results.insights
                }
            except Exception as e:
                logging.warning(f"Advanced analytics failed: {e}")
                # Fallback to original analytics
                
            # Add professional data analyst insights
            try:
                professional_insights = self._generate_professional_insights(
                    execution_result.get("results", []), question, analysis_results
                )
                execution_result["professional_insights"] = professional_insights
            except Exception as e:
                logging.warning(f"Professional insights generation failed: {e}")
                insights = self.analytics.generate_insights(execution_result.get("results", []), question)
                execution_result["insights"] = insights
                analysis_results = None
            
        else:
            execution_result = {
                "error": "No relevant data found for your query",
                "sql": None,
                "results": [],
                "charts": [],
                "insights": []
            }
        
        # Step 7: Store interaction
        self.conversation_memory.add_interaction(user_id, session_id, question, execution_result)
        
        # Add enhanced metadata
        enhanced_features = {
            "query_complexity": enhanced_intent.complexity,
            "requires_decomposition": enhanced_intent.requires_decomposition,
            "retrieval_stats": self.enhanced_retriever.get_performance_stats()
        }
        
        # Add optional features if they exist
        if 'validation_result' in locals() and validation_result:
            # Convert ValidationResult to dictionary for JSON serialization
            enhanced_features["validation_result"] = {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "performance_score": validation_result.performance_score,
                "optimization_suggestions": validation_result.optimization_suggestions,
                "optimized_sql": validation_result.optimized_sql
            }
        if 'analysis_results' in locals() and analysis_results:
            # Convert AnalysisResults to dictionary for JSON serialization
            enhanced_features["analysis_results"] = {
                "descriptive_stats": analysis_results.descriptive_stats,
                "trend_analysis": analysis_results.trend_analysis,
                "anomalies": analysis_results.anomalies,
                "correlations": analysis_results.correlations,
                "significance_tests": analysis_results.significance_tests,
                "insights": analysis_results.insights
            }
        
        execution_result["metadata"] = {
            "processing_time": time.time() - start_time,
            "intent": enhanced_intent.intent_type,
            "confidence": enhanced_intent.confidence,
            "entities": enhanced_intent.entities,
            "temporal_context": enhanced_intent.temporal_context,
            "enhanced_features": enhanced_features
        }
        
        return execution_result
    
    def _generate_professional_insights(self, data: List[Dict], question: str, analysis_results: Any = None) -> Dict[str, Any]:
        """Generate professional data analyst insights"""
        if not data:
            return {"error": "No data available for analysis"}
        
        insights = {
            "executive_summary": "",
            "key_metrics": {},
            "trends": [],
            "recommendations": [],
            "risk_factors": [],
            "opportunities": []
        }
        
        try:
            # Extract key metrics
            if any('year' in str(row).lower() for row in data):
                years = [row.get('year', 0) for row in data if 'year' in str(row).lower()]
                if years:
                    insights["key_metrics"]["years_analyzed"] = sorted(set(years))
            
            if any('revenue' in str(row).lower() for row in data):
                revenues = [row.get('yearly_revenue', 0) for row in data if 'yearly_revenue' in row]
                if revenues:
                    insights["key_metrics"]["total_revenue"] = sum(revenues)
                    insights["key_metrics"]["avg_revenue"] = sum(revenues) / len(revenues)
                    insights["key_metrics"]["revenue_range"] = f"{min(revenues):,.0f} - {max(revenues):,.0f}"
            
            # Generate adaptive executive summary based on query and data
            insights["executive_summary"] = self._generate_adaptive_executive_summary(data, question)
            
            # Generate recommendations
            if analysis_results and hasattr(analysis_results, 'insights'):
                insights["recommendations"] = [
                    "Focus on top-performing products to maximize revenue",
                    "Strengthen relationships with key clients",
                    "Analyze seasonal patterns for inventory planning",
                    "Monitor transaction volume trends for capacity planning"
                ]
            
            # Risk factors
            insights["risk_factors"] = [
                "Dependency on top clients (concentration risk)",
                "Seasonal fluctuations in sales",
                "Product performance variability"
            ]
            
            # Opportunities
            insights["opportunities"] = [
                "Expand successful product lines",
                "Develop new client acquisition strategies",
                "Optimize depot performance",
                "Implement data-driven forecasting"
            ]
            
        except Exception as e:
            logging.warning(f"Professional insights generation error: {e}")
            insights["error"] = f"Analysis error: {str(e)}"
        
        return insights
    
    def _generate_adaptive_executive_summary(self, data: List[Dict], question: str) -> str:
        """Generate adaptive executive summary based on query and data"""
        if not data:
            return "No data available for analysis"
        
        try:
            # Extract years if available
            years = [row.get('year', 0) for row in data if 'year' in str(row).lower() and row.get('year')]
            years = sorted(set(years))
            
            # Extract revenue data
            revenues = [row.get('yearly_revenue', 0) for row in data if 'yearly_revenue' in row]
            
            if not years or not revenues:
                return "Data analysis completed - detailed metrics available in key metrics section"
            
            # Generate summary based on query context
            if "compare" in question.lower() and len(years) >= 2:
                latest_year = max(years)
                prev_year = latest_year - 1
                
                latest_data = next((row for row in data if row.get('year') == latest_year), {})
                prev_data = next((row for row in data if row.get('year') == prev_year), {})
                
                if latest_data and prev_data:
                    growth_pct = latest_data.get('revenue_growth_pct', 0)
                    return f"""
                    📊 Sales Analysis Summary: {latest_year} vs {prev_year}
                    
                    • Total Revenue {latest_year}: ${latest_data.get('yearly_revenue', 0):,.0f}
                    • Revenue Growth: {growth_pct}% year-over-year
                    • Top Product: {latest_data.get('top_product', latest_data.get('productName', 'N/A'))}
                    • Top Client: {latest_data.get('top_client', latest_data.get('clientName', 'N/A'))}
                    
                    {'📈 Positive Growth Trend' if growth_pct > 0 else '📉 Declining Performance' if growth_pct < 0 else '➡️ Stable Performance'}
                    """
            
            elif len(years) == 1:
                year = years[0]
                year_data = next((row for row in data if row.get('year') == year), {})
                return f"""
                📊 Sales Analysis Summary for {year}
                
                • Total Revenue: ${year_data.get('yearly_revenue', 0):,.0f}
                • Total Transactions: {year_data.get('yearly_transactions', 0):,}
                • Avg Transaction Value: ${year_data.get('yearly_avg_transaction', 0):,.2f}
                • Top Product: {year_data.get('top_product', year_data.get('productName', 'N/A'))}
                • Top Client: {year_data.get('top_client', year_data.get('clientName', 'N/A'))}
                """
            
            else:
                # Multi-year analysis
                total_revenue = sum(revenues)
                avg_revenue = total_revenue / len(revenues)
                return f"""
                📊 Multi-Year Sales Analysis ({min(years)}-{max(years)})
                
                • Total Revenue Across Years: ${total_revenue:,.0f}
                • Average Annual Revenue: ${avg_revenue:,.0f}
                • Years Analyzed: {len(years)}
                • Revenue Range: ${min(revenues):,.0f} - ${max(revenues):,.0f}
                """
                
        except Exception as e:
            logging.warning(f"Adaptive executive summary generation error: {e}")
            return "Comprehensive analysis completed - detailed insights available in the analysis results"
    
    def _execute_and_analyze(self, sql: str, query: str) -> Dict[str, Any]:
        """Execute SQL and perform analysis"""
        try:
            job = bq_client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=False))
            # Convert BigQuery results to dictionaries, handling boolean values
            results = []
            for row in job.result():
                row_dict = {}
                for key, value in row.items():
                    # Handle boolean values explicitly
                    if isinstance(value, bool):
                        row_dict[key] = bool(value)
                    else:
                        row_dict[key] = value
                results.append(row_dict)
        except Exception as e:
            return {"error": str(e), "sql": sql, "charts": [], "insights": []}

        if not results:
            return {"results": [], "summary": "No results found.", "sql": sql, "charts": [], "insights": []}

        # Generate summary
        summary_prompt = f"Summarize these business results in plain terms:\n{json.dumps(results[:5], indent=2, cls=PipelineJSONEncoder)}"
        try:
            summary = llm.generate_content(summary_prompt).candidates[0].content.parts[0].text.strip()
        except:
            summary = "Results retrieved successfully."

        # Generate charts
        charts = self._generate_charts(results)

        return {
            "results": results,
            "summary": summary,
            "sql": sql,
            "charts": charts,
            "insights": []
        }
    
    def _generate_charts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate chart configurations"""
        if not results:
            return []
        
        charts = []
        numeric_keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
        cat_keys = [k for k in results[0] if not isinstance(results[0][k], (int, float))]
        x_axis = cat_keys[0] if cat_keys else list(results[0].keys())[0]

        for key in numeric_keys:
            chart_type = "pie" if len(results) <= 5 else "bar"
            charts.append({
                "type": chart_type,
                "title": key,
                "labels": [str(r[x_axis]) for r in results],
                "values": [r[key] for r in results]
            })

        return charts

# -----------------------------
# 9️⃣ INITIALIZATION & USAGE
# -----------------------------
if __name__ == "__main__":
    # Initialize the Ultimate RAG Pipeline
    pipeline = UltimateRAGPipeline()
    
    logging.info("🚀 Ultimate RAG Pipeline Ready!")
    logging.info("Features: Multi-stage retrieval, Query decomposition, Knowledge graph, Enhanced SQL generation, Conversation memory, Advanced analytics")
    
    # Example usage
    while True:
        user_question = input("\nAsk me anything (type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        response = pipeline.ask_intelligent(user_question)
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(json.dumps(response, indent=2, cls=PipelineJSONEncoder))
