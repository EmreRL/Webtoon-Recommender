"""
Query classification module to detect query intent.
Classifies queries as attribute-based, content-based, or hybrid.
"""
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """Represents the classified intent of a user query."""
    query_type: str  # 'attribute', 'content', 'hybrid'
    filters: Dict[str, Any]  # Extracted filters (genre, popularity, etc.)
    semantic_query: str  # The part to use for semantic search
    confidence: float  # Confidence score


class QueryClassifier:
    """Classifies user queries to determine search strategy."""
    
    # Attribute keywords mapping
    POPULARITY_KEYWORDS = {
        'popular': ['Popular', 'VeryPopular'],
        'very popular': ['VeryPopular'],
        'trending': ['VeryPopular'],
        'famous': ['Popular', 'VeryPopular'],
        'well-known': ['Popular', 'VeryPopular'],
        'mainstream': ['Popular', 'VeryPopular'],
        # Negative keywords
        'unpopular': ['Unpopular', 'LessPopular'],
        'not popular': ['Unpopular', 'LessPopular'],
        'not so popular': ['Unpopular', 'LessPopular'],
        'less popular': ['LessPopular'],
        'unknown': ['Unpopular', 'LessPopular'],
        'hidden gem': ['Unpopular', 'LessPopular'],
        'underrated': ['Unpopular', 'LessPopular'],
        'niche': ['Unpopular', 'LessPopular'],
    }
    
    QUALITY_KEYWORDS = {
        'excellent': ['Excellent'],
        'best': ['Excellent'],
        'top': ['Excellent'],
        'highest quality': ['Excellent'],
        'highly rated': ['Excellent'],
        'great': ['Excellent', 'Good'],
        'good': ['Good', 'Excellent'],
        'quality': ['Good', 'Excellent'],
        'decent': ['Good'],
        # Negative keywords
        'poor': ['Poor'],
        'bad': ['Poor'],
        'low quality': ['Poor'],
    }
    
    GENRE_KEYWORDS = {
        'action': 'Action',
        'romance': 'Romance',
        'fantasy': 'Fantasy',
        'drama': 'Drama',
        'thriller': 'Thriller',
        'horror': 'Horror',
        'comedy': 'Comedy',
        'supernatural': 'Supernatural',
        'sci-fi': 'Sci-Fi',
        'school': 'School',
        'slice of life': 'Slice of Life'
    }
    
    # Content-based keywords (triggers semantic search)
    CONTENT_KEYWORDS = [
        'mc', 'protagonist', 'character', 'plot', 'story',
        'about', 'where', 'revenge', 'power', 'crazy',
        'overpowered', 'weak', 'strong', 'smart', 'funny',
        'sad', 'dark', 'wholesome', 'toxic', 'betrayal',
        'friendship', 'family', 'underdog', 'villain',
        'hero', 'martial arts', 'regression', 'reincarnation',
        'system', 'game', 'level up', 'dungeon', 'tower'
    ]
    
    # Negation patterns
    NEGATION_PATTERNS = [
        r'\bnot\s+(?:so\s+)?',  # "not", "not so"
        r'\bun',                # "unpopular", "unknown"
        r'\bnever\s+',          # "never popular"
        r'\bavoiding?\s+',      # "avoid", "avoiding"
        r'\bwithout\s+',        # "without"
        r'\bisn[\'t]*\s+',      # "isn't"
    ]
    
    def classify(self, user_query: str) -> QueryIntent:
        """
        Classify the user query and extract filters.
        
        Args:
            user_query: The user's query string
            
        Returns:
            QueryIntent object with classification results
        """
        query_lower = user_query.lower().strip()
        
        # Extract filters with negation awareness
        filters = {}
        
        # Check for popularity filters
        popularity_filter = self._extract_popularity(query_lower)
        if popularity_filter:
            filters['popularity'] = popularity_filter
        
        # Check for quality filters
        quality_filter = self._extract_quality(query_lower)
        if quality_filter:
            filters['quality'] = quality_filter
        
        # Check for genre filters
        genre_filter = self._extract_genre(query_lower)
        if genre_filter:
            filters['genre'] = genre_filter
        
        # Check if query has content-based keywords
        has_content_keywords = any(
            keyword in query_lower 
            for keyword in self.CONTENT_KEYWORDS
        )
        
        # Build semantic query (remove attribute keywords)
        semantic_query = self._build_semantic_query(query_lower, filters)
        
        # Determine query type
        if filters and not has_content_keywords:
            # Pure attribute query (e.g., "popular webtoon")
            query_type = 'attribute'
            confidence = 0.9
        elif filters and has_content_keywords:
            # Hybrid query (e.g., "popular webtoon with crazy mc")
            query_type = 'hybrid'
            confidence = 0.85
        elif has_content_keywords:
            # Pure content query (e.g., "webtoon where mc is crazy")
            query_type = 'content'
            confidence = 0.8
        else:
            # Default to content-based with lower confidence
            query_type = 'content'
            confidence = 0.5
            semantic_query = user_query  # Use full query
        
        return QueryIntent(
            query_type=query_type,
            filters=filters,
            semantic_query=semantic_query,
            confidence=confidence
        )
    
    def _extract_popularity(self, query: str) -> Optional[List[str]]:
        """Extract popularity filter from query with negation awareness."""
        # Sort keywords by length (longest first) to match specific phrases before general ones
        # This ensures "not popular" is checked before "popular"
        sorted_keywords = sorted(
            self.POPULARITY_KEYWORDS.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for keyword, values in sorted_keywords:
            if keyword in query:
                # Found a match, return the corresponding values
                return values
        
        return None
    
    def _extract_quality(self, query: str) -> Optional[List[str]]:
        """Extract quality filter from query with negation awareness."""
        # Sort keywords by length (longest first) to match specific phrases before general ones
        sorted_keywords = sorted(
            self.QUALITY_KEYWORDS.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for keyword, values in sorted_keywords:
            if keyword in query:
                return values
        
        return None
    
    def _extract_genre(self, query: str) -> Optional[str]:
        """Extract genre filter from query."""
        for keyword, genre in self.GENRE_KEYWORDS.items():
            if keyword in query:
                return genre
        return None
    
    def _build_semantic_query(self, query: str, filters: Dict) -> str:
        """
        Build semantic query by removing filter keywords.
        Keeps the content-focused part of the query.
        """
        # Start with the original query
        semantic_query = query
        
        # Remove popularity keywords
        for keyword in self.POPULARITY_KEYWORDS.keys():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            semantic_query = re.sub(pattern, '', semantic_query, flags=re.IGNORECASE)
        
        # Remove quality keywords
        for keyword in self.QUALITY_KEYWORDS.keys():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            semantic_query = re.sub(pattern, '', semantic_query, flags=re.IGNORECASE)
        
        # Remove genre keywords
        for keyword in self.GENRE_KEYWORDS.keys():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            semantic_query = re.sub(pattern, '', semantic_query, flags=re.IGNORECASE)
        
        # Remove common filler words
        filler_words = ['webtoon', 'manhwa', 'manga', 'give me', 'show me', 
                       'i want', 'looking for', 'recommend', 'find', 'a', 'an', 'the']
        for filler in filler_words:
            pattern = r'\b' + re.escape(filler) + r'\b'
            semantic_query = re.sub(pattern, '', semantic_query, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        semantic_query = ' '.join(semantic_query.split()).strip()
        
        # If empty after filtering, return original
        return semantic_query if semantic_query else query


# Singleton instance
_classifier_instance = None

def get_classifier() -> QueryClassifier:
    """Get or create the global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance