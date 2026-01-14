"""
Intelligent rejection handler that provides helpful, conversational responses
when no results are found, with analysis of what's available in the database.
"""
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from config import Config


class SmartRejectionHandler:
    """Handles no-result scenarios with helpful, natural language responses."""
    
    def __init__(self):
        """Initialize with Gemini Flash for fast rejection responses."""
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Smart Rejection Handler initialized")
    
    def handle_no_results(
        self,
        user_query: str,
        filters: Dict[str, Any],
        query_type: str,
        database_stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a helpful, conversational response when no results are found.
        
        Args:
            user_query: Original user query
            filters: Extracted filters that didn't match
            query_type: Type of query (attribute/content/hybrid)
            database_stats: Statistics about what's available in the database
            
        Returns:
            Natural language explanation and suggestions
        """
        # Build context about what went wrong
        missing_context = self._build_missing_context(filters, database_stats)
        
        # Generate natural response using LLM
        prompt = self._build_rejection_prompt(
            user_query, 
            filters, 
            query_type,
            missing_context
        )
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ LLM rejection generation failed: {e}")
            # Fallback to basic message
            return self._fallback_message(filters)
    
    def _build_missing_context(
        self, 
        filters: Dict[str, Any],
        database_stats: Optional[Dict[str, Any]]
    ) -> str:
        """Build context about what's missing in the database."""
        context_parts = []
        
        if 'genre' in filters:
            genre = filters['genre']
            context_parts.append(f"- You searched for '{genre}' genre")
            if database_stats and 'available_genres' in database_stats:
                available = database_stats['available_genres']
                context_parts.append(f"- Available genres: {', '.join(available)}")
        
        if 'popularity' in filters:
            pop_levels = filters['popularity']
            context_parts.append(f"- You wanted popularity levels: {', '.join(pop_levels)}")
            if database_stats and 'available_popularity' in database_stats:
                available = database_stats['available_popularity']
                context_parts.append(f"- Available popularity: {', '.join(available)}")
        
        if 'quality' in filters:
            quality_levels = filters['quality']
            context_parts.append(f"- You wanted quality levels: {', '.join(quality_levels)}")
            if database_stats and 'available_quality' in database_stats:
                available = database_stats['available_quality']
                context_parts.append(f"- Available quality: {', '.join(available)}")
        
        return "\n".join(context_parts) if context_parts else "No specific filters provided"
    
    def _build_rejection_prompt(
        self,
        user_query: str,
        filters: Dict[str, Any],
        query_type: str,
        missing_context: str
    ) -> str:
        """Build a prompt for generating a helpful rejection message."""
        prompt = f"""You are a friendly webtoon recommendation assistant. A user searched for something, but there are NO matching webtoons in the database.

USER QUERY: "{user_query}"

EXTRACTED FILTERS: {filters}

WHY NO RESULTS:
{missing_context}

Your task is to write a warm, conversational response that:
1. Acknowledges their request with empathy
2. Explains clearly WHY no results were found (e.g., "we don't have Comedy genre in our database")
3. Suggests alternatives based on what IS available
4. Keeps a positive, helpful tone (never robotic or error-like)
5. Is concise (2-3 sentences max)

Examples:

Bad (robotic): "Error: No webtoons found matching your criteria: {{'genre': 'Comedy'}}. Try different attributes."

Good (conversational): "I'd love to recommend some Comedy webtoons, but unfortunately our current database doesn't include that genre yet! However, we have some great Action and Romance titles that might make you laugh with their lighter moments. Would you like to explore those instead?"

Bad (technical): "Search returned 0 results. Adjust your query parameters."

Good (helpful): "Hmm, I couldn't find any webtoons that are both VeryPopular and Poor quality - that's a pretty rare combination! Most popular webtoons tend to be at least Good quality. Would you like to see popular webtoons regardless of quality, or focus on finding hidden gems?"

Now write a response for the user's query above. Be warm and helpful:"""
        
        return prompt
    
    def _fallback_message(self, filters: Dict[str, Any]) -> str:
        """Simple fallback message if LLM fails."""
        filter_str = ", ".join(f"{k}: {v}" for k, v in filters.items())
        return (
            f"I couldn't find any webtoons matching {filter_str}. "
            "This might be because our database doesn't have those specific combinations yet. "
            "Try adjusting your search or let me know what you're looking for!"
        )


# Singleton instance
_rejection_handler = None

def get_rejection_handler() -> SmartRejectionHandler:
    """Get or create the global rejection handler."""
    global _rejection_handler
    if _rejection_handler is None:
        _rejection_handler = SmartRejectionHandler()
    return _rejection_handler