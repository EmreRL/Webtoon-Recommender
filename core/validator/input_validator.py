"""
Input validation module for the Webtoon RAG System.
Validates user queries for relevance and quality.
"""
import re
from typing import Tuple
from config import Config


class InputValidator:
    """Validates user input for webtoon recommendation queries."""
    
    # Keywords that suggest webtoon-related queries
    WEBTOON_KEYWORDS = {
        'webtoon', 'manhwa', 'manga', 'comic', 'story', 'genre', 
        'action', 'romance', 'fantasy', 'drama', 'thriller', 'horror',
        'comedy', 'adventure', 'school', 'supernatural', 'sci-fi',
        'recommend', 'suggestion', 'similar', 'like', 'find',
        'hero', 'villain', 'protagonist', 'character', 'plot'
    }
    
    # Invalid patterns (spam, gibberish, etc.)
    INVALID_PATTERNS = [
        r'^[^a-zA-Z0-9\s]+$',  # Only special characters
        r'^(.)\1{10,}$',        # Repeated characters
    ]
    
    @staticmethod
    def validate(user_input: str) -> Tuple[bool, str]:
        """
        Validate user input for webtoon recommendation queries.
        
        Args:
            user_input: The user's query string
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if input is valid, False otherwise
            - error_message: Empty if valid, error description if invalid
        """
        # Check if input is empty or too short
        if not user_input or len(user_input.strip()) < Config.MIN_INPUT_LENGTH:
            return False, "Input is too short. Please provide a meaningful query."
        
        # Check if input is too long
        if len(user_input) > Config.MAX_INPUT_LENGTH:
            return False, f"Input is too long. Maximum {Config.MAX_INPUT_LENGTH} characters."
        
        # Clean and normalize input
        cleaned_input = user_input.strip().lower()
        
        # Check for invalid patterns (gibberish, spam)
        for pattern in InputValidator.INVALID_PATTERNS:
            if re.match(pattern, cleaned_input):
                return False, "Input appears to be invalid or contains only special characters."
        
        # Check if input has actual content (not just whitespace/special chars)
        has_alphanumeric = bool(re.search(r'[a-zA-Z0-9]', cleaned_input))
        if not has_alphanumeric:
            return False, "Please provide a query with actual content."
        
        # Optional: Check for webtoon relevance
        # This is a soft check - we'll let the LLM handle edge cases
        words = set(re.findall(r'\b\w+\b', cleaned_input))
        has_webtoon_context = bool(words & InputValidator.WEBTOON_KEYWORDS)
        
        # If no obvious webtoon keywords, check if it's a reasonable query
        if not has_webtoon_context:
            # Allow general story/theme queries (e.g., "revenge story", "school life")
            if len(words) >= 2:  # At least 2 words
                return True, ""
            else:
                return False, (
                    "Your query doesn't seem to be about webtoon recommendations. "
                    "Please ask about webtoon genres, themes, or specific preferences."
                )
        
        # Input is valid
        return True, ""
    
    @staticmethod
    def sanitize(user_input: str) -> str:
        """
        Sanitize user input by removing extra whitespace and normalizing.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Sanitized input string
        """
        # Remove extra whitespace
        sanitized = ' '.join(user_input.split())
        
        # Remove potentially harmful characters (basic XSS prevention)
        sanitized = re.sub(r'[<>]', '', sanitized)
        
        return sanitized.strip()