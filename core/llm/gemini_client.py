"""
Google Gemini client for generating final recommendations.
Handles LLM interaction for RAG responses with rate limiting and retry logic.
"""
import time
from typing import List, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions
from config import Config


class GeminiClient:
    """Handles interaction with Google Gemini API with rate limiting."""
    
    def __init__(self):
        """Initialize Gemini client with rate limiting."""
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        # Rate limiting configuration
        self.last_request_time = 0
        self.min_request_interval = 12  # 12 seconds = max 5 requests/minute (free tier limit)
        
        print(f"✅ Gemini model initialized: {Config.GEMINI_MODEL}")
        print(f"⏱️  Rate limiting enabled: {self.min_request_interval}s between requests")
    
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        if self.last_request_time > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed
                print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate a response from Gemini with retry logic.
        
        Args:
            prompt: The complete prompt including context and query
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If generation fails after all retries
        """
        # Apply rate limiting before request
        self._wait_for_rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                self.last_request_time = time.time()
                return response.text
                
            except exceptions.ResourceExhausted as e:
                # Handle quota/rate limit errors
                if attempt == max_retries - 1:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts")
                    raise Exception(
                        "Gemini API quota exceeded. Please try again in a few minutes. "
                        "If this persists, consider enabling billing on your Google Cloud project."
                    )
                
                # Calculate exponential backoff: 60s, 120s, 240s
                wait_time = 60 * (2 ** attempt)
                print(f"⚠️  Rate limit hit. Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                print(f"    Error: {str(e)[:100]}...")
                time.sleep(wait_time)
                
            except exceptions.InvalidArgument as e:
                # Handle invalid prompt errors (don't retry)
                print(f"❌ Invalid prompt: {e}")
                raise Exception(f"Invalid prompt format: {str(e)}")
                
            except Exception as e:
                # Handle other errors
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    # Treat as rate limit error
                    if attempt == max_retries - 1:
                        raise Exception(
                            "Gemini API quota exceeded. Please wait a few minutes and try again."
                        )
                    wait_time = 60 * (2 ** attempt)
                    print(f"⚠️  Quota error detected. Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Unknown error, don't retry
                    print(f"❌ Gemini generation failed: {e}")
                    raise
        
        # Should never reach here, but just in case
        raise Exception("Generation failed after all retries")


class ResponseBuilder:
    """Builds prompts for Gemini based on retrieved context."""
    
    @staticmethod
    def build_rag_prompt(
        user_query: str, 
        retrieved_webtoons: List[Dict[str, Any]]
    ) -> str:
        """
        Build a RAG prompt with retrieved context.
        
        Args:
            user_query: Original user query
            retrieved_webtoons: List of similar webtoons from database
            
        Returns:
            Complete prompt string for Gemini
        """
        # Format retrieved webtoons as context
        context_parts = []
        for i, webtoon in enumerate(retrieved_webtoons, 1):
            similarity = webtoon.get('similarity', 0)
            context_parts.append(
                f"{i}. **{webtoon['title']}** by {webtoon['author']}\n"
                f"   - Genre: {webtoon['genre']}\n"
                f"   - Summary: {webtoon['summary']}\n"
                f"   - Quality: {webtoon['quality']} | Popularity: {webtoon['popularity']}\n"
                f"   - Released: {webtoon['released_date']}\n"
                f"   - Relevance Score: {similarity:.2f}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Build the complete prompt (keep it concise to reduce token usage)
        prompt = f"""You are a webtoon recommendation expert. Based on the user's query and the retrieved similar webtoons from the database, provide personalized recommendations.

USER QUERY:
{user_query}

RETRIEVED SIMILAR WEBTOONS FROM DATABASE:
{context}

INSTRUCTIONS:
1. Recommend 3-5 webtoons from the list above that best match the user's query
2. Explain WHY each recommendation fits their request
3. Highlight key themes, genres, or story elements that match their interests
4. Be enthusiastic and engaging, but concise
5. ONLY recommend webtoons from the list above - DO NOT hallucinate or suggest webtoons not in the database
6. Order recommendations by relevance to the user's query

RESPONSE FORMAT:
Provide a natural, conversational response that includes:
- A brief introduction acknowledging their request
- 3-5 specific recommendations with titles and authors
- Short explanations for each recommendation
- A friendly closing

Begin your response now:"""
        
        return prompt


# Singleton instance
_gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create the global Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client