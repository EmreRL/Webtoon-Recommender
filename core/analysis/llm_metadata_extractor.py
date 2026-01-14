"""
LLM-based metadata extraction using Google Gemini Flash.
More robust than keyword matching for understanding user intent.
Updated to map quality queries to popularity + likes sorting.
Now supports 5-tier popularity system including "Hit" (top 3%).
"""
import json
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from config import Config


@dataclass
class ExtractedMetadata:
    """Structured metadata extracted from user query."""
    genre: Optional[str] = None
    popularity: Optional[list] = None
    quality_intent: Optional[str] = None  # New: track quality intent separately
    content_keywords: Optional[str] = None
    query_type: str = 'content'  # 'attribute', 'content', 'hybrid'
    confidence: float = 0.0
    sort_by_likes: bool = False  # New: flag for likes-based sorting


class LLMMetadataExtractor:
    """Uses a mini LLM to extract metadata from user queries."""
    
    def __init__(self):
        """Initialize Gemini Flash for fast metadata extraction."""
        genai.configure(api_key=Config.GEMINI_API_KEY)
        # Use Gemini Flash for speed and cost efficiency
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ LLM Metadata Extractor initialized (Gemini Flash)")
    
    def extract(self, user_query: str) -> ExtractedMetadata:
        """
        Extract metadata from user query using LLM.
        
        Args:
            user_query: User's query string
            
        Returns:
            ExtractedMetadata object with extracted filters
        """
        prompt = self._build_extraction_prompt(user_query)
        
        try:
            # Request JSON response from LLM
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            result_text = re.sub(r'```json\s*|\s*```', '', result_text).strip()
            
            metadata_dict = json.loads(result_text)
            
            # Map quality to popularity intelligently
            popularity, sort_by_likes = self._map_quality_to_popularity(
                quality_intent=metadata_dict.get('quality_intent'),
                popularity=metadata_dict.get('popularity')
            )
            
            # Convert to ExtractedMetadata object
            metadata = ExtractedMetadata(
                genre=metadata_dict.get('genre'),
                popularity=popularity,
                quality_intent=metadata_dict.get('quality_intent'),
                content_keywords=metadata_dict.get('content_keywords'),
                query_type=metadata_dict.get('query_type', 'content'),
                confidence=metadata_dict.get('confidence', 0.8),
                sort_by_likes=sort_by_likes
            )
            
            print(f"✅ LLM extracted metadata:")
            print(f"   Genre: {metadata.genre}")
            print(f"   Popularity: {metadata.popularity}")
            print(f"   Quality Intent: {metadata.quality_intent}")
            print(f"   Sort by Likes: {metadata.sort_by_likes}")
            print(f"   Content: {metadata.content_keywords}")
            print(f"   Type: {metadata.query_type}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse LLM response as JSON: {e}")
            print(f"   Response: {result_text[:200]}")
            # Fallback to empty metadata
            return ExtractedMetadata(query_type='content', confidence=0.3)
            
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            # Fallback to empty metadata
            return ExtractedMetadata(query_type='content', confidence=0.3)
    
    def _map_quality_to_popularity(
        self, 
        quality_intent: Optional[str],
        popularity: Optional[list]
    ) -> tuple[Optional[list], bool]:
        """
        Map quality intent to popularity tiers + likes sorting.
        
        5-Tier Popularity System:
        - Hit (top 3% - absolute best)
        - VeryPopular
        - Popular  
        - LessPopular
        - Unpopular
        
        Quality Mapping Strategy:
        - Excellent quality = Hit (top 3%, proven masterpieces)
        - Good quality = Popular/VeryPopular (solid mainstream)
        - Hidden gems = Popular/LessPopular (mid-tier with high likes)
        - Poor quality = Unpopular (low engagement)
        
        Args:
            quality_intent: User's quality preference (excellent, good, poor)
            popularity: Extracted popularity preference
            
        Returns:
            (popularity_list, sort_by_likes_flag)
        """
        sort_by_likes = False
        
        if not quality_intent:
            # No quality intent, keep original popularity
            return popularity, False
        
        # If user specified both quality AND popularity, combine intelligently
        if popularity and quality_intent:
            if quality_intent == "poor":
                # Want unpopular + bad quality → stick with unpopular, sort by likes ASC
                if "Unpopular" in popularity or "LessPopular" in popularity:
                    sort_by_likes = True
                    return popularity, sort_by_likes
            else:
                # Want popular + good quality → stick with popular, sort by likes DESC
                sort_by_likes = True
                return popularity, sort_by_likes
        
        # Quality-only queries → map to popularity
        if quality_intent == "excellent":
            # Excellent = Hit (top 3% masterpieces) + high likes
            popularity = ["Hit"]
            sort_by_likes = True
            
        elif quality_intent == "good":
            # Good = Popular or VeryPopular + high likes (solid mainstream)
            popularity = ["Popular", "VeryPopular"]
            sort_by_likes = True
            
        elif quality_intent == "unpopular_but_good":
            # Hidden gems = Popular/LessPopular + high likes (underrated quality)
            popularity = ["Popular", "LessPopular"]
            sort_by_likes = True
            
        elif quality_intent == "poor":
            # Poor = Unpopular + low likes
            popularity = ["Unpopular"]
            sort_by_likes = True
        
        return popularity, sort_by_likes
    
    def _build_extraction_prompt(self, user_query: str) -> str:
        """Build a prompt for metadata extraction."""
        prompt = f"""You are a query parser for a webtoon recommendation system. Extract metadata from the user's query.

USER QUERY: "{user_query}"

Extract the following information:

1. **genre**: One of: Action, Romance, Fantasy, Drama, Thriller, Horror, Comedy, Supernatural, Sci-Fi, School, Slice of Life
   - Return null if not specified

2. **popularity**: List of acceptable popularity levels based on user intent
   - We have 5 tiers: Hit (top 3%), VeryPopular, Popular, LessPopular, Unpopular
   - If user wants HIT/MASTERPIECE/LEGENDARY/ABSOLUTE BEST: ["Hit"]
   - If user wants VERY POPULAR/EXTREMELY POPULAR: ["VeryPopular", "Hit"]
   - If user wants POPULAR/FAMOUS/TRENDING/MAINSTREAM: ["Popular", "VeryPopular"]
   - If user wants UNPOPULAR/HIDDEN GEM/UNDERRATED/NOT POPULAR/UNKNOWN/NICHE: ["Unpopular", "LessPopular"]
   - If user wants LESS POPULAR: ["LessPopular"]
   - Return null if not specified
   - IMPORTANT: Pay attention to negations like "NOT popular"

3. **quality_intent**: Extract quality preference (we'll map it to popularity + likes)
   - If user wants EXCELLENT/BEST/TOP/HIGHEST/MASTERPIECE QUALITY: "excellent"
   - If user wants GOOD/QUALITY/GREAT/DECENT: "good"
   - If user wants UNPOPULAR BUT GOOD/HIDDEN GEM WITH QUALITY: "unpopular_but_good"
   - If user wants POOR/BAD/LOW QUALITY: "poor"
   - Return null if not specified
   - IMPORTANT: "bad quality" means "poor", not "good"!

4. **content_keywords**: Extract content-related themes (revenge, overpowered mc, crazy character, etc.)
   - Return null if query is only about attributes

5. **query_type**: One of:
   - "attribute": Query only asks about metadata (genre, popularity, quality)
   - "content": Query asks about plot, characters, themes
   - "hybrid": Query asks about both

6. **confidence**: Float between 0 and 1 indicating how confident you are in the extraction

IMPORTANT RULES:
- Pay close attention to negations: "NOT popular" means unpopular, "bad quality" means poor quality
- "but" often indicates contrast: "popular but bad" means popular AND poor quality
- "hidden gem" means unpopular but good quality
- "masterpiece" / "legendary" / "absolute best" should map to Hit tier (top 3%)
- Multiple attributes can be combined: "popular action with crazy MC" has all three

Return ONLY a valid JSON object with these exact keys, no explanation:

{{
  "genre": null or "GenreName",
  "popularity": null or ["Level1", "Level2"],
  "quality_intent": null or "excellent" or "good" or "unpopular_but_good" or "poor",
  "content_keywords": null or "extracted themes",
  "query_type": "attribute" or "content" or "hybrid",
  "confidence": 0.0 to 1.0
}}

Examples:

Query: "popular webtoon"
{{
  "genre": null,
  "popularity": ["Popular", "VeryPopular"],
  "quality_intent": null,
  "content_keywords": null,
  "query_type": "attribute",
  "confidence": 0.95
}}

Query: "masterpiece webtoon" or "legendary quality"
{{
  "genre": null,
  "popularity": null,
  "quality_intent": "excellent",
  "content_keywords": null,
  "query_type": "attribute",
  "confidence": 0.95
}}

Query: "good quality webtoon"
{{
  "genre": null,
  "popularity": null,
  "quality_intent": "good",
  "content_keywords": null,
  "query_type": "attribute",
  "confidence": 0.95
}}

Query: "hidden gem" or "unpopular but good quality"
{{
  "genre": null,
  "popularity": null,
  "quality_intent": "unpopular_but_good",
  "content_keywords": null,
  "query_type": "attribute",
  "confidence": 0.95
}}

Query: "very popular but bad quality"
{{
  "genre": null,
  "popularity": ["VeryPopular", "Hit"],
  "quality_intent": "poor",
  "content_keywords": null,
  "query_type": "attribute",
  "confidence": 0.95
}}

Query: "webtoon where mc is crazy"
{{
  "genre": null,
  "popularity": null,
  "quality_intent": null,
  "content_keywords": "crazy mc",
  "query_type": "content",
  "confidence": 0.9
}}

Query: "hit action webtoon with overpowered mc"
{{
  "genre": "Action",
  "popularity": ["Hit"],
  "quality_intent": null,
  "content_keywords": "overpowered mc",
  "query_type": "hybrid",
  "confidence": 0.9
}}

Now extract metadata from the user query above. Return ONLY the JSON object:"""
        
        return prompt


# Singleton instance
_extractor_instance = None

def get_llm_extractor() -> LLMMetadataExtractor:
    """Get or create the global LLM extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LLMMetadataExtractor()
    return _extractor_instance