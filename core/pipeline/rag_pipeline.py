"""
Enhanced RAG Pipeline with query classification and hybrid search.
Updated to include image URLs in recommendations.
"""
from typing import Dict, Any, List
from ..validator.input_validator import InputValidator
from ..embeddings.embedder import get_embedder
from ..database.hybrid_retriever import get_hybrid_retriever
from ..llm.gemini_client import get_gemini_client, ResponseBuilder
from ..analysis.llm_metadata_extractor import get_llm_extractor
from ..analysis.smart_rejection_handler import get_rejection_handler
from ..utils.database_stats import get_stats_collector
import json


class EnhancedRAGPipeline:
    """Orchestrates the complete RAG workflow with hybrid search."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize pipeline components.
        
        Args:
            verbose: If True, print detailed progress logs
        """
        self.verbose = verbose
        
        if verbose:
            print("="*60)
            print("Initializing Enhanced Webtoon RAG Pipeline...")
            print("="*60)
        
        self.validator = InputValidator()
        self.llm_extractor = get_llm_extractor()
        self.embedder = get_embedder()
        self.retriever = get_hybrid_retriever()
        self.gemini_client = get_gemini_client()
        self.response_builder = ResponseBuilder()
        self.rejection_handler = get_rejection_handler()
        self.stats_collector = get_stats_collector()
        
        if verbose:
            print("="*60)
            print("✅ Enhanced RAG Pipeline ready!")
            print("="*60)
    
    def _log(self, message: str) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Execute the enhanced RAG pipeline with query classification.
        
        Args:
            user_query: User's recommendation request
            
        Returns:
            Dictionary containing the final response and metadata
        """
        self._log(f"\n{'='*60}")
        self._log(f"Processing query: '{user_query}'")
        self._log(f"{'='*60}")
        
        # Step 1: Validate input
        self._log("\n[1/6] Validating input...")
        is_valid, error_message = self.validator.validate(user_query)
        
        if not is_valid:
            return {
                'success': False,
                'error': error_message,
                'stage': 'validation'
            }
        
        clean_query = self.validator.sanitize(user_query)
        self._log(f"✅ Input validated: '{clean_query}'")
        
        # Step 2: Extract metadata using LLM
        self._log("\n[2/6] Extracting metadata with LLM...")
        metadata = self.llm_extractor.extract(clean_query)
        
        # Convert ExtractedMetadata to filters dict (no quality!)
        filters = {}
        if metadata.genre:
            filters['genre'] = metadata.genre
        if metadata.popularity:
            filters['popularity'] = metadata.popularity
        
        query_type = metadata.query_type
        semantic_query = metadata.content_keywords if metadata.content_keywords else clean_query
        sort_by_likes = metadata.sort_by_likes
        
        self._log(f"   Query type: {query_type}")
        self._log(f"   Confidence: {metadata.confidence:.2f}")
        self._log(f"   Sort by likes: {sort_by_likes}")
        
        # Step 3: Generate embedding (if needed for semantic search)
        query_embedding = None
        if query_type in ['content', 'hybrid']:
            self._log("\n[3/6] Generating query embedding...")
            try:
                embed_text = semantic_query if semantic_query else clean_query
                query_embedding = self.embedder.embed(embed_text)
                self._log(f"✅ Embedding generated (dim={len(query_embedding)})")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Embedding generation failed: {str(e)}",
                    'stage': 'embedding'
                }
        else:
            self._log("\n[3/6] Skipping embedding (attribute-only query)")
        
        # Step 4: Retrieve with hybrid search
        self._log("\n[4/6] Retrieving webtoons with hybrid search...")
        try:
            retrieved_webtoons = self.retriever.retrieve_with_filters(
                query_embedding=query_embedding,
                filters=filters,
                query_type=query_type,
                sort_by_likes=sort_by_likes
            )
            
            if not retrieved_webtoons:
                # Use smart rejection handler for better user experience
                self._log("🤔 No results found, generating helpful response...")
                
                # Get database statistics for context
                db_stats = self.stats_collector.get_stats()
                
                # Generate natural, helpful rejection message
                rejection_message = self.rejection_handler.handle_no_results(
                    user_query=clean_query,
                    filters=filters,
                    query_type=query_type,
                    database_stats=db_stats
                )
                
                self._log(f"💬 Generated smart rejection message")
                
                return {
                    'success': False,
                    'error': rejection_message,
                    'stage': 'retrieval',
                    'query_type': query_type,
                    'filters': filters,
                    'is_smart_rejection': True,
                    'database_stats': db_stats
                }
            
            self._log(f"✅ Found {len(retrieved_webtoons)} webtoons")
            
            # Log top results
            if self.verbose:
                for i, webtoon in enumerate(retrieved_webtoons[:3], 1):
                    similarity = webtoon.get('similarity', 0)
                    likes = webtoon.get('likes', 0)
                    self._log(f"  {i}. {webtoon['title']} (similarity: {similarity:.3f}, likes: {likes:,})")
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Retrieval failed: {str(e)}",
                'stage': 'retrieval'
            }
        
        # Step 5: Generate explanations for top recommendations
        self._log("\n[5/6] Generating personalized explanations...")
        try:
            top_webtoons = retrieved_webtoons[:5]  # Take top 5
            structured_recommendations = self._generate_structured_recommendations(
                clean_query,
                top_webtoons,
                metadata,
                filters
            )
            self._log(f"✅ Generated {len(structured_recommendations)} recommendations")
        except Exception as e:
            self._log(f"⚠️ Failed to generate explanations: {str(e)}")
            # Fallback: return webtoons without explanations
            structured_recommendations = self._create_fallback_recommendations(retrieved_webtoons[:5])
        
        self._log(f"\n{'='*60}")
        self._log("✅ Pipeline completed successfully!")
        self._log(f"{'='*60}\n")
        
        return {
            'success': True,
            'query': clean_query,
            'query_type': query_type,
            'filters': filters,
            'sort_by_likes': sort_by_likes,
            'response': structured_recommendations,  # Now returns structured list
            'retrieved_count': len(retrieved_webtoons),
            'retrieved_webtoons': retrieved_webtoons,
            'stage': 'complete'
        }
    
    def _generate_structured_recommendations(
        self,
        user_query: str,
        webtoons: List[Dict[str, Any]],
        metadata: 'ExtractedMetadata',
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate structured recommendations with LLM-powered explanations.
        
        Returns:
            List of recommendation dictionaries with explanations
        """
        # Build prompt for generating explanations
        prompt = self._build_explanation_prompt(user_query, webtoons, metadata, filters)
        
        try:
            # Get LLM response
            llm_response = self.gemini_client.generate(prompt)
            
            # Try to parse JSON response
            explanations = self._parse_llm_explanations(llm_response, len(webtoons))
            
        except Exception as e:
            self._log(f"⚠️ LLM explanation failed: {str(e)}")
            explanations = [None] * len(webtoons)
        
        # Build structured recommendations
        recommendations = []
        for i, webtoon in enumerate(webtoons):
            rec = {
                'title': webtoon.get('title', 'Unknown'),
                'description': webtoon.get('summary', 'No description available.'),
                'genre': webtoon.get('genre', ''),
                'popularity': webtoon.get('popularity', ''),
                'author': webtoon.get('author', 'Unknown'),
                'released_date': webtoon.get('released_date', ''),
                'likes': webtoon.get('likes', 0),
                'views': webtoon.get('view', 0),
                'similarity_score': webtoon.get('similarity', 0.0),
                'explanation': explanations[i] if i < len(explanations) else None,
                # Add image URL - adjust field name based on your Supabase column
                'image_url': webtoon.get('cover_url') or webtoon.get('cover_image') or webtoon.get('thumbnail') or None
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _build_explanation_prompt(
        self,
        user_query: str,
        webtoons: List[Dict[str, Any]],
        metadata: 'ExtractedMetadata',
        filters: Dict[str, Any]
    ) -> str:
        """Build prompt for generating explanations."""
        # Format webtoons
        webtoons_context = []
        for i, w in enumerate(webtoons, 1):
            likes = w.get('likes', 0)
            views = w.get('view', 0)
            webtoons_context.append(
                f"{i}. {w['title']} - Genre: {w['genre']}, "
                f"Popularity: {w['popularity']}, Likes: {likes:,}, Views: {views:,}\n"
                f"   Summary: {w['summary']}"
            )
        
        context = "\n\n".join(webtoons_context)
        
        prompt = f"""You are a webtoon recommendation expert. Generate personalized explanations for why each webtoon matches the user's query.

USER QUERY: {user_query}

WEBTOONS TO EXPLAIN:
{context}

TASK:
For each webtoon above, write a 1-2 sentence explanation of why it matches the user's preferences.
Focus on the specific aspects they're looking for.

Return your response as a JSON array of explanations in this exact format:
[
  "Explanation for webtoon 1",
  "Explanation for webtoon 2",
  "Explanation for webtoon 3"
]

IMPORTANT: Return ONLY the JSON array, no other text."""
        
        return prompt
    
    def _parse_llm_explanations(self, llm_response: str, expected_count: int) -> List[str]:
        """
        Parse LLM response to extract explanations.
        
        Args:
            llm_response: Raw LLM response
            expected_count: Number of explanations expected
            
        Returns:
            List of explanation strings
        """
        try:
            # Try to find JSON in the response
            response = llm_response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
                response = response.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            explanations = json.loads(response)
            
            if isinstance(explanations, list):
                return explanations[:expected_count]
            else:
                return [None] * expected_count
                
        except json.JSONDecodeError:
            # Fallback: try to split by newlines
            lines = [line.strip() for line in llm_response.split('\n') if line.strip()]
            
            # Remove numbering if present
            cleaned = []
            for line in lines:
                # Remove leading numbers like "1. " or "1) "
                import re
                cleaned_line = re.sub(r'^\d+[\.)]\s*', '', line)
                if cleaned_line:
                    cleaned.append(cleaned_line)
            
            return cleaned[:expected_count] if cleaned else [None] * expected_count
    
    def _create_fallback_recommendations(
        self, 
        webtoons: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create recommendations without LLM explanations (fallback).
        
        Args:
            webtoons: List of retrieved webtoons
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        for webtoon in webtoons:
            rec = {
                'title': webtoon.get('title', 'Unknown'),
                'description': webtoon.get('summary', 'No description available.'),
                'genre': webtoon.get('genre', ''),
                'popularity': webtoon.get('popularity', ''),
                'author': webtoon.get('author', 'Unknown'),
                'released_date': webtoon.get('released_date', ''),
                'likes': webtoon.get('likes', 0),
                'views': webtoon.get('view', 0),
                'similarity_score': webtoon.get('similarity', 0.0),
                'explanation': None,
                # Add image URL - adjust field name based on your Supabase column
                'image_url': webtoon.get('image_url') or webtoon.get('cover_image') or webtoon.get('thumbnail') or None
            }
            recommendations.append(rec)
        
        return recommendations


# Singleton instance
_pipeline_instance = None

def get_pipeline(verbose: bool = True) -> EnhancedRAGPipeline:
    """
    Get or create the global pipeline instance.
    
    Args:
        verbose: If True, print detailed progress logs
        
    Returns:
        EnhancedRAGPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EnhancedRAGPipeline(verbose=verbose)
    return _pipeline_instance