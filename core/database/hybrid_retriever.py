"""
Enhanced Supabase retrieval module with hybrid search.
Combines semantic search with metadata filtering.
Updated to remove quality column and use likes-based ranking.
"""
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from config import Config


class HybridRetriever:
    """Handles hybrid search: semantic similarity + metadata filtering + likes ranking."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Client = create_client(
            Config.SUPABASE_URL,
            Config.SUPABASE_SERVICE_KEY
        )
        self.table_name = Config.SUPABASE_TABLE
        print(f"✅ Connected to Supabase table: {self.table_name}")
    
    def retrieve_with_filters(
        self,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None,
        query_type: str = 'content',
        sort_by_likes: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve webtoons using hybrid search strategy.
        
        Args:
            query_embedding: Embedding vector (None for pure attribute queries)
            filters: Metadata filters (genre, popularity)
            top_k: Number of results to return
            query_type: 'attribute', 'content', or 'hybrid'
            sort_by_likes: If True, sort results by likes (for quality proxy)
            
        Returns:
            List of webtoon records with similarity scores
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        print(f"🔍 Query type: {query_type}")
        print(f"🔍 Filters: {filters}")
        print(f"🔍 Sort by likes: {sort_by_likes}")
        
        # Route to appropriate retrieval method
        if query_type == 'attribute':
            results = self._retrieve_by_attributes(filters, top_k * 2)  # Get more for sorting
        elif query_type == 'hybrid':
            results = self._retrieve_hybrid(query_embedding, filters, top_k * 3)  # More candidates
        else:  # content
            # Get more candidates for re-ranking (3x for better selection)
            results = self._retrieve_semantic(query_embedding, filters, top_k * 3)
        
        # Apply smart re-ranking for content queries
        if query_type == 'content' and results and not sort_by_likes:
            results = self._smart_rerank(results)
        
        # Apply likes-based sorting if requested (for quality queries)
        if sort_by_likes and results:
            results = self._sort_by_likes(results)
        
        return results[:top_k]
    
    def _sort_by_likes(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort results by likes (descending) to find quality content.
        
        Args:
            results: List of webtoon records
            
        Returns:
            Sorted list by likes
        """
        # Sort by likes, handling None values
        sorted_results = sorted(
            results,
            key=lambda x: x.get('likes', 0) or 0,
            reverse=True
        )
        
        print(f"📊 Sorted {len(sorted_results)} results by likes")
        if sorted_results:
            top_likes = sorted_results[0].get('likes', 0)
            print(f"   Top result has {top_likes:,} likes")
        
        return sorted_results
    
    def _smart_rerank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Smart re-ranking for content queries: boost popular items when similarity is close.
        
        Strategy:
        - Primary: Semantic similarity (most important)
        - Secondary: Popularity tier + likes (tiebreaker for similar scores)
        - Boost Hit/VeryPopular items by small amount when similarity is within 0.05
        
        Args:
            results: List of webtoon records with similarity scores
            
        Returns:
            Re-ranked list
        """
        if not results:
            return results
        
        # Define popularity weights
        popularity_weights = {
            'Hit': 0.03,           # Small boost for masterpieces
            'VeryPopular': 0.02,   # Smaller boost
            'Popular': 0.01,       # Tiny boost
            'LessPopular': 0.0,
            'Unpopular': 0.0
        }
        
        # Calculate boosted scores
        for result in results:
            similarity = result.get('similarity', 0.0)
            popularity = result.get('popularity', 'Unpopular')
            likes = result.get('likes', 0) or 0
            
            # Apply small popularity boost (max 0.03)
            popularity_boost = popularity_weights.get(popularity, 0.0)
            
            # Apply tiny likes boost (normalized, max 0.02)
            # Use log scale to prevent huge webtoons from dominating
            import math
            likes_boost = min(0.02, math.log10(likes + 1) / 100) if likes > 0 else 0
            
            # Final score = similarity + small boosts
            result['boosted_score'] = similarity + popularity_boost + likes_boost
        
        # Sort by boosted score
        sorted_results = sorted(
            results,
            key=lambda x: x.get('boosted_score', 0.0),
            reverse=True
        )
        
        print(f"🎯 Smart re-ranking applied:")
        if sorted_results:
            for i, r in enumerate(sorted_results[:3], 1):
                orig_sim = r.get('similarity', 0.0)
                boosted = r.get('boosted_score', 0.0)
                pop = r.get('popularity', '')
                print(f"   {i}. {r['title'][:30]} - Orig: {orig_sim:.3f} → Boosted: {boosted:.3f} ({pop})")
        
        return sorted_results
    
    def _retrieve_by_attributes(
        self,
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Pure attribute-based retrieval (no semantic search).
        Used for queries like "popular webtoon" or "action genre".
        """
        try:
            # Build query with filters
            query = self.client.table(self.table_name).select('*')
            
            # Apply filters
            if 'genre' in filters:
                query = query.eq('genre', filters['genre'])
            
            if 'popularity' in filters:
                # popularity is a list of acceptable values
                query = query.in_('popularity', filters['popularity'])
            
            # Execute query with ORDER BY for consistent results
            response = query.order('likes', desc=True).limit(top_k).execute()
            results = response.data if response.data else []
            
            # Add fake similarity scores for consistency
            for result in results:
                result['similarity'] = 0.95  # High score for exact matches
            
            print(f"✅ Retrieved {len(results)} webtoons (attribute-based)")
            return results
            
        except Exception as e:
            print(f"❌ Attribute retrieval failed: {e}")
            return []
    
    def _retrieve_hybrid(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: semantic search + metadata filtering.
        Used for queries like "popular webtoon with crazy MC".
        """
        try:
            # First, get candidates using semantic search
            # Use higher match_count to get more candidates before filtering
            candidates = self._retrieve_semantic(
                query_embedding,
                filters=None,  # Don't filter yet
                top_k=top_k * 3  # Get 3x more candidates
            )
            
            if not candidates:
                # Fallback to pure attribute search
                print("⚠️ No semantic matches, falling back to attributes")
                return self._retrieve_by_attributes(filters, top_k)
            
            # Filter candidates by metadata
            filtered_results = []
            for candidate in candidates:
                matches = True
                
                if 'genre' in filters:
                    if candidate.get('genre') != filters['genre']:
                        matches = False
                
                if 'popularity' in filters:
                    if candidate.get('popularity') not in filters['popularity']:
                        matches = False
                
                if matches:
                    filtered_results.append(candidate)
                
                if len(filtered_results) >= top_k:
                    break
            
            # If not enough results after filtering, add attribute-only matches
            if len(filtered_results) < top_k:
                print(f"⚠️ Only {len(filtered_results)} hybrid matches, adding attribute matches")
                attribute_results = self._retrieve_by_attributes(filters, top_k)
                
                # Merge results (avoid duplicates)
                existing_titles = {r['title'] for r in filtered_results}
                for result in attribute_results:
                    if result['title'] not in existing_titles:
                        result['similarity'] = 0.7  # Lower score for attribute-only
                        filtered_results.append(result)
                    if len(filtered_results) >= top_k:
                        break
            
            print(f"✅ Retrieved {len(filtered_results)} webtoons (hybrid)")
            return filtered_results[:top_k]
            
        except Exception as e:
            print(f"❌ Hybrid retrieval failed: {e}")
            # Fallback to attribute-based
            return self._retrieve_by_attributes(filters, top_k)
    
    def _retrieve_semantic(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Pure semantic search (original functionality).
        Used for content-based queries.
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        try:
            # Try RPC function first
            response = self.client.rpc(
                'match_webtoons',
                {
                    'query_embedding': query_embedding,
                    'match_count': top_k,
                    'match_threshold': Config.SIMILARITY_THRESHOLD
                }
            ).execute()
            
            results = response.data if response.data else []
            print(f"✅ Retrieved {len(results)} webtoons (semantic)")
            return results
            
        except Exception as e:
            print(f"⚠️ RPC function not found, using fallback: {e}")
            return self._retrieve_manual(query_embedding, top_k)
    
    def _retrieve_manual(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback manual retrieval (same as original)."""
        try:
            response = self.client.table(self.table_name)\
                .select('*')\
                .not_.is_('embedding', 'null')\
                .execute()
            
            records = response.data
            
            import numpy as np
            query_vec = np.array(query_embedding)
            
            for record in records:
                if record.get('embedding'):
                    doc_vec = np.array(record['embedding'])
                    similarity = float(np.dot(query_vec, doc_vec))
                    record['similarity'] = similarity
                else:
                    record['similarity'] = 0.0
            
            records.sort(key=lambda x: x['similarity'], reverse=True)
            results = records[:top_k]
            
            print(f"✅ Retrieved {len(results)} webtoons (manual fallback)")
            return results
            
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []


# Singleton instance
_retriever_instance = None

def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance