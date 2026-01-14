"""
Database statistics collector for understanding what's available.
Helps provide better rejection messages and suggestions.
"""
from typing import Dict, Any, List, Set
from supabase import Client
from config import Config
from ..database.hybrid_retriever import get_hybrid_retriever


class DatabaseStatsCollector:
    """Collects statistics about available webtoons in the database."""
    
    def __init__(self):
        """Initialize with retriever for database access."""
        self.retriever = get_hybrid_retriever()
        self._cache = None  # Cache stats to avoid repeated queries
    
    def get_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get database statistics about available attributes.
        
        Args:
            force_refresh: Force refresh the cache
            
        Returns:
            Dictionary with available genres, popularity levels, quality levels
        """
        # Return cached stats if available
        if self._cache and not force_refresh:
            return self._cache
        
        try:
            # Query all records to collect stats
            response = self.retriever.client.table(self.retriever.table_name)\
                .select('genre, popularity, quality')\
                .execute()
            
            records = response.data
            
            # Collect unique values
            genres: Set[str] = set()
            popularity: Set[str] = set()
            quality: Set[str] = set()
            
            for record in records:
                if record.get('genre'):
                    genres.add(record['genre'])
                if record.get('popularity'):
                    popularity.add(record['popularity'])
                if record.get('quality'):
                    quality.add(record['quality'])
            
            # Build stats dictionary
            stats = {
                'available_genres': sorted(list(genres)),
                'available_popularity': sorted(list(popularity)),
                'available_quality': sorted(list(quality)),
                'total_webtoons': len(records)
            }
            
            # Cache the results
            self._cache = stats
            
            print(f"📊 Database stats collected:")
            print(f"   Genres: {stats['available_genres']}")
            print(f"   Popularity: {stats['available_popularity']}")
            print(f"   Quality: {stats['available_quality']}")
            print(f"   Total webtoons: {stats['total_webtoons']}")
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Failed to collect database stats: {e}")
            # Return empty stats
            return {
                'available_genres': [],
                'available_popularity': [],
                'available_quality': [],
                'total_webtoons': 0
            }
    
    def get_suggestions(
        self, 
        failed_filters: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Get suggestions for alternative filters based on what's available.
        
        Args:
            failed_filters: Filters that returned no results
            
        Returns:
            Dictionary with suggested alternatives
        """
        stats = self.get_stats()
        suggestions = {}
        
        # Suggest alternative genres
        if 'genre' in failed_filters:
            requested_genre = failed_filters['genre']
            available = stats['available_genres']
            if available:
                suggestions['alternative_genres'] = available
        
        # Suggest alternative popularity
        if 'popularity' in failed_filters:
            requested = failed_filters['popularity']
            available = stats['available_popularity']
            if available:
                suggestions['alternative_popularity'] = available
        
        # Suggest alternative quality
        if 'quality' in failed_filters:
            requested = failed_filters['quality']
            available = stats['available_quality']
            if available:
                suggestions['alternative_quality'] = available
        
        return suggestions
    
    def check_filter_exists(self, filters: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check which filters have matching values in the database.
        
        Args:
            filters: Filters to check
            
        Returns:
            Dictionary with existence status for each filter
        """
        stats = self.get_stats()
        exists = {}
        
        if 'genre' in filters:
            genre = filters['genre']
            exists['genre'] = genre in stats['available_genres']
        
        if 'popularity' in filters:
            pop_levels = filters['popularity']
            exists['popularity'] = any(
                p in stats['available_popularity'] for p in pop_levels
            )
        
        if 'quality' in filters:
            quality_levels = filters['quality']
            exists['quality'] = any(
                q in stats['available_quality'] for q in quality_levels
            )
        
        return exists


# Singleton instance
_stats_collector = None

def get_stats_collector() -> DatabaseStatsCollector:
    """Get or create the global stats collector."""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = DatabaseStatsCollector()
    return _stats_collector