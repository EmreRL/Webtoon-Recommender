"""
Main entry point for the Webtoon RAG Recommendation System.
Run this file to start the interactive recommendation system.
"""
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline.rag_pipeline import get_pipeline
from config import Config


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("🎨 WEBTOON RAG RECOMMENDATION SYSTEM")
    print("="*60)
    print("Powered by: MiniLM-L6-v2 + Supabase + Gemini 2.0 Flash")
    print("="*60 + "\n")


def print_help():
    """Print usage instructions."""
    print("\n📖 How to use:")
    print("  - Describe what kind of webtoon you're looking for")
    print("  - Mention genres, themes, or story elements you enjoy")
    print("  - Ask for recommendations similar to a specific style")
    print("\nExamples:")
    print('  • "I want an action webtoon with a strong female lead"')
    print('  • "Recommend something like a revenge story"')
    print('  • "Fantasy webtoon with good plot twists"')
    print('  • "School life romance with comedy"')
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for this message\n")


def format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """
    Format structured recommendations into a beautiful display.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Formatted string for display
    """
    if not recommendations:
        return "No recommendations found."
    
    output = []
    
    for i, rec in enumerate(recommendations, 1):
        # Header with title and popularity
        popularity_emoji = {
            'Hit': '🔥',
            'VeryPopular': '⭐',
            'Popular': '👍',
            'LessPopular': '📚',
            'Unpopular': '💎'
        }
        emoji = popularity_emoji.get(rec.get('popularity', ''), '📖')
        
        output.append(f"\n{i}. {emoji} {rec['title']}")
        output.append("   " + "-" * 55)
        
        # Author and genre
        author = rec.get('author', 'Unknown')
        genre = rec.get('genre', 'Unknown')
        output.append(f"   By {author} | Genre: {genre}")
        
        # Stats line
        popularity = rec.get('popularity', 'Unknown')
        likes = rec.get('likes', 0)
        views = rec.get('views', 0)
        
        # Format numbers nicely
        likes_str = f"{likes:,}" if likes else "N/A"
        views_str = f"{views:,}" if views else "N/A"
        
        output.append(f"   Popularity: {popularity} | Likes: {likes_str} | Views: {views_str}")
        
        # Similarity score
        sim_score = rec.get('similarity_score', 0.0)
        output.append(f"   Match Score: {sim_score:.1%}")
        
        # Description
        description = rec.get('description', 'No description available.')
        # Wrap description at ~70 chars
        desc_lines = _wrap_text(description, 70)
        output.append(f"\n   📝 {desc_lines[0]}")
        for line in desc_lines[1:]:
            output.append(f"      {line}")
        
        # Explanation (if available)
        explanation = rec.get('explanation')
        if explanation:
            output.append(f"\n   💡 Why this matches: {explanation}")
        
        output.append("")  # Blank line between recommendations
    
    return "\n".join(output)


def _wrap_text(text: str, width: int) -> List[str]:
    """
    Wrap text to specified width, breaking at word boundaries.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        
    Returns:
        List of wrapped lines
    """
    if len(text) <= width:
        return [text]
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length <= width:
            current_line.append(word)
            current_length += word_length
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word) + 1
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines


def display_result(result: Dict[str, Any]) -> None:
    """
    Display pipeline result in a user-friendly format.
    
    Args:
        result: Pipeline result dictionary
    """
    print("\n" + "="*60)
    
    if result['success']:
        # Success: Show recommendations
        recommendations = result['response']
        
        print("✨ RECOMMENDATIONS FOR YOU")
        print("="*60)
        
        # Format and display recommendations
        formatted = format_recommendations(recommendations)
        print(formatted)
        
        print("="*60)
        print(f"📊 Found {result['retrieved_count']} matching webtoons")
        print(f"🎯 Query type: {result.get('query_type', 'unknown')}")
        
        # Show applied filters if any
        filters = result.get('filters', {})
        if filters:
            filter_parts = []
            if filters.get('genre'):
                filter_parts.append(f"Genre: {filters['genre']}")
            if filters.get('popularity'):
                filter_parts.append(f"Popularity: {', '.join(filters['popularity'])}")
            if filter_parts:
                print(f"🔍 Filters applied: {' | '.join(filter_parts)}")
    
    elif result.get('is_smart_rejection'):
        # Smart rejection: Show as helpful message (NOT an error)
        print("💬 LET ME HELP YOU FIND SOMETHING")
        print("="*60)
        print(result['error'])  # This is actually a helpful message
        print("\n" + "="*60)
        print("💡 Tip: Try asking for genres we have, or describe plot themes!")
    
    else:
        # Actual system error
        print("❌ OOPS, SOMETHING WENT WRONG")
        print("="*60)
        
        stage_names = {
            'validation': 'Input Validation',
            'embedding': 'Query Processing',
            'retrieval': 'Database Search',
            'generation': 'Response Generation'
        }
        
        stage = result.get('stage', 'unknown')
        stage_display = stage_names.get(stage, stage.title())
        
        print(f"Issue at: {stage_display}")
        print(f"\n{result['error']}")
        print("\nPlease try:")
        print("  • Rephrasing your query")
        print("  • Being more specific")
        print("  • Using simpler language")
    
    print("="*60)


def run_interactive():
    """Run the interactive CLI interface."""
    print_banner()
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease create a .env file with:")
        print("  SUPABASE_URL=your_url")
        print("  SUPABASE_SERVICE_KEY=your_key")
        print("  GEMINI_API_KEY=your_key")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        pipeline = get_pipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    print_help()
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_query = input("\n💭 What kind of webtoon are you looking for?\n> ").strip()
            
            # Handle commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the Webtoon Recommender! Goodbye!")
                break
            
            if user_query.lower() in ['help', 'h', '?']:
                print_help()
                continue
            
            if not user_query:
                continue
            
            # Run pipeline
            result = pipeline.run(user_query)
            
            # Display results using smart handler
            display_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or type 'quit' to exit.")


def run_single_query(query: str):
    """
    Run a single query (useful for testing or API integration).
    
    Args:
        query: The user's recommendation request
        
    Returns:
        Result dictionary from the pipeline
    """
    try:
        Config.validate()
        pipeline = get_pipeline()
        return pipeline.run(query)
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'stage': 'initialization'
        }


if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Run with command-line query
        query = " ".join(sys.argv[1:])
        result = run_single_query(query)
        
        if result['success']:
            # Format recommendations nicely for command-line too
            formatted = format_recommendations(result['response'])
            print("\n" + "="*60)
            print("✨ RECOMMENDATIONS")
            print("="*60)
            print(formatted)
            print("="*60)
        else:
            print(f"Error ({result['stage']}): {result['error']}")
            sys.exit(1)
    else:
        # Run interactive mode
        run_interactive()