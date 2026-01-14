"""
Display helper for formatting pipeline results in a user-friendly way.
Handles both success and failure cases with appropriate styling.
"""
from typing import Dict, Any


class DisplayHelper:
    """Formats pipeline results for console display."""
    
    @staticmethod
    def display_result(result: Dict[str, Any]) -> None:
        """
        Display the pipeline result in a user-friendly format.
        
        Args:
            result: Pipeline result dictionary
        """
        print("\n" + "="*60)
        
        if result['success']:
            DisplayHelper._display_success(result)
        else:
            DisplayHelper._display_failure(result)
        
        print("="*60 + "\n")
    
    @staticmethod
    def _display_success(result: Dict[str, Any]) -> None:
        """Display successful recommendation result."""
        print("✨ RECOMMENDATIONS FOR YOU:")
        print("="*60)
        print(result['response'])
        print("="*60)
        
        # Show metadata
        count = result.get('retrieved_count', 0)
        print(f"📊 Based on {count} similar webtoons from our database")
    
    @staticmethod
    def _display_failure(result: Dict[str, Any]) -> None:
        """Display failure result with appropriate styling."""
        # Check if it's a smart rejection (conversational) or system error
        is_smart_rejection = result.get('is_smart_rejection', False)
        
        if is_smart_rejection:
            # Display as a helpful message, not an error
            print("💬 SORRY, NO MATCHES FOUND")
            print("="*60)
            print(result['error'])
        else:
            # Display as a system error
            stage = result.get('stage', 'unknown')
            
            # Map stages to user-friendly names
            stage_names = {
                'validation': 'Input Validation',
                'embedding': 'Query Processing',
                'retrieval': 'Database Search',
                'generation': 'Response Generation'
            }
            
            stage_display = stage_names.get(stage, stage.title())
            
            print("❌ OOPS, SOMETHING WENT WRONG")
            print("="*60)
            print(f"Issue occurred at: {stage_display}")
            print(f"\nDetails: {result['error']}")
            print("\nPlease try:")
            print("  • Rephrasing your query")
            print("  • Being more specific about what you're looking for")
            print("  • Asking for different genres or themes")
    
    @staticmethod
    def display_welcome() -> None:
        """Display welcome message."""
        print("\n" + "🎨"*30)
        print("     WEBTOON RECOMMENDATION SYSTEM")
        print("🎨"*30)
        print("\nTell me what kind of webtoon you're looking for!")
        print("Examples:")
        print("  • 'popular action webtoon'")
        print("  • 'webtoon where the MC is crazy'")
        print("  • 'unpopular but good quality'")
        print("  • 'revenge story with complex characters'")
        print("\n" + "-"*60 + "\n")
    
    @staticmethod
    def display_query_prompt() -> str:
        """Display query input prompt."""
        return "\n💭 What kind of webtoon are you looking for?\n> "
    
    @staticmethod
    def display_thinking(stage: str) -> None:
        """Display thinking indicator for long operations."""
        indicators = {
            'extracting': '🤔 Understanding your request...',
            'searching': '🔍 Searching database...',
            'generating': '✨ Crafting recommendations...'
        }
        message = indicators.get(stage, '⏳ Processing...')
        print(f"\n{message}", end='', flush=True)
    
    @staticmethod
    def clear_thinking() -> None:
        """Clear thinking indicator."""
        print("\r" + " "*60 + "\r", end='', flush=True)


# Example usage in main.py
def example_main():
    """Example main loop with display helper."""
    from core.pipeline.rag_pipeline import get_pipeline
    
    display = DisplayHelper()
    pipeline = get_pipeline()
    
    display.display_welcome()
    
    while True:
        try:
            # Get user input
            query = input(display.display_query_prompt()).strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the recommendation system!\n")
                break
            
            # Show thinking indicator
            display.display_thinking('searching')
            
            # Run pipeline
            result = pipeline.run(query)
            
            # Clear thinking indicator
            display.clear_thinking()
            
            # Display result
            display.display_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            display.clear_thinking()
            print(f"\n❌ Unexpected error: {e}\n")


if __name__ == "__main__":
    example_main()