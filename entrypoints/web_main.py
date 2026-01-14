"""
Web Application Launcher for Webtoon RAG Recommendation System

Run this file from the app/ directory:
    python web_main.py

Or from project root:
    python -m app.web_main
"""

import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web import create_app

# ASCII Art Banner
BANNER = """
============================================================
🎨 WEBTOON RAG RECOMMENDATION SYSTEM - WEB INTERFACE
============================================================
Powered by: MiniLM-L6-v2 + Supabase + Gemini 2.0 Flash
============================================================
"""


def main():
    """Main entry point for the web application"""
    
    print(BANNER)
    print("🚀 Starting Webtoon RAG Web Server...")
    print("=" * 60)
    
    # Create Flask app
    app = create_app()
    
    # Configuration
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\n✅ Server Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"\n🌐 Access the application at:")
    print(f"   👉 http://{host}:{port}")
    print(f"\n💡 Tips:")
    print(f"   - Press Ctrl+C to stop the server")
    print(f"   - Refresh the page to see updates")
    print(f"   - Check the terminal for request logs")
    print("=" * 60)
    print()
    
    try:
        # Run the Flask development server
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()