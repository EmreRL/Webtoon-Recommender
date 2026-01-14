"""
Flask application factory for Webtoon RAG system
"""

from flask import Flask, render_template
from flask_cors import CORS
import os


def create_app():
    """
    Create and configure the Flask application
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Initialize Flask app with correct template/static paths
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
    
    # Enable CORS for local development
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
            "methods": ["GET", "POST"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Main route - serve the index page
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'webtoon-rag'}, 200
    
    return app