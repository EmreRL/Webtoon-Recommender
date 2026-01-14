from flask import Blueprint, request, jsonify
import sys
import os

# Add parent directory to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.pipeline.rag_pipeline import get_pipeline

api_bp = Blueprint('api', __name__)

# Use the pipeline's built-in singleton
def get_pipeline_instance():
    return get_pipeline(verbose=False)

@api_bp.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'No query provided'}), 400

        query = data['query'].strip()
        if not query:
            return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400

        pipeline = get_pipeline_instance()
        result = pipeline.run(query)

        if result.get('is_smart_rejection'):
            return jsonify({
                'success': False,
                'message': result['error'],
                'query_type': result['query_type'],
                'filters_applied': result['filters'],
                'metadata': {},
                'database_stats': result.get('database_stats', {})
            }), 200

        return jsonify({
            'success': True,
            'query': result['query'],
            'recommendations': result['response'],
            'retrieved_webtoons': result['retrieved_webtoons'],
            'metadata': {
                'query_type': result['query_type'],
                'filters_applied': result['filters'],
                'total_found': result['retrieved_count'],
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@api_bp.route('/stats', methods=['GET'])
def get_database_stats():
    try:
        pipeline = get_pipeline_instance()

        if hasattr(pipeline, 'stats_collector'):
            stats = pipeline.stats_collector.get_stats()
        else:
            stats = {}

        return jsonify({'success': True, 'stats': stats}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
