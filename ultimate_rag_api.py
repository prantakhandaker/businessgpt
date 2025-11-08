from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import time
import numpy as np
from datetime import datetime
from decimal import Decimal
from ultimate_rag_pipeline import UltimateRAGPipeline

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types, dataclasses, BigQuery Row objects, date objects, and Decimal types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool) or isinstance(obj, np.bool_):
            # Handle boolean values (including numpy booleans)
            return bool(obj)
        elif hasattr(obj, '__dataclass_fields__'):
            # Handle dataclass objects
            return obj.__dict__
        elif hasattr(obj, 'keys') and hasattr(obj, 'values'):
            # Handle BigQuery Row objects and similar dict-like objects
            try:
                return dict(obj)
            except:
                return str(obj)
        elif hasattr(obj, 'isoformat'):
            # Handle date and datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'strftime'):
            # Handle other date-like objects
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, Decimal) or hasattr(obj, 'as_tuple'):
            # Handle Decimal objects
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the Ultimate RAG Pipeline
pipeline = UltimateRAGPipeline()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_charts_data(data):
    """Helper function to save charts data"""
    try:
        context = pipeline.conversation_memory.get_context(data['user_id'], data['session_id'])
        if not hasattr(context, 'saved_charts'):
            context.saved_charts = []
        
        chart_data = {
            'timestamp': datetime.now().isoformat(),
            'query': data['query'],
            'charts': data['charts'],
            'session_id': data['session_id']
        }
        
        context.saved_charts.append(chart_data)
        
        # Keep only last 50 chart sets per user
        if len(context.saved_charts) > 50:
            context.saved_charts = context.saved_charts[-50:]
        
        pipeline.conversation_memory.save_conversations()
        logger.info(f"Auto-saved {len(data['charts'])} charts for user {data['user_id']}")
    except Exception as e:
        logger.error(f"Error in save_charts_data: {e}")

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint for asking questions"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        user_id = data.get('user_id', 'default')
        session_id = data.get('session_id', 'default')
        
        if not question:
            return jsonify({
                'error': 'No question provided',
                'answer': 'Please provide a question to analyze.',
                'results': [],
                'charts': [],
                'insights': []
            })
        
        logger.info(f"Processing question: {question}")
        
        # Process the question through the Ultimate RAG Pipeline
        response = pipeline.ask_intelligent(question, user_id, session_id)
        
        # Add unique IDs to charts for selection
        charts = response.get('charts', [])
        for i, chart in enumerate(charts):
            chart['id'] = f"{user_id}_{session_id}_{int(time.time())}_{i}"
        
        # Format response for frontend
        formatted_response = {
            'answer': response.get('summary', 'No summary available'),
            'results': response.get('results', []),
            'generated_sql': response.get('sql'),
            'sql_display': response.get('sql'),  # Add this for frontend display
            'charts': charts,
            'insights': response.get('insights', []),
            'metadata': response.get('metadata', {}),
            'error': response.get('error')
        }
        # Surface dry-run stats and simple citations when available
        meta = formatted_response.get('metadata') or {}
        dry = meta.get('dry_run') if isinstance(meta, dict) else None
        formatted_response['dry_run'] = dry
        # Simple citation: list tables referenced in SQL (best-effort)
        sql_text = formatted_response.get('generated_sql') or ''
        try:
            import re
            tables = list(set(re.findall(r'`([^`]+\.[^`]+\.[^`]+)`', sql_text)))
        except Exception:
            tables = []
        formatted_response['citations'] = {'tables': tables}
        
        # Auto-save charts if any were generated
        if charts:
            try:
                save_data = {
                    'user_id': user_id,
                    'session_id': session_id,
                    'charts': charts,
                    'query': question
                }
                # Save charts asynchronously (don't wait for response)
                import threading
                threading.Thread(target=lambda: save_charts_data(save_data)).start()
            except Exception as e:
                logger.warning(f"Failed to auto-save charts: {e}")
        
        logger.info(f"Response generated successfully")
        # Convert numpy types before jsonify
        formatted_response = json.loads(json.dumps(formatted_response, cls=NumpyEncoder))
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'answer': 'Sorry, I encountered an error processing your question.',
            'results': [],
            'charts': [],
            'insights': []
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pipeline': 'Ultimate RAG Pipeline',
        'features': [
            'Multi-stage retrieval',
            'Query decomposition',
            'Knowledge graph',
            'Enhanced SQL generation',
            'Conversation memory',
            'Advanced analytics'
        ]
    })

@app.route('/conversation/<user_id>/<session_id>', methods=['GET'])
def get_conversation_history(user_id, session_id):
    """Get conversation history for a user session"""
    try:
        context = pipeline.conversation_memory.get_context(user_id, session_id)
        return jsonify({
            'user_id': user_id,
            'session_id': session_id,
            'history': context.history,
            'preferences': context.preferences,
            'last_updated': context.last_updated.isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/insights', methods=['POST'])
def generate_insights():
    """Generate insights from provided data"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        query = data.get('query', '')
        
        insights = pipeline.analytics.generate_insights(results, query)
        
        return jsonify({
            'insights': insights,
            'count': len(insights)
        })
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/decompose', methods=['POST'])
def decompose_query():
    """Decompose complex queries into sub-queries"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        sub_queries = pipeline.query_understanding.decompose_complex_query(query)
        
        return jsonify({
            'original_query': query,
            'sub_queries': sub_queries,
            'count': len(sub_queries)
        })
        
    except Exception as e:
        logger.error(f"Error decomposing query: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/intent', methods=['POST'])
def classify_intent():
    """Classify query intent"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        intent = pipeline.query_understanding.classify_intent(query)
        
        return jsonify({
            'query': query,
            'intent_type': intent.intent_type,
            'confidence': intent.confidence,
            'entities': intent.entities,
            'temporal_context': intent.temporal_context
        })
        
    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/charts/save', methods=['POST'])
def save_charts():
    """Save charts from a chat session for dashboard creation"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        session_id = data.get('session_id', 'default')
        charts = data.get('charts', [])
        query = data.get('query', '')
        
        # Store charts in conversation memory with metadata
        chart_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'charts': charts,
            'session_id': session_id
        }
        
        # Get or create user's chart collection
        context = pipeline.conversation_memory.get_context(user_id, session_id)
        if not hasattr(context, 'saved_charts'):
            context.saved_charts = []
        
        context.saved_charts.append(chart_data)
        
        # Keep only last 50 chart sets per user
        if len(context.saved_charts) > 50:
            context.saved_charts = context.saved_charts[-50:]
        
        pipeline.conversation_memory.save_conversations()
        
        return jsonify({
            'success': True,
            'message': f'Saved {len(charts)} charts',
            'total_charts': len(context.saved_charts)
        })
        
    except Exception as e:
        logger.error(f"Error saving charts: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/charts/list/<user_id>/<session_id>', methods=['GET'])
def list_saved_charts(user_id, session_id):
    """Get list of saved charts for a user session"""
    try:
        context = pipeline.conversation_memory.get_context(user_id, session_id)
        saved_charts = getattr(context, 'saved_charts', [])
        
        # Format for frontend
        chart_list = []
        for chart_set in saved_charts:
            chart_list.append({
                'timestamp': chart_set['timestamp'],
                'query': chart_set['query'],
                'chart_count': len(chart_set['charts']),
                'charts': chart_set['charts']
            })
        
        return jsonify({
            'success': True,
            'charts': chart_list
        })
        
    except Exception as e:
        logger.error(f"Error listing charts: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/dashboard/create', methods=['POST'])
def create_dashboard():
    """Create a custom dashboard from selected charts"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        session_id = data.get('session_id', 'default')
        selected_charts = data.get('selected_charts', [])
        dashboard_name = data.get('dashboard_name', 'Custom Dashboard')
        
        # Validate selected charts
        if not isinstance(selected_charts, list) or not selected_charts:
            return jsonify({'error': 'No charts selected'}), 400
        
        # Get chart data from saved charts
        context = pipeline.conversation_memory.get_context(user_id, session_id)
        saved_charts = getattr(context, 'saved_charts', [])
        
        dashboard_charts = []
        for chart_ref in selected_charts:
            # Find the chart in saved charts
            for chart_set in saved_charts:
                for chart in chart_set['charts']:
                    if chart.get('id') == chart_ref['chart_id']:
                        dashboard_charts.append({
                            **chart,
                            'source_query': chart_set['query'],
                            'source_timestamp': chart_set['timestamp']
                        })
                        break
        
        # If nothing matched, return helpful error
        if not dashboard_charts:
            return jsonify({'error': 'No matching charts found for the current session. Generate charts first and try again.'}), 400

        # Save dashboard configuration
        dashboard_config = {
            'name': dashboard_name,
            'created_at': datetime.now().isoformat(),
            'charts': dashboard_charts,
            'user_id': user_id,
            'session_id': session_id
        }
        
        # Store dashboard (you might want to persist this differently)
        if not hasattr(context, 'dashboards'):
            context.dashboards = []
        
        context.dashboards.append(dashboard_config)
        pipeline.conversation_memory.save_conversations()
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_config,
            'message': f'Created dashboard with {len(dashboard_charts)} charts'
        })
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/dashboards/list/<user_id>/<session_id>', methods=['GET'])
def list_dashboards(user_id, session_id):
    """List saved dashboards for a user/session."""
    try:
        context = pipeline.conversation_memory.get_context(user_id, session_id)
        dashboards = getattr(context, 'dashboards', [])
        return jsonify({
            'success': True,
            'dashboards': dashboards,
            'count': len(dashboards)
        })
    except Exception as e:
        logger.error(f"Error listing dashboards: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for learning"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        response = data.get('response', {})
        rating = data.get('rating', 0)
        feedback_text = data.get('feedback_text', '')
        user_id = data.get('user_id', 'default')
        
        if not query or not response:
            return jsonify({"error": "Query and response are required"}), 400
        
        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating must be between 1 and 5"}), 400
        
        # Create feedback object
        from ultimate_rag_pipeline import UserFeedback
        feedback = UserFeedback(
            query=query,
            response=response,
            rating=rating,
            feedback_text=feedback_text
        )
        
        # Process feedback
        pipeline.feedback_learning.process_feedback(query, response, feedback)
        
        return jsonify({
            "status": "success",
            "message": "Feedback submitted successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in submit_feedback: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics/performance', methods=['GET'])
def get_performance_analytics():
    """Get system performance analytics"""
    try:
        user_id = request.args.get('user_id', 'default')
        
        # Get retrieval performance stats
        retrieval_stats = pipeline.enhanced_retriever.get_performance_stats()
        
        # Get conversation analytics
        context = pipeline.conversation_memory.get_context(user_id, 'default')
        conversation_stats = {
            'total_interactions': len(context.get('interactions', [])),
            'recent_queries': [i.get('query', '') for i in context.get('interactions', [])[-5:]],
            'avg_response_time': sum(i.get('processing_time', 0) for i in context.get('interactions', [])) / max(len(context.get('interactions', [])), 1)
        }
        
        return jsonify({
            "retrieval_performance": retrieval_stats,
            "conversation_analytics": conversation_stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_performance_analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics/insights', methods=['POST'])
def get_advanced_insights():
    """Get advanced analytics insights"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id', 'default')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Get enhanced response with analytics
        response = pipeline.ask_intelligent(query, user_id, 'default')
        
        # Extract analytics results
        analysis_results = response.get('metadata', {}).get('enhanced_features', {}).get('analysis_results')
        
        if analysis_results:
            return jsonify({
                "query": query,
                "insights": analysis_results.insights,
                "descriptive_stats": analysis_results.descriptive_stats,
                "trend_analysis": analysis_results.trend_analysis,
                "anomalies": analysis_results.anomalies,
                "correlations": analysis_results.correlations,
                "significance_tests": analysis_results.significance_tests,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "query": query,
                "message": "No advanced analytics available for this query",
                "timestamp": datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error in get_advanced_insights: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/knowledge-graph/entities', methods=['GET'])
def get_knowledge_graph_entities():
    """Get knowledge graph entities"""
    try:
        query = request.args.get('query', '')
        top_k = int(request.args.get('top_k', 10))
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        # Get contextual entities from dynamic knowledge graph
        entities = pipeline.dynamic_knowledge_graph.get_contextual_entities(query, top_k)
        
        return jsonify({
            "query": query,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_knowledge_graph_entities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sql/validate', methods=['POST'])
def validate_sql():
    """Validate and optimize SQL query"""
    try:
        data = request.get_json()
        sql = data.get('sql', '')
        user_id = data.get('user_id', 'default')
        
        if not sql:
            return jsonify({"error": "SQL query is required"}), 400
        
        # Validate SQL
        from ultimate_rag_pipeline import QueryContext
        validation_result = pipeline.advanced_sql_validator.validate_and_optimize(
            sql, QueryContext(sql, user_id, 'default')
        )
        
        return jsonify({
            "sql": sql,
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "performance_score": validation_result.performance_score,
                "optimization_suggestions": validation_result.optimization_suggestions,
                "optimized_sql": validation_result.optimized_sql
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in validate_sql: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced Ultimate RAG API Server...")
    logger.info("Available endpoints:")
    logger.info("  POST /ask - Main question processing")
    logger.info("  GET /health - Health check")
    logger.info("  GET /conversation/<user_id>/<session_id> - Get conversation history")
    logger.info("  POST /insights - Generate insights from data")
    logger.info("  POST /decompose - Decompose complex queries")
    logger.info("  POST /intent - Classify query intent")
    logger.info("  POST /charts/save - Save charts for dashboard creation")
    logger.info("  GET /charts/list/<user_id>/<session_id> - List saved charts")
    logger.info("  POST /dashboard/create - Create custom dashboard")
    logger.info("  POST /feedback - Submit user feedback for learning")
    logger.info("  GET /analytics/performance - Get system performance analytics")
    logger.info("  POST /analytics/insights - Get advanced analytics insights")
    logger.info("  GET /knowledge-graph/entities - Get knowledge graph entities")
    logger.info("  POST /sql/validate - Validate and optimize SQL queries")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
