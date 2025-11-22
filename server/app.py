"""Flask application for FedShield API server."""
import json
import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from marshmallow import ValidationError

# Add project root to path
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from config.settings import get_config
    from api.schemas import ThreatReportSchema, ThreatQuerySchema
    from utils.helpers import setup_logging, get_logger
    from utils.validators import validate_file_path, sanitize_input
except ImportError as e:
    # Fallback for development - create minimal implementations
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import new modules: {e}. Using fallback implementations.")
    
    # Minimal fallback config
    class Config:
        SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-in-production')
        API_HOST = os.getenv('API_HOST', '0.0.0.0')
        API_PORT = int(os.getenv('API_PORT', '5000'))
        DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        MAX_LOG_SIZE = int(os.getenv('MAX_LOG_SIZE', '10000'))
        CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    def get_config():
        return Config()
    
    # Fallback validators
    def validate_file_path(file_path: str):
        return True, ""
    
    def sanitize_input(data: dict):
        return data
    
    def setup_logging():
        pass
    
    def get_logger(name: str):
        return logging.getLogger(name)
    
    # Fallback schemas
    class ThreatReportSchema:
        def load(self, data):
            return data
    
    class ThreatQuerySchema:
        def load(self, data):
            return data

# Initialize configuration
config = get_config()

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Enable CORS
CORS(app, origins=config.CORS_ORIGINS)

# File paths
DATA_FILE = os.path.join(BASE_DIR, 'global_logs.json')
MODEL_INFO_FILE = os.path.join(BASE_DIR, 'models', 'global_model_info.json')
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)


def _load_logs():
    """Load threat logs from JSON file with error handling."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Logs file contains non-list data, resetting to empty list")
                return []
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse logs file: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error loading logs: {e}")
        return []


def _save_logs(logs):
    """Save threat logs to JSON file with error handling."""
    try:
        # Limit log size to prevent memory issues
        if len(logs) > config.MAX_LOG_SIZE:
            logger.warning(f"Log size ({len(logs)}) exceeds limit ({config.MAX_LOG_SIZE}), truncating")
            logs = logs[-config.MAX_LOG_SIZE:]
        
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.exception(f"Failed to save logs: {e}")
        return False


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    return jsonify({'error': 'Bad Request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    return jsonify({'error': 'Not Found', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error."""
    logger.exception("Internal server error")
    return jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred'}), 500


@app.route('/api/report_threat', methods=['POST'])
def report_threat():
    """Report a threat detection event."""
    try:
        # Get and validate JSON payload
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        raw_payload = request.get_json(force=True)
        if not raw_payload:
            return jsonify({'error': 'Empty request body'}), 400
        
        # Sanitize input
        sanitized = sanitize_input(raw_payload)
        
        # Validate using schema
        try:
            schema = ThreatReportSchema()
            payload = schema.load(sanitized)
        except ValidationError as err:
            logger.warning(f"Validation error: {err.messages}")
            return jsonify({'error': 'Validation failed', 'details': err.messages}), 400
        
        # Validate file path if provided
        if 'file_path' in payload:
            is_valid, error_msg = validate_file_path(payload['file_path'])
            if not is_valid:
                logger.warning(f"Invalid file path: {error_msg}")
                return jsonify({'error': 'Invalid file path', 'details': error_msg}), 400
        
        # Add metadata
        payload['received_at'] = datetime.utcnow().isoformat() + 'Z'
        if 'id' not in payload:
            import uuid
            payload['id'] = str(uuid.uuid4())
        
        # Save to logs
        logs = _load_logs()
        logs.append(payload)
        
        if not _save_logs(logs):
            return jsonify({'error': 'Failed to save threat report'}), 500
        
        logger.info(f"Threat reported: client_id={payload.get('client_id')}, is_threat={payload.get('is_threat')}")
        
        return jsonify({
            'status': 'ok',
            'saved': True,
            'id': payload.get('id')
        }), 201
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        logger.exception(f"Unexpected error in report_threat: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/threats', methods=['GET'])
def get_threats():
    """Get threat logs with pagination and filtering."""
    try:
        # Parse query parameters
        try:
            query_schema = ThreatQuerySchema()
            query_params = query_schema.load(request.args.to_dict())
        except ValidationError as err:
            return jsonify({'error': 'Invalid query parameters', 'details': err.messages}), 400
        
        # Load all logs
        logs = _load_logs()
        
        # Apply filters
        if query_params.get('client_id'):
            logs = [l for l in logs if l.get('client_id') == query_params['client_id']]
        
        if query_params.get('is_threat') is not None:
            logs = [l for l in logs if l.get('is_threat') == query_params['is_threat']]
        
        # Calculate pagination
        page = query_params.get('page', 1)
        per_page = query_params.get('per_page', 50)
        total = len(logs)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_logs = logs[start:end]
        
        return jsonify({
            'data': paginated_logs,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        }), 200
        
    except Exception as e:
        logger.exception(f"Unexpected error in get_threats: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/system_summary', methods=['GET'])
def system_summary():
    """Get system summary statistics."""
    try:
        logs = _load_logs()
        
        # Calculate statistics
        unique_clients = len(set(l.get('client_id') for l in logs if l.get('client_id')))
        threats = len([l for l in logs if l.get('is_threat')])
        isolations = len([l for l in logs if l.get('action') == 'quarantine'])
        
        # Calculate recent activity (last 24 hours)
        now = datetime.utcnow()
        recent_logs = [
            l for l in logs
            if l.get('received_at') and
            (now - datetime.fromisoformat(l['received_at'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 86400
        ]
        recent_threats = len([l for l in recent_logs if l.get('is_threat')])
        
        return jsonify({
            'clients': unique_clients,
            'threats': threats,
            'isolations': isolations,
            'recent_threats_24h': recent_threats,
            'total_events': len(logs),
            'timestamp': now.isoformat() + 'Z'
        }), 200
        
    except Exception as e:
        logger.exception(f"Unexpected error in system_summary: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/global_model_info', methods=['GET'])
def global_model_info():
    """Get global model information."""
    try:
        if not os.path.exists(MODEL_INFO_FILE):
            return jsonify({'status': 'no-model', 'message': 'No global model available'}), 404
        
        with open(MODEL_INFO_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return jsonify(data), 200
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse model info file: {e}")
        return jsonify({'error': 'Invalid model info file'}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in global_model_info: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'version': '1.0.0'
        }), 200
    except Exception as e:
        logger.exception(f"Unexpected error in health check: {e}")
        return jsonify({'status': 'error'}), 500


if __name__ == '__main__':
    logger.info(f"Starting FedShield API server on {config.API_HOST}:{config.API_PORT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )
