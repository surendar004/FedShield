"""FedSIG+ API server (Flask) – maintains the global signature pool."""
import json
import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from marshmallow import ValidationError
import threading

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

# Lock to protect concurrent load/save of the logs file
LOG_LOCK = threading.Lock()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

# Enable CORS
CORS(app, origins=config.CORS_ORIGINS)

# File paths (Global Signature Pool storage)
DATA_FILE = os.path.join(BASE_DIR, 'global_logs.json')
MODEL_INFO_FILE = os.path.join(BASE_DIR, 'models', 'global_model_info.json')
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)


def _load_logs():
    """Load signature contributions from JSON with error handling."""
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
        # Try to recover from concatenated JSON objects or NDJSON
        logger.warning(f"Failed to parse logs file as single JSON array: {e}; attempting tolerant parsing")
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            decoder = json.JSONDecoder()
            idx = 0
            objs = []
            content_len = len(content)
            while idx < content_len:
                # Skip whitespace
                while idx < content_len and content[idx].isspace():
                    idx += 1
                if idx >= content_len:
                    break
                try:
                    obj, offset = decoder.raw_decode(content, idx)
                    idx += offset
                    # If the object is a list, extend; if dict, append
                    if isinstance(obj, list):
                        objs.extend(obj)
                    else:
                        objs.append(obj)
                except json.JSONDecodeError:
                    # Cannot decode at current position, give up and return empty
                    logger.error("Tolerant parsing failed at position %d; returning current objects", idx)
                    break
            # If we successfully recovered one or more objects, attempt to rewrite
            # the logs file as a proper JSON array to avoid repeated decode errors.
            if objs:
                try:
                    logger.info(f"Recovered %d log objects from malformed file, rewriting clean JSON array", len(objs))
                    _save_logs(objs)
                except Exception as e3:
                    logger.warning(f"Failed to persist recovered logs file: {e3}")
            return objs
        except Exception as e2:
            logger.exception(f"Failed tolerant parsing of logs file: {e2}")
            return []
    except Exception as e:
        logger.exception(f"Unexpected error loading logs: {e}")
        return []


def _save_logs(logs):
    """Persist signature contributions to JSON with error handling."""
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
    """Report a FedSIG+ signature contribution (poster Step 3 & 4)."""
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
            # Attempt best-effort normalization for common client-provided errors
            logger.warning(f"Validation error: {err.messages}")
            try:
                # Normalize threat_type if present and retry
                if isinstance(sanitized, dict) and 'threat_type' in sanitized and isinstance(sanitized.get('threat_type'), str):
                    tt = sanitized.get('threat_type')
                    tt_norm = tt.strip().lower().replace(' ', '_')
                    allowed = {'malware', 'unauthorized_access', 'data_leak', 'phishing', 'system_anomaly'}
                    if tt_norm in allowed:
                        sanitized['threat_type'] = tt_norm
                    else:
                        # If normalization doesn't match allowed set, drop the field to avoid validation failure
                        sanitized.pop('threat_type', None)
                payload = schema.load(sanitized)
            except ValidationError as err2:
                logger.warning(f"Validation failed after normalization attempt: {err2.messages}")
                return jsonify({'error': 'Validation failed', 'details': err2.messages}), 400
        
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
        # Defensive: do not trust client-provided threat_type if the event is not marked as a threat
        try:
            if not payload.get('is_threat'):
                payload.pop('threat_type', None)
        except Exception:
            # If payload shape is unexpected, continue without blocking ingestion
            logger.debug('Could not sanitize threat_type for payload')

        # Accept and persist client metrics if provided (stored under a safe key)
        try:
            cm = None
            # payload may include 'client_metrics' field coming from clients
            if isinstance(payload, dict) and 'client_metrics' in payload:
                cm = payload.pop('client_metrics')
            # normalize numeric types where possible
            if isinstance(cm, dict):
                try:
                    if 'access_count' in cm:
                        cm['access_count'] = int(cm.get('access_count') or 0)
                except Exception:
                    cm['access_count'] = 0
                try:
                    if 'cpu_percent' in cm:
                        cm['cpu_percent'] = float(cm.get('cpu_percent') or 0.0)
                except Exception:
                    cm['cpu_percent'] = 0.0
                try:
                    if 'memory_mb' in cm:
                        cm['memory_mb'] = float(cm.get('memory_mb') or 0.0)
                except Exception:
                    cm['memory_mb'] = 0.0
            if cm:
                payload['_client_metrics'] = cm
        except Exception:
            logger.debug('Failed to extract client_metrics from payload')
        
        # Save to logs (serialized to avoid concurrent write corruption)
        with LOG_LOCK:
            logs = _load_logs()
            logs.append(payload)
            if not _save_logs(logs):
                return jsonify({'error': 'Failed to save threat report'}), 500
        
        logger.info(
            "Signature ingested: client_id=%s trust_flag=%s",
            payload.get('client_id'),
            payload.get('is_threat'),
        )
        
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
    """Get signature contributions with pagination and filtering."""
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
    """Get FedSIG+ KPI summary (trusted clients, isolations, etc.)."""
    try:
        logs = _load_logs()
        # Only process dict entries
        valid_logs = [l for l in logs if isinstance(l, dict)]
        unique_clients = len(set(l.get('client_id') for l in valid_logs if l.get('client_id')))
        threats = len([l for l in valid_logs if l.get('is_threat')])
        isolations = len([l for l in valid_logs if l.get('action') == 'quarantine'])
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Build recent_logs robustly: support 'Z' suffix and skip malformed timestamps
        recent_logs = []
        for l in valid_logs:
            ra = l.get('received_at')
            if not ra:
                continue
            try:
                ts = ra
                if isinstance(ts, str) and ts.endswith('Z'):
                    ts = ts.replace('Z', '+00:00')
                dt = datetime.fromisoformat(ts)
                # If dt is offset-naive, make it UTC-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if (now - dt).total_seconds() < 86400:
                recent_logs.append(l)
        recent_threats = len([l for l in recent_logs if l.get('is_threat')])
        return jsonify({
            'clients': unique_clients,
            'threats': threats,
            'isolations': isolations,
            'recent_threats_24h': recent_threats,
            'total_events': len(valid_logs),
            'timestamp': now.isoformat() + 'Z'
        }), 200
        
    except Exception as e:
        logger.exception(f"Unexpected error in system_summary: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/global_model_info', methods=['GET'])
def global_model_info():
    """Get global model information (Flower aggregation snapshot)."""
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


@app.route('/api/metrics_summary', methods=['GET'])
def metrics_summary():
    """Return aggregated client metrics comparing threat vs non-threat events."""
    try:
        logs = _load_logs()
        groups = {
            'threat': {'count': 0, 'cpu_sum': 0.0, 'access_sum': 0, 'net_sum': 0},
            'non_threat': {'count': 0, 'cpu_sum': 0.0, 'access_sum': 0, 'net_sum': 0},
        }
        for e in logs:
            if not isinstance(e, dict):
                continue
            cm = e.get('_client_metrics') or e.get('client_metrics') or {}
            try:
                cpu = float(cm.get('cpu_percent', 0.0) or 0.0)
            except Exception:
                cpu = 0.0
            try:
                access = int(cm.get('access_count', 0) or 0)
            except Exception:
                access = 0
            try:
                net = int(cm.get('net_bytes', e.get('net_bytes', 0)) or 0)
            except Exception:
                net = 0
            key = 'threat' if e.get('is_threat') else 'non_threat'
            groups[key]['count'] += 1
            groups[key]['cpu_sum'] += cpu
            groups[key]['access_sum'] += access
            groups[key]['net_sum'] += net

        def avg(g):
            if g['count'] == 0:
                return {'count': 0, 'avg_cpu': 0.0, 'avg_access': 0.0, 'avg_net': 0}
            return {
                'count': g['count'],
                'avg_cpu': round(g['cpu_sum'] / g['count'], 2),
                'avg_access': round(g['access_sum'] / g['count'], 2),
                'avg_net': int(g['net_sum'] / g['count'])
            }

        return jsonify({'threat': avg(groups['threat']), 'non_threat': avg(groups['non_threat'])}), 200
    except Exception as e:
        logger.exception(f"Unexpected error in metrics_summary: {e}")
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
