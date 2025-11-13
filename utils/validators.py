"""Input validation and sanitization utilities."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from config.settings import get_config
except ImportError:
    # Fallback if config not available
    def get_config():
        class Config:
            QUARANTINE_DIR = 'data/quarantined'
        return Config()

def validate_file_path(file_path: str):
    """
    Validate file path and prevent path traversal attacks.
    
    Returns:
        (is_valid, error_message)
    """
    if not file_path or not isinstance(file_path, str):
        return False, "File path must be a non-empty string"
    
    try:
        # Normalize the path
        normalized = os.path.normpath(file_path)
        
        # Check for path traversal attempts
        if '..' in normalized or normalized.startswith('/'):
            return False, "Path traversal detected"
        
        # Check if path is within allowed directories
        config = get_config()
        base_path = Path(config.QUARANTINE_DIR).parent.resolve()
        full_path = (base_path / normalized).resolve()
        
        # Ensure resolved path is within base directory
        if not str(full_path).startswith(str(base_path)):
            return False, "File path outside allowed directory"
        
        return True, ""
    except Exception as e:
        return False, f"Invalid file path: {str(e)}"


def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize input data by removing potentially dangerous fields.
    
    Args:
        data: Input dictionary to sanitize
        
    Returns:
        Sanitized dictionary with only allowed fields
    """
    allowed_fields = {
        'client_id', 'cpu_pct', 'net_bytes', 'file_access_count',
        'file_path', 'is_threat', 'action', 'timestamp',
        'quarantined_path', 'id', 'received_at'
    }
    
    sanitized = {}
    for key, value in data.items():
        if key in allowed_fields:
            # Additional sanitization for string fields
            if isinstance(value, str):
                # Remove null bytes and control characters
                value = value.replace('\x00', '').replace('\r', '').replace('\n', '')
                # Limit string length
                if len(value) > 1000:
                    value = value[:1000]
            sanitized[key] = value
    
    return sanitized


def validate_client_id(client_id: str) -> bool:
    """Validate client ID format."""
    if not client_id or not isinstance(client_id, str):
        return False
    if len(client_id) > 50:
        return False
    # Allow alphanumeric, dash, underscore
    return all(c.isalnum() or c in ['-', '_'] for c in client_id)

