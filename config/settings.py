"""Configuration settings for FedShield."""
import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Config:
    """Application configuration."""
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '5000'))
    API_BASE_URL: str = os.getenv('API_BASE_URL', 'http://localhost:5000/api')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'change-me-in-production-use-strong-random-key')
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///threats.db')
    USE_DATABASE: bool = os.getenv('USE_DATABASE', 'False').lower() == 'true'
    
    # Data Management
    MAX_LOG_SIZE: int = int(os.getenv('MAX_LOG_SIZE', '10000'))
    DATA_RETENTION_DAYS: int = int(os.getenv('DATA_RETENTION_DAYS', '30'))
    
    # File Operations
    QUARANTINE_DIR: str = os.getenv('QUARANTINE_DIR', 'data/quarantined')
    ALLOWED_FILE_PATHS: List[str] = field(default_factory=list)
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/app.log')
    
    # CORS
    CORS_ORIGINS: List[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', '*').split(','))
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if not self.ALLOWED_FILE_PATHS:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.ALLOWED_FILE_PATHS = [
                os.path.join(base_dir, 'data'),
            ]


_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

