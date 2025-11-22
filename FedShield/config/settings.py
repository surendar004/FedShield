"""Configuration settings for the FedSIG+ (FedShield) prototype."""
import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Config:
    """Application configuration aligned with FedSIG+ methodology."""
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Ensure file/directory paths are absolute relative to project base_dir
        if not self.ALLOWED_FILE_PATHS:
            self.ALLOWED_FILE_PATHS = [os.path.join(base_dir, 'data')]
        else:
            resolved = []
            for p in self.ALLOWED_FILE_PATHS:
                if not os.path.isabs(p):
                    resolved.append(os.path.join(base_dir, p))
                else:
                    resolved.append(p)
            self.ALLOWED_FILE_PATHS = resolved

        # Make QUARANTINE_DIR absolute if it's a relative path
        if self.QUARANTINE_DIR and not os.path.isabs(self.QUARANTINE_DIR):
            self.QUARANTINE_DIR = os.path.join(base_dir, self.QUARANTINE_DIR)

        # Make LOG_FILE absolute if relative
        if self.LOG_FILE and not os.path.isabs(self.LOG_FILE):
            self.LOG_FILE = os.path.join(base_dir, self.LOG_FILE)

        # If using a sqlite file URL and it's relative, convert to absolute
        if isinstance(self.DATABASE_URL, str) and self.DATABASE_URL.startswith('sqlite:///'):
            db_path = self.DATABASE_URL.replace('sqlite:///', '', 1)
            if not os.path.isabs(db_path):
                abs_db = os.path.join(base_dir, db_path)
                self.DATABASE_URL = 'sqlite:///' + abs_db


_config_instance: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

