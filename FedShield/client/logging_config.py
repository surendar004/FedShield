"""Configure logging for the client modules."""
import logging
import warnings


def setup_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Try to reduce sklearn warnings if sklearn is available; else continue silently
    try:
        from sklearn.exceptions import DataConversionWarning
        warnings.filterwarnings('ignore', category=DataConversionWarning)
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('joblib').setLevel(logging.WARNING)
    except Exception:
        # sklearn not available or failed to import (low-memory environment); proceed without it
        pass