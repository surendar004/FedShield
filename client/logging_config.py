"""Configure logging for the client modules."""
import logging
import warnings
from sklearn.exceptions import DataConversionWarning

def setup_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce sklearn warnings
    warnings.filterwarnings('ignore', category=DataConversionWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    # Set specific logger levels
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('joblib').setLevel(logging.WARNING)