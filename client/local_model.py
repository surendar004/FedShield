"""Train a simple IsolationForest on sample log data and save the local model.
This script can be imported by client nodes to load a local model.
"""
import os
import sys
import logging

# Verify third-party dependencies at runtime and give a clear install hint if missing.
_missing = []
import importlib

# Import third-party modules at runtime using importlib to avoid static "could not be resolved" warnings.
try:
    joblib = importlib.import_module('joblib')
except Exception:
    _missing.append('joblib')
try:
    pd = importlib.import_module('pandas')
except Exception:
    _missing.append('pandas')
try:
    np = importlib.import_module('numpy')
except Exception:
    _missing.append('numpy')
try:
    _sklearn_ensemble = importlib.import_module('sklearn.ensemble')
    IsolationForest = _sklearn_ensemble.IsolationForest
except Exception:
    _missing.append('scikit-learn')

if _missing:
    print(
        "Missing required packages: %s\nInstall them with: pip install %s"
        % (', '.join(_missing), ' '.join(_missing)),
        file=sys.stderr,
    )
    sys.exit(1)

# Import local modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from client.logging_config import setup_logging

# Set up logging configuration
setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'local_model.pkl')


def generate_or_load_data(csv_path=None):
    """Load sample logs for training. Returns a DataFrame.

    If no CSV exists, this function will create a small synthetic dataset.
    """
    if csv_path is None:
        csv_path = os.path.join(ROOT, 'data', 'sample_logs.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # generate small synthetic dataset
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-11-01', periods=200, freq='S').astype(str),
            'cpu_pct': rng.normal(15, 5, 200).clip(0, 100),
            'net_bytes': rng.normal(2000, 800, 200).clip(0),
            'file_access_count': rng.poisson(1, 200),
            'file_path': ['data/sample_file_%d.txt' % i for i in range(200)]
        })
    return df


def train_and_save(csv_path=None):
    """Train IsolationForest and save model to disk."""
    df = generate_or_load_data(csv_path)
    feature_cols = ['cpu_pct', 'net_bytes', 'file_access_count']
    features = df[feature_cols]
    
    # Create IsolationForest with feature names
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(features)
    
    # Store feature names with the model as an attribute
    setattr(iso, 'feature_names_', feature_cols)
    
    # Save the model
    joblib.dump(iso, MODEL_PATH)
    print('Saved local model to', MODEL_PATH)
    return MODEL_PATH


def load_local_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)


if __name__ == '__main__':
    train_and_save()
