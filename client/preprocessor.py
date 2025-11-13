"""
Standardized feature preprocessing for FedShield.
Ensures all clients send normalized, comparable features.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Standardizes features across all clients."""
    
    FEATURE_NAMES = [
        'src_port', 'dst_port', 'protocol', 'packet_count', 'byte_count',
        'duration_sec', 'flow_rate', 'cpu_usage_percent', 'memory_usage_percent',
        'disk_io_read_mb', 'disk_io_write_mb', 'network_in_mbps', 'network_out_mbps',
        'process_count', 'open_connections', 'failed_login_attempts',
        'http_requests', 'https_requests', 'dns_queries', 'unique_domains',
        'suspicious_ports_contacted', 'encryption_weak_tls', 'packet_size_avg',
        'packet_size_std', 'inter_arrival_time_avg', 'entropy_src_port', 'entropy_dst_port'
    ]
    
    LABEL_MAPPING = {
        'NORMAL': 0,
        'MALWARE': 1,
        'PHISHING': 2,
        'UNAUTHORIZED_ACCESS': 3,
        'DATA_LEAK': 4,
        'ANOMALY': 5
    }
    
    def __init__(self):
        self.feature_names = self.FEATURE_NAMES
        self.label_mapping = self.LABEL_MAPPING
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Fit z-score normalization parameters on local data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {X.shape[0]} samples")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        if not self.is_fitted:
            logger.warning("Preprocessor not fitted; fitting on provided data")
            self.fit(X)
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the normalization for interpretation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet")
        return X * self.std + self.mean
    
    def load_and_preprocess_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV and return normalized features and labels."""
        df = pd.read_csv(csv_path)
        
        # Extract features and labels
        X = df[self.feature_names].values.astype(np.float32)
        y = df['label'].map(self.label_mapping).values.astype(np.int64)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        X_norm = self.fit_transform(X)
        
        logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Label distribution: {np.bincount(y)}")
        
        return X_norm, y
    
    def get_stats(self) -> Dict[str, Any]:
        """Return preprocessing statistics."""
        return {
            'mean': self.mean,
            'std': self.std,
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping
        }


def create_feature_matrix(data_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert dict of features to normalized array.
    
    Args:
        data_dict: Dictionary with feature names as keys
        
    Returns:
        Normalized feature array of shape (1, 27)
    """
    preprocessor = FeaturePreprocessor()
    
    # Extract values in correct order
    values = [data_dict.get(f, 0.0) for f in preprocessor.feature_names]
    X = np.array(values, dtype=np.float32).reshape(1, -1)
    
    # Fit and transform
    return preprocessor.fit_transform(X)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample data
    preprocessor = FeaturePreprocessor()
    X_sample = np.random.randn(100, 27).astype(np.float32)
    X_normalized = preprocessor.fit_transform(X_sample)
    print(f"Original shape: {X_sample.shape}, Normalized shape: {X_normalized.shape}")
    print(f"Mean: {X_normalized.mean(axis=0)[:5]}")
    print(f"Std: {X_normalized.std(axis=0)[:5]}")
