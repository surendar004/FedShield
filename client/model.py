"""
FedShield model architectures for threat detection.
Supports FedAvg, FedProx, and personalization.
"""
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class ThreatDetectionModel:
    """Light MLP model for federated threat detection."""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (128, 64, 32),
                 learning_rate: float = 0.001,
                 max_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize threat detection model.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            learning_rate: Initial learning rate
            max_iter: Maximum iterations for local training
            random_state: Random seed
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classes_ = np.array([0, 1, 2, 3, 4, 5])  # 6 threat classes
    
    def build(self):
        """Build the MLP model."""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            solver='adam',
            batch_size=32,
            verbose=0
        )
        logger.info(f"Built model with layers: {self.hidden_layers}")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1,
              global_weights: Optional[Dict[str, np.ndarray]] = None,
              mu: float = 0.0) -> Dict[str, float]:
        """
        Train model locally for one or more epochs.
        
        Supports both FedAvg and FedProx training:
        - FedAvg (mu=0.0): Standard local SGD
        - FedProx (mu>0.0): Adds proximal term to regularize toward global model
        
        FedProx loss: L(θ) + (mu/2) * ||θ - θ_global||^2
        
        Args:
            X: Feature array (batch_size, 27)
            y: Label array (batch_size,)
            epochs: Number of local epochs
            global_weights: Global model weights (for FedProx). If None, uses FedAvg.
            mu: FedProx regularization coefficient (0 = FedAvg, >0 = FedProx)
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self.build()
        
        # Reset scaler for each training round to avoid state issues
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Store global weights if FedProx is enabled
        if mu > 0.0 and global_weights is not None:
            self._global_weights = global_weights
            self._mu = mu
            # Note: For sklearn MLPClassifier with FedProx, we simply train the model
            # then apply a single proximal gradient step after training
            self.model.fit(X_scaled, y)
            self._apply_fedprox_regularization()
        else:
            # Standard FedAvg training
            self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        
        # Compute metrics
        train_acc = self.model.score(X_scaled, y)
        logger.info(f"Local training accuracy: {train_acc:.4f} (FedProx mu={mu})" if mu > 0 else f"Local training accuracy: {train_acc:.4f}")
        
        return {'accuracy': train_acc, 'epochs': epochs, 'samples': len(X), 'fedprox_mu': mu}
    
    def _apply_fedprox_regularization(self) -> None:
        """
        Apply FedProx proximal regularization to model weights.
        
        Adjusts weights toward global weights using:
        w_local = w_local - (mu * learning_rate) * (w_local - w_global)
        """
        if not hasattr(self, '_mu') or not hasattr(self, '_global_weights'):
            return
        
        # Apply regularization to coefficients
        for i, coef in enumerate(self.model.coefs_):
            global_coef = self._global_weights['coefs'][i]
            # Proximal term: scale the difference from global weights
            self.model.coefs_[i] = coef - (self._mu * self.learning_rate) * (coef - global_coef)
        
        # Apply regularization to intercepts
        for i, intercept in enumerate(self.model.intercepts_):
            global_intercept = self._global_weights['intercepts'][i]
            self.model.intercepts_[i] = intercept - (self._mu * self.learning_rate) * (intercept - global_intercept)
        
        logger.debug(f"Applied FedProx regularization with mu={self._mu}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy score on provided (already-scaled) data.

        This is a thin wrapper around the sklearn estimator's score method
        that ensures the model is fitted and uses the stored scaler.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Extract model weights for federated averaging."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        weights = {}
        weights['coefs'] = [coef.copy() for coef in self.model.coefs_]
        weights['intercepts'] = [intercept.copy() for intercept in self.model.intercepts_]
        weights['scaler_mean'] = self.scaler.mean_.copy()
        weights['scaler_scale'] = self.scaler.scale_.copy()
        
        return weights
    
    def set_weights(self, weights: Dict[str, List[np.ndarray]]) -> None:
        """Set model weights from federated server."""
        if self.model is None:
            self.build()
        
        self.model.coefs_ = [w.copy() for w in weights['coefs']]
        self.model.intercepts_ = [w.copy() for w in weights['intercepts']]
        self.scaler.mean_ = weights['scaler_mean'].copy()
        self.scaler.scale_ = weights['scaler_scale'].copy()
        self.is_fitted = True
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': {
                'hidden_layers': self.hidden_layers,
                'learning_rate': self.learning_rate
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.hidden_layers = data['config']['hidden_layers']
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")

    def personalize(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, learning_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Perform per-client fine-tuning (personalization) starting from current model weights.

        This does a lightweight local adaptation step to improve per-client performance.

        Args:
            X: Local features
            y: Local labels
            epochs: Number of personalization epochs (small, e.g. 1-3)
            learning_rate: Optional override learning rate for personalization

        Returns:
            Dictionary with personalization metrics
        """
        if self.model is None:
            self.build()

        # Use a fresh scaler fit on personalization data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Temporarily adjust learning rate and max_iter if requested
        orig_lr = getattr(self.model, 'learning_rate_init', None)
        orig_max_iter = getattr(self.model, 'max_iter', None)

        if learning_rate is not None:
            try:
                self.model.learning_rate_init = learning_rate
            except Exception:
                # sklearn MLPClassifier may not expose direct setter for some versions
                pass

        # Run a short training session for personalization
        try:
            self.model.max_iter = epochs if epochs > 0 else self.model.max_iter
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        finally:
            # Restore original hyperparameters when possible
            if orig_lr is not None:
                try:
                    self.model.learning_rate_init = orig_lr
                except Exception:
                    pass
            if orig_max_iter is not None:
                try:
                    self.model.max_iter = orig_max_iter
                except Exception:
                    pass

        acc = self.model.score(X_scaled, y)
        logger.info(f"Personalization accuracy: {acc:.4f} (epochs={epochs})")
        return {'personalization_accuracy': acc, 'personalization_epochs': epochs, 'samples': len(X)}


class ServerEnsemble:
    """Server-side ensemble of client models for improved accuracy."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_models: Dict[str, Dict[str, np.ndarray]] = {}
        self.aggregated_weights = None
    
    def aggregate(self, client_updates: Dict[str, Dict[str, List[np.ndarray]]],
                  weights: Dict[str, float] = None) -> Dict[str, List[np.ndarray]]:
        """
        Aggregate client updates using weighted averaging.
        
        Args:
            client_updates: Dict of client_id -> weights
            weights: Dict of client_id -> weight (e.g., num_samples)
            
        Returns:
            Aggregated weights
        """
        if weights is None:
            # Uniform weights
            weights = {cid: 1.0 / len(client_updates) for cid in client_updates.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {cid: w / total_weight for cid, w in weights.items()}
        
        # Aggregate coefs and intercepts
        aggregated = {
            'coefs': [],
            'intercepts': [],
            'scaler_mean': None,
            'scaler_scale': None
        }
        
        clients_list = list(client_updates.keys())
        first_client = clients_list[0]
        num_coefs = len(client_updates[first_client]['coefs'])
        
        # Aggregate coefficient matrices
        for layer_idx in range(num_coefs):
            agg_coef = np.zeros_like(client_updates[first_client]['coefs'][layer_idx])
            for cid, update in client_updates.items():
                agg_coef += weights[cid] * update['coefs'][layer_idx]
            aggregated['coefs'].append(agg_coef)
        
        # Aggregate intercepts
        num_intercepts = len(client_updates[first_client]['intercepts'])
        for layer_idx in range(num_intercepts):
            agg_intercept = np.zeros_like(client_updates[first_client]['intercepts'][layer_idx])
            for cid, update in client_updates.items():
                agg_intercept += weights[cid] * update['intercepts'][layer_idx]
            aggregated['intercepts'].append(agg_intercept)
        
        # Aggregate scaler parameters
        agg_mean = np.zeros_like(client_updates[first_client]['scaler_mean'])
        agg_scale = np.zeros_like(client_updates[first_client]['scaler_scale'])
        for cid, update in client_updates.items():
            agg_mean += weights[cid] * update['scaler_mean']
            agg_scale += weights[cid] * update['scaler_scale']
        aggregated['scaler_mean'] = agg_mean
        aggregated['scaler_scale'] = agg_scale
        
        self.aggregated_weights = aggregated
        logger.info("Aggregated weights from {} clients".format(len(client_updates)))
        return aggregated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model
    model = ThreatDetectionModel()
    X_train = np.random.randn(200, 27).astype(np.float32)
    y_train = np.random.randint(0, 6, 200)
    
    metrics = model.train(X_train, y_train)
    print(f"Training metrics: {metrics}")
    
    X_test = np.random.randn(50, 27).astype(np.float32)
    preds = model.predict(X_test)
    print(f"Predictions shape: {preds.shape}")
