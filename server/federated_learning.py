"""
Federated Averaging (FedAvg) - Baseline Algorithm
Implements server-side orchestration and client-side training.
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from utils.compression import decompress_weights, compress_weights

logger = logging.getLogger(__name__)


@dataclass
class FLConfig:
    """Federated Learning Configuration."""
    num_rounds: int = 10
    clients_per_round: int = 3
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    clip_norm: float = 1.0
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    model_architecture: Tuple[int, ...] = (128, 64, 32)
    seed: int = 42
    # Compression settings
    compression_enabled: bool = False
    compression_top_k: float = 0.1  # fraction of weights to keep (top-k)
    quantization_bits: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_rounds': self.num_rounds,
            'clients_per_round': self.clients_per_round,
            'local_epochs': self.local_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'clip_norm': self.clip_norm,
            'dp_epsilon': self.dp_epsilon,
            'dp_delta': self.dp_delta,
            'model_architecture': self.model_architecture,
            'seed': self.seed
        }


class FederatedServer:
    """Server coordinating federated learning rounds."""
    
    def __init__(self, config: FLConfig, num_clients: int):
        self.config = config
        self.num_clients = num_clients
        self.current_round = 0
        self.global_model_weights = None
        self.round_history = []
    
    def client_selection(self) -> List[int]:
        """Select clients for current round."""
        available_clients = list(range(self.num_clients))
        selected = np.random.choice(
            available_clients,
            size=min(self.config.clients_per_round, self.num_clients),
            replace=False
        )
        logger.info(f"Round {self.current_round}: Selected clients {selected.tolist()}")
        return selected.tolist()
    
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, np.ndarray]],
                         num_samples: Dict[int, int]) -> Dict[str, np.ndarray]:
        """
        Aggregate client model updates using weighted averaging.
        
        FedAvg: θ_t+1 = Σ (n_k / n) * θ_k
        where n_k is samples on client k, n is total samples
        """
        # Calculate weights
        total_samples = sum(num_samples.values())
        weights = {cid: num_samples[cid] / total_samples for cid in num_samples.keys()}

        logger.info(f"Aggregating {len(client_updates)} clients, total samples: {total_samples}")

        # Pre-process: detect and decompress any compressed client payloads
        for cid, update in list(client_updates.items()):
            # Build a payload dict consisting of keys that appear compressed
            payload = {k: v for k, v in update.items() if isinstance(v, dict) and 'quantized' in v}
            if payload:
                try:
                    decompr = decompress_weights(payload)
                except Exception as e:
                    raise ValueError(f"Failed to decompress payload from client {cid}: {e}")
                # Replace compressed keys with decompressed arrays
                for k, v in decompr.items():
                    client_updates[cid][k] = v

        # Initialize aggregated weights
        aggregated = {}
        first_client = list(client_updates.keys())[0]
        
        for key in client_updates[first_client].keys():
            # Skip metrics - they are metadata, not model parameters
            if key == 'metrics':
                aggregated[key] = client_updates[first_client][key]
                continue
                
            if key in ['coefs', 'intercepts']:
                # These are lists of arrays
                agg_list = []
                num_layers = len(client_updates[first_client][key])
                
                for layer_idx in range(num_layers):
                    agg_matrix = np.zeros_like(client_updates[first_client][key][layer_idx], dtype=np.float64)
                    
                    for cid, update in client_updates.items():
                        agg_matrix += weights[cid] * update[key][layer_idx]
                    
                    agg_list.append(agg_matrix)
                aggregated[key] = agg_list
            else:
                # Scalar or 1D array - convert to float to avoid dtype casting issues
                agg_value = np.zeros_like(client_updates[first_client][key], dtype=np.float64)
                for cid, update in client_updates.items():
                    # Support compressed payloads (dict) - if compressed, skip here (handled externally)
                    val = update[key]
                    if isinstance(val, dict) and 'quantized' in val:
                        # Decompression should happen before aggregation; raise for clarity
                        raise ValueError("Compressed weights detected during aggregation; decompress before calling aggregate_updates")
                    agg_value += weights[cid] * np.asarray(val, dtype=np.float64)
                aggregated[key] = agg_value
        
        return aggregated
    
    def run_round(self, client_metrics: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute one federated learning round."""
        self.current_round += 1
        selected_clients = self.client_selection()
        
        round_info = {
            'round': self.current_round,
            'timestamp': datetime.now().isoformat(),
            'selected_clients': selected_clients,
            'config': self.config.to_dict(),
            'client_metrics': {}
        }
        
        logger.info(f"=== FL Round {self.current_round} ===")
        logger.info(f"Config: {round_info['config']}")
        
        # In real execution, this would:
        # 1. Send global model to clients
        # 2. Clients train locally
        # 3. Clients send updates back
        # 4. Server aggregates
        
        self.round_history.append(round_info)
        return round_info
    
    def save_checkpoint(self, path: str) -> None:
        """Save server state."""
        checkpoint = {
            'current_round': self.current_round,
            'num_clients': self.num_clients,
            'config': self.config.to_dict(),
            'history': self.round_history
        }
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved to {path}")


class FederatedClient:
    """Client-side federated learning participant."""
    
    def __init__(self, client_id: int, local_data_size: int):
        self.client_id = client_id
        self.local_data_size = local_data_size
        self.local_model = None
        self.global_model_weights = None
        self.training_history = []
    
    def set_model(self, model):
        """Set the model instance."""
        self.local_model = model
    
    def receive_global_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Receive aggregated model from server."""
        if self.local_model is not None:
            self.local_model.set_weights(weights)
        self.global_model_weights = weights
        logger.info(f"Client {self.client_id}: Received global weights")
    
    def local_training(self, X: np.ndarray, y: np.ndarray, 
                      config: FLConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Train locally for config.local_epochs.
        
        Returns:
            Tuple of (model_weights, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local training "
                   f"({config.local_epochs} epochs, {len(X)} samples)")
        
        # Train model
        metrics = self.local_model.train(X, y, epochs=config.local_epochs)
        
        # Get updated weights
        weights = self.local_model.get_weights()
        
        # Optionally compress weights to save bandwidth
        if getattr(config, 'compression_enabled', False):
            try:
                payload = compress_weights({
                    'coefs': weights['coefs'],
                    'intercepts': weights['intercepts'],
                    'scaler_mean': weights['scaler_mean'],
                    'scaler_scale': weights['scaler_scale']
                }, bits=config.quantization_bits, k_fraction=config.compression_top_k)

                # Replace arrays with compressed payload entries
                for k, v in payload.items():
                    weights[k] = v
            except Exception as e:
                logger.warning(f"Client {self.client_id}: Compression failed, sending raw weights: {e}")

        # Add client metadata
        weights['client_id'] = self.client_id
        weights['num_samples'] = len(X)
        weights['metrics'] = metrics
        
        logger.info(f"Client {self.client_id}: Local training complete")
        self.training_history.append(metrics)
        
        return weights, metrics

    def personalize(self, X: np.ndarray, y: np.ndarray, config: FLConfig,
                    personalization_epochs: Optional[int] = None,
                    personalization_lr: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Perform per-client personalization (fine-tuning) starting from global weights.

        Returns (personalized_weights, metrics)
        """
        personalization_epochs = personalization_epochs or max(1, config.local_epochs // 2)
        personalization_lr = personalization_lr or config.learning_rate

        logger.info(f"Client {self.client_id}: Starting personalization (epochs={personalization_epochs})")
        metrics = self.local_model.personalize(X, y, epochs=personalization_epochs, learning_rate=personalization_lr)
        personalized_weights = self.local_model.get_weights()
        # Add metadata
        personalized_weights['client_id'] = self.client_id
        personalized_weights['num_samples'] = len(X)
        personalized_weights['metrics'] = metrics

        self.training_history.append(metrics)
        logger.info(f"Client {self.client_id}: Personalization complete")
        return personalized_weights, metrics
    
    def compute_gradient_norm(self, weights: Dict[str, np.ndarray]) -> float:
        """Compute L2 norm of weight updates for anomaly detection."""
        norm = 0.0
        if 'coefs' in weights:
            for coef in weights['coefs']:
                norm += np.sum(coef ** 2)
        return np.sqrt(norm)
    
    def local_training_fedprox(self, X: np.ndarray, y: np.ndarray,
                               config: 'FLConfig',
                               global_weights: Optional[Dict[str, np.ndarray]] = None,
                               mu: float = 0.01) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Train locally with FedProx regularization.
        
        Args:
            X: Training features
            y: Training labels
            config: FL configuration
            global_weights: Global model weights
            mu: FedProx regularization coefficient
            
        Returns:
            Tuple of (model_weights, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting FedProx training "
                   f"(mu={mu}, {config.local_epochs} epochs, {len(X)} samples)")
        
        # Train model with FedProx
        metrics = self.local_model.train(
            X, y,
            epochs=config.local_epochs,
            global_weights=global_weights,
            mu=mu
        )
        
        # Get updated weights
        weights = self.local_model.get_weights()
        
        # Add client metadata
        weights['client_id'] = self.client_id
        weights['num_samples'] = len(X)
        weights['metrics'] = metrics
        
        logger.info(f"Client {self.client_id}: FedProx training complete")
        self.training_history.append(metrics)
        
        return weights, metrics


class FedAvgOrchestrator:
    """Orchestrate full FedAvg training loop."""
    
    def __init__(self, config: FLConfig, num_clients: int):
        self.config = config
        self.server = FederatedServer(config, num_clients)
        self.clients = {i: FederatedClient(i, 0) for i in range(num_clients)}
        self.global_history = []
    
    def simulate_round(self, client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                      num_samples: Dict[int, int]) -> Dict[str, Any]:
        """
        Simulate one complete FL round with given client data.
        
        Args:
            client_data: Dict of client_id -> (X, y)
            num_samples: Dict of client_id -> number of samples
        """
        # Select clients
        selected = self.server.client_selection()
        
        # Collect updates
        client_updates = {}
        client_metrics = {}
        
        for cid in selected:
            if cid in client_data:
                X, y = client_data[cid]
                weights, metrics = self.clients[cid].local_training(X, y, self.config)
                client_updates[cid] = weights
                client_metrics[cid] = metrics
        
        # Aggregate
        aggregated = self.server.aggregate_updates(
            {cid: client_updates[cid] for cid in client_updates.keys() if 'coefs' in client_updates[cid]},
            num_samples
        )
        
        # Update global model
        for cid in self.clients.keys():
            self.clients[cid].receive_global_weights(aggregated)
        
        round_summary = {
            'round': self.server.current_round,
            'selected_clients': selected,
            'num_updates': len(client_updates),
            'metrics': client_metrics
        }
        
        self.global_history.append(round_summary)
        return round_summary


class FedOptimizer:
    """
    Server-side optimizer for FedOpt algorithms.
    
    Supports multiple server-side optimization methods:
    - FedAvg: Standard averaging
    - FedAdam: Adam optimizer on server gradients
    - FedYogi: Yogi optimizer (adaptive learning rate)
    """
    
    def __init__(self, optimizer_name: str = "adam",
                 learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-7):
        """
        Initialize server-side optimizer.
        
        Args:
            optimizer_name: "adam", "yogi", or "sgd"
            learning_rate: Server learning rate
            beta1: Exponential decay rate for 1st moment (Adam)
            beta2: Exponential decay rate for 2nd moment (Adam)
            epsilon: Numerical stability constant
        """
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Optimizer states
        self.m_t = {}  # First moment estimate
        self.v_t = {}  # Second moment estimate
        self.t = 0  # Timestep
        
        logger.info(f"Initialized FedOpt with optimizer: {optimizer_name}")
    
    def step(self, aggregated_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one optimization step on aggregated weights.
        
        Args:
            aggregated_weights: Weighted average of client updates
            
        Returns:
            Optimized weights
        """
        self.t += 1
        updated = {}
        
        if self.optimizer_name == "adam":
            return self._adam_step(aggregated_weights)
        elif self.optimizer_name == "yogi":
            return self._yogi_step(aggregated_weights)
        else:  # sgd
            return self._sgd_step(aggregated_weights)
    
    def _sgd_step(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Standard SGD update."""
        updated = {}
        for key, value in weights.items():
            # Skip metadata that isn't an ndarray or list of arrays
            if isinstance(value, dict) or (isinstance(value, (int, float, str)) and not isinstance(value, np.ndarray)):
                updated[key] = value
                continue
            
            if isinstance(value, list):
                updated[key] = [v - self.learning_rate * np.asarray(v) for v in value]
            else:
                updated[key] = value - self.learning_rate * np.asarray(value)
        return updated
    
    def _adam_step(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Adam optimization step.
        
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
        """
        updated = {}
        
        for key, grad in weights.items():
            # Skip metadata that isn't an ndarray or list of arrays
            if isinstance(grad, dict) or (isinstance(grad, (int, float, str)) and not isinstance(grad, np.ndarray)):
                updated[key] = grad
                continue
            
            if isinstance(grad, list):
                # Handle lists of arrays (coefs, intercepts)
                if key not in self.m_t:
                    self.m_t[key] = [np.zeros_like(g) for g in grad]
                    self.v_t[key] = [np.zeros_like(g) for g in grad]
                
                updated_list = []
                for i, g in enumerate(grad):
                    g = np.asarray(g)
                    # Update biased first moment estimate
                    self.m_t[key][i] = self.beta1 * self.m_t[key][i] + (1 - self.beta1) * g
                    # Update biased second raw moment estimate
                    self.v_t[key][i] = self.beta2 * self.v_t[key][i] + (1 - self.beta2) * (g ** 2)
                    # Compute bias-corrected first moment estimate
                    m_hat = self.m_t[key][i] / (1 - self.beta1 ** self.t)
                    # Compute bias-corrected second raw moment estimate
                    v_hat = self.v_t[key][i] / (1 - self.beta2 ** self.t)
                    # Update parameters
                    update = g - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    updated_list.append(update)
                updated[key] = updated_list
            else:
                # Handle scalar/1D arrays
                g = np.asarray(grad)
                if key not in self.m_t:
                    self.m_t[key] = np.zeros_like(g)
                    self.v_t[key] = np.zeros_like(g)
                
                # Update biased first moment estimate
                self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * g
                # Update biased second raw moment estimate
                self.v_t[key] = self.beta2 * self.v_t[key] + (1 - self.beta2) * (g ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = self.m_t[key] / (1 - self.beta1 ** self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v_t[key] / (1 - self.beta2 ** self.t)
                # Update parameters
                updated[key] = g - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated
    
    def _yogi_step(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Yogi optimization step (adaptive variant).
        
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = v_{t-1} - (1 - β₂) * sign(v_{t-1} - g_t²) * g_t²
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
        """
        updated = {}
        
        for key, grad in weights.items():
            # Skip metadata
            if isinstance(grad, dict) or (isinstance(grad, (int, float, str)) and not isinstance(grad, np.ndarray)):
                updated[key] = grad
                continue
            
            if isinstance(grad, list):
                if key not in self.m_t:
                    self.m_t[key] = [np.zeros_like(g) for g in grad]
                    self.v_t[key] = [np.zeros_like(g) for g in grad]
                
                updated_list = []
                for i, g in enumerate(grad):
                    g = np.asarray(g)
                    g_sq = g ** 2
                    # Update biased first moment estimate
                    self.m_t[key][i] = self.beta1 * self.m_t[key][i] + (1 - self.beta1) * g
                    # Update biased second moment estimate (Yogi variant)
                    self.v_t[key][i] = self.v_t[key][i] - (1 - self.beta2) * np.sign(self.v_t[key][i] - g_sq) * g_sq
                    # Compute bias-corrected estimates
                    m_hat = self.m_t[key][i] / (1 - self.beta1 ** self.t)
                    v_hat = self.v_t[key][i] / (1 - self.beta2 ** self.t)
                    # Update parameters
                    update = g - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    updated_list.append(update)
                updated[key] = updated_list
            else:
                g = np.asarray(grad)
                g_sq = g ** 2
                if key not in self.m_t:
                    self.m_t[key] = np.zeros_like(g)
                    self.v_t[key] = np.zeros_like(g)
                
                # Update biased first moment estimate
                self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * g
                # Update biased second moment estimate (Yogi variant)
                self.v_t[key] = self.v_t[key] - (1 - self.beta2) * np.sign(self.v_t[key] - g_sq) * g_sq
                # Compute bias-corrected estimates
                m_hat = self.m_t[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v_t[key] / (1 - self.beta2 ** self.t)
                # Update parameters
                updated[key] = g - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated


class FedProxOrchestrator:
    """
    Orchestrate FedProx training loop.
    
    FedProx adds a proximal term to regularize client updates toward the global model,
    making it more robust to systems/statistical heterogeneity.
    
    Loss function: L(θ) + (mu/2) * ||θ - θ_global||^2
    """
    
    def __init__(self, config: FLConfig, num_clients: int, mu: float = 0.01):
        """
        Initialize FedProx orchestrator.
        
        Args:
            config: FL configuration
            num_clients: Number of clients
            mu: Proximal coefficient (higher = stronger regularization toward global model)
        """
        self.config = config
        self.mu = mu
        self.server = FederatedServer(config, num_clients)
        self.clients = {i: FederatedClient(i, 0) for i in range(num_clients)}
        self.global_history = []
        logger.info(f"Initialized FedProx with mu={mu}")
    
    def simulate_round(self, client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                      num_samples: Dict[int, int],
                      global_weights: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Simulate one complete FedProx round.
        
        Args:
            client_data: Dict of client_id -> (X, y)
            num_samples: Dict of client_id -> number of samples
            global_weights: Current global model weights
            
        Returns:
            Round summary with metrics
        """
        # Select clients
        selected = self.server.client_selection()
        
        # Collect updates with FedProx regularization
        client_updates = {}
        client_metrics = {}
        
        for cid in selected:
            if cid in client_data:
                X, y = client_data[cid]
                # Pass global weights for FedProx term
                if global_weights is not None:
                    self.clients[cid].receive_global_weights(global_weights)
                # Train with FedProx (mu > 0)
                weights, metrics = self.clients[cid].local_training_fedprox(
                    X, y, self.config, global_weights, self.mu
                )
                client_updates[cid] = weights
                client_metrics[cid] = metrics
        
        # Aggregate
        aggregated = self.server.aggregate_updates(
            {cid: client_updates[cid] for cid in client_updates.keys() if 'coefs' in client_updates[cid]},
            num_samples
        )
        
        # Update all clients with new global model
        for cid in self.clients.keys():
            self.clients[cid].receive_global_weights(aggregated)
        
        round_summary = {
            'round': self.server.current_round,
            'selected_clients': selected,
            'num_updates': len(client_updates),
            'metrics': client_metrics,
            'algorithm': 'FedProx',
            'mu': self.mu
        }
        
        self.global_history.append(round_summary)
        return round_summary


class FedOptOrchestrator:
    """
    Orchestrate FedOpt training loop with server-side optimization.
    
    FedOpt uses server-side optimizers (Adam, Yogi) to accelerate convergence
    by adaptively adjusting server learning rates based on gradient statistics.
    """
    
    def __init__(self, config: FLConfig, num_clients: int,
                 server_optimizer: str = "adam",
                 server_lr: float = 0.01):
        """
        Initialize FedOpt orchestrator.
        
        Args:
            config: FL configuration
            num_clients: Number of clients
            server_optimizer: "adam", "yogi", or "sgd"
            server_lr: Server-side learning rate
        """
        self.config = config
        self.server = FederatedServer(config, num_clients)
        self.clients = {i: FederatedClient(i, 0) for i in range(num_clients)}
        self.optimizer = FedOptimizer(
            optimizer_name=server_optimizer,
            learning_rate=server_lr
        )
        self.global_history = []
        logger.info(f"Initialized FedOpt with server optimizer: {server_optimizer}, lr={server_lr}")
    
    def simulate_round(self, client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                      num_samples: Dict[int, int]) -> Dict[str, Any]:
        """
        Simulate one complete FedOpt round.
        
        Args:
            client_data: Dict of client_id -> (X, y)
            num_samples: Dict of client_id -> number of samples
            
        Returns:
            Round summary with metrics
        """
        # Select clients
        selected = self.server.client_selection()
        
        # Collect updates
        client_updates = {}
        client_metrics = {}
        
        for cid in selected:
            if cid in client_data:
                X, y = client_data[cid]
                weights, metrics = self.clients[cid].local_training(X, y, self.config)
                client_updates[cid] = weights
                client_metrics[cid] = metrics
        
        # Aggregate client updates
        aggregated = self.server.aggregate_updates(
            {cid: client_updates[cid] for cid in client_updates.keys() if 'coefs' in client_updates[cid]},
            num_samples
        )
        
        # Apply server-side optimization
        optimized = self.optimizer.step(aggregated)
        
        # Update all clients with optimized model
        for cid in self.clients.keys():
            self.clients[cid].receive_global_weights(optimized)
        
        round_summary = {
            'round': self.server.current_round,
            'selected_clients': selected,
            'num_updates': len(client_updates),
            'metrics': client_metrics,
            'algorithm': 'FedOpt',
            'optimizer': self.optimizer.optimizer_name,
            'server_step': self.optimizer.t
        }
        
        self.global_history.append(round_summary)
        return round_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = FLConfig(
        num_rounds=5,
        clients_per_round=2,
        local_epochs=2
    )
    
    print("FLConfig:", config.to_dict())
    
    server = FederatedServer(config, num_clients=4)
    for _ in range(config.num_rounds):
        server.run_round({})
    
    print(f"\nCompleted {server.current_round} rounds")
