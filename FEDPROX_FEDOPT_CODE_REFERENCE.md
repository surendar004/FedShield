# FedProx & FedOpt Code Reference

## Key Implementation Files

### 1. FedProx Integration in client/model.py

```python
# Extended ThreatDetectionModel.train() with FedProx support
def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1,
          global_weights: Optional[Dict[str, np.ndarray]] = None,
          mu: float = 0.0) -> Dict[str, float]:
    """
    Supports both FedAvg (mu=0.0) and FedProx (mu>0.0) training.
    
    FedProx loss: L(θ) + (mu/2) * ||θ - θ_global||^2
    """
    # ... model training ...
    if mu > 0.0 and global_weights is not None:
        self._global_weights = global_weights
        self._mu = mu
        self.model.fit(X_scaled, y)
        self._apply_fedprox_regularization()
    else:
        self.model.fit(X_scaled, y)
    # ...

def _apply_fedprox_regularization(self) -> None:
    """Apply proximal regularization to model weights."""
    for i, coef in enumerate(self.model.coefs_):
        global_coef = self._global_weights['coefs'][i]
        # w_local = w_local - (mu * lr) * (w_local - w_global)
        self.model.coefs_[i] = coef - (self._mu * self.learning_rate) * (coef - global_coef)
```

### 2. FedOptimizer Class in server/federated_learning.py

```python
class FedOptimizer:
    """Server-side optimizer for FedOpt algorithms."""
    
    def __init__(self, optimizer_name: str = "adam",
                 learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-7):
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_t = {}  # First moment estimate
        self.v_t = {}  # Second moment estimate
        self.t = 0     # Timestep
    
    def step(self, aggregated_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform one optimization step on aggregated weights."""
        self.t += 1
        if self.optimizer_name == "adam":
            return self._adam_step(aggregated_weights)
        elif self.optimizer_name == "yogi":
            return self._yogi_step(aggregated_weights)
        else:  # sgd
            return self._sgd_step(aggregated_weights)
    
    def _adam_step(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Adam: m_t = β₁*m_{t-1} + (1-β₁)*g_t
              v_t = β₂*v_{t-1} + (1-β₂)*g_t²
              θ = θ - α * m̂ / (√v̂ + ε)
        """
        updated = {}
        for key, grad in weights.items():
            # Skip metadata (dicts, strings, etc.)
            if isinstance(grad, dict) or not isinstance(grad, (np.ndarray, list)):
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
                    self.m_t[key][i] = self.beta1 * self.m_t[key][i] + (1 - self.beta1) * g
                    self.v_t[key][i] = self.beta2 * self.v_t[key][i] + (1 - self.beta2) * (g ** 2)
                    m_hat = self.m_t[key][i] / (1 - self.beta1 ** self.t)
                    v_hat = self.v_t[key][i] / (1 - self.beta2 ** self.t)
                    update = g - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    updated_list.append(update)
                updated[key] = updated_list
            else:
                # Handle scalar/1D arrays
                g = np.asarray(grad)
                if key not in self.m_t:
                    self.m_t[key] = np.zeros_like(g)
                    self.v_t[key] = np.zeros_like(g)
                
                self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * g
                self.v_t[key] = self.beta2 * self.v_t[key] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m_t[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v_t[key] / (1 - self.beta2 ** self.t)
                updated[key] = g - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated
```

### 3. FedProxOrchestrator Class

```python
class FedProxOrchestrator:
    """
    Orchestrate FedProx training loop with proximal regularization.
    
    FedProx loss: L(θ) + (mu/2) * ||θ - θ_global||^2
    """
    
    def __init__(self, config: FLConfig, num_clients: int, mu: float = 0.01):
        """Initialize with proximal coefficient mu."""
        self.config = config
        self.mu = mu
        self.server = FederatedServer(config, num_clients)
        self.clients = {i: FederatedClient(i, 0) for i in range(num_clients)}
        self.global_history = []
        logger.info(f"Initialized FedProx with mu={mu}")
    
    def simulate_round(self, client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                      num_samples: Dict[int, int],
                      global_weights: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Simulate one complete FedProx round."""
        selected = self.server.client_selection()
        
        client_updates = {}
        client_metrics = {}
        
        for cid in selected:
            if cid in client_data:
                X, y = client_data[cid]
                # Train with FedProx regularization
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
        
        # Update all clients
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
```

### 4. FedOptOrchestrator Class

```python
class FedOptOrchestrator:
    """
    Orchestrate FedOpt training with server-side optimization.
    
    Applies Adam, Yogi, or SGD to aggregated weights for accelerated convergence.
    """
    
    def __init__(self, config: FLConfig, num_clients: int,
                 server_optimizer: str = "adam",
                 server_lr: float = 0.01):
        """Initialize with server-side optimizer."""
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
        """Simulate one complete FedOpt round."""
        selected = self.server.client_selection()
        
        client_updates = {}
        client_metrics = {}
        
        for cid in selected:
            if cid in client_data:
                X, y = client_data[cid]
                # Standard local training (no FedProx)
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
```

### 5. FederatedClient FedProx Method

```python
def local_training_fedprox(self, X: np.ndarray, y: np.ndarray,
                           config: 'FLConfig',
                           global_weights: Optional[Dict[str, np.ndarray]] = None,
                           mu: float = 0.01) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Train locally with FedProx regularization."""
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
```

## Algorithm Comparison Code

### FedAvg Training Loop
```python
# Standard federated averaging
orchestrator = FedAvgOrchestrator(config, num_clients=20)
for round in range(config.num_rounds):
    result = orchestrator.simulate_round(client_data, num_samples)
    # result contains: round, selected_clients, metrics, num_updates
```

### FedProx Training Loop  
```python
# With proximal regularization
orchestrator = FedProxOrchestrator(config, num_clients=20, mu=0.01)
orchestrator.clients[0].local_model.train(init_data[0], init_labels[0])
global_weights = orchestrator.clients[0].local_model.get_weights()

for round in range(config.num_rounds):
    result = orchestrator.simulate_round(client_data, num_samples, global_weights)
    # result includes: algorithm='FedProx', mu=0.01
```

### FedOpt Training Loop
```python
# With server-side optimization
orchestrator = FedOptOrchestrator(
    config, num_clients=20,
    server_optimizer="adam",
    server_lr=0.01
)

for round in range(config.num_rounds):
    result = orchestrator.simulate_round(client_data, num_samples)
    # result includes: algorithm='FedOpt', optimizer='adam', server_step=round
```

## Test Examples

### Testing FedProx Initialization
```python
def test_fedprox_initialization(self):
    """Test FedProx orchestrator initialization."""
    config = FLConfig(num_rounds=3, clients_per_round=2)
    orchestrator = FedProxOrchestrator(config, num_clients=4, mu=0.01)
    
    assert orchestrator.mu == 0.01
    assert len(orchestrator.clients) == 4
```

### Testing FedOpt Adam Optimizer
```python
def test_adam_step_sequence(self):
    """Test Adam converges over multiple steps."""
    optimizer = FedOptimizer(optimizer_name="adam", learning_rate=0.01)
    
    weights = {'w': np.array([1.0, 1.0, 1.0])}
    
    norms = []
    for _ in range(10):
        updated = optimizer.step(weights)
        norm = np.linalg.norm(updated['w'])
        norms.append(norm)
        weights = updated
    
    # Adam should reduce magnitude over time
    assert norms[0] > norms[-1]
```

## Configuration Examples

### FedProx Configuration
```python
# In code
config = FLConfig(num_rounds=20, clients_per_round=5)
orchestrator = FedProxOrchestrator(config, num_clients=100, mu=0.01)

# In YAML (future)
algorithm: fedprox
rounds: 20
clients_per_round: 5
fedprox:
  mu: 0.01  # Proximal coefficient
  # mu=0: Behaves like FedAvg
  # mu=0.01-0.1: Good for non-IID data
  # mu>0.5: Strong regularization
```

### FedOpt Configuration
```python
# In code
config = FLConfig(num_rounds=20, clients_per_round=5)
orchestrator = FedOptOrchestrator(
    config, num_clients=100,
    server_optimizer="adam",
    server_lr=0.01
)

# In YAML (future)
algorithm: fedopt
rounds: 20
clients_per_round: 5
fedopt:
  server_optimizer: adam  # Options: adam, yogi, sgd
  server_lr: 0.01
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-7
```

## Integration with Privacy

```python
# FedProx + DP-SGD
config = FLConfig(
    num_rounds=20,
    dp_epsilon=1.0,
    dp_delta=1e-5
)
fedprox = FedProxOrchestrator(config, num_clients=100, mu=0.01)
privacy_manager = PrivacyManager(epsilon=1.0, delta=1e-5)

# Client training applies both FedProx regularization and DP-SGD noise
```

## Performance Tuning Tips

### FedProx Tuning
1. **Start with mu=0.01**: Mild regularization
2. **Increase mu for higher non-IID**: mu=0.05-0.1
3. **Decrease mu for IID data**: mu=0.001
4. **Monitor convergence**: Should stabilize faster on non-IID

### FedOpt Tuning
1. **Adam usually works well**: Safe default choice
2. **Use Yogi for sparse gradients**: Better stability
3. **Tune server_lr**: Similar to regular SGD tuning
4. **Monitor server_step**: Tracks optimization progress

## Troubleshooting

### FedProx Issues
- **Slow convergence**: Increase mu value (but not too much)
- **Non-convergence**: Decrease mu, check global_weights initialization
- **Memory issues**: FedProx overhead is minimal, check client data size

### FedOpt Issues  
- **Optimizer diverging**: Decrease server_lr
- **Slow convergence**: Increase server_lr (cautiously)
- **Memory issues**: Adam/Yogi maintain m_t, v_t states (larger with more parameters)

## References

- **FedProx Paper**: Li et al., "Federated Optimization for Heterogeneous Networks"
  https://arxiv.org/abs/1812.06127
  
- **FedOpt Paper**: Reddi et al., "Adaptive Federated Optimization"
  https://arxiv.org/abs/2003.00295
  
- **Adam**: Kingma & Ba, "Adam: A Method for Stochastic Optimization"
  https://arxiv.org/abs/1412.6980
  
- **Yogi**: Zaheer et al., "Adaptive Methods for Nonconvex Optimization"
  https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
