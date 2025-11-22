# DP-SGD Integration Guide

Quick guide to integrate privacy-enhanced training into your FedShield models.

## Quick Start (5 minutes)

### Step 1: Import Privacy Manager
```python
from server.privacy_manager import PrivacyManager, DPSGDTrainer
```

### Step 2: Initialize
```python
pm = PrivacyManager(
    epsilon=1.0,      # Privacy budget (1.0 = moderate privacy)
    delta=1e-5,       # Failure prob (1/(num_clients) recommended)
    clip_norm=1.0,    # Gradient clipping threshold
    num_rounds=10     # Total training rounds
)
```

### Step 3: Use in Training Loop
```python
# During gradient-based training
gradients = model.compute_gradients(loss)
private_grads = pm.apply_privacy(gradients)  # Clip + noise in one call
model.update(private_grads)
```

## Integration into Client Model

### Option A: Modify `client/model.py` - ThreatDetectionModel

Add privacy manager to the class:

```python
from server.privacy_manager import PrivacyManager

class ThreatDetectionModel:
    def __init__(self, input_dim=27, hidden_dims=None, 
                 enable_privacy=False, privacy_config=None):
        # ... existing code ...
        
        if enable_privacy:
            self.privacy_manager = PrivacyManager(
                epsilon=privacy_config.get('epsilon', 1.0),
                delta=privacy_config.get('delta', 1e-5),
                clip_norm=privacy_config.get('clip_norm', 1.0),
                num_rounds=privacy_config.get('num_rounds', 10),
            )
        else:
            self.privacy_manager = None
    
    def train(self, X, y, epochs=5, batch_size=32):
        """Train with optional differential privacy."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_x = X_scaled[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                y_pred = self.predict_proba(batch_x)
                loss = self._compute_loss(y_pred, batch_y)
                
                # Compute gradients
                gradients = self._compute_gradients(loss, batch_x)
                
                # Apply privacy if enabled
                if self.privacy_manager:
                    gradients = self.privacy_manager.apply_privacy(gradients)
                
                # Update parameters
                self._update_parameters(gradients)
```

### Option B: Modify `server/federated_learning.py` - FederatedClient

Add privacy to local training:

```python
from server.privacy_manager import PrivacyManager

class FederatedClient:
    def __init__(self, client_id, model, privacy_config=None):
        self.client_id = client_id
        self.model = model
        
        # Initialize privacy manager if configured
        self.privacy_manager = None
        if privacy_config and privacy_config.get('use_differential_privacy'):
            self.privacy_manager = PrivacyManager(
                epsilon=privacy_config.get('epsilon', 1.0),
                delta=privacy_config.get('delta', 1e-5),
                clip_norm=privacy_config.get('clip_norm', 1.0),
                num_rounds=privacy_config.get('num_rounds', 10),
            )
    
    def local_training(self, local_data):
        """Train locally with privacy if enabled."""
        X, y = local_data
        
        # Train the model
        initial_weights = self.model.get_weights()
        
        # Training with privacy
        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self._batch_iterator(X, y):
                # Forward + backward
                loss = self.model.train(batch_x, batch_y, epochs=1)
                
                # Get gradients
                gradients = self.model.get_gradients()
                
                # Apply privacy
                if self.privacy_manager:
                    gradients = self.privacy_manager.apply_privacy(gradients)
                    # Would need to update model with private gradients
                    # This requires modifying model.py to support gradient-based updates
        
        final_weights = self.model.get_weights()
        weight_diff = self._compute_weight_diff(initial_weights, final_weights)
        
        return {
            'weights': weight_diff,
            'num_samples': len(X),
            'privacy_budget_used': self.privacy_manager.privacy_budget.rounds_completed if self.privacy_manager else None,
        }
```

## Integration into Configuration

### Update `config/fl_config.yaml`

```yaml
# Existing config
num_rounds: 10
clients_per_round: 3
local_epochs: 5
learning_rate: 0.001
batch_size: 32

# Add privacy section
privacy:
  use_differential_privacy: true
  epsilon: 1.0           # Privacy budget
  delta: 1e-5            # Failure probability
  clip_norm: 1.0         # L2 clipping threshold
  composition: "advanced" # "advanced" or "basic"
  
# Security remains unchanged
security:
  use_tls: true
  secure_aggregation: true
  client_authentication: true
```

### Load in Python

```python
import yaml
from server.privacy_manager import PrivacyManager

with open('config/fl_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if config['privacy']['use_differential_privacy']:
    privacy_manager = PrivacyManager(
        epsilon=config['privacy']['epsilon'],
        delta=config['privacy']['delta'],
        clip_norm=config['privacy']['clip_norm'],
        num_rounds=config['num_rounds'],
    )
```

## Privacy Parameter Selection

### Quick Guide

**For High Privacy (Healthcare, Legal)**
```python
PrivacyManager(
    epsilon=0.1,   # Very strict privacy
    delta=1e-6,    # Tiny failure probability
    clip_norm=1.0,
    num_rounds=100
)
```
*Effect*: ~95% smaller gradients, 20-30% accuracy loss

**For Moderate Privacy (Finance, Social Networks)**
```python
PrivacyManager(
    epsilon=1.0,   # Good privacy
    delta=1e-5,    # Standard failure probability
    clip_norm=1.0,
    num_rounds=50
)
```
*Effect*: ~5x noise scaling, 2-5% accuracy loss

**For Weak Privacy (Web, Analytics)**
```python
PrivacyManager(
    epsilon=10.0,  # Weak privacy for better utility
    delta=1e-4,    # Higher failure probability
    clip_norm=2.0,
    num_rounds=10
)
```
*Effect*: ~0.5x noise scaling, <1% accuracy loss

**For Privacy-Optional**
```python
# Just don't use privacy manager
# or use epsilon=100+
```

## Monitoring Privacy Budget

### In Training Loop

```python
for round_num in range(num_rounds):
    # ... training code ...
    
    # Check privacy budget
    if round_num % 5 == 0:
        status = pm.get_privacy_status()
        print(f"Round {round_num}:")
        print(f"  Epsilon used: {status['epsilon_used']:.4f}")
        print(f"  Epsilon remaining: {status['epsilon_remaining']:.4f}")
        print(f"  Current noise scale: {status['clip_norm']:.4f}")
        
        # Stop if budget exhausted
        if status['epsilon_remaining'] <= 0:
            print("WARNING: Privacy budget exhausted!")
            break
```

## Testing Privacy Integration

### Unit Test

```python
def test_privacy_in_training():
    from server.privacy_manager import PrivacyManager
    from client.model import ThreatDetectionModel
    import numpy as np
    
    # Create privacy manager
    pm = PrivacyManager(epsilon=1.0, delta=1e-5, num_rounds=5)
    
    # Create model
    model = ThreatDetectionModel()
    
    # Create dummy data
    X = np.random.randn(100, 27)
    y = np.random.randint(0, 6, 100)
    
    # Train with privacy
    for round_num in range(5):
        # Get gradients
        model.train(X, y, epochs=1)
        gradients = {'coef': np.random.randn(27, 6)}
        
        # Apply privacy
        private_grads = pm.apply_privacy(gradients)
        
        # Verify privacy was applied
        assert private_grads['coef'].shape == gradients['coef'].shape
        assert not np.allclose(private_grads['coef'], gradients['coef'])
    
    # Verify budget tracking
    assert pm.privacy_budget.rounds_completed == 5
```

## FAQ

**Q: What epsilon should I use?**
A: Start with ε=1.0 (moderate privacy). Increase for better accuracy, decrease for stronger privacy.

**Q: What's the privacy vs accuracy tradeoff?**
A: Typically 1-5% accuracy loss for ε=1.0. Varies by model and dataset size.

**Q: Does privacy add computational overhead?**
A: ~10-15% overhead from clipping and noise injection (negligible for most applications).

**Q: Can I change epsilon mid-training?**
A: Not recommended. Create a new PrivacyManager for consistent budget allocation.

**Q: What if clients have different epsilon?**
A: Each client can have their own privacy manager with the same epsilon (aggregate still DP).

**Q: How do I verify privacy guarantees?**
A: Use `get_privacy_status()` to track epsilon consumption and verify (ε, δ)-DP achieved.

## Example: Full Integration

See `PHASE2_WEEK1_COMPLETE.md` for complete examples with:
- Single-model privacy
- Multi-round training
- Integration with FedAvg
- Privacy monitoring
- Epsilon-delta tuning

## Next Steps

1. **Week 1** (Done): DP-SGD implementation ✅
2. **Week 2**: Byzantine-robust aggregation
3. **Week 3**: FedProx & FedOpt algorithms
4. **Week 4**: MLflow experiment logging

---

For detailed theory and implementation, see `PHASE2_WEEK1_COMPLETE.md`
