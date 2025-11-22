# Compression and Personalization in FedShield

This guide demonstrates how to use FedShield's compression and personalization features for efficient federated learning.

## Overview

FedShield now includes two key enhancements:

1. **Update Compression**: Reduce communication bandwidth using quantization and top-k sparsification
2. **Personalization**: Per-client fine-tuning from global model weights for local adaptation

## Compression

### How It Works

Compression uses two techniques:
- **Quantization**: Reduce precision (e.g., float32 → 8-bit integers)
- **Top-k Sparsification**: Keep only the top-k fraction of weights by absolute value

This combination significantly reduces model update size while maintaining accuracy.

### Configuration

Enable compression in your FL config:

```python
from server.federated_learning import FLConfig, FederatedServer, FederatedClient
from client.model import ThreatDetectionModel

# Create config with compression enabled
config = FLConfig(
    num_rounds=10,
    clients_per_round=3,
    local_epochs=5,
    compression_enabled=True,           # Enable compression
    compression_top_k=0.2,              # Keep top 20% of weights
    quantization_bits=8                 # Quantize to 8 bits
)

print(f"Compression config: bits={config.quantization_bits}, top_k={config.compression_top_k}")
```

### Client-Side Compression

When a client trains with compression enabled, model updates are automatically compressed:

```python
from server.federated_learning import FederatedClient, FLConfig
from client.model import ThreatDetectionModel
import numpy as np

# Setup client
model = ThreatDetectionModel()
model.build()
client = FederatedClient(client_id=0, local_data_size=100)
client.set_model(model)

# Create config with compression
config = FLConfig(compression_enabled=True, quantization_bits=8, compression_top_k=0.2)

# Train - weights are automatically compressed
X_train = np.random.randn(100, 27)
y_train = np.random.randint(0, 6, size=100)

weights, metrics = client.local_training(X_train, y_train, config)

# weights['coefs'] and weights['intercepts'] are now compressed dicts with 'quantized' key
print(f"Compressed coefs keys: {weights['coefs'].keys()}")  # {'quantized': [...], 'meta': [...]}
```

### Server-Side Decompression and Aggregation

The server automatically decompresses compressed client payloads before aggregation:

```python
from server.federated_learning import FederatedServer

# Server detects and decompresses compressed payloads automatically
server = FederatedServer(config, num_clients=4)

# client_updates contains compressed payloads
client_updates = {0: compressed_update_1, 1: compressed_update_2, ...}
num_samples = {0: 100, 1: 100, ...}

# aggregate_updates automatically decompresses before averaging
aggregated = server.aggregate_updates(client_updates, num_samples)

# aggregated weights are now decompressed and ready for the next round
print(f"Aggregated coefs shape: {aggregated['coefs'][0].shape}")
```

### Compression Benefits

```
Example: Model with 50K parameters

Uncompressed (float32):     50K × 4 bytes = 200 KB per client update
Compressed (8-bit + 20% sparsity): 50K × 0.2 × 1 byte = 10 KB per client update

Compression ratio: ~20x reduction in bandwidth per round
With 100 clients: 20 MB → 1 MB per round
Over 10 rounds: 200 MB → 10 MB total communication
```

## Personalization

### How It Works

Personalization allows each client to fine-tune the global model on their local data, adapting it to their specific distribution while maintaining global knowledge.

Process:
1. Client receives global model weights from server
2. Client fine-tunes locally using personalization data
3. Client returns personalized weights (or compresses before sending)

### Configuration

Personalization is controlled via `FedAvgOrchestrator` or manually in `FederatedClient`:

```python
from server.federated_learning import FLConfig, FederatedClient
from client.model import ThreatDetectionModel
import numpy as np

# Setup
config = FLConfig(local_epochs=5)
model = ThreatDetectionModel()
model.build()
client = FederatedClient(client_id=0, local_data_size=100)
client.set_model(model)

# Global training data
X_train = np.random.randn(100, 27)
y_train = np.random.randint(0, 6, size=100)

# Personalization data (smaller, local subset)
X_personal = X_train[:20]
y_personal = y_train[:20]

# Train locally first
weights, metrics = client.local_training(X_train, y_train, config)
print(f"Global training accuracy: {metrics['accuracy']:.4f}")

# Then personalize on local data
personalized_weights, p_metrics = client.personalize(X_personal, y_personal, config)
print(f"Personalization accuracy: {p_metrics['personalization_accuracy']:.4f}")
```

### Personalization + Compression

You can combine personalization and compression:

```python
from server.federated_learning import FLConfig
import numpy as np

# Config with both personalization and compression
config = FLConfig(
    local_epochs=5,
    compression_enabled=True,
    compression_top_k=0.2,
    quantization_bits=8
)

# Client trains on global data
weights, metrics = client.local_training(X_train, y_train, config)

# Client personalizes on local data
personalized_weights, p_metrics = client.personalize(X_personal, y_personal, config)

# personalized_weights contains compressed updates if compression_enabled=True
# Server will auto-decompress before aggregation
```

### Personalization Benefits

```
Scenario: Heterogeneous client data distributions

Without personalization:
- Global model: 85% accuracy across all clients
- Client A (different data): 72% (poor fit)
- Client B (different data): 78% (poor fit)

With personalization:
- Global model: 85% accuracy (maintained)
- Client A (after personalization): 92% (local adaptation)
- Client B (after personalization): 88% (local adaptation)

Trade-off: ~2x communication overhead but better per-client accuracy
```

## End-to-End Example

```python
from server.federated_learning import (
    FLConfig, FederatedServer, FedAvgOrchestrator, FederatedClient
)
from client.model import ThreatDetectionModel
import numpy as np

# Configuration with both compression and personalization
config = FLConfig(
    num_rounds=3,
    clients_per_round=2,
    local_epochs=5,
    compression_enabled=True,
    compression_top_k=0.2,
    quantization_bits=8
)

# Simulate 4 clients with local data
client_data = {}
for i in range(4):
    X = np.random.randn(100, 27)
    y = np.random.randint(0, 6, size=100)
    client_data[i] = (X, y)

# Initialize orchestrator
orchestrator = FedAvgOrchestrator(config, num_clients=4)
for cid in range(4):
    model = ThreatDetectionModel()
    model.build()
    orchestrator.clients[cid].set_model(model)

# Run federated learning rounds
for round_num in range(config.num_rounds):
    print(f"\n=== Round {round_num + 1} ===")
    
    # Simulate round (clients train, compress, send to server)
    num_samples = {cid: len(client_data[cid][0]) for cid in range(4)}
    round_summary = orchestrator.simulate_round(client_data, num_samples)
    
    print(f"Selected clients: {round_summary['selected_clients']}")
    print(f"Updates received: {round_summary['num_updates']}")
    print(f"Metrics: {round_summary['metrics']}")

print(f"\nTotal rounds completed: {orchestrator.server.current_round}")
print("All updates were compressed and decompressed automatically!")
```

## Testing

Unit tests for compression and personalization are in `tests/`:

- `tests/test_personalization_and_compression.py` - Personalization and compression roundtrip tests
- `tests/test_aggregation_with_compressed_payloads.py` - Server-side decompression test

Run tests:

```bash
python -m pytest tests/test_personalization_and_compression.py -v
python -m pytest tests/test_aggregation_with_compressed_payloads.py -v
```

## Performance Notes

1. **Compression overhead**: ~5-10% CPU time increase (quantization/sparsification)
2. **Communication savings**: ~20x bandwidth reduction typical
3. **Accuracy impact**: <1-2% for 8-bit quantization + 20% sparsity
4. **Personalization overhead**: ~2x communication (compressed personalized updates sent back)
5. **Personalization benefit**: 5-15% accuracy improvement on heterogeneous clients

## Advanced: Custom Compression Settings

Fine-tune compression for your use case:

```python
# Aggressive compression (smaller, faster, lower accuracy)
config_aggressive = FLConfig(
    compression_enabled=True,
    quantization_bits=4,           # 4-bit quantization
    compression_top_k=0.1          # Keep only top 10%
)

# Conservative compression (larger, slower, higher accuracy)
config_conservative = FLConfig(
    compression_enabled=True,
    quantization_bits=16,          # 16-bit quantization
    compression_top_k=0.5          # Keep top 50%
)

# Trade-off: Recommended for most cases
config_balanced = FLConfig(
    compression_enabled=True,
    quantization_bits=8,           # 8-bit quantization
    compression_top_k=0.2          # Keep top 20%
)
```

## References

- FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- Compression: Top-k sparsification + quantization (standard FL compression techniques)
- Personalization: Federated multi-task learning / per-client adaptation

---

**Questions?** Check the test files for more examples, or see the main README.md for overall FedShield documentation.
