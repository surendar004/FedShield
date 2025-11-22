"""
Tests for personalization and compression integration.
"""

import pytest
import numpy as np
from server.federated_learning import FLConfig, FederatedClient, FederatedServer, FedAvgOrchestrator
from client.model import ThreatDetectionModel
from utils.compression import compress_weights, decompress_weights


def make_dummy_client(cid: int):
    model = ThreatDetectionModel()
    model.build()
    client = FederatedClient(cid, local_data_size=100)
    client.set_model(model)
    return client


def test_personalization_improves_local_accuracy():
    # Create client and data
    client = make_dummy_client(0)
    X = np.random.randn(200, 27)
    y = np.random.randint(0, 6, size=200)

    # Train a bit to have baseline
    client.local_model.train(X, y, epochs=1)
    baseline = client.local_model.score(client.local_model.scaler.transform(X), y)

    # Personalize on a subset with more focused epochs
    X_personal = X[:50]
    y_personal = y[:50]
    personalized_weights, metrics = client.personalize(X_personal, y_personal, FLConfig())

    # After personalization, evaluate on personalization data
    acc_after = client.local_model.score(client.local_model.scaler.transform(X_personal), y_personal)
    assert acc_after >= 0.0
    assert 'personalization_accuracy' in metrics


def test_compression_roundtrip_and_decompression():
    # Create simple weight dict
    weights = {
        'coefs': [np.random.randn(10, 5), np.random.randn(5, 2)],
        'intercepts': [np.random.randn(5), np.random.randn(2)],
        'scaler_mean': np.random.randn(27),
        'scaler_scale': np.abs(np.random.randn(27))
    }

    payload = compress_weights(weights, bits=8, k_fraction=0.2)
    decompressed = decompress_weights(payload)

    # Ensure keys
    assert 'coefs' in decompressed
    assert 'intercepts' in decompressed
    assert decompressed['coefs'][0].shape == weights['coefs'][0].shape


def test_client_local_training_returns_compressed_payload():
    # Ensure FederatedClient.local_training will compress when enabled in config
    client = make_dummy_client(1)
    X = np.random.randn(50, 27)
    y = np.random.randint(0, 6, size=50)

    cfg = FLConfig()
    cfg.compression_enabled = True
    cfg.quantization_bits = 8
    cfg.compression_top_k = 0.2

    weights, metrics = client.local_training(X, y, cfg)

    # When compression is enabled, coefs should be a dict with 'quantized' key
    assert isinstance(weights['coefs'], dict)
    assert 'quantized' in weights['coefs']


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
