"""
Test that server aggregates correctly when clients send compressed payloads.
"""

import numpy as np
from server.federated_learning import FederatedServer, FLConfig
from utils.compression import compress_weights


def make_weights():
    return {
        'coefs': [np.random.randn(10, 5), np.random.randn(5, 2)],
        'intercepts': [np.random.randn(5), np.random.randn(2)],
        'scaler_mean': np.random.randn(27),
        'scaler_scale': np.abs(np.random.randn(27))
    }


def test_aggregate_decompresses_and_aggregates():
    server = FederatedServer(FLConfig(), num_clients=2)

    w1 = make_weights()
    w2 = make_weights()

    p1 = compress_weights(w1, bits=8, k_fraction=0.3)
    p2 = compress_weights(w2, bits=8, k_fraction=0.3)

    # Simulate client payloads where coefs/intercepts are compressed
    client_updates = {
        0: {'coefs': p1['coefs'], 'intercepts': p1['intercepts'], 'scaler_mean': p1['scaler_mean'], 'scaler_scale': p1['scaler_scale']},
        1: {'coefs': p2['coefs'], 'intercepts': p2['intercepts'], 'scaler_mean': p2['scaler_mean'], 'scaler_scale': p2['scaler_scale']}
    }

    num_samples = {0: 100, 1: 100}

    aggregated = server.aggregate_updates(client_updates, num_samples)

    # Ensure aggregated keys exist
    assert 'coefs' in aggregated and 'intercepts' in aggregated
    assert len(aggregated['coefs']) == len(w1['coefs'])
    assert aggregated['coefs'][0].shape == w1['coefs'][0].shape
