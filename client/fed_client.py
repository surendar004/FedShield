"""Flower NumPyClient wrapper that can send local model parameters.
Note: IsolationForest isn't a neural network; we will send/get parameters in a simple serialized form for the prototype.
"""
import os
try:
    import joblib
except Exception:
    try:
        # Some environments expose joblib via sklearn.externals (older scikit-learn)
        from sklearn.externals import joblib  # type: ignore
    except Exception:
        # Minimal fallback using pickle to avoid hard dependency for prototyping.
        # This provides load/dump methods used in this file.
        import pickle as _pickle
        class _JoblibFallback:
            @staticmethod
            def load(path):
                with open(path, "rb") as f:
                    return _pickle.load(f)
            @staticmethod
            def dump(obj, path):
                with open(path, "wb") as f:
                    return _pickle.dump(obj, f)
        joblib = _JoblibFallback()

import numpy as np
from typing import Dict, Tuple

try:
    import flwr as fl
    from flwr.common import Scalar
except Exception:
    fl = None
    Scalar = float  # fallback for type hints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'local_model.pkl')


class FedClient:
    def __init__(self, client_id: str):
        self.client_id = client_id

    def get_parameters(self):
        # For a sklearn model, return a serialized representation (placeholder)
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return [np.array([1])]  # placeholder
        return [np.array([0])]

    def fit(self, parameters, config):
        # In a real implementation you'd update local model weights here
        # Return updated parameters, number of training examples, and optional metrics
        updated = self.get_parameters()
        num_examples = int(config.get("num_examples", 1)) if isinstance(config, dict) else 1
        return updated, num_examples, {}

    def evaluate(self, parameters, config):
        # Return dummy loss/metrics
        num_examples = int(config.get("num_examples", 1)) if isinstance(config, dict) else 1
        return 0.1, num_examples, {"accuracy": Scalar(0.9)}


def start_flwr_client(client_id: str):
    if fl is None:
        print('flwr not installed; skipping federated client start')
        return

    class NumPyClient(fl.client.NumPyClient):
        def __init__(self, fc: FedClient):
            self.fc = fc

        def get_parameters(self, config=None):
            return self.fc.get_parameters()

        def fit(self, parameters, config):
            return self.fc.fit(parameters, config)
        def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
            loss, num_examples, metrics = self.fc.evaluate(parameters, config)
            # Convert float metrics to Scalar type to match the base class signature
            scalar_metrics = {k: float(v) for k, v in metrics.items()}
            return loss, num_examples, scalar_metrics

    fl.client.start_numpy_client(server_address='localhost:8080', client=NumPyClient(FedClient(client_id)))
