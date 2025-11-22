"""Simple Flower federated server that waits for clients and performs aggregation.
This demo script expects clients to connect to the default Flower port.
"""
import os
import json
from pathlib import Path
import joblib

try:
    import flwr as fl
except Exception:
    fl = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, 'global_model.pkl')
GLOBAL_INFO_PATH = os.path.join(MODEL_DIR, 'global_model_info.json')

os.makedirs(MODEL_DIR, exist_ok=True)


def save_global_model(dummy_model):
    # For this prototype we just save a placeholder dict
    joblib.dump(dummy_model, GLOBAL_MODEL_PATH)
    with open(GLOBAL_INFO_PATH, 'w') as f:
        json.dump({'saved_at': str(Path(GLOBAL_MODEL_PATH).stat().st_mtime), 'path': GLOBAL_MODEL_PATH}, f)


def start_server(rounds: int = 2):
    if fl is None:
        print('Flower (flwr) is not installed. Install with pip install flwr')
        # Save placeholder model info
        save_global_model({'note': 'flwr not installed - placeholder global model'})
        return

    def fit_config(rnd: int):
        return {"rnd": rnd}

    strategy = fl.server.strategy.FedAvg(min_available_clients=1)

    print('Starting Flower server...')
    fl.server.start_server(server_address='0.0.0.0:8080', config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy)

    # After training ends, save placeholder global model (in a real run you'd extract and save weights)
    save_global_model({'note': 'global model produced after flwr run'})


if __name__ == '__main__':
    start_server()
