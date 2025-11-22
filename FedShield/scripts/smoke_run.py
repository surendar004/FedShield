"""Quick smoke runner: loads model and processes a few rows locally without network calls.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from client.local_model import load_local_model
import pandas as pd

DATA_CSV = os.path.join(ROOT, 'data', 'sample_logs.csv')


def main(rows=5):
    model = load_local_model()
    df = pd.read_csv(DATA_CSV)
    for i, row in df.head(rows).iterrows():
        # Values from iterrows are already Python scalars; no .item() needed
        features = [[
            float(row['cpu_pct']),
            float(row['net_bytes']),
            int(row['file_access_count']),
        ]]
        pred = model.predict(features)[0]
        is_threat = pred == -1
        print(f"Row {i}: cpu={row['cpu_pct']} net_bytes={row['net_bytes']} file_access={row['file_access_count']} threat={is_threat}")


if __name__ == '__main__':
    main()
