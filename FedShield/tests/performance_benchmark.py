"""Simple benchmark measuring detection and reporting latency for the client loop."""
import time
import requests
import subprocess
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def measure():
    # Ensure server is running
    try:
        r = requests.get('http://localhost:5000/api/health', timeout=3)
        if r.status_code != 200:
            print('Server not healthy')
            return
    except Exception:
        print('Server not available')
        return

    # Trigger client to run one pass and measure total time
    t0 = time.time()
    subprocess.run(['python', os.path.join(ROOT, 'client', 'client_node.py'), '--id', 'bench_client'])
    t1 = time.time()
    print('Client run duration (s):', t1 - t0)


if __name__ == '__main__':
    measure()
