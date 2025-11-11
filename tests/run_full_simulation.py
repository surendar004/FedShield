"""Run a simple end-to-end simulation: start Flask server, train local model, run clients, then print summary."""
import subprocess
import time
import requests
import os
import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def start_flask():
    print('Starting Flask server...')
    p = subprocess.Popen(['python', os.path.join(ROOT, 'server', 'app.py')])
    return p


def start_client(client_id):
    return subprocess.Popen(['python', os.path.join(ROOT, 'client', 'client_node.py'), '--id', client_id])


def main():
    s = start_flask()
    time.sleep(1)
    # ensure local model exists
    subprocess.run(['python', os.path.join(ROOT, 'client', 'local_model.py')])

    clients = [start_client('clientA'), start_client('clientB')]
    # wait for clients to complete (they stream finite CSV rows)
    for c in clients:
        c.wait()

    # fetch threats
    try:
        r = requests.get('http://localhost:5000/api/threats', timeout=5)
        print('Threats:', r.json())
    except Exception as e:
        print('Failed to fetch threats:', e)

    # cleanup
    try:
        os.kill(s.pid, signal.SIGINT)
    except Exception:
        pass


if __name__ == '__main__':
    main()
