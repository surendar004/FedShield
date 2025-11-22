"""Launch helper to run the full FedShield demo (Flask API, Flower, clients, Streamlit)

Usage (PowerShell):
    Set-Location "<path-to-repo>\\FedShield"
    python .\\lunch.py    # runs using .venv if present, else creates it

This script is best run from Windows PowerShell. It will:
 - ensure a virtualenv at `./.venv` (attempts `py -3.12` then `python`)
 - install requirements into the venv
 - train the local model if missing
 - start Flask server, Flower server, 2 clients, and Streamlit dashboard
 - write logs into `./logs` and PIDs into `./logs/pids.json`

Note: Starting all services will run background processes. Press Ctrl+C in this
terminal to stop them and perform cleanup.
"""
from __future__ import annotations
import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).parent
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
PIDS_FILE = LOG_DIR / 'pids.json'


def find_python_executable() -> str:
    """Return a python executable to create/use the venv.
    Prefer `py -3.12` if available, else fallback to `python` on PATH.
    """
    # prefer launcher
    try:
        out = subprocess.check_output(['py', '-3.12', '--version'], stderr=subprocess.STDOUT)
        return 'py -3.12'
    except Exception:
        pass
    # fallback to 'python'
    return shutil.which('python') or 'python'


def venv_python(venv_path: Path) -> str:
    exe = venv_path / 'Scripts' / 'python.exe'
    return str(exe)


def run_cmd_background(cmd, cwd=None, logpath: Path | None = None):
    log = None
    if logpath:
        log = open(logpath, 'ab')
    # On Windows, run in same console but backgrounded; Popen is fine.
    proc = subprocess.Popen(cmd, cwd=cwd or str(ROOT), stdout=log or subprocess.PIPE, stderr=log or subprocess.PIPE, shell=isinstance(cmd, str))
    return proc


def ensure_venv(venv_dir: Path):
    if venv_dir.exists():
        print('Using existing venv:', venv_dir)
        return
    python = find_python_executable()
    print('Creating venv with', python)
    if python == 'py -3.12':
        subprocess.check_call(['py', '-3.12', '-m', 'venv', str(venv_dir)])
    else:
        subprocess.check_call([python, '-m', 'venv', str(venv_dir)])


def pip_install(venv_py: str, requirements: str = 'requirements.txt'):
    print('Upgrading pip and installing requirements...')
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    subprocess.check_call([venv_py, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    subprocess.check_call([venv_py, '-m', 'pip', 'install', '-r', req_path])


def train_model(venv_py: str):
    # train using the project's module so paths are resolved correctly
    print('Checking/training local model...')
    subprocess.check_call([venv_py, '-c', "import sys; sys.path.insert(0,r'{}'); from client.local_model import train_and_save; train_and_save()".format(str(ROOT).replace('\\','\\\\'))])


def start_services(venv_py: str):
    procs = {}

    # Flask API
    flask_out = LOG_DIR / 'flask_server.out'
    flask_err = LOG_DIR / 'flask_server.err'
    print('Starting Flask API ->', flask_out)
    p = run_cmd_background([venv_py, str(ROOT / 'server' / 'app.py')], cwd=str(ROOT), logpath=flask_out)
    procs['flask'] = p
    time.sleep(1)

    # Flower server
    flower_out = LOG_DIR / 'flwr_server.out'
    flower_err = LOG_DIR / 'flwr_server.err'
    print('Starting Flower server ->', flower_out)
    p = run_cmd_background([venv_py, str(ROOT / 'server' / 'federated_server.py')], cwd=str(ROOT), logpath=flower_out)
    procs['flower'] = p
    time.sleep(1)

    # Use threat types that match the API schema (deterministic round-robin)
    threat_types = ['malware', 'unauthorized_access', 'data_leak', 'phishing', 'system_anomaly']
    procs['clients'] = []
    # Reduce initial concurrency to avoid MemoryError under heavy load
    initial = 10
    base_threat = 0.01
    threat_increment = 0.02
    for i in range(1, initial + 1):
        client_out = LOG_DIR / f'client_{i}.out'
        client_id = f'client{i}'
        threat_rate = base_threat + (i - 1) * threat_increment
        # Assign threat type round-robin to ensure variety in dashboard
        threat_type = threat_types[(i - 1) % len(threat_types)]
        print('Starting', client_id, f'(threat_rate={threat_rate:.2f}, threat_type={threat_type}) ->', client_out)
        env = os.environ.copy()
        env['CLIENT_THREAT_RATE'] = str(threat_rate)
        env['CLIENT_THREAT_TYPE'] = threat_type
        # Increase per-client sleep to reduce request rate
        env['CLIENT_SLEEP'] = '3.0'
        # Do not load sklearn-backed model in each client to avoid memory pressure
        env['CLIENT_USE_MODEL'] = 'false'
        p = subprocess.Popen([venv_py, str(ROOT / 'client' / 'client_node.py'), '--id', client_id], cwd=str(ROOT), stdout=open(client_out, 'ab'), stderr=open(client_out, 'ab'), env=env)
        procs['clients'].append(p)
        # Stagger client starts to avoid burst load
        time.sleep(1.0)

    # Streamlit dashboard
    stream_out = LOG_DIR / 'streamlit.out'
    stream_err = LOG_DIR / 'streamlit.err'
    print('Starting Streamlit dashboard ->', stream_out)
    p = run_cmd_background([venv_py, '-m', 'streamlit', 'run', str(ROOT / 'dashboard' / 'dashboard_app.py'), '--server.port', '8501'], cwd=str(ROOT), logpath=stream_out)
    procs['streamlit'] = p

    # record pids
    pid_map = {k: (v.pid if isinstance(v, subprocess.Popen) else [x.pid for x in v]) for k, v in ((k, v) for k, v in procs.items())}
    with open(PIDS_FILE, 'w', encoding='utf8') as f:
        json.dump(pid_map, f, indent=2)
    print('Wrote PIDs to', PIDS_FILE)
    return procs


def scheduled_client_adder(venv_py: str, procs: dict, max_clients: int = 40, interval_seconds: int = 120):
    """Spawn an additional client every interval_seconds, increasing its threat rate.

    This runs in the background (thread-like) by the main loop in lunch.py.
    """
    import threading

    def worker():
        # use same threat types as the API schema to avoid validation errors
        threat_types = ['malware', 'unauthorized_access', 'data_leak', 'phishing', 'system_anomaly']
        idx = len(procs.get('clients', []))
        base_threat = 0.01
        threat_increment = 0.02
        while idx < max_clients:
            idx += 1
            client_id = f'client{idx}'
            threat_rate = base_threat + (idx - 1) * threat_increment
            # round-robin assignment so dashboard shows a mix of types
            threat_type = threat_types[(idx - 1) % len(threat_types)]
            client_out = LOG_DIR / f'client_{idx}.out'
            print(f'[scheduler] Adding {client_id} (threat_rate={threat_rate:.2f}, threat_type={threat_type})')
            env = os.environ.copy()
            env['CLIENT_THREAT_RATE'] = str(threat_rate)
            env['CLIENT_THREAT_TYPE'] = threat_type
            env['CLIENT_SLEEP'] = '2.0'
            env['CLIENT_USE_MODEL'] = 'false'
            p = subprocess.Popen([venv_py, str(ROOT / 'client' / 'client_node.py'), '--id', client_id], cwd=str(ROOT), stdout=open(client_out, 'ab'), stderr=open(client_out, 'ab'), env=env)
            procs.setdefault('clients', []).append(p)
            # update pid file
            try:
                with open(PIDS_FILE, 'w', encoding='utf8') as f:
                    json.dump({k: (v.pid if isinstance(v, subprocess.Popen) else [x.pid for x in v]) for k, v in procs.items()}, f, indent=2)
            except Exception:
                pass
            time.sleep(interval_seconds)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def stop_procs(procs: dict):
    print('Stopping processes...')
    for k, v in procs.items():
        if isinstance(v, list):
            for p in v:
                try:
                    p.terminate()
                except Exception:
                    pass
        else:
            try:
                v.terminate()
            except Exception:
                pass


def main():
    venv_dir = ROOT / '.venv'
    ensure_venv(venv_dir)
    venv_py = venv_python(venv_dir)
    if not Path(venv_py).exists():
        print('ERROR: venv python not found at', venv_py)
        sys.exit(1)

    pip_install(venv_py)
    train_model(venv_py)

    procs = start_services(venv_py)
    # start background scheduler to add clients every 2 minutes
    # start background scheduler to add clients every 2 minutes (120s)
    # Add one client every 3 minutes to scale slowly
    scheduled_client_adder(venv_py, procs, max_clients=40, interval_seconds=180)

    print('\nServices started:')
    print('  - Flask:     http://localhost:5000/api')
    print('  - Flower:    http://localhost:8080')
    print('  - Dashboard: http://localhost:8501')
    print('\nPress Ctrl+C here to stop all services.')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_procs(procs)
        if PIDS_FILE.exists():
            PIDS_FILE.unlink()
        print('\nServices stopped.')


if __name__ == '__main__':
    main()
