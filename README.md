# FedShield — Federated Threat Isolation & Detection System

Overview
--------
FedShield is a prototype demonstrating federated anomaly detection and automated isolation with a real-time dashboard. It uses local IsolationForest models on clients, Flower for federated learning, a Flask backend for reporting, and a Streamlit dashboard for visualization.

Quick start
-----------
1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/requirements.txt"
```

Windows (recommended) quick start
-------------------------------
If you're on Windows PowerShell, we validated this project using Python 3.12. To reproduce the working environment:

```powershell
# Create a dedicated venv (Python 3.12 recommended)
py -3.12 -m venv C:\Users\K.Pavithra\OneDrive\Desktop\vscode\FedShield\fedshield_env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\Users\K.Pavithra\OneDrive\Desktop\vscode\FedShield\fedshield_env\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Run the quick smoke test (no network calls):

```powershell
python scripts\smoke_run.py
```

2. Run the demo script (Linux/macOS/WSL or Git Bash recommended for shell scripts):

```bash
bash c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/start_demo.sh
```

3. Open dashboard: http://localhost:8501

4. API endpoint: http://localhost:5000/api/threats

Note: This is a prototype; run in an environment where running subprocesses and network sockets is permitted.

Files
-----
- `server/` — Flask app and federated server
- `client/` — client model, federated client, isolation and logging
- `dashboard/` — Streamlit dashboard and components
- `data/` — sample logs and quarantined folder
- `tests/` — simulation and benchmarks
