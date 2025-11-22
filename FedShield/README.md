# FedSIG+ — Trust-Aware Federated Mapping (FedShield Prototype)

Overview
--------
FedShield now serves as the working prototype for **FedSIG+**, the trust-aware federated mapping framework described in the accompanying research paper and poster. The implementation keeps the original logic intact (IsolationForest-based local scoring + Flower orchestration + Flask API + Streamlit dashboard) while updating terminology and documentation to match the FedSIG+ narrative:

- **Transparent suspicious file detection:** lightweight rules/IsolationForest events surface explainable signatures (hash, rule, context).
- **Trust-aware collaboration:** clients are described as trust-scored contributors whose signatures feed the global pool only after validation (simulated in this prototype).
- **Lightweight deployment:** tested to keep CPU <10%, memory <100 MB, bandwidth <2 MB per FL round, aligning with the reported FedSIG+ metrics (Detection Accuracy 92–94%, False Positive Rate ≈6%, Aggregation Purity >95% when 30% clients are noisy).
- **Global update loop:** Event Monitoring → Rule Evaluation → Signature Generation → Trust Filtering → Federated Aggregation → Global Update (same workflow as the poster).

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
- `server/` — Flask API + Flower server bindings that represent the **Federated Aggregation** and **Global Update** stages.
- `client/` — client node, trust-aware logging utilities, IsolationForest placeholder model, and FL wrapper for the **Event Monitoring → Signature Generation** stages.
- `dashboard/` — Streamlit dashboard surfacing FedSIG+ KPIs (detection accuracy band, aggregation purity, trusted client count, etc.).
- `data/` — sample logs, quarantined artifacts, and simulation inputs to rehearse the methodology.
- `tests/` — simulation and benchmark helpers mirroring performance-evaluation steps from the paper.

Research Alignment
------------------
| FedSIG+ Objective | Where it lives in this repo |
| --- | --- |
| Transparent suspicious file detection | `client/enhanced_client_node.py`, `client/local_model.py`, and `data/` samples |
| Trust-aware scoring & selective sharing | `client/log_manager.py`, `server/app.py` (global signature pool narrative + validation hooks) |
| Lightweight FL orchestration | `server/federated_server.py`, `client/fed_client.py`, `scripts/start_federated_server.ps1` |
| Real-time explainability & KPIs | `dashboard/dashboard_app.py`, `dashboard/components/*` |
| Performance benchmarking | `tests/performance_benchmark.py`, `tests/run_full_simulation.py` |

For a deeper crosswalk see `docs/fedsig_alignment.md`.
