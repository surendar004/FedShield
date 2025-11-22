# FedShield Project Report

Abstract
--------
FedShield demonstrates privacy-preserving collaborative anomaly detection using federated learning. Clients train local IsolationForest models, report detected threats to a central Flask service, and optionally isolate suspicious files. A Streamlit dashboard visualizes metrics and alerts in real time.

Methods
-------
- Local detection: IsolationForest trained on synthetic system logs.
- Federated averaging using Flower: clients send model parameters, server aggregates to global model.
- Isolation: when a client flags a log as malicious, it moves the associated file to `data/quarantined`.

Results
-------
This prototype includes scripts to simulate threats and demonstrate end-to-end flow.

Limitations
-----------
- Demo uses simulated data; not production hardened.
- Proper security, authentication, and signed model updates are out of scope for this prototype.
