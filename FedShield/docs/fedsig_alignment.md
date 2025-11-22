# FedSIG+ Alignment Crosswalk

This document maps the research paper/poster narrative to the existing FedShield prototype so contributors can extend the project without altering core behaviour.

## Objectives → Implementation

| FedSIG+ Objective | Repository Anchor | Notes |
| --- | --- | --- |
| Transparent suspicious file detection | `client/enhanced_client_node.py`, `client/local_model.py` | IsolationForest + deterministic rules simulate explainable scoring; comments now describe signature makeup (hash, violated rule, timestamp). |
| Trust-aware intelligence sharing | `client/log_manager.py`, `server/app.py`, `utils/validators.py` | Payload sanitization, trust-score placeholders, and selective logging match the “validated contribution” narrative. |
| Lightweight scalable FL | `client/fed_client.py`, `server/federated_server.py`, `scripts/start_federated_server.ps1` | Flower FedAvg orchestrates sharing without raw data; docstrings explain its role as the aggregation layer referenced in the paper. |
| Global signature pool & dashboard | `server/global_logs.json`, `dashboard/dashboard_app.py`, `dashboard/components/*` | The dashboard displays aggregation purity, trusted clients, and isolation counts consistent with poster KPIs. |
| Performance evaluation & benchmarking | `tests/performance_benchmark.py`, `tests/run_full_simulation.py` | Scripts emulate noisy clients and log CPU/memory usage to replicate the reported metrics (Accuracy 92–94%, False Positive Rate ≈6%, Aggregation Purity >95%). |

## Workflow Trace (Poster Methodology)

1. **Event Monitoring** – `client/enhanced_client_node.py` simulates file/process telemetry per threat profile.
2. **Rule Evaluation** – `client/local_model.py` and the threat profile thresholds translate telemetry into anomaly scores.
3. **Signature Generation** – `client/log_manager.py` persists signatures locally before attempting remote submission.
4. **Trust Filtering** – inline comments describe the trust-score gating; `utils/validators.py` enforces structural checks to mimic this stage.
5. **Federated Aggregation** – `server/federated_server.py` and `client/fed_client.py` run Flower FedAvg rounds.
6. **Global Update** – `server/app.py` stores validated signatures and `dashboard/dashboard_app.py` visualizes them in Streamlit.

## Terminology Updates

- “Threat report” → “Signature contribution” in documentation and UI copy (schemas retain original field names for backward compatibility).
- “Global logs” → “Global signature pool” when describing outputs.
- “Clients” → “Trust-scored contributors” in narrative contexts to highlight their role in FedSIG+.

## Future Enhancements (per paper)

The following backlog items capture the “Future Enhancements” section of the paper:

- Integrate unsupervised anomaly detection to complement deterministic rules.
- Apply homomorphic encryption or secure aggregation to the Flower exchange.
- Add cross-organizational federation support with differential aggregation for confidentiality.
- Automate rule synthesis and multi-source trust validation to detect compromised high-trust clients faster.

Keep these themes in mind when proposing new features to ensure the prototype stays aligned with the research roadmap.

