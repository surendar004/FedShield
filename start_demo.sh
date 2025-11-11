#!/usr/bin/env bash
# Demo runner: starts Flask + Flower + 2 clients + dashboard (best run in Git Bash / WSL)
echo "Running start_demo: Flask + Flower + 2 clients + Streamlit"

bash "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/start_server.sh" &
sleep 2
bash "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/start_clients.sh" &
sleep 2
bash "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/start_dashboard.sh" &

echo "Demo started."
