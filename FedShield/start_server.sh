#!/usr/bin/env bash
echo "Starting Flask API server..."
python "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/server/app.py" &
sleep 1
echo "Starting Flower federated server (in-process)..."
python "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/server/federated_server.py"

echo "Server startup script finished."
