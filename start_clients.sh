#!/usr/bin/env bash
echo "Starting two client nodes..."
python "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/client/client_node.py" --id client1 &
python "c:/Users/K.Pavithra/OneDrive/Desktop/vscode/FedShield/client/client_node.py" --id client2 &
echo "Clients launched." 
