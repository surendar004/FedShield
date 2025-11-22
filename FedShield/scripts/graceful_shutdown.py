"""Helper script for graceful process shutdown.
This script attempts to gracefully stop processes by sending shutdown signals
before they are forcefully terminated by the cleanup.ps1 script.
"""
import os
import sys
import signal
import requests
import time
import psutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.absolute()

def shutdown_flask_server():
    """Attempt to gracefully shutdown Flask server"""
    try:
        requests.post('http://localhost:5000/shutdown')
        print("Sent shutdown signal to Flask server")
    except requests.exceptions.ConnectionError:
        print("Flask server not running")

def shutdown_streamlit():
    """Attempt to gracefully shutdown Streamlit dashboard"""
    try:
        requests.get('http://localhost:8501/_stcore/health', timeout=0.1)
        print("Streamlit appears to be running, will be terminated by PowerShell script")
    except requests.exceptions.ConnectionError:
        print("Streamlit dashboard not running")

def find_python_processes():
    """Find all Python processes related to FedShield"""
    fedshield_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline'])
                if any(x in cmdline for x in ['flask', 'streamlit', 'client_node.py', 'fed_client.py']):
                    fedshield_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return fedshield_processes

def main():
    print("Starting graceful shutdown...")
    
    # Try graceful shutdown of services
    shutdown_flask_server()
    shutdown_streamlit()
    
    # Give services a moment to shutdown gracefully
    time.sleep(2)
    
    # Get list of processes still running
    processes = find_python_processes()
    if processes:
        print(f"\nFound {len(processes)} FedShield processes still running:")
        for proc in processes:
            print(f"PID {proc.pid}: {' '.join(proc.cmdline())}")
    else:
        print("\nNo FedShield processes found running")

if __name__ == '__main__':
    main()