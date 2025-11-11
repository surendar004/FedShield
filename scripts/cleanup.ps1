#!/usr/bin/env pwsh
# Cleanup script to gracefully stop all FedShield processes

# Define the processes to look for
$processNames = @(
    "python"  # Python processes running Flask, Streamlit, clients
)

# Define the ports used by services
$ports = @(
    5000,  # Flask API server
    8501,  # Streamlit dashboard
    8080   # Flower server
)

Write-Host "Stopping FedShield processes..."

# Function to stop processes by port
function Stop-ProcessByPort {
    param (
        [int]$port
    )
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connection) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    } else {
        $process = $null
    }
    if ($process) {
        Write-Host "Stopping process on port $port (PID: $($process.Id))"
        Stop-Process -Id $process.Id -Force
    }
}

# Stop processes by port first (more targeted)
foreach ($port in $ports) {
    Stop-ProcessByPort -port $port
}

# Then look for any remaining Python processes with specific command lines
$pythonProcesses = Get-WmiObject Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and (
        $_.CommandLine -like "*flask*" -or 
        $_.CommandLine -like "*streamlit*" -or
        $_.CommandLine -like "*client_node.py*" -or
        $_.CommandLine -like "*fed_client.py*"
    )
}

foreach ($process in $pythonProcesses) {
    Write-Host "Stopping Python process: $($process.ProcessId) - $($process.CommandLine)"
    Stop-Process -Id $process.ProcessId -Force
}

Write-Host "Cleanup complete!"

# Optional: Verify no processes are still running on target ports
$remainingConnections = $ports | ForEach-Object {
    Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue
}

if ($remainingConnections) {
    Write-Host "`nWarning: Some ports are still in use:"
    $remainingConnections | ForEach-Object {
        Write-Host "Port $($_.LocalPort) is still in use by process $($_.OwningProcess)"
    }
} else {
    Write-Host "`nAll target ports are free."
}