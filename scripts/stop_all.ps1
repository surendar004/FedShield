#!/usr/bin/env pwsh
# Master cleanup script that combines graceful shutdown with forceful cleanup

$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Starting FedShield cleanup process..."

# First try graceful shutdown using Python script
Write-Host "`nAttempting graceful shutdown..."
try {
    python "$scriptPath\graceful_shutdown.py"
} catch {
    Write-Host "Warning: Graceful shutdown script failed: $_"
}

# Wait a moment for graceful shutdowns to complete
Start-Sleep -Seconds 2

# Then run the forceful cleanup
Write-Host "`nRunning final cleanup..."
try {
    & "$scriptPath\cleanup.ps1"
} catch {
    Write-Host "Error during cleanup: $_"
    exit 1
}

Write-Host "`nCleanup completed successfully!"