<#
PowerShell helper to start backend services and the Streamlit dashboard for FedShield.
Usage:
  ./scripts/start_all.ps1            # start Flask + Streamlit
  ./scripts/start_all.ps1 -WithFLWR  # also start federated server (Flower)

Behavior:
- Uses the project's virtualenv Python (`.venv\Scripts\python.exe`) and Streamlit (`.venv\Scripts\streamlit.exe`).
- Writes stdout/stderr to `logs/*.out` and `logs/*.err` files.
- Waits for `http://127.0.0.1:5000/api/health` to return 200 before starting Streamlit.
#>
param(
    [switch]$WithFLWR
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
# Project root is the parent of the scripts directory
$root = Split-Path -Parent $scriptDir
Set-Location $root

# Ensure logs directory exists
$logDir = Join-Path $root 'logs'
if (-not (Test-Path $logDir)) { New-Item -Path $logDir -ItemType Directory | Out-Null }

$python = Join-Path $root '.venv\Scripts\python.exe'
$streamlit = Join-Path $root '.venv\Scripts\streamlit.exe'

if (-not (Test-Path $python)) { Write-Error "Python executable not found at $python. Activate your venv or create .venv."; exit 1 }
if (-not (Test-Path $streamlit)) { Write-Warning "Streamlit executable not found at $streamlit. Streamlit may not be installed in the venv." }

# Helper to stop processes listening on a port
function Stop-ByPort {
    param([int]$Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        if ($connections) {
            $owningPids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
            foreach ($procId in $owningPids) {
                Write-Output "Stopping process $procId listening on port $Port"
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Verbose "Stop-ByPort: $_"
    }
}

# Stop previous server instances if any
Stop-ByPort -Port 5000
Stop-ByPort -Port 8501

# Start Flask API
$flaskOut = Join-Path $logDir 'flask_server.out'
$flaskErr = Join-Path $logDir 'flask_server.err'
Write-Output "Starting Flask API (stdout -> $flaskOut, stderr -> $flaskErr)"
Start-Process -FilePath $python -ArgumentList 'server\app.py' -WindowStyle Hidden -RedirectStandardOutput $flaskOut -RedirectStandardError $flaskErr

# Optionally start federated server
if ($WithFLWR) {
    $flwrOut = Join-Path $logDir 'flwr_server.out'
    $flwrErr = Join-Path $logDir 'flwr_server.err'
    Write-Output "Starting federated server (stdout -> $flwrOut, stderr -> $flwrErr)"
    Start-Process -FilePath $python -ArgumentList 'server\federated_server.py' -WindowStyle Hidden -RedirectStandardOutput $flwrOut -RedirectStandardError $flwrErr
}

# Wait for Flask API to be healthy
$health = 'http://127.0.0.1:5000/api/health'
$maxAttempts = 30
$attempt = 0
while ($attempt -lt $maxAttempts) {
    try {
        $resp = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { Write-Output "API healthy after $attempt attempts."; break }
    } catch {
        Start-Sleep -Seconds 1
        $attempt++
    }
}
if ($attempt -ge $maxAttempts) {
    Write-Warning "API did not become healthy after $maxAttempts seconds. Check logs: $flaskErr"
} else {
    # Start Streamlit dashboard once API is up (or we've tried)
    $streamOut = Join-Path $logDir 'streamlit.out'
    $streamErr = Join-Path $logDir 'streamlit.err'
    Write-Output "Starting Streamlit dashboard (stdout -> $streamOut, stderr -> $streamErr)"
    Start-Process -FilePath $streamlit -ArgumentList 'run','dashboard\dashboard_app.py','--server.address','127.0.0.1','--server.port','8501' -WindowStyle Hidden -RedirectStandardOutput $streamOut -RedirectStandardError $streamErr
    Write-Output "Streamlit started. Visit http://127.0.0.1:8501"
}

Write-Output "Done. Logs available under: $logDir"
