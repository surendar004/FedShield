# FedShield Setup and Run Script
# This script sets up the environment and runs all services

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "=== FedShield Setup and Run ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/9] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Trying 'py' launcher..." -ForegroundColor Yellow
    try {
        $pythonVersion = py --version 2>&1
        $pythonCmd = "py"
        Write-Host "Found: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Python not found. Please install Python 3.8+ and try again." -ForegroundColor Red
        exit 1
    }
}

if (-not $pythonCmd) {
    $pythonCmd = "python"
}

# Step 2: Check/Create Virtual Environment
Write-Host "[2/9] Checking virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $root "fedshield_env"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    & $pythonCmd -m venv $venvPath
    if (-not (Test-Path $venvPython)) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment found." -ForegroundColor Green
}

# Step 3: Upgrade pip
Write-Host "[3/9] Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "pip upgraded." -ForegroundColor Green

# Step 4: Install dependencies
Write-Host "[4/9] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
$requirementsFile = Join-Path $root "requirements.txt"
if (Test-Path $requirementsFile) {
    $pipLog = Join-Path $root "logs\pip_install.log"
    Write-Host "  (logging output to $pipLog)" -ForegroundColor Gray
    # Use direct invocation so both stdout and stderr can be redirected to the same file
    try {
        & $venvPython -m pip install -r $requirementsFile *> $pipLog
    } catch {
        Write-Warning "pip install produced warnings or failed; continuing. See $pipLog for details."
    }
    Write-Host "Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements.txt not found!" -ForegroundColor Red
    exit 1
}

# Step 5: Create required directories
Write-Host "[5/9] Creating required directories..." -ForegroundColor Yellow
$dirs = @(
    "logs",
    "server\models",
    "client\logs",
    "data\quarantined"
)

foreach ($dir in $dirs) {
    $fullPath = Join-Path $root $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}
Write-Host "Directories ready." -ForegroundColor Green

# Step 6: Check/Train local model
Write-Host "[6/9] Checking local model..." -ForegroundColor Yellow
$modelPath = Join-Path $root "client\local_model.pkl"
if (-not (Test-Path $modelPath)) {
    Write-Host "Model not found. Training model..." -ForegroundColor Yellow
    & $venvPython -c "import sys; sys.path.insert(0, r'$root'); from client.local_model import train_and_save; train_and_save()"
    Write-Host "Model trained and saved." -ForegroundColor Green
} else {
    Write-Host "Model exists." -ForegroundColor Green
}

# Step 7: Verify sample data
Write-Host "[7/9] Verifying sample data..." -ForegroundColor Yellow
$sampleData = Join-Path $root "data\sample_logs.csv"
if (Test-Path $sampleData) {
    Write-Host "Sample data found." -ForegroundColor Green
} else {
    Write-Host "WARNING: sample_logs.csv not found. Model training will generate it." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""

# Step 8: Start services
Write-Host "[8/9] Starting services..." -ForegroundColor Yellow

# Stop any existing processes first
Write-Host "  Stopping existing processes..." -ForegroundColor Gray
Get-Process | Where-Object { $_.Path -like "*fedshield_env*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Start Flask API server
Write-Host "  Starting Flask API server (port 5000)..." -ForegroundColor Gray
$flaskOut = Join-Path $root "logs\flask_server.out"
$flaskErr = Join-Path $root "logs\flask_server.err"
Start-Process -FilePath $venvPython -ArgumentList "server\app.py" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $flaskOut -RedirectStandardError $flaskErr
Start-Sleep -Seconds 2

# Start Flower federated server
Write-Host "  Starting Flower federated server (port 8080)..." -ForegroundColor Gray
$flowerOut = Join-Path $root "logs\flwr_server.out"
$flowerErr = Join-Path $root "logs\flwr_server.err"
Start-Process -FilePath $venvPython -ArgumentList "server\federated_server.py" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $flowerOut -RedirectStandardError $flowerErr
Start-Sleep -Seconds 2

# Start client nodes
Write-Host "  Starting client nodes..." -ForegroundColor Gray
for ($i = 1; $i -le 2; $i++) {
    $clientId = "client$i"
    $clientOut = Join-Path $root "logs\client_$i.out"
    $clientErr = Join-Path $root "logs\client_$i.err"
    Start-Process -FilePath $venvPython -ArgumentList "client\client_node.py", "--id", $clientId -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $clientOut -RedirectStandardError $clientErr
    Write-Host "    Started $clientId" -ForegroundColor Gray
    Start-Sleep -Seconds 1
}

# Step 9: Start Streamlit dashboard
Write-Host "[9/9] Starting Streamlit dashboard (port 8501)..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
Start-Process -FilePath $venvPython -ArgumentList "-m", "streamlit", "run", "dashboard\dashboard_app.py", "--server.port", "8501" -WorkingDirectory $root

Write-Host ""
Write-Host "=== All Services Started ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services are running:" -ForegroundColor Green
Write-Host "  - Flask API:      http://localhost:5000/api" -ForegroundColor White
Write-Host "  - Flower Server:  http://localhost:8080" -ForegroundColor White
Write-Host "  - Dashboard:      http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Logs are available in the 'logs' directory." -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Yellow

