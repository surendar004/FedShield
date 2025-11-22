# Git Setup and Push Script for FedShield Project
# Run this script after installing Git

Write-Host "=== FedShield GitHub Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git first:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "2. Or install via winget: winget install --id Git.Git -e --source winget" -ForegroundColor Yellow
    Write-Host "3. Restart PowerShell after installation" -ForegroundColor Yellow
    exit 1
}

# Navigate to project root
$projectRoot = "D:\WEB FED\CAPSTONE"
Set-Location $projectRoot

Write-Host ""
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Check if already a git repository
if (Test-Path .git) {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}

# Check if remote exists
$remoteUrl = git remote get-url origin 2>$null
if ($remoteUrl) {
    Write-Host "✓ Remote 'origin' already exists: $remoteUrl" -ForegroundColor Green
    $updateRemote = Read-Host "Do you want to update it to https://github.com/surendar004/FedShield.git? (y/n)"
    if ($updateRemote -eq 'y' -or $updateRemote -eq 'Y') {
        git remote set-url origin https://github.com/surendar004/FedShield.git
        Write-Host "✓ Remote updated" -ForegroundColor Green
    }
} else {
    Write-Host "Adding remote 'origin'..." -ForegroundColor Yellow
    git remote add origin https://github.com/surendar004/FedShield.git
    Write-Host "✓ Remote added" -ForegroundColor Green
}

# Check git status
Write-Host ""
Write-Host "Checking repository status..." -ForegroundColor Cyan
git status

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review the files to be committed (above)" -ForegroundColor Yellow
Write-Host "2. Add files: git add ." -ForegroundColor Yellow
Write-Host "3. Commit: git commit -m 'Initial commit: FedShield project'" -ForegroundColor Yellow
Write-Host "4. Create repository on GitHub if not exists: https://github.com/surendar004/FedShield" -ForegroundColor Yellow
Write-Host "5. Push: git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or run the automated push script after reviewing changes." -ForegroundColor Cyan
Write-Host ""

$proceed = Read-Host "Do you want to proceed with add, commit, and push now? (y/n)"
if ($proceed -eq 'y' -or $proceed -eq 'Y') {
    Write-Host ""
    Write-Host "Adding all files..." -ForegroundColor Yellow
    git add .
    
    Write-Host "Committing changes..." -ForegroundColor Yellow
    $commitMessage = Read-Host "Enter commit message (or press Enter for default)"
    if ([string]::IsNullOrWhiteSpace($commitMessage)) {
        $commitMessage = "Initial commit: FedShield - Federated Cybersecurity System"
    }
    git commit -m $commitMessage
    
    Write-Host ""
    Write-Host "Checking if main branch exists..." -ForegroundColor Yellow
    $currentBranch = git branch --show-current 2>$null
    if (-not $currentBranch) {
        git branch -M main
        Write-Host "✓ Created and switched to 'main' branch" -ForegroundColor Green
    } elseif ($currentBranch -ne 'main') {
        git branch -M main
        Write-Host "✓ Renamed branch to 'main'" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "Note: You may need to authenticate. Use GitHub Personal Access Token if prompted." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "View your repository at: https://github.com/surendar004/FedShield" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "✗ Push failed. Common issues:" -ForegroundColor Red
        Write-Host "  - Repository doesn't exist on GitHub. Create it first at https://github.com/new" -ForegroundColor Yellow
        Write-Host "  - Authentication required. Use Personal Access Token instead of password" -ForegroundColor Yellow
        Write-Host "  - Branch name mismatch. Try: git push -u origin main:main" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "Setup complete. Run git commands manually when ready." -ForegroundColor Cyan
}

