# GitHub Setup Instructions for FedShield

## Prerequisites

### 1. Install Git

If Git is not installed on your system:

**Option A: Download Git for Windows**
- Visit: https://git-scm.com/download/win
- Download and run the installer
- Use default settings (recommended)
- Restart PowerShell after installation

**Option B: Install via winget (Windows Package Manager)**
```powershell
winget install --id Git.Git -e --source winget
```

**Option C: Install via Chocolatey**
```powershell
choco install git
```

### 2. Verify Git Installation

After installation, restart PowerShell and verify:
```powershell
git --version
```

## Quick Setup (Automated)

1. **Run the setup script:**
   ```powershell
   cd "D:\WEB FED\CAPSTONE"
   .\setup_git_and_push.ps1
   ```

2. **Follow the prompts** - the script will:
   - Initialize Git repository (if needed)
   - Add remote origin
   - Stage all files
   - Commit changes
   - Push to GitHub

## Manual Setup

If you prefer to do it manually:

### Step 1: Initialize Git Repository

```powershell
cd "D:\WEB FED\CAPSTONE"
git init
```

### Step 2: Create Repository on GitHub

1. Go to https://github.com/surendar004
2. Click "New repository" or go to https://github.com/new
3. Repository name: `FedShield`
4. Description: "FedShield - Federated Cybersecurity System"
5. Choose Public or Private
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### Step 3: Add Remote and Configure

```powershell
# Add remote origin
git remote add origin https://github.com/surendar004/FedShield.git

# Or if remote already exists, update it:
git remote set-url origin https://github.com/surendar004/FedShield.git

# Verify remote
git remote -v
```

### Step 4: Stage and Commit Files

```powershell
# Check what will be committed
git status

# Add all files (respects .gitignore)
git add .

# Commit with a message
git commit -m "Initial commit: FedShield - Federated Cybersecurity System"
```

### Step 5: Push to GitHub

```powershell
# Create and switch to main branch (if not already)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Authentication

When pushing, GitHub may ask for credentials:

### Option 1: Personal Access Token (Recommended)
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use the token as password when prompted
4. Username: your GitHub username

### Option 2: GitHub CLI
```powershell
# Install GitHub CLI
winget install --id GitHub.cli

# Authenticate
gh auth login
```

### Option 3: SSH Keys (Advanced)
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
3. Change remote URL: `git remote set-url origin git@github.com:surendar004/FedShield.git`

## Troubleshooting

### Repository Already Exists on GitHub
If the repository already exists and has content:
```powershell
git pull origin main --allow-unrelated-histories
# Resolve any conflicts, then:
git push -u origin main
```

### Authentication Failed
- Use Personal Access Token instead of password
- Ensure token has `repo` scope
- Check if 2FA is enabled (requires token)

### Branch Name Issues
```powershell
# If your branch is named 'master' instead of 'main'
git branch -M main
git push -u origin main
```

### Large Files
If you have large files (>100MB), consider:
- Using Git LFS: `git lfs install`
- Or excluding them in `.gitignore`

## Verify Upload

After pushing, visit:
https://github.com/surendar004/FedShield

You should see all your project files there!

## Future Updates

To push future changes:
```powershell
cd "D:\WEB FED\CAPSTONE"
git add .
git commit -m "Your commit message describing changes"
git push
```

