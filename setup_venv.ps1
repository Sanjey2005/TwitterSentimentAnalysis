# PowerShell script to setup virtual environment
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up ML Project Virtual Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Remove old virtual environment if it exists
if (Test-Path ".venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# Create new virtual environment
Write-Host "Step 1: Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Step 2: Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Step 3: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install packages
Write-Host "Step 4: Installing required packages..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes. Please wait..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel --quiet
python -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Install Jupyter kernel:" -ForegroundColor White
Write-Host "   python -m ipykernel install --user --name=ml_proj --display-name='Python (ml_proj)'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start Jupyter:" -ForegroundColor White
Write-Host "   jupyter notebook" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Select kernel: Kernel -> Change Kernel -> Python (ml_proj)" -ForegroundColor White
Write-Host ""

