@echo off
echo ========================================
echo Setting up ML Project Virtual Environment
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call .venv\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo Step 4: Installing required packages...
echo This may take several minutes. Please wait...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   .venv\Scripts\activate
echo.
echo To install Jupyter kernel, run:
echo   python -m ipykernel install --user --name=ml_proj --display-name="Python (ml_proj)"
echo.
pause

