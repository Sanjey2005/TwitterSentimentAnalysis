@echo off
echo ========================================
echo Twitter Sentiment Analysis - Streamlit App
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Checking Python installation...
python --version

echo.
echo Checking if Streamlit is installed...
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)" 2>nul
if errorlevel 1 (
    echo.
    echo Streamlit is not installed. Installing now...
    python -m pip install streamlit
    if errorlevel 1 (
        echo ERROR: Failed to install streamlit
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the app
echo.
python -m streamlit run app.py

pause

