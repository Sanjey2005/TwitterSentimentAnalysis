@echo off
echo ========================================
echo Starting Streamlit App
echo ========================================
echo.

REM Check if virtual environment exists and activate it
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing...
    python -m pip install streamlit
)

echo.
echo Starting Streamlit app...
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the app
echo.
python -m streamlit run app.py

pause

