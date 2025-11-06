# Troubleshooting Guide - Streamlit Installation

## Problem: "no module named streamlit" Error

If you're getting this error, here are several solutions:

### Solution 1: Install Streamlit in Your Current Environment

Run this command in your terminal:
```bash
python -m pip install streamlit
```

### Solution 2: If Using a Virtual Environment

If you have a virtual environment (`.venv` folder), you need to activate it first:

**Windows (CMD):**
```bash
.venv\Scripts\activate
python -m pip install streamlit
python -m streamlit run app.py
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
python -m pip install streamlit
python -m streamlit run app.py
```

### Solution 3: Use the Check Script

I've created a script that automatically checks and installs streamlit if needed:

```bash
run_app_with_check.bat
```

This script will:
1. Check if Python is available
2. Check if Streamlit is installed
3. Install Streamlit if missing
4. Run the app

### Solution 4: Verify Python Installation

Check which Python you're using:
```bash
python --version
where python
```

Make sure you're using the same Python that has your packages installed.

### Solution 5: Reinstall All Requirements

If nothing works, try reinstalling all requirements:
```bash
python -m pip install -r requirements.txt
```

### Solution 6: Create a Fresh Virtual Environment

If you want to start fresh with a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows CMD)
.venv\Scripts\activate

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install all requirements
python -m pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

## Quick Test

To verify Streamlit is installed correctly, run:
```bash
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

If this works, you can run:
```bash
python -m streamlit run app.py
```

## Still Having Issues?

1. Make sure you're in the project directory: `C:\Users\acer\Desktop\ML\ml_proj`
2. Check that you're using the correct Python interpreter
3. Try using the full path to Python if needed
4. Check if you have multiple Python installations

