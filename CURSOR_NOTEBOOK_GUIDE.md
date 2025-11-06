# Running Jupyter Notebooks in Cursor

## ✅ Setup Complete!
Your Jupyter kernel is installed and ready to use.

## How to Run Notebooks in Cursor

### Step 1: Select the Python Interpreter
1. Open any `.ipynb` file in Cursor
2. Look at the **top-right corner** of the notebook - you'll see a kernel selector
3. Click on it and select: **"Python (ml_proj)"** or **".venv"**
4. If you don't see it, click **"Select Kernel"** button

### Step 2: Verify the Kernel
Once you select the kernel, you should see:
- The kernel name in the top-right corner
- Status indicator (idle/busy)

### Step 3: Run Cells
There are several ways to run cells:

**Option A: Run Current Cell**
- Click the ▶️ **"Run Cell"** button above the cell
- Or press **`Shift + Enter`** while the cell is selected

**Option B: Run All Cells**
- Click **"Run All"** button in the toolbar
- Or use **`Ctrl + Shift + P`** → Type "Run All Cells"

**Option C: Run Cell and Below**
- Press **`Ctrl + Enter`** to run current cell and move to next

### Step 4: Run Your First Notebook

1. **Open** `notebooks/01_data_exploration.ipynb`
2. **Select Kernel**: Click top-right → Select "Python (ml_proj)"
3. **Run First Cell**: Click the ▶️ button on the first code cell (Cell 1)
4. **Wait for Output**: It should print "Libraries imported successfully!"
5. **Continue**: Run cells one by one or click "Run All"

## Troubleshooting

### ❌ "No Kernel Selected" or "Kernel Not Found"
**Solution:**
```powershell
# Make sure you're in the project folder
cd C:\Users\acer\Desktop\ml_proj

# Activate environment
.\.venv\Scripts\Activate.ps1

# Reinstall kernel
python -m ipykernel install --user --name=ml_proj --display-name="Python (ml_proj)"
```
Then **reload Cursor** (Ctrl+Shift+P → "Reload Window")

### ❌ "Module not found" errors
**Solution:**
1. Make sure you selected the correct kernel (Python ml_proj)
2. Check if packages are installed:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip list
   ```
3. If packages are missing:
   ```powershell
   pip install -r requirements.txt
   ```

### ❌ Cell keeps running forever
**Solution:**
- Click **"Interrupt"** button in the toolbar
- Or **`Ctrl + C`** to stop execution
- Check if you're using the right kernel

### ❌ Can't see notebook preview
**Solution:**
- Right-click on `.ipynb` file → "Open With" → "Jupyter Notebook"
- Or install Jupyter extension in Cursor if not already installed

## Quick Test

Run this in a new cell to verify everything works:

```python
import sys
print(f"Python: {sys.executable}")
print(f"Kernel: {sys.executable}")

import pandas as pd
import numpy as np
print(f"✅ Pandas: {pd.__version__}")
print(f"✅ NumPy: {np.__version__}")
```

## Running All Notebooks in Order

1. **01_data_exploration.ipynb** - Loads 20k tweets, does EDA
2. **02_preprocessing.ipynb** - Cleans text, creates features
3. **03_baseline_models.ipynb** - Trains Logistic Regression & Random Forest
4. **04_deep_learning.ipynb** - Trains LSTM model
5. **05_fairness_analysis.ipynb** - Analyzes model fairness
6. **06_adversarial_testing.ipynb** - Tests model robustness

**Important:** Run them in order! Each notebook depends on outputs from previous ones.

## Tips

- **Save frequently**: Notebooks auto-save, but you can also Ctrl+S
- **Restart kernel**: If things get stuck, use "Restart Kernel" button
- **Clear outputs**: Right-click → "Clear All Outputs" to clean up
- **View variables**: Check the "Variables" panel to see loaded data
- **Execution order matters**: Run cells in order from top to bottom

## Keyboard Shortcuts

- **Shift + Enter**: Run cell and move to next
- **Ctrl + Enter**: Run cell and stay on it
- **Esc**: Enter command mode
- **A**: Insert cell above
- **B**: Insert cell below
- **DD**: Delete cell (press D twice)

