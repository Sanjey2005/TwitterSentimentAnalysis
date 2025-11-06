# Quick Start Guide

## If Your Virtual Environment is Not Working

### Option 1: Use the Setup Script (Recommended for Windows)

1. **Delete the old virtual environment** (if it exists):
   ```
   Remove-Item -Recurse -Force .venv
   ```

2. **Run the setup script**:
   ```
   setup_venv.bat
   ```

3. **Activate the environment**:
   ```
   .venv\Scripts\activate
   ```

4. **Install Jupyter kernel**:
   ```
   python -m ipykernel install --user --name=ml_proj --display-name="Python (ml_proj)"
   ```

### Option 2: Manual Setup

1. **Delete old virtual environment**:
   ```powershell
   Remove-Item -Recurse -Force .venv
   ```

2. **Create new virtual environment**:
   ```powershell
   python -m venv .venv
   ```

3. **Activate it**:
   ```powershell
   .venv\Scripts\activate
   ```

4. **Upgrade pip**:
   ```powershell
   python -m pip install --upgrade pip
   ```

5. **Install packages** (this will take 5-10 minutes):
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```

6. **Install Jupyter kernel**:
   ```powershell
   python -m ipykernel install --user --name=ml_proj --display-name="Python (ml_proj)"
   ```

## Using Jupyter Notebooks

1. **Activate the environment**:
   ```powershell
   .venv\Scripts\activate
   ```

2. **Start Jupyter**:
   ```powershell
   jupyter notebook
   ```

3. **Select the kernel**: In Jupyter, go to Kernel → Change Kernel → Select "Python (ml_proj)"

## Important Notes

- **First run will be slow**: Installing packages takes time, especially TensorFlow
- **Memory requirements**: The code is optimized for 16GB RAM - uses 20k tweets
- **Run notebooks in order**: 01 → 02 → 03 → 04 → 05 → 06

## Troubleshooting

### If Jupyter keeps loading:
- Make sure you've selected the correct kernel
- Try restarting the kernel: Kernel → Restart
- Check that all packages are installed: `pip list`

### If TensorFlow errors occur:
- The code now handles TensorFlow version checks gracefully
- Make sure TensorFlow is installed: `pip show tensorflow`

### If memory errors occur:
- The dataset is already limited to 20k tweets
- Close other applications to free up RAM
- Restart the kernel if it gets stuck

## Project Structure

```
ml_proj/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_deep_learning.ipynb
│   ├── 05_fairness_analysis.ipynb
│   └── 06_adversarial_testing.ipynb
├── data/
│   ├── processed/        (generated after running notebooks)
│   └── augmented/        (empty)
├── models/
│   └── saved_models/     (generated after training)
├── reports/
│   └── figures/          (generated plots)
├── requirements.txt
└── sentiment140.csv     (your dataset)
```

