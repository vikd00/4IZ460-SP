# Advanced Knowledge Discovery from Databases - Project

Data analysis project using CleverMiner and Python for advanced database knowledge mining.

## Project Structure

- `data/` - Data files (CSV format)
  - `acc_20.csv` - Account data
  - `pers_20.csv` - Person data
  - `veh_20.csv` - Vehicle data
  - `US Holiday Dates (2004-2021).csv` - Holiday reference data
- `notebooks/` - Jupyter notebooks for analysis
  - `analysis.ipynb` - Main analysis notebook
- `docs/` - Documentation
  - `cleverminer_reference.md` - CleverMiner reference guide

## Setup Instructions for VS Code

### Prerequisites

- Python 3.8 or higher
- Git
- VS Code

### Required VS Code Extensions

Install these extensions in VS Code:

- **Python** (Microsoft) - For Python support and debugging
- **Jupyter** (Microsoft) - For Jupyter notebook support

You can install them from the Extensions marketplace.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SP/code
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

**Activate the virtual environment:**

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

### 3. Extract Data Files

Unzip the data files in the `data/` folder:

Verify the CSV files are extracted to `data/` folder.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure VS Code

Create or update `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "python.analysis.typeCheckingMode": "basic"
}
```

### 6. Open Jupyter Notebook

- Open `notebooks/analysis.ipynb` in VS Code
- Select the `.venv` interpreter when prompted
- Run cells to execute the analysis

## Dependencies

Key packages:

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` & `seaborn` - Visualization
- `jupyter` - Interactive notebooks
- `cleverminer` - Knowledge mining engine

See `requirements.txt` for full list.

## Usage

1. Activate the virtual environment
2. Open VS Code
3. Open `analysis.ipynb` notebook
4. Select kernel from `.venv` environment
5. Execute cells to run analysis

## Notes

- Virtual environment (`.venv/`) is not tracked in Git
- **Data files are zipped** - Extract them to the `data/` folder before running analysis
- Make sure to activate the virtual environment before running anything
