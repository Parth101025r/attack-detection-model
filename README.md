# Attack Detection Model

This repository trains multiple machine learning models for attack detection and saves the trained models as pickle files.

**What this repo does**
- Trains K-Nearest Neighbors (KNN), Decision Tree, Logistic Regression, and Support Vector Machine (SVM) and RandomForest models.
- Prints model accuracy on a test split.
- Saves each trained model as a `.pkl` file in the `pickel files/` folder.

**Files of interest**
- `code.py` - Main training script. Trains models, evaluates accuracy, and saves trained models.
- `pickel files/` - Folder where trained model pickle files are written (e.g. `knn_model.pkl`).
- `UNSW_NB15_training-set.csv` -  dataset files present in the workspace.

Prerequisites
- Python 3.8+ recommended
- pip

Recommended packages
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib` (optional; the script may use `pickle` or `joblib` to save models)


Installation (PowerShell)

```powershell
# create and activate virtual environment (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# install dependencies
pip install -r requirements.txt
```

Run the training script (PowerShell)

```powershell
# from the repository root (D:\Research)
python code.py
```

Expected output
- Console output showing training progress and accuracy values for each model.
- Pickle files written to `pickel files/` (for example: `knn_model.pkl`, `decision_tree_model.pkl`, `logreg_model.pkl`, `svm_model.pkl`).

Notes and tips
- If you cloned the repository containing pre-saved model pickle files in a separate `pickel files/` folder, copy that `pickel files/` folder into the repository root before running `code.py` so the script can find the saved models.
- If `code.py` expects specific column names or a particular CSV, open `code.py` and adjust dataset paths/column names accordingly.
- If you prefer `joblib` for saving/loading large sklearn models, install `joblib` and update `code.py` accordingly.

