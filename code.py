import numpy as np
import pandas as pd
import joblib
import pickle
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load data
dataframe = pd.read_csv("UNSW_NB15_training-set.csv")

# 2. Drop binary label (we will use attack_cat as target)
dataframe.drop("label", axis=1, inplace=True, errors="ignore")

# 3. Clean service column
dataframe["service"] = dataframe["service"].replace("-", "none")

# 4. Define features and target
X = dataframe.drop("attack_cat", axis=1)
y = dataframe["attack_cat"]

# 5. Separate categorical & numerical columns from X (not from whole dataframe)
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# 6. Pipelines
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# 7. ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ]
)

# 8. Stratified train-test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):
    X_train = X.iloc[train_index]
    X_test  = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test  = y.iloc[test_index]

# 9. Define multiple models (KNN, Decision Tree, Logistic Regression, SVM)
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

# Directory where pickle files are stored / looked for
PICKLE_DIR = "pickel files"
os.makedirs(PICKLE_DIR, exist_ok=True)

def model_key(name: str) -> str:
    return name.replace(' ', '_').lower()

def find_pickle(name: str):
    """Look for a model pickle in PICKLE_DIR first, then in repo root."""
    key = model_key(name)
    candidates = [os.path.join(PICKLE_DIR, f"{key}_model.pkl"), f"{key}_model.pkl"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

all_models_path = os.path.join(PICKLE_DIR, "all_trained_models.pkl")

trained_models = {}
results = {}

if os.path.exists(all_models_path):
    # Load all models at once if available
    try:
        trained_models = joblib.load(all_models_path)
        print(f"Loaded combined models from: {all_models_path}")
        for model_name, pipeline in trained_models.items():
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc
            print(f"{model_name} (loaded) Accuracy: {acc*100:.4f}%")
    except Exception as e:
        print(f"Failed to load combined models from {all_models_path}: {e}")
        trained_models = {}

for model_name, model in models.items():
    if model_name in trained_models:
        # Already loaded from combined file
        continue

    print(f"\nProcessing {model_name}...")
    pkl_path = find_pickle(model_name)
    if pkl_path:
        try:
            pipeline = joblib.load(pkl_path)
            trained_models[model_name] = pipeline
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[model_name] = acc
            print(f"Loaded {model_name} from {pkl_path} (Accuracy: {acc*100:.4f}%)")
            continue
        except Exception as e:
            print(f"Failed to load {pkl_path}: {e} -- will retrain")

    # If we reach here, train the model
    print(f"Training {model_name}...")
    fullPipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    fullPipeline.fit(X_train, y_train)
    y_pred = fullPipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    trained_models[model_name] = fullPipeline
    results[model_name] = acc
    print(f"{model_name} Accuracy: {acc*100:.4f}%")
    


# Summary
print("\n" + "="*50)
print("MODEL ACCURACY SUMMARY")
print("="*50)
for model_name, accuracy in results.items():
    print(f"{model_name:20s}: {accuracy*100:.4f}%")

# Save individual model pickles and combined file (only if they don't already exist)
print("\nSaving models to pickle files (skip if file exists)...")

for model_name, pipeline in trained_models.items():
    key = model_key(model_name)
    filename = os.path.join(PICKLE_DIR, f"{key}_model.pkl")
    if os.path.exists(filename):
        print(f"Skipping save (exists): {filename}")
        continue
    try:
        joblib.dump(pipeline, filename, compress=0)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

# Save combined models only if combined file doesn't already exist
if os.path.exists(all_models_path):
    print(f"Skipping save of combined models (exists): {all_models_path}")
else:
    try:
        joblib.dump(trained_models, all_models_path, compress=0)
        print(f"Saved combined models: {all_models_path}")
    except Exception as e:
        print(f"Failed to save combined models: {e}")
