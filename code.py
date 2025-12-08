import numpy as np
import pandas as pd
import joblib
import pickle

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

# 9. Define multiple models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

# 9. Train and evaluate all models
trained_models = {}
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Create full pipeline with current model
    fullPipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    
    fullPipeline.fit(X_train, y_train)
    
    
    # Predict & evaluate
    y_pred = fullPipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Store results
    trained_models[model_name] = fullPipeline
    results[model_name] = acc
    
    print(f"{model_name} Accuracy: {acc*100:.4f}%")

# 10. Display summary
print("\n" + "="*50)
print("MODEL ACCURACY SUMMARY")
print("="*50)
for model_name, accuracy in results.items():
    print(f"{model_name:20s}: {accuracy*100:.4f}%")

# 11. Save all trained models to pickle files
print("\n" + "="*50)
print("Saving trained models...")
print("="*50)

for model_name, pipeline in trained_models.items():
    filename = f"model_{model_name.replace(' ', '_').lower()}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved: {filename}")

# 12. Save all models in a single pickle file
with open("all_trained_models.pkl", 'wb') as f:
    pickle.dump(trained_models, f)
print(f"Saved: all_trained_models.pkl")
