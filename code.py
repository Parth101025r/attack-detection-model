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
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
TEST_DATA_FILE = "test_data.pkl"

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])


if not os.path.exists(MODEL_FILE):
    print("Files do not exists")
    dataframe = pd.read_csv("UNSW_NB15_training-set.csv")
    
    sample_size = 1  
    dataframe_sample = dataframe.sample(frac=sample_size, random_state=42)
    dataframe_sample.drop("label", axis=1, inplace=True, errors="ignore")
  
    dataframe_sample["service"] = dataframe_sample["service"].replace("-", "none")
    X = dataframe_sample.drop("attack_cat", axis=1)
    y = dataframe_sample["attack_cat"]

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(X, y):
        X_train = X.iloc[train_index]
        X_test  = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test  = y.iloc[test_index]

    joblib.dump((X_test, y_test), TEST_DATA_FILE)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42,max_depth=10),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "RandomForest":RandomForestClassifier(n_estimators=100,min_samples_leaf=2,min_samples_split=10,max_features=None,max_depth=20,random_state=42),
        "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=16,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        )
            }
    trained_models = {}
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        fullPipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", model)
                        ])
        if model_name == "XGBoost":
            fullPipeline.fit(X_train, y_train_enc)
            y_pred = fullPipeline.predict(X_test)
            y_pred = le.inverse_transform(y_pred)
            acc = accuracy_score(y_test, y_pred)
        else:
            fullPipeline.fit(X_train, y_train)
            y_pred = fullPipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        trained_models[model_name] = fullPipeline
        results[model_name] = acc
        print(f"{model_name} Accuracy: {acc*100:.4f}%")
    joblib.dump(trained_models,MODEL_FILE)
    print("Saved all trained models into a pickel")
else:
    print("Files already exists")
    trained_models = joblib.load(MODEL_FILE)
    X_test, y_test = joblib.load(TEST_DATA_FILE)
    results = {}
    for model_name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[model_name] = acc
        print(f"{model_name} (loaded) Accuracy: {acc*100:.4f}%")
