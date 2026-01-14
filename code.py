import numpy as np
import pandas as pd
import joblib
import pickle
import os

from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

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
    dataframe = pd.read_csv("UNSW_dataset.csv",low_memory=False)
    dataframe['attack_cat'] = dataframe['attack_cat'].str.strip()

    # 2. (Optional) Force everything to lowercase to catch "Fuzzers" vs "fuzzers"
    dataframe['attack_cat'] = dataframe['attack_cat'].str.lower()
    # 1. Calculate frequencies
    series = dataframe['attack_cat'].value_counts(normalize=True)

    # 2. Identify infrequent labels
    threshold = 0.05
    rare_classes = series[series < threshold].index

    # 3. Replace them
    dataframe['attack_cat'] = dataframe['attack_cat'].replace(rare_classes, 'Other')
    sample_size = 0.01  
    dataframe_sample = dataframe.sample(frac=sample_size, random_state=42)
    dataframe_sample.drop("label", axis=1, inplace=True, errors="ignore")
    dataframe_sample.drop("srcip", axis=1, inplace=True, errors="ignore")
    dataframe_sample.drop("dstip", axis=1, inplace=True, errors="ignore")
  
    dataframe_sample["service"] = dataframe_sample["service"].replace("-", "none")
    dataframe_sample["attack_cat"] = dataframe_sample["attack_cat"].fillna("normal")
    dataframe_sample["ct_flw_http_mthd"] = dataframe_sample["ct_flw_http_mthd"].fillna(0)
    dataframe_sample["is_ftp_login"] = dataframe_sample["is_ftp_login"].fillna(0)

    print(dataframe_sample["attack_cat"].value_counts())
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
    "Decision Tree": DecisionTreeClassifier( max_depth=10, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.5908,solver='lbfgs',max_iter=500,random_state=42    ),
    "SVM": LinearSVC(C=0.3058, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=2,max_features='sqrt',random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.05,subsample=0.6,colsample_bytree=1.0,eval_metric="mlogloss",random_state=42)
    }
    strategy = {'generic': 500}
    trained_models = {}
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        fullPipeline = Pipeline([
                    ("preprocess", preprocessor),
                   ("smote",SMOTE(sampling_strategy="not majority")),
                    ("RUS",RandomUnderSampler(sampling_strategy="not minority")),
                    ("model", model)
                        ])
        start_time = time.perf_counter()
        if model_name == "XGBoost":
            fullPipeline.fit(X_train, y_train_enc)
        else:
            fullPipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time

        y_pred = fullPipeline.predict(X_test)
        # If predictions are encoded (e.g., XGBoost trained on integers), convert back to original labels
        if model_name == "XGBoost":
            y_pred = le.inverse_transform(y_pred)

        # Compute metrics (weighted average for multiclass)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        trained_models[model_name] = fullPipeline
        report = classification_report(y_test, y_pred, zero_division=0)
        results[model_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_time_sec": train_time,
            "classification_report": report
        }
        print(f"{model_name} -> Time: {train_time:.4f}s | Acc: {acc*100:.4f}% | Precision: {precision*100:.4f}% | Recall: {recall*100:.4f}% | F1: {f1*100:.4f}%")
        print(f"Classification report for {model_name}:\n{report}")
    joblib.dump(trained_models, MODEL_FILE)
    print("Saved all trained models into a pickle")
else:
    print("Files already exist")
    trained_models = joblib.load(MODEL_FILE)
    X_test, y_test = joblib.load(TEST_DATA_FILE)
    results = {}
    for model_name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        # If predictions appear encoded as integers while y_test are strings, try inverse transforming
        try:
            if hasattr(y_pred, "dtype") and y_pred.dtype.kind in 'iu':
                le = LabelEncoder()
                le.fit(y_test)
                y_pred = le.inverse_transform(y_pred)
            elif len(y_pred) > 0 and isinstance(y_pred[0], (int,)):
                le = LabelEncoder()
                le.fit(y_test)
                y_pred = le.inverse_transform(y_pred)
        except Exception:
            pass

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        report = classification_report(y_test, y_pred, zero_division=0)
        results[model_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_time_sec": None,
            "classification_report": report
        }
        print(f"{model_name} (loaded) -> Time: N/A | Acc: {acc*100:.4f}% | Precision: {precision*100:.4f}% | Recall: {recall*100:.4f}% | F1: {f1*100:.4f}%")
        print(f"Classification report for {model_name}:\n{report}")





