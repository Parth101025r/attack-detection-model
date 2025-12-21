import numpy as np
import pandas as pd
import joblib
import time
import os

# ===================== CONFIG =====================
FEATURE_FILE = "capture/features.csv"
MODEL_FILE = "model.pkl"

MIN_ABNORMAL_RATIO = 0.30      # % of rows that must be abnormal per model
MIN_MODELS_VOTING = 2          # how many models must agree
CONSECUTIVE_ALERTS = 3         # how many times in a row
STATE_FILE = "alert_state.txt"
# =================================================

print("Running prediction")

def beep():
    for _ in range(10):
        print('\a')
        time.sleep(0.4)


def deleteFile():
    if os.path.exists(FEATURE_FILE):
        os.remove(FEATURE_FILE)


def get_consecutive_count():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return int(f.read())
    return 0


def save_consecutive_count(count):
    with open(STATE_FILE, "w") as f:
        f.write(str(count))


def predict():

    if not os.path.exists(FEATURE_FILE):
        return

    # Load features
    df = pd.read_csv(FEATURE_FILE)
    length = len(df)
    if length>1000:
    	df = df.sample(frac=0.1)
    # Drop unwanted columns
    for col in ("attack_cat", "label"):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Fix categorical value
    if "service" in df.columns:
        df["service"] = df["service"].replace("-", "none")

    # Safety against NaNs
    df.fillna(0, inplace=True)

    # Load ensemble models
    models = joblib.load(MODEL_FILE)

    abnormal_models = 0

    # Majority voting logic
    for model_name, pipeline in models.items():
        preds = pipeline.predict(df)
        abnormal_ratio = np.mean(preds != "Normal")

        if abnormal_ratio >= MIN_ABNORMAL_RATIO:
            abnormal_models += 1

    # Consecutive confirmation logic
    count = get_consecutive_count()

    if abnormal_models >= MIN_MODELS_VOTING:
        count += 1
    else:
        count = 0

    save_consecutive_count(count)

    if count >= CONSECUTIVE_ALERTS:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  INTRUSION DETECTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        beep()
        save_consecutive_count(0)

    deleteFile()
    print("Files deleted")

# Run prediction
predict()
print("prediction complete")

