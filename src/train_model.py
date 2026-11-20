"""
Trains a sign classifier on the landmarks collected by collect_data.py.

Usage:
    python src/train_model.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the plot to a file
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    LANDMARKS_CSV_PATH,
    CLASSIFIER_PATH,
    CONFUSION_MATRIX_PATH,
    MODEL_DIR,
)


def main():
    if not os.path.exists(LANDMARKS_CSV_PATH):
        print(f"No data found at {LANDMARKS_CSV_PATH}. Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(LANDMARKS_CSV_PATH)
    if df.empty:
        print("Data file is empty. Collect some samples first.")
        sys.exit(1)

    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Loaded {len(df)} samples across {y.nunique()} labels: {sorted(y.unique())}")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Classification report (held-out test set) ===")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"\nSaved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    joblib.dump(model, CLASSIFIER_PATH)
    print(f"Saved trained model to {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
