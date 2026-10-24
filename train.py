# -*- coding: utf-8 -*-

"""
train.py — Train a Random Forest classifier on the NSL-KDD dataset
and save the model to disk.

Usage:
    python train.py
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "data", "KDDTrain+.txt")
TEST_FILE  = os.path.join(BASE_DIR, "data", "KDDTest+.txt")
MODEL_FILE = os.path.join(BASE_DIR, "models", "rf_intrusion_detector.pkl")
SCALER_FILE= os.path.join(BASE_DIR, "models", "scaler.pkl")


def train():
    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("=" * 60)
    print("  Network Intrusion Detector - Training")
    print("=" * 60)

    if not os.path.exists(TRAIN_FILE):
        print(f"\n[error] Training file not found: {TRAIN_FILE}")
        print("        Run:  python data/download_dataset.py\n")
        return

    print("\n[1/4] Loading dataset ...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_dataset(
        TRAIN_FILE, TEST_FILE
    )
    print(f"      Train samples : {len(X_train):,}")
    print(f"      Test  samples : {len(X_test):,}")
    print(f"      Features      : {len(feature_names)}")
    normal_pct = (y_train == 0).mean() * 100
    attack_pct = (y_train == 1).mean() * 100
    print(f"      Train label split - normal: {normal_pct:.1f}%  attack: {attack_pct:.1f}%")

    # ── 2. Train model ────────────────────────────────────────────────────────
    print("\n[2/4] Training Random Forest (100 trees) ...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("      Training complete.")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/4] Evaluating on test set ...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Accuracy : {acc * 100:.2f}%\n")
    print("      Classification Report:")
    print("      " + "-" * 50)
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"]
    )
    for line in report.splitlines():
        print("      " + line)

    print("\n      Confusion Matrix  (rows=actual, cols=predicted):")
    print("               Normal   Attack")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Normal", "Attack"]
    for i, row in enumerate(cm):
        print(f"      {labels[i]:<8}  {row[0]:>6}   {row[1]:>6}")

    # ── 4. Top-10 important features ─────────────────────────────────────────
    print("\n      Top-10 Feature Importances:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"        {rank:>2}. {feature_names[idx]:<35} {importances[idx]:.4f}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving model  -> {MODEL_FILE}")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("      Done.\n")


if __name__ == "__main__":
    train()
