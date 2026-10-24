# -*- coding: utf-8 -*-

"""
visualise.py — Plot evaluation charts for the trained model.

Generates and saves:
  1. Confusion matrix heatmap
  2. Feature importance bar chart (top 15)

Usage:
    python visualise.py
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from preprocess import load_dataset

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE  = os.path.join(BASE_DIR, "data", "KDDTrain+.txt")
TEST_FILE   = os.path.join(BASE_DIR, "data", "KDDTest+.txt")
MODEL_FILE  = os.path.join(BASE_DIR, "models", "rf_intrusion_detector.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")


def plot_confusion_matrix(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Normal", "Attack"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Confusion Matrix — NSL-KDD Test Set", fontsize=13, pad=12)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")
    plt.show()


def plot_feature_importance(clf, feature_names, top_n=15):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names   = [feature_names[i] for i in indices]
    values  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names[::-1], values[::-1], color="#3b82d4", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")
    plt.show()


def main():
    if not os.path.exists(MODEL_FILE):
        print("[error] Model not found. Run: python train.py")
        return

    print("[load] model & dataset ...")
    clf    = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    _, X_test, _, y_test, feature_names, _ = load_dataset(TRAIN_FILE, TEST_FILE)

    plot_confusion_matrix(clf, X_test, y_test)
    plot_feature_importance(clf, feature_names)


if __name__ == "__main__":
    main()
