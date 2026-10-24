# -*- coding: utf-8 -*-

"""
predict.py — Run the trained model on a new (or existing) sample
and print the prediction.

Usage:
    python predict.py                    # predicts on 5 random test samples
    python predict.py --sample <index>   # predicts on a specific test row
"""

import os
import sys
import argparse
import joblib
import numpy as np

from preprocess import load_dataset

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TEST_FILE   = os.path.join(BASE_DIR, "data", "KDDTest+.txt")
TRAIN_FILE  = os.path.join(BASE_DIR, "data", "KDDTrain+.txt")
MODEL_FILE  = os.path.join(BASE_DIR, "models", "rf_intrusion_detector.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.pkl")

LABEL_MAP = {0: "Normal  [OK]", 1: "Attack  [!!]"}


def load_model():
    if not os.path.exists(MODEL_FILE):
        print("[error] Model not found. Run: python train.py")
        sys.exit(1)
    clf    = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return clf, scaler


def predict_samples(indices=None, n=5):
    clf, scaler = load_model()

    _, X_test, _, y_test, feature_names, _ = load_dataset(TRAIN_FILE, TEST_FILE)

    if indices is None:
        rng     = np.random.default_rng(seed=7)
        indices = rng.choice(len(X_test), size=n, replace=False)

    print("\n" + "=" * 60)
    print("  Network Intrusion Detector - Prediction")
    print("=" * 60)
    print(f"{'Index':<8} {'Actual':<20} {'Predicted':<20} {'Correct'}")
    print("-" * 60)

    for idx in indices:
        sample   = X_test[idx].reshape(1, -1)
        pred     = clf.predict(sample)[0]
        actual   = y_test[idx]
        correct  = "OK" if pred == actual else "WRONG"
        print(f"{idx:<8} {LABEL_MAP[actual]:<20} {LABEL_MAP[pred]:<20} {correct}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict network intrusion")
    parser.add_argument("--sample", type=int, nargs="+",
                        help="Index/indices of test samples to predict")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of random samples (default 5)")
    args = parser.parse_args()

    predict_samples(indices=args.sample, n=args.n)
