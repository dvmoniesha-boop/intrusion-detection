# -*- coding: utf-8 -*-

"""
preprocess.py — Load and preprocess the NSL-KDD dataset.

Responsibilities:
  - Assign column names
  - Encode categorical features (protocol_type, service, flag)
  - Binarise the label column (normal vs. attack)
  - Scale numeric features with StandardScaler
  - Return train/test splits ready for model training
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Column names for NSL-KDD ──────────────────────────────────────────────────
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_raw(filepath: str) -> pd.DataFrame:
    """Read a raw NSL-KDD .txt file into a DataFrame."""
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    return df


def binarise_label(df: pd.DataFrame) -> pd.DataFrame:
    """Convert multi-class attack labels to binary: 0=normal, 1=attack."""
    df = df.copy()
    df["label"] = df["label"].apply(lambda x: 0 if x.strip() == "normal" else 1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the three categorical columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def load_dataset(train_path: str, test_path: str):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test : np.ndarray  — scaled feature matrices
    y_train, y_test : np.ndarray  — binary labels (0=normal, 1=attack)
    feature_names   : list[str]
    """
    train_df = load_raw(train_path)
    test_df  = load_raw(test_path)

    # Drop the difficulty column — not a feature
    train_df = train_df.drop(columns=["difficulty"])
    test_df  = test_df.drop(columns=["difficulty"])

    # Binarise labels
    train_df = binarise_label(train_df)
    test_df  = binarise_label(test_df)

    # Encode categoricals
    train_df = encode_categoricals(train_df)
    test_df  = encode_categoricals(test_df)

    feature_names = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_names].values
    y_train = train_df["label"].values
    X_test  = test_df[feature_names].values
    y_test  = test_df["label"].values

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, scaler
