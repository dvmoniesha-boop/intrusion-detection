"""
main.py - One-shot entry point.

Runs the full pipeline end-to-end:
  1. Download dataset (if not present)
  2. Preprocess
  3. Train & evaluate
  4. (Optional) Predict on random samples

Usage:
    python main.py            # full pipeline
    python main.py --skip-viz # skip matplotlib charts
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def step_download():
    print("\n-- Step 1: Dataset ------------------------------------------")
    train_file = os.path.join(BASE_DIR, "data", "KDDTrain+.txt")
    if os.path.exists(train_file):
        print("   Dataset already present - skipping download.")
        return

    print("   Downloading NSL-KDD dataset ...")
    # Add data dir to path so we can import download_dataset
    sys.path.insert(0, os.path.join(BASE_DIR, "data"))
    import download_dataset
    download_dataset.download()


def step_train():
    print("\n-- Step 2: Train --------------------------------------------")
    import train
    train.train()


def step_predict():
    print("\n-- Step 3: Sample Predictions -------------------------------")
    import predict
    predict.predict_samples(n=5)


def step_visualise():
    print("\n-- Step 4: Visualisations -----------------------------------")
    import visualise
    visualise.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Network Intrusion Detector - full pipeline"
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip matplotlib visualisation step"
    )
    args = parser.parse_args()

    # Make sure project root is on sys.path
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    step_download()
    step_train()
    step_predict()

    if not args.skip_viz:
        step_visualise()

    print("\n" + "=" * 60)
    print("  All done!")
    print("=" * 60 + "\n")
