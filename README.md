# Network Intrusion Detector

A beginner-level machine learning project that trains a **Random Forest** classifier on the **NSL-KDD** dataset to classify network connections as **Normal** or **Attack**.

## Project Structure

```
network-intrusion-detector/
├── data/
│   └── download_dataset.py   # Downloads NSL-KDD train & test files
├── preprocess.py             # Column names, encoding, scaling
├── train.py                  # Train, evaluate and save the model
├── predict.py                # Predict on test samples
├── visualise.py              # Confusion matrix & feature importance charts
├── main.py                   # Runs the full pipeline
└── requirements.txt
```

Files created when you run the project (not stored in the repo): `data/KDDTrain+.txt`, `data/KDDTest+.txt`, `models/*.pkl`, and the chart PNGs.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (download -> train -> predict -> charts)
python main.py

# Skip the matplotlib charts
python main.py --skip-viz
```

Run steps individually:

```bash
python data/download_dataset.py     # download dataset
python train.py                     # train & evaluate
python predict.py                   # 5 random test samples
python predict.py --sample 10 42 99 # specific test rows
python visualise.py                 # charts (train first)
```

## Dataset

[NSL-KDD](https://github.com/defcom17/NSL_KDD), a cleaned version of KDD Cup 1999 with duplicate records removed.

| Property      | Value                                  |
|---------------|----------------------------------------|
| Train samples | 125,973                                |
| Test samples  | 22,544                                 |
| Features      | 41                                     |
| Label         | Binary: 0 = Normal, 1 = Attack         |

## Model

| Setting   | Value          |
|-----------|----------------|
| Algorithm | Random Forest  |
| Trees     | 100            |
| Max depth | 20             |
| Scaling   | StandardScaler |

## Results

Test set (KDDTest+):

| Metric          | Value  |
|-----------------|--------|
| Accuracy        | 77.17% |
| Normal recall   | 0.97   |
| Attack recall   | 0.62   |
| Attack precision| 0.97   |

Confusion matrix (rows = actual, columns = predicted):

|        | Normal | Attack |
|--------|--------|--------|
| Normal | 9439   | 272    |
| Attack | 4875   | 7958   |

Top features: `src_bytes`, `dst_bytes`, `same_srv_rate`, `dst_host_same_srv_rate`, `flag`.

### Why is accuracy only ~77%?

The NSL-KDD test set contains attack types that **do not appear in the training set**, so the model misses many of them (4,875 attacks predicted as Normal). This is a known property of this benchmark. When the model does flag an attack, it is right 97% of the time.

## Sample Output

Terminal output of `python main.py`:

![Terminal output](images/output.png)

Confusion matrix:

![Confusion matrix](images/confusion_matrix.png)

Top 15 feature importances:

![Feature importance](images/feature_importance.png)

## Ideas for Improvement

- Fit the label encoders on the training set and reuse them on the test set
- Try `GradientBoostingClassifier` or tune with `GridSearchCV`
- Multi-class labels (DoS, Probe, R2L, U2R)
- Handle unseen attack types, e.g. with anomaly detection
- Serve the model as a REST API
