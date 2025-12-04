import os
import glob
import numpy as np
from explore_data import load_motion
from extract_features import extract_features
from knn import knn

DEBUG = False

"""
This script:
    - loads all motion CSV files
    - extracts features
    - creates training/validation splits
    - runs the kNN classifier
    - prints accuracy and confusion matrix

It is used to evaluate the baseline kNN method.
To view the results, set DEBUG = True.
"""

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_label_from_filename(path):
    """
    Extract digit label from filename.

    Expected filename pattern: stroke_<digit>_XXXX.csv
    Example: stroke_3_0123.csv -> label = 3
    """
    base = os.path.basename(path)          # e.g. "stroke_3_0123.csv"
    name, _ = os.path.splitext(base)       # "stroke_3_0123"
    parts = name.split("_")                # ["stroke", "3", "0123"]

    digit_str = parts[1]                   # "3"
    return int(digit_str)


def build_dataset(num_points=50):
    """
    Load all CSV files, extract features and labels.

    Returns:
        X : array, shape (n_samples, num_points * 3)
        y : array, shape (n_samples,)
    """
    pattern = os.path.join(DATA_DIR, "*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    X_list = []
    y_list = []

    for path in csv_files:
        motion = load_motion(path)                     # N x 3
        features = extract_features(motion, num_points)  # (num_points*3,)
        label = get_label_from_filename(path)

        X_list.append(features)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    return X, y


def train_val_split(X, y, val_ratio=0.2, seed=42):
    """
    Simple random train/validation split.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split = int((1.0 - val_ratio) * n_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, y_train, X_val, y_val


def train_knn():

    if DEBUG:
        # 1. Build dataset (features + labels)
        print("Building dataset from CSV files...")
        X, y = build_dataset(num_points=50)
        print("Feature matrix shape:", X.shape)
        print("Labels shape:", y.shape)

        # 2. Split into train/validation
        X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)
        print(f"Train size: {X_train.shape[0]}, validation size: {X_val.shape[0]}")

        # 3. Run kNN baseline
        k = 5
        print(f"\nRunning kNN with k={k}...")
        y_pred = knn(X_train, y_train, X_val, k)

        # 4. Compute accuracy
        accuracy = np.mean(y_pred == y_val)
        print(f"Validation accuracy: {accuracy:.4f}")

        # Confusion matrix for more detail
        num_classes = 10
        conf = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(y_val, y_pred):
            conf[true_label, pred_label] += 1

        print("\nConfusion matrix (rows=true, cols=pred):")
        print(conf)


if __name__ == "__main__":
    train_knn()
