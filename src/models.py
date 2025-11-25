#!/usr/bin/env python3
"""
src/models.py

Train classification models for the Consumer Complaints project.

Produces:
 - outputs/models/<model_name>.joblib            (saved model)
 - outputs/models/<model_name>_metrics.json      (metrics)
 - outputs/models/<model_name>_confusion.png     (confusion matrix)

Usage (examples):
    # Train all default models
    python src/models.py --features-dir outputs/features --out-dir outputs/models --train all

    # Train only XGBoost with hyperparameter tuning
    python src/models.py --features-dir outputs/features --out-dir outputs/models --train xgb --tune

    # Train logistic regression only
    python src/models.py --train lr

Notes:
 - Expects feature matrices created by src/features.py:
   outputs/features/train_X.npz, train_y.npy, val_X.npz, val_y.npy, test_X.npz, test_y.npy
 - Labels with value -1 are treated as unlabeled and ignored for training/eval.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# -----------------------
# Utilities
# -----------------------
def safe_load_npz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return sparse.load_npz(str(path))

def safe_load_npy(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    return np.load(str(path), allow_pickle=True)

def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def filter_labeled(X, y):
    mask = (y != -1) & ~np.isnan(y)
    if sparse.issparse(X):
        Xf = X[mask]
    else:
        Xf = X[mask]
    yf = y[mask].astype(int)
    return Xf, yf

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2)

def plot_and_save_confusion(y_true, y_pred, out_path: Path, normalize=False, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='pred' if normalize else None)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], '.2f') if isinstance(cm[i, j], float) else int(cm[i,j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------
# Models (train/eval wrappers)
# -----------------------
def train_logistic(X_train, y_train, X_val=None, y_val=None, params=None):
    params = params or {"C":1.0, "solver":"saga", "max_iter":2000, "penalty":"l2", "n_jobs":-1}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, params=None):
    params = params or {"n_estimators":200, "max_depth":20, "n_jobs":-1, "random_state":42}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train, params=None):
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is not installed. Install xgboost to use this trainer.")
    params = params or {"n_estimators":200, "max_depth":6, "learning_rate":0.1, "use_label_encoder":False, "eval_metric":"logloss", "n_jobs":-1, "random_state":42}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

# -----------------------
# Evaluation
# -----------------------
def evaluate_model(model, X, y, dataset_name="test"):
    y_pred = model.predict(X)
    # Some classifiers produce float labels for binary but sklearn handles int comparisons
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y, y_pred))
    metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y, y_pred, zero_division=0))
    # ROC-AUC only if we have binary labels and predicted probabilities
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:,1]
            metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None
    metrics["classification_report"] = classification_report(y, y_pred, output_dict=True, zero_division=0)
    metrics["dataset"] = dataset_name
    return metrics, y_pred

# -----------------------
# Hyperparameter tuning (lightweight)
# -----------------------
def randomized_search(model, param_distributions, X, y, n_iter=20, cv=3, scoring="f1", random_state=42, n_jobs=-1):
    rnd = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, scoring=scoring, cv=cv, random_state=random_state, n_jobs=n_jobs, verbose=1)
    rnd.fit(X, y)
    return rnd

# -----------------------
# CLI / Orchestration
# -----------------------
def load_features(features_dir: Path):
    features_dir = Path(features_dir)
    X_train = safe_load_npz(features_dir / "train_X.npz")
    X_val = safe_load_npz(features_dir / "val_X.npz")
    X_test = safe_load_npz(features_dir / "test_X.npz")
    y_train = safe_load_npy(features_dir / "train_y.npy")
    y_val = safe_load_npy(features_dir / "val_y.npy")
    y_test = safe_load_npy(features_dir / "test_y.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_save(model_name: str, X_train, y_train, X_val, y_val, X_test, y_test, out_dir: Path, tune=False):
    out_dir = Path(out_dir)
    ensure_outdir(out_dir)

    # filter unlabeled rows
    X_tr, y_tr = filter_labeled(X_train, y_train)
    X_val_f, y_val_f = filter_labeled(X_val, y_val)
    X_test_f, y_test_f = filter_labeled(X_test, y_test)

    if len(y_tr) == 0:
        raise RuntimeError("No labeled training examples found (labels are -1 or NaN).")

    print(f"Training {model_name} on {len(y_tr)} labeled rows; val:{len(y_val_f)} test:{len(y_test_f)}")

    model = None
    if model_name.lower() in ("lr", "logistic", "logisticregression"):
        if tune:
            # simple tuning for C
            param_dist = {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}
            base = LogisticRegression(solver="saga", penalty="l2", max_iter=2000, n_jobs=-1)
            rnd = randomized_search(base, param_dist, X_tr, y_tr, n_iter=5, cv=3, scoring="f1")
            model = rnd.best_estimator_
            print("Best params:", rnd.best_params_)
        else:
            model = train_logistic(X_tr, y_tr)
    elif model_name.lower() in ("rf", "randomforest"):
        if tune:
            param_dist = {"n_estimators":[100,200,300], "max_depth":[6,10,20,None], "min_samples_split":[2,5,10]}
            base = RandomForestClassifier(n_jobs=-1, random_state=42)
            rnd = randomized_search(base, param_dist, X_tr, y_tr, n_iter=10, cv=3, scoring="f1")
            model = rnd.best_estimator_
            print("Best params:", rnd.best_params_)
        else:
            model = train_random_forest(X_tr, y_tr)
    elif model_name.lower() in ("xgb", "xgboost"):
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed. Install xgboost to train this model.")
        if tune:
            param_dist = {"n_estimators":[100,200,300], "max_depth":[3,6,8], "learning_rate":[0.01,0.05,0.1]}
            base = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42)
            rnd = randomized_search(base, param_dist, X_tr, y_tr, n_iter=10, cv=3, scoring="f1")
            model = rnd.best_estimator_
            print("Best params:", rnd.best_params_)
        else:
            model = train_xgb(X_tr, y_tr)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Evaluate on val and test
    metrics_val, y_pred_val = evaluate_model(model, X_val_f, y_val_f, dataset_name="val")
    metrics_test, y_pred_test = evaluate_model(model, X_test_f, y_test_f, dataset_name="test")

    # Save model and metrics
    model_path = out_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print("Saved model ->", model_path)

    metrics_out = {
        "model": model_name,
        "val": metrics_val,
        "test": metrics_test,
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_val_f)),
        "test_size": int(len(y_test_f))
    }
    save_json(metrics_out, out_dir / f"{model_name}_metrics.json")
    print("Saved metrics ->", out_dir / f"{model_name}_metrics.json")

    # confusion matrices
    plot_and_save_confusion(y_val_f, y_pred_val, out_dir / f"{model_name}_confusion_val.png")
    plot_and_save_confusion(y_test_f, y_pred_test, out_dir / f"{model_name}_confusion_test.png")
    print("Saved confusion matrices.")

    return {
        "model_path": str(model_path),
        "metrics_path": str(out_dir / f"{model_name}_metrics.json")
    }

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="outputs/features", help="Directory with features (train_X.npz, train_y.npy, ...)")
    parser.add_argument("--out-dir", type=str, default="outputs/models", help="Where to save models and metrics")
    parser.add_argument("--train", type=str, default="all", help="Which model to train: all | lr | rf | xgb")
    parser.add_argument("--tune", action="store_true", help="Whether to run a small randomized hyperparameter search (slower)")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_features(Path(args.features_dir))

    to_train = []
    if args.train.lower() == "all":
        to_train = ["lr", "rf", "xgb"]
    else:
        to_train = [args.train.lower()]

    results = {}
    for model_name in to_train:
        try:
            res = train_and_save(model_name, X_train, y_train, X_val, y_val, X_test, y_test, Path(args.out_dir), tune=args.tune)
            results[model_name] = res
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    print("Done. Summary:", results)

if __name__ == "__main__":
    main_cli()
