#!/usr/bin/env python3
"""
retrain_class_weighted.py

Retrains LR, RF and XGBoost with class balancing and saves models + metrics.

Usage:
    # with uv
    uv run python src/retrain_class_weighted.py

    # or plain python
    python src/retrain_class_weighted.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/models")
FEATURES_DIR = Path("outputs/features")

def safe_load_npz(p): return sparse.load_npz(str(p))
def safe_load_npy(p): return np.load(str(p), allow_pickle=True)

def filter_labeled(X, y):
    mask = (y != -1) & ~np.isnan(y)
    if sparse.issparse(X):
        Xf = X[mask]
    else:
        Xf = X[mask]
    yf = y[mask].astype(int)
    return Xf, yf

def evaluate_and_save(model, X_val, y_val, X_test, y_test, model_name, outdir):
    # filter labeled
    Xv, yv = filter_labeled(X_val, y_val)
    Xt, yt = filter_labeled(X_test, y_test)

    res = {}
    def make_metrics(model, X, y, name):
        res_local = {}
        preds = model.predict(X)
        res_local['accuracy'] = float(accuracy_score(y, preds))
        res_local['precision'] = float(precision_score(y, preds, zero_division=0))
        res_local['recall'] = float(recall_score(y, preds, zero_division=0))
        res_local['f1'] = float(f1_score(y, preds, zero_division=0))
        # ROC-AUC
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:,1]
                res_local['roc_auc'] = float(roc_auc_score(y, probs))
            else:
                res_local['roc_auc'] = None
        except Exception:
            res_local['roc_auc'] = None
        # PR-AUC
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:,1]
            else:
                probs = model.decision_function(X)
            p, r, _ = precision_recall_curve(y, probs)
            res_local['pr_auc'] = float(auc(r, p))
        except Exception:
            res_local['pr_auc'] = None

        res_local['classification_report'] = classification_report(y, preds, output_dict=True, zero_division=0)
        # confusion matrix (raw)
        res_local['confusion_matrix'] = confusion_matrix(y, preds).tolist()
        return res_local

    res['val'] = make_metrics(model, Xv, yv, 'val')
    res['test'] = make_metrics(model, Xt, yt, 'test')
    # save metrics
    metrics_path = outdir / f"{model_name}_balanced_metrics.json"
    with open(metrics_path, "w", encoding="utf8") as f:
        json.dump(res, f, indent=2)
    print("Saved metrics:", metrics_path)

    # save confusion matrix figures for val and test
    def plot_cm(cm, title, path):
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels([0,1]); ax.set_yticklabels([0,1])
        ax.set_ylabel('True label'); ax.set_xlabel('Predicted label'); ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    cm_val = np.array(res['val']['confusion_matrix'])
    cm_test = np.array(res['test']['confusion_matrix'])
    plot_cm(cm_val, f"{model_name} (val) CM", outdir / f"{model_name}_balanced_confusion_val.png")
    plot_cm(cm_test, f"{model_name} (test) CM", outdir / f"{model_name}_balanced_confusion_test.png")
    print("Saved confusion images.")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # load features
    X_train = safe_load_npz(FEATURES_DIR / "train_X.npz")
    X_val = safe_load_npz(FEATURES_DIR / "val_X.npz")
    X_test = safe_load_npz(FEATURES_DIR / "test_X.npz")
    y_train = safe_load_npy(FEATURES_DIR / "train_y.npy")
    y_val = safe_load_npy(FEATURES_DIR / "val_y.npy")
    y_test = safe_load_npy(FEATURES_DIR / "test_y.npy")

    # filter labeled training set
    X_tr, y_tr = filter_labeled(X_train, y_train)
    print("Labeled training rows:", len(y_tr))
    # compute scale_pos_weight for xgboost
    num_pos = int((y_tr==1).sum())
    num_neg = int((y_tr==0).sum())
    scale_pos_weight = 1.0
    if num_pos > 0:
        scale_pos_weight = num_neg / max(1, num_pos)
    print("Class counts (train): pos=", num_pos, "neg=", num_neg, "scale_pos_weight=", scale_pos_weight)

    # 1) Logistic Regression (balanced)
    print("Training LogisticRegression with class_weight='balanced' ...")
    lr = LogisticRegression(class_weight='balanced', solver='saga', max_iter=3000, n_jobs=-1)
    lr.fit(X_tr, y_tr)
    joblib.dump(lr, OUT_DIR / "lr_balanced.joblib")
    print("Saved", OUT_DIR / "lr_balanced.joblib")
    evaluate_and_save(lr, X_val, y_val, X_test, y_test, "lr", OUT_DIR)

    # 2) Random Forest (balanced)
    print("Training RandomForestClassifier with class_weight='balanced' ...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    joblib.dump(rf, OUT_DIR / "rf_balanced.joblib")
    print("Saved", OUT_DIR / "rf_balanced.joblib")
    evaluate_and_save(rf, X_val, y_val, X_test, y_test, "rf", OUT_DIR)

    # 3) XGBoost with scale_pos_weight (if available)
    if XGBClassifier is not None:
        print("Training XGBoost with scale_pos_weight ...")
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss",
                             scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=42)
        xgb.fit(X_tr, y_tr)
        joblib.dump(xgb, OUT_DIR / "xgb_balanced.joblib")
        print("Saved", OUT_DIR / "xgb_balanced.joblib")
        evaluate_and_save(xgb, X_val, y_val, X_test, y_test, "xgb", OUT_DIR)
    else:
        print("XGBoost not installed; skip XGB training.")

if __name__ == "__main__":
    main()
