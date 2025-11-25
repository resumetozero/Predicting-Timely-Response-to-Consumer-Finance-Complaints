#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")   # ensure non-GUI backend
from pathlib import Path
import joblib, json
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

FEATURES_DIR = Path("outputs/features")
OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_X(path): return sparse.load_npz(str(path))
def load_y(path): return np.load(str(path), allow_pickle=True)

def filter_labeled(X, y):
    mask = (y!=-1) & ~np.isnan(y)
    return X[mask], y[mask].astype(int)

# load
X_train = load_X(FEATURES_DIR / "train_X.npz")
X_val = load_X(FEATURES_DIR / "val_X.npz")
X_test = load_X(FEATURES_DIR / "test_X.npz")
y_train = load_y(FEATURES_DIR / "train_y.npy")
y_val = load_y(FEATURES_DIR / "val_y.npy")
y_test = load_y(FEATURES_DIR / "test_y.npy")

X_tr, y_tr = filter_labeled(X_train, y_train)
X_val_f, y_val_f = filter_labeled(X_val, y_val)
X_test_f, y_test_f = filter_labeled(X_test, y_test)

print("Training RF on", len(y_tr), "rows (balanced class weight).")

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
joblib.dump(rf, OUT_DIR / "rf_balanced.joblib")
print("Saved", OUT_DIR / "rf_balanced.joblib")

# Evaluate and save metrics
def eval_and_save(model, X, y, name):
    preds = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "classification_report": classification_report(y, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds).tolist()
    }
    with open(OUT_DIR / f"rf_balanced_{name}_metrics.json", "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    # plot confusion matrix
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels([0,1]); ax.set_yticklabels([0,1])
    ax.set_xlabel('Pred'); ax.set_ylabel('True'); ax.set_title(f'rf_balanced_{name}_cm')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"rf_balanced_{name}_confusion.png", dpi=150)
    plt.close(fig)
    print("Saved", OUT_DIR / f"rf_balanced_{name}_metrics.json")

eval_and_save(rf, X_val_f, y_val_f, "val")
eval_and_save(rf, X_test_f, y_test_f, "test")
