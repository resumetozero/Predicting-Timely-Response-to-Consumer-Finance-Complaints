#!/usr/bin/env python3
"""
Tune threshold to detect class 0 (timely_response == 0).

Usage:
    uv run python src/tune_threshold_class0.py --model outputs/models/lr_balanced.joblib
    (or python src/tune_threshold_class0.py --model outputs/models/rf_balanced.joblib)
"""
import argparse, joblib, json, numpy as np
from scipy import sparse
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, classification_report, auc

def load_features(path_prefix="outputs/features"):
    X_val = sparse.load_npz(f"{path_prefix}/val_X.npz")
    X_test = sparse.load_npz(f"{path_prefix}/test_X.npz")
    y_val = np.load(f"{path_prefix}/val_y.npy", allow_pickle=True)
    y_test = np.load(f"{path_prefix}/test_y.npy", allow_pickle=True)
    return X_val, y_val, X_test, y_test

def filter_labeled(X, y):
    mask = (y != -1) & ~np.isnan(y)
    return X[mask], y[mask].astype(int)

def evaluate_at_threshold(y_true, prob_pos_class1, thr_for_class0):
    # prob_pos_class1 = P(y==1). We want to predict class0 if (1 - prob1) >= thr -> i.e. prob1 <= (1-thr)
    cutoff = 1.0 - thr_for_class0
    preds = (prob_pos_class1 > cutoff).astype(int)  # 1 if prob1 > cutoff else 0
    # compute metrics for both classes
    metrics = {}
    metrics['confusion_matrix'] = confusion_matrix(y_true, preds).tolist()
    metrics['classification_report'] = classification_report(y_true, preds, output_dict=True, zero_division=0)
    # also return F1 for class 0 explicitly
    p0 = precision_score(y_true==0, preds==0, zero_division=0)
    r0 = recall_score(y_true==0, preds==0, zero_division=0)
    f10 = f1_score(y_true==0, preds==0, zero_division=0)
    metrics['class0_precision'] = float(p0)
    metrics['class0_recall'] = float(r0)
    metrics['class0_f1'] = float(f10)
    return metrics, preds

def main(args):
    X_val, y_val, X_test, y_test = load_features(args.features_dir)
    Xv, yv = filter_labeled(X_val, y_val)
    Xt, yt = filter_labeled(X_test, y_test)

    model = joblib.load(args.model)
    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        raise RuntimeError("Model has neither predict_proba nor decision_function.")

    # get probabilities for class 1 on validation
    if hasattr(model, "predict_proba"):
        prob_val = model.predict_proba(Xv)[:,1]
    else:
        prob_val = model.decision_function(Xv)
        prob_val = 1/(1+np.exp(-prob_val))

    # score0 = 1 - prob_val
    score0 = 1.0 - prob_val
    precisions, recalls, thresholds = precision_recall_curve(np.where(yv==0, 1, 0), score0)
    pr_auc = auc(recalls, precisions)
    print("PR-AUC for class0 (on val):", pr_auc)

    # choose threshold that maximizes F1 for class0
    best_f1 = -1.0
    best_thr = None
    best_stats = None
    # thresholds returned by precision_recall_curve are thresholds on score (score0) except last point.
    for i, thr in enumerate(np.append(thresholds, 1.0)):  # include top endpoint
        # thr is on score0
        metrics, _ = evaluate_at_threshold(yv, 1.0 - score0, thr)  # but evaluate_at_threshold expects prob1; we'll compute prob1 differently below
        # The above is a bit awkward; simpler: compute preds from score0 directly:
        preds0 = (score0 >= thr).astype(int)  # 1 means predict class0
        # Build class0 preds: predicted label is 0 when preds0==1, else 1
        predicted = np.where(preds0==1, 0, 1)
        # compute f1 for class0
        # Use sklearn: f1_score(y_true==0, predicted==0)
        f10 = f1_score(yv==0, predicted==0, zero_division=0)
        if f10 > best_f1:
            best_f1 = f10
            best_thr = float(thr)
    print("Best threshold on score0 (val) maximizing F1 for class0:", best_thr, "best_f1:", best_f1)

    # Evaluate chosen threshold on test set
    if hasattr(model, "predict_proba"):
        prob_test = model.predict_proba(Xt)[:,1]
    else:
        prob_test = model.decision_function(Xt)
        prob_test = 1/(1+np.exp(-prob_test))
    score0_test = 1.0 - prob_test
    # apply threshold
    preds0_test = (score0_test >= best_thr).astype(int)
    predicted_test = np.where(preds0_test==1, 0, 1)
    # compute and print metrics
    cm = confusion_matrix(yt, predicted_test)
    cr = classification_report(yt, predicted_test, zero_division=0)
    p0 = precision_score(yt==0, predicted_test==0, zero_division=0)
    r0 = recall_score(yt==0, predicted_test==0, zero_division=0)
    f10 = f1_score(yt==0, predicted_test==0, zero_division=0)
    out = {
        "best_threshold_on_score0": best_thr,
        "val_best_f1_class0": best_f1,
        "test_class0_precision": float(p0),
        "test_class0_recall": float(r0),
        "test_class0_f1": float(f10),
        "confusion_matrix_test": cm.tolist(),
        "classification_report_test": cr
    }
    print(json.dumps(out, indent=2))
    # save
    with open(args.out_json, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    print("Saved results to", args.out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.joblib) e.g. outputs/models/lr_balanced.joblib")
    parser.add_argument("--features-dir", default="outputs/features", help="Folder with train/val/test X/y")
    parser.add_argument("--out-json", default="outputs/models/threshold_tune_results.json", help="Where to save results")
    args = parser.parse_args()
    main(args)
