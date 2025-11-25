#!/usr/bin/env python3
"""
infer_and_explain.py

Usage (examples):

# single text input
python src/infer_and_explain.py --model outputs/models/rf.joblib \
  --vectorizer outputs/models/tfidf_vectorizer.joblib \
  --features outputs/features --text "My bank incorrectly charged me late fees and won't refund."

# CSV of texts (column 'raw_text' or 'text_clean')
python src/infer_and_explain.py --model outputs/models/xgb.joblib \
  --vectorizer outputs/models/tfidf_vectorizer.joblib \
  --features outputs/features \
  --input-csv data/new_samples.csv --text-col raw_text

Outputs:
 - Prints predictions and probabilities to stdout.
 - Saves per-example explanations (SHAP bar plots) to outputs/models/shap_<model>_<i>.png when SHAP is available.
 - If SHAP is not available or model unsupported, prints feature importances (when available).
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import sys
import os

from scipy import sparse
from sklearn.preprocessing import StandardScaler

# plotting
import matplotlib.pyplot as plt

# Try importing shap but handle if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# -------------------------
# Helpers to reconstruct same feature space used during training
# -------------------------
def load_metadata(features_dir: Path):
    meta_path = features_dir / "feature_metadata.json"
    enc_path = features_dir / "encoders.json"
    scaler_path = features_dir / "scaler.joblib"
    if not meta_path.exists() or not enc_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("One of feature metadata/scaler/encoders is missing in outputs/features. Run src/features.py first.")
    meta = json.loads(meta_path.read_text(encoding='utf8'))
    enc = json.loads(enc_path.read_text(encoding='utf8'))
    scaler = joblib.load(scaler_path)
    return meta, enc, scaler

def load_tfidf(vec_path: Path):
    if not vec_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vec_path}")
    tfidf = joblib.load(vec_path)
    return tfidf

def make_dense_features_from_row(row, encoders):
    """
    Given a pandas Series (one row) produce dense features array in same order as feature_metadata.dense_feature_names
    The features created here must match those used by src/features.dense_features_from_df
    """
    # defensive access with defaults
    text_len = len(str(row.get('text_clean', row.get('raw_text', ""))))
    num_exclaims = str(row.get('raw_text', "")).count('!')
    num_questions = str(row.get('raw_text', "")).count('?')
    num_words = len(str(row.get('text_clean', row.get('raw_text', ""))).split())
    # product freq encoding - we don't have counts here so fallback to 0
    product_freq_enc = 0.0
    if 'product' in row and encoders.get('product_freq_keys'):
        # not ideal — this script cannot know global freq without train metadata
        if row['product'] in encoders.get('product_freq_keys'):
            product_freq_enc = 1.0  # crude placeholder
    # company mapped freq — similarly fallback
    company_mapped_freq = 0.0
    # state index if available from encoders
    state_idx = -1
    if 'state' in row and 'states' in encoders:
        try:
            state_idx = encoders['states'].index(row['state'])
        except ValueError:
            state_idx = -1

    dense = np.array([text_len, num_exclaims, num_questions, num_words,
                      product_freq_enc, company_mapped_freq, float(state_idx)], dtype=float)
    return dense.reshape(1, -1)

def combine_tfidf_and_dense(tfidf_vec, dense_np):
    # tfidf_vec: sparse (1, n_tfidf), dense_np: (1, n_dense)
    dense_sparse = sparse.csr_matrix(dense_np)
    combined = sparse.hstack([tfidf_vec, dense_sparse], format='csr')
    return combined

# -------------------------
# Explain / predict helpers
# -------------------------
def try_shap_explain(model, X_combined, feature_meta, vec, dense_names, out_path: Path, index_label="0"):
    """
    Try to produce SHAP explanation and save plot.
    For tree models we use TreeExplainer; for linear models use LinearExplainer if available.
    """
    if not SHAP_AVAILABLE:
        print("SHAP not installed — skip SHAP explanations.")
        return None

    explainer = None
    try:
        # prefer TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
    except Exception:
        try:
            explainer = shap.Explainer(model)
        except Exception as e:
            print("Could not build SHAP explainer for model:", e)
            return None

    # compute shap values (may be slow for large models)
    shap_values = explainer.shap_values(X_combined)
    # shap_values may be list (for multiclass) or array
    # For binary classification with scikit-learn XGBoost: shap_values[1] is positive class contribution
    vals = shap_values
    if isinstance(shap_values, list):
        # pick the class 1 if binary
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Build feature names: vocab terms (only top N) + dense names
    # For readability we will show top 20 contributors by absolute value
    # Obtain TF-IDF vocab terms (if available)
    vocab = {}
    try:
        # vec is tfidf vectorizer instance
        if hasattr(vec, 'get_feature_names_out'):
            vocab = vec.get_feature_names_out()
        elif hasattr(vec, 'vocabulary_'):
            # building array sorted by index
            inv = {v:k for k,v in vec.vocabulary_.items()}
            vocab = [inv[i] for i in range(len(inv))]
    except Exception:
        vocab = []

    # Combine names
    tfidf_len = X_combined.shape[1] - len(dense_names)
    tfidf_names = [vocab[i] if i < len(vocab) else f"tfidf_{i}" for i in range(tfidf_len)]
    combined_names = list(tfidf_names) + dense_names

    # shap values for this single sample
    sv = vals if isinstance(vals, np.ndarray) and vals.ndim == 1 else np.array(vals).reshape(-1)
    if sv.shape[0] != len(combined_names):
        # Try to flatten if it's (1,n)
        sv = sv.flatten()
    # pick top contributors
    topk = 30
    idxs = np.argsort(np.abs(sv))[-topk:][::-1]
    top_names = [combined_names[i] for i in idxs]
    top_vals = sv[idxs]

    # plot horizontal bar
    fig, ax = plt.subplots(figsize=(8, min(6, len(top_names)*0.3 + 1)))
    colors = ['green' if v>0 else 'red' for v in top_vals]
    ax.barh(range(len(top_names))[::-1], top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(f"SHAP explanation (top {topk}) for sample {index_label}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--vectorizer", required=True, help="Path to TF-IDF vectorizer (.joblib)")
    parser.add_argument("--features", default="outputs/features", help="Directory where features metadata/scaler/encoders live")
    parser.add_argument("--input-csv", help="Optional CSV file containing texts to predict (column name provided by --text-col)")
    parser.add_argument("--text-col", default="raw_text", help="Column in CSV that contains raw text")
    parser.add_argument("--text", help="Single text input to predict (use quotes). If both CSV and text provided, CSV will be used first.")
    parser.add_argument("--out-dir", default="outputs/models", help="Where to save explanations/plots")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of rows from CSV to explain")
    args = parser.parse_args()

    model_path = Path(args.model)
    vec_path = Path(args.vectorizer)
    features_dir = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("Model not found:", model_path)
        sys.exit(1)
    if not vec_path.exists():
        print("TF-IDF vectorizer not found:", vec_path)
        sys.exit(1)

    # load model
    model = joblib.load(model_path)
    # load metadata/scaler/encoders
    feature_meta, encoders, scaler = load_metadata(features_dir)
    tfidf = load_tfidf(vec_path)

    dense_names = feature_meta.get("dense_feature_names", ["text_len","num_exclaims","num_questions","num_words","product_freq_enc","company_mapped_freq","state_idx"])

    # prepare input dataframe
    if args.input_csv:
        df_in = pd.read_csv(args.input_csv)
        if args.text_col not in df_in.columns:
            print(f"Warning: text column {args.text_col} not found in CSV. Trying raw_text/text_clean columns.")
            if 'raw_text' in df_in.columns:
                text_col = 'raw_text'
            elif 'text_clean' in df_in.columns:
                text_col = 'text_clean'
            else:
                raise ValueError("No usable text column found in CSV.")
        else:
            text_col = args.text_col
        df_in = df_in.head(args.limit).copy()
        df_in['raw_text'] = df_in[text_col].astype(str)
        df_in['text_clean'] = df_in.get('text_clean', df_in['raw_text']).astype(str)
    elif args.text:
        df_in = pd.DataFrame([{ 'raw_text': args.text, 'text_clean': args.text }])
    else:
        print("Provide either --input-csv or --text")
        sys.exit(1)

    # For each row: build TF-IDF vector, dense features, scale, combine -> predict -> explain
    texts = df_in['text_clean'].fillna("").astype(str).tolist()
    X_tfidf = tfidf.transform(texts)  # sparse (n_samples, n_tfidf)
    dense_list = []
    for idx, row in df_in.iterrows():
        dense_np = make_dense_features_from_row(row, encoders)  # (1, n_dense)
        dense_list.append(dense_np)
    X_dense = np.vstack(dense_list)  # (n, n_dense)
    # scale dense
    X_dense_scaled = scaler.transform(X_dense)
    # combine
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_dense_scaled)], format='csr')

    # predictions
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_combined)[:,1]
    else:
        # fallback to decision_function or raw predictions
        try:
            probs = model.decision_function(X_combined)
            # try to map to [0,1] via logistic
            probs = 1/(1+np.exp(-probs))
        except Exception:
            probs = model.predict(X_combined)

    preds = model.predict(X_combined)

    # show results
    for i, (pred, prob, text) in enumerate(zip(preds, probs, df_in['raw_text'].tolist())):
        print(f"=== SAMPLE {i} ===")
        print("Text:", text[:400].replace("\n"," "))
        print("Predicted timely_response:", int(pred), "  Prob:", float(prob))
        # explanation
        shap_out = out_dir / f"shap_{model_path.stem}_{i}.png"
        try:
            expl_path = try_shap_explain(model, X_combined[i], feature_meta, tfidf, dense_names, shap_out, index_label=str(i))
            if expl_path:
                print("Saved SHAP plot ->", expl_path)
            else:
                # fallback: print simple feature importances (if available)
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    # show top 10 features (note: mapping features back to names for TF-IDF is heavy)
                    top_idx = np.argsort(fi)[-10:][::-1]
                    print("Top feature indices and importances (tfidf+dense):")
                    for ti in top_idx:
                        print(f" idx {ti}: importance {fi[ti]:.6f}")
                else:
                    print("No SHAP and model has no feature_importances_.")
        except Exception as e:
            print("Could not compute explanation:", e)

    print("Done.")

if __name__ == "__main__":
    main()
