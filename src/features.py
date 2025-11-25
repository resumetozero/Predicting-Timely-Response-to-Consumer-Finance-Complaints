#!/usr/bin/env python3
"""
src/features.py

Build feature matrices for modelling from the processed Kaggle complaints dataset.

Usage (CLI example):
    python src/features.py \
      --processed-dir data/processed \
      --vectorizer outputs/models/tfidf_vectorizer.joblib \
      --out-dir outputs/features \
      --top-companies 100

What it produces (in --out-dir):
 - train_X.npz, val_X.npz, test_X.npz  (sparse feature matrices: [tfidf | dense features])
 - train_y.npy, val_y.npy, test_y.npy  (labels array, np.nan replaced with -1)
 - scaler.joblib                         (StandardScaler fitted on dense features of train)
 - encoders.json                         (mappings used: company_map, product_freqs)
 - feature_metadata.json                 (names & ordering of dense features)
"""

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_TFIDF = "outputs/models/tfidf_vectorizer.joblib"

# ------- Helper utilities -------

def load_tfidf(path: Path) -> TfidfVectorizer:
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {path}")
    tfidf = joblib.load(path)
    return tfidf

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def safe_label_array(df: pd.DataFrame, label_col="timely_response"):
    # If label is missing, convert to -1 (useful to detect unlabeled rows).
    if label_col in df.columns:
        y = df[label_col].fillna(-1).astype(float).to_numpy()
    else:
        y = np.full(len(df), -1.0)
    return y

# ------- Encoders: frequency encoding & top-K mapping -------

def build_encoders(train_df: pd.DataFrame, top_k_companies=100):
    encoders = {}
    # product frequency encoding
    if 'product' in train_df.columns:
        prod_counts = train_df['product'].fillna("UNKNOWN").astype(str).value_counts()
        prod_freq = (prod_counts / prod_counts.sum()).to_dict()
        encoders['product_freq'] = prod_freq
    else:
        encoders['product_freq'] = {}

    # company: keep top_k as individual, map rest to 'OTHER'
    if 'company' in train_df.columns:
        comps = train_df['company'].fillna("UNKNOWN").astype(str)
        topk = list(comps.value_counts().nlargest(top_k_companies).index)
        company_map = {c: c if c in topk else 'OTHER' for c in comps.unique()}
        encoders['company_map'] = company_map
        encoders['top_companies'] = topk
    else:
        encoders['company_map'] = {}
        encoders['top_companies'] = []

    # state: we won't one-hot by default (too many states), but save unique states
    if 'state' in train_df.columns:
        states = list(train_df['state'].fillna("UNK").astype(str).unique())
        encoders['states'] = states
    else:
        encoders['states'] = []

    return encoders

def apply_encoders(df: pd.DataFrame, encoders: dict):
    # product_freq
    if 'product' in df.columns and encoders.get('product_freq'):
        df['product'] = df['product'].fillna("UNKNOWN").astype(str)
        df['product_freq_enc'] = df['product'].map(encoders['product_freq']).fillna(0.0)
    else:
        df['product_freq_enc'] = 0.0

    # company_map -> company_topk one-hot-ish via label mapping (we will one-hot later if needed)
    if 'company' in df.columns and encoders.get('company_map'):
        df['company_raw'] = df['company'].fillna("UNKNOWN").astype(str)
        df['company_mapped'] = df['company_raw'].map(encoders['company_map']).fillna('OTHER')
        # frequency of mapped company (as numeric)
        counts = {}
        for c in df['company_mapped'].unique():
            counts[c] = (df['company_mapped'] == c).sum()
        df['company_mapped_freq'] = df['company_mapped'].map(lambda x: counts.get(x, 0) / max(1, len(df)))
    else:
        df['company_mapped_freq'] = 0.0

    # state -> ordinal mapping of US states (not ordered by anything meaningful)
    if 'state' in df.columns and encoders.get('states'):
        state_to_idx = {s: i for i, s in enumerate(encoders['states'])}
        df['state_idx'] = df['state'].fillna("UNK").astype(str).map(lambda s: state_to_idx.get(s, -1))
    else:
        df['state_idx'] = -1

    return df

# ------- Dense features creation -------

def dense_features_from_df(df: pd.DataFrame, include_counts=True):
    """
    Returns dense features DataFrame (numeric) and feature_names list.
    Keep this deterministic and lightweight.
    """
    feats = {}
    # text-derived numeric features (these were created by preprocessing)
    feats['text_len'] = df.get('text_len', df.get('raw_text', pd.Series([""]*len(df))).map(lambda s: len(str(s))))
    feats['num_exclaims'] = df.get('num_exclaims', 0).fillna(0)
    feats['num_questions'] = df.get('num_questions', 0).fillna(0)
    feats['num_words'] = df.get('num_words', df.get('text_clean', df.get('raw_text', pd.Series([""]*len(df)))).map(lambda s: len(str(s).split())))
    # encoded categorical numeric
    feats['product_freq_enc'] = df.get('product_freq_enc', 0).fillna(0)
    feats['company_mapped_freq'] = df.get('company_mapped_freq', 0).fillna(0)
    feats['state_idx'] = df.get('state_idx', -1).fillna(-1)

    dense_df = pd.DataFrame(feats, index=df.index).astype(float)
    feature_names = list(dense_df.columns)
    return dense_df, feature_names

# ------- Main pipeline -------

def build_and_save_features(processed_dir: Path,
                            vectorizer_path: Path = Path(DEFAULT_TFIDF),
                            out_dir: Path = Path("outputs/features"),
                            top_companies: int = 100):
    """
    processed_dir should contain train.csv, val.csv, test.csv and processed_all.csv as produced
    by preprocessing.py.
    """
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TF-IDF vectorizer
    tfidf = load_tfidf(Path(vectorizer_path))

    # Load splits
    train_df = load_csv(processed_dir / "train.csv")
    val_df = load_csv(processed_dir / "val.csv")
    test_df = load_csv(processed_dir / "test.csv")

    # Build encoders from train
    encoders = build_encoders(train_df, top_k_companies=top_companies)
    # Apply encoders
    train_df = apply_encoders(train_df, encoders)
    val_df = apply_encoders(val_df, encoders)
    test_df = apply_encoders(test_df, encoders)

    # Dense features
    X_train_dense_df, dense_feature_names = dense_features_from_df(train_df)
    X_val_dense_df, _ = dense_features_from_df(val_df)
    X_test_dense_df, _ = dense_features_from_df(test_df)

    # Fit scaler on train dense features
    scaler = StandardScaler()
    scaler.fit(X_train_dense_df)
    X_train_dense = scaler.transform(X_train_dense_df)
    X_val_dense = scaler.transform(X_val_dense_df)
    X_test_dense = scaler.transform(X_test_dense_df)

    # TF-IDF sparse matrices (use text_clean column)
    def tfidf_transform(df):
        if 'text_clean' in df.columns:
            texts = df['text_clean'].fillna("").astype(str).tolist()
        elif 'raw_text' in df.columns:
            texts = df['raw_text'].fillna("").astype(str).tolist()
        else:
            texts = [""] * len(df)
        X = tfidf.transform(texts)
        return X

    X_train_tfidf = tfidf_transform(train_df)
    X_val_tfidf = tfidf_transform(val_df)
    X_test_tfidf = tfidf_transform(test_df)

    # Combine sparse TF-IDF and dense features into a single sparse matrix
    def combine_sparse_and_dense(X_sparse, X_dense_np):
        # convert dense to sparse CSR
        X_dense_sparse = sparse.csr_matrix(X_dense_np)
        combined = sparse.hstack([X_sparse, X_dense_sparse], format='csr')
        return combined

    X_train = combine_sparse_and_dense(X_train_tfidf, X_train_dense)
    X_val = combine_sparse_and_dense(X_val_tfidf, X_val_dense)
    X_test = combine_sparse_and_dense(X_test_tfidf, X_test_dense)

    # Labels
    y_train = safe_label_array(train_df)
    y_val = safe_label_array(val_df)
    y_test = safe_label_array(test_df)

    # Persist everything
    sparse.save_npz(out_dir / "train_X.npz", X_train)
    sparse.save_npz(out_dir / "val_X.npz", X_val)
    sparse.save_npz(out_dir / "test_X.npz", X_test)

    np.save(out_dir / "train_y.npy", y_train)
    np.save(out_dir / "val_y.npy", y_val)
    np.save(out_dir / "test_y.npy", y_test)

    # Save metadata: scaler, encoders, dense feature names, tfidf feature size
    joblib.dump(scaler, out_dir / "scaler.joblib")
    with open(out_dir / "encoders.json", "w", encoding="utf8") as f:
        json.dump({
            "product_freq_keys": list(encoders.get('product_freq', {}).keys()),
            "top_companies": encoders.get('top_companies', []),
            "states": encoders.get('states', []),
        }, f, indent=2)
    # Save feature metadata
    feature_meta = {
        "dense_feature_names": dense_feature_names,
        "tfidf_vocab_size": len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else None,
        "sparse_plus_dense_shape_train": (X_train.shape[0], X_train.shape[1])
    }
    with open(out_dir / "feature_metadata.json", "w", encoding="utf8") as f:
        json.dump(feature_meta, f, indent=2)

    print("Saved features to", out_dir)
    return out_dir

# ------- CLI entrypoint -------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory with train.csv val.csv test.csv")
    parser.add_argument("--vectorizer", type=str, default=DEFAULT_TFIDF, help="Path to saved TF-IDF vectorizer (.joblib)")
    parser.add_argument("--out-dir", type=str, default="outputs/features", help="Where to save feature matrices and metadata")
    parser.add_argument("--top-companies", type=int, default=100, help="Keep top-K companies as separate categories; rest -> OTHER")
    args = parser.parse_args()
    build_and_save_features(args.processed_dir, Path(args.vectorizer), Path(args.out_dir), top_companies=args.top_companies)
