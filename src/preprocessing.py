#!/usr/bin/env python3
"""
preprocessing.py  (Kaggle: US Consumer Finance Complaints)

Usage:
    python src/preprocessing.py data/raw/complaints.csv data/processed/

What it does:
 - Loads the raw CSV (single file or directory of CSVs)
 - Normalizes column names (several common variants)
 - Creates a clean text field: combines 'Consumer complaint narrative' with 'Issue' and 'Sub-issue'
 - Creates a binary target: timely_response = 1 if 'Timely response?' == 'Yes' else 0 (handles missing)
 - Builds simple numeric features: text_length, num_exclaims, num_questions
 - Fits a TF-IDF vectorizer on the complaint text and saves it to outputs/models/
 - Splits into train/val/test by date (configurable); default: 70/15/15 chronological split
 - Saves processed CSV(s) and vectorizer
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------
# Helpers
# ---------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lowercase columns for easier matching
    df = df.rename(columns={c: c.strip() for c in df.columns})
    cols_lower = {c: c.lower() for c in df.columns}
    # mapping of common names in the Kaggle dataset
    rename_map = {}
    inv = {k.lower(): k for k in df.columns}
    def has(k): 
        return k in inv
    # Map canonical names we use in pipeline
    if 'date received' in cols_lower:
        rename_map[inv['date received']] = 'date_received'
    if 'date_sent_to_company' in cols_lower or 'date sent to company' in cols_lower:
        # prefer 'Date sent to company' variations
        if 'date sent to company' in cols_lower:
            rename_map[inv['date sent to company']] = 'date_sent_to_company'
        else:
            rename_map[inv['date_sent_to_company']] = 'date_sent_to_company'
    if 'product' in cols_lower:
        rename_map[inv['product']] = 'product'
    if 'sub-product' in cols_lower:
        rename_map[inv['sub-product']] = 'sub_product'
    if 'issue' in cols_lower:
        rename_map[inv['issue']] = 'issue'
    if 'sub-issue' in cols_lower:
        rename_map[inv['sub-issue']] = 'sub_issue'
    # narrative column has different names sometimes
    if 'consumer complaint narrative' in cols_lower:
        rename_map[inv['consumer complaint narrative']] = 'consumer_complaint_narrative'
    elif 'complaint_what_happened' in cols_lower:
        rename_map[inv['complaint_what_happened']] = 'consumer_complaint_narrative'
    elif 'consumer_complaint_narrative' in cols_lower:
        rename_map[inv['consumer_complaint_narrative']] = 'consumer_complaint_narrative'
    if 'company' in cols_lower:
        rename_map[inv['company']] = 'company'
    if 'state' in cols_lower:
        rename_map[inv['state']] = 'state'
    if 'zip code' in cols_lower or 'zipcode' in cols_lower:
        if 'zip code' in cols_lower:
            rename_map[inv['zip code']] = 'zip_code'
        else:
            rename_map[inv['zipcode']] = 'zip_code'
    if 'submitted via' in cols_lower:
        rename_map[inv['submitted via']] = 'submitted_via'
    if 'company response to consumer' in cols_lower:
        rename_map[inv['company response to consumer']] = 'company_response'
    if 'timely response?' in cols_lower:
        rename_map[inv['timely response?']] = 'timely_response'
    if 'consumer disputed?' in cols_lower:
        rename_map[inv['consumer disputed?']] = 'consumer_disputed'
    # Apply renames
    df = df.rename(columns=rename_map)
    return df

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    # basic cleaning
    s = str(s)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'http\S+', ' ', s)             # remove URLs
    s = re.sub(r'\s+', ' ', s)                 # collapse whitespace
    s = s.strip()
    return s

def text_preprocess_pipeline(df: pd.DataFrame, text_col: str) -> pd.Series:
    # Lowercase and minimal cleaning (keep punctuation for features)
    texts = df[text_col].fillna("").astype(str).map(clean_text).map(lambda t: t.lower())
    return texts

# ---------------------------
# Main processing function
# ---------------------------
def process_file(in_path: Path, out_dir: Path, vectorizer_out: Path, test_size=0.15, val_size=0.15, random_state=42):
    print("Loading:", in_path)
    df = pd.read_csv(in_path, parse_dates=[col for col in ['Date received','date_received'] if col in pd.read_csv(in_path, nrows=0).columns], infer_datetime_format=True)
    df = normalize_cols(df)
    # Ensure date field exists
    if 'date_received' not in df.columns:
        # try variants
        for c in df.columns:
            if 'date' in c.lower():
                df = df.rename(columns={c: 'date_received'})
                break
    # parse date_received to datetime
    if 'date_received' in df.columns:
        df['date_received'] = pd.to_datetime(df['date_received'], errors='coerce')
    else:
        # create a fake date if missing (not ideal)
        df['date_received'] = pd.NaT

    # Compose a single text field
    narrative_col = 'consumer_complaint_narrative' if 'consumer_complaint_narrative' in df.columns else None
    pieces = []
    if 'issue' in df.columns:
        pieces.append(df['issue'].fillna("").astype(str))
    if 'sub_issue' in df.columns:
        pieces.append(df['sub_product'].fillna("").astype(str) if 'sub_product' in df.columns else df['sub_issue'].fillna("").astype(str))
    if narrative_col:
        pieces.append(df[narrative_col].fillna("").astype(str))
    # fallback: combine all text-like columns
    if not pieces:
        txt_cols = [c for c in df.columns if df[c].dtype == object][:3]
        pieces = [df[c].fillna("").astype(str) for c in txt_cols]

    df['raw_text'] = [" ".join(parts).strip() for parts in zip(*pieces)]

    # cleaned, lowercased text for vectorization
    df['text_clean'] = text_preprocess_pipeline(df, 'raw_text')

    # Target: timely_response -> binary
    if 'timely_response' in df.columns:
        df['timely_response'] = df['timely_response'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    else:
        # If missing, try to infer from other columns or create NaN
        df['timely_response'] = np.nan

    # Simple engineered features
    df['text_len'] = df['text_clean'].map(len)
    df['num_exclaims'] = df['raw_text'].map(lambda s: s.count('!'))
    df['num_questions'] = df['raw_text'].map(lambda s: s.count('?'))
    df['num_words'] = df['text_clean'].map(lambda s: len(s.split()))

    # Categorical simplifications
    if 'product' in df.columns:
        df['product'] = df['product'].fillna("UNKNOWN").astype(str)
    if 'company' in df.columns:
        df['company'] = df['company'].fillna("UNKNOWN").astype(str)
    if 'state' in df.columns:
        df['state'] = df['state'].fillna("UNK").astype(str)

    # Drop rows with no text
    df = df[df['text_clean'].str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    # Save a copy of processed raw table
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_csv = out_dir / "processed_all.csv"
    df.to_csv(proc_csv, index=False)
    print("Saved processed CSV ->", proc_csv)

    # Fit TF-IDF vectorizer on the cleaned text
    print("Fitting TF-IDF vectorizer on text (may take a moment)...")
    tfidf = TfidfVectorizer(max_features=20000, min_df=3, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df['text_clean'])
    vectorizer_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, vectorizer_out)
    print("Saved TF-IDF vectorizer ->", vectorizer_out)

    # Optionally create a train/val/test chronological split
    if 'date_received' in df.columns and df['date_received'].notna().sum() > 0:
        df_sorted = df.sort_values('date_received').reset_index(drop=True)
        n = len(df_sorted)
        n_test = int(np.ceil(n * test_size))
        n_val = int(np.ceil(n * val_size))
        n_train = n - n_val - n_test
        if n_train < 10:
            print("Warning: small training set with chronological split; falling back to random split")
            train, temp = train_test_split(df, test_size=(test_size + val_size), random_state=random_state, stratify=None)
            val, test = train_test_split(temp, test_size=test_size/(test_size+val_size), random_state=random_state)
        else:
            train = df_sorted.iloc[:n_train].reset_index(drop=True)
            val = df_sorted.iloc[n_train:n_train + n_val].reset_index(drop=True)
            test = df_sorted.iloc[n_train + n_val:].reset_index(drop=True)
    else:
        # random stratified split by target if possible
        if df['timely_response'].notna().sum() > 0:
            train, temp = train_test_split(df, test_size=(test_size + val_size), random_state=random_state, stratify=df['timely_response'].fillna(0))
            val, test = train_test_split(temp, test_size=test_size/(test_size+val_size), random_state=random_state, stratify=temp['timely_response'].fillna(0))
        else:
            train, temp = train_test_split(df, test_size=(test_size + val_size), random_state=random_state)
            val, test = train_test_split(temp, test_size=test_size/(test_size+val_size), random_state=random_state)

    # Save splits (without TF-IDF expansion) - will be used by modelling step
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved splits -> train:{len(train)} val:{len(val)} test:{len(test)}")

    # Also save sparse TF-IDF matrices if user wants (optional); here we save as .npz
    try:
        from scipy import sparse
        sparse_out_dir = out_dir / "tfidf"
        sparse_out_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(sparse_out_dir / "X_tfidf.npz", X_tfidf)
        print("Saved TF-IDF matrix ->", sparse_out_dir / "X_tfidf.npz")
    except Exception as e:
        print("Could not save TF-IDF matrix (scipy maybe missing):", e)

    return {
        "processed_csv": str(proc_csv),
        "train_csv": str(out_dir / "train.csv"),
        "val_csv": str(out_dir / "val.csv"),
        "test_csv": str(out_dir / "test.csv"),
        "tfidf_vectorizer": str(vectorizer_out)
    }

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess consumer complaints CSV")
    parser.add_argument("input", help="Path to raw CSV (or directory with CSVs) - e.g. data/raw/complaints.csv")
    parser.add_argument("outdir", help="Output directory for processed data - e.g. data/processed/")
    parser.add_argument("--vectorizer-out", default="outputs/models/tfidf_vectorizer.joblib", help="Path to save TF-IDF vectorizer")
    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.outdir)
    vec_out = Path(args.vectorizer_out)
    if inp.is_dir():
        # if directory, try to find a single CSV inside
        csvs = list(inp.glob("*.csv"))
        if not csvs:
            print("No CSV files found in directory:", inp)
            sys.exit(1)
        inp_file = csvs[0]
    else:
        inp_file = inp
    result = process_file(inp_file, out, vec_out)
    print("Done. Artifacts:", result)
