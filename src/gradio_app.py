#!/usr/bin/env python3
"""
src/gradio_app.py

Small Gradio demo for the Consumer Complaints classifier + explanations.

Requirements:
  pip install gradio joblib scipy scikit-learn numpy pandas matplotlib

Run:
  uv run python src/gradio_app.py
  OR
  python src/gradio_app.py

Open the printed local link (http://127.0.0.1:7860 by default) to use the UI.

Notes:
 - Expects these artifacts (produced earlier):
    - outputs/models/<model>.joblib       (one or more saved models)
    - outputs/models/tfidf_vectorizer.joblib
    - outputs/features/feature_metadata.json
    - outputs/features/encoders.json
    - outputs/features/scaler.joblib
 - SHAP plots are produced if `shap` is installed and the model is supported.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import time

from scipy import sparse

# try optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import gradio as gr
except Exception:
    raise RuntimeError("gradio not installed. Install with `pip install gradio`")

# default paths (adjust if you used different directories)
FEATURES_DIR = Path("outputs/features")
VECTORIZER_PATH = Path("outputs/models/tfidf_vectorizer.joblib")
MODELS_DIR = Path("outputs/models")

# -------------------------
# Load metadata, vectorizer, scaler, encoders
# -------------------------
def load_metadata(features_dir=FEATURES_DIR):
    meta_f = features_dir / "feature_metadata.json"
    enc_f = features_dir / "encoders.json"
    scaler_f = features_dir / "scaler.joblib"
    if not meta_f.exists() or not enc_f.exists() or not scaler_f.exists():
        raise FileNotFoundError(f"Missing feature artifacts in {features_dir}. Run src/features.py first.")
    meta = json.loads(meta_f.read_text(encoding="utf8"))
    enc = json.loads(enc_f.read_text(encoding="utf8"))
    scaler = joblib.load(scaler_f)
    return meta, enc, scaler

def load_tfidf(vec_path=VECTORIZER_PATH):
    if not vec_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vec_path}. Run src/preprocessing.py first.")
    return joblib.load(vec_path)

def list_models(models_dir=MODELS_DIR):
    if not models_dir.exists():
        return []
    return sorted([p.name for p in models_dir.glob("*.joblib")])

# load once
FEATURE_META, ENCODERS, SCALER = load_metadata()
TFIDF = load_tfidf()
AVAILABLE_MODELS = list_models()

# -------------------------
# Small feature reconstruction utilities (must match src/features.py)
# -------------------------
def make_dense_from_text(text, row_meta=None, encoders=ENCODERS):
    """
    Create dense features array for a single text example.
    Must match dense_features_from_df order used in src/features.py:
    ['text_len','num_exclaims','num_questions','num_words','product_freq_enc','company_mapped_freq','state_idx']
    """
    text_clean = str(text or "")
    raw_text = text_clean
    text_len = len(text_clean)
    num_exclaims = raw_text.count('!')
    num_questions = raw_text.count('?')
    num_words = len(text_clean.split())
    # placeholders for product/company/state — UI doesn't collect these; fallbacks to 0 / -1
    product_freq_enc = 0.0
    company_mapped_freq = 0.0
    state_idx = -1
    dense = np.array([text_len, num_exclaims, num_questions, num_words,
                      product_freq_enc, company_mapped_freq, float(state_idx)], dtype=float).reshape(1, -1)
    return dense

def combine_tfidf_and_dense(texts, dense_np, tfidf=TFIDF):
    X_tfidf = tfidf.transform(texts)
    dense_sparse = sparse.csr_matrix(dense_np)
    X_comb = sparse.hstack([X_tfidf, dense_sparse], format='csr')
    return X_comb

# -------------------------
# SHAP explanation helper
# -------------------------
def shap_explain_for_single(model, X_combined, vec=TFIDF, dense_names=None, topk=25):
    """
    Returns path to a saved PNG with SHAP-like bar plot (top contributors).
    If SHAP not available or explainer fails, returns None.
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        # build explainer (prefer TreeExplainer)
        explainer = None
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            try:
                explainer = shap.Explainer(model)
            except Exception:
                explainer = None

        if explainer is None:
            return None

        # compute shap values for the single sample
        sv = explainer.shap_values(X_combined)
        vals = sv
        if isinstance(sv, list):
            # pick class 1 if binary
            vals = sv[1] if len(sv) > 1 else sv[0]
        # ensure vector
        arr = np.array(vals).flatten()
        # build names for top features: try to extract TFIDF terms for the first part
        vocab = []
        try:
            if hasattr(vec, "get_feature_names_out"):
                vocab = vec.get_feature_names_out()
            elif hasattr(vec, "vocabulary_"):
                inv = {v:k for k,v in vec.vocabulary_.items()}
                vocab = [inv[i] for i in range(len(inv))]
        except Exception:
            vocab = []
        n_tfidf = (X_combined.shape[1] - (len(dense_names) if dense_names else 0))
        # produce combined names list (tfidf then dense)
        tfidf_names = []
        for i in range(n_tfidf):
            if i < len(vocab):
                tfidf_names.append(vocab[i])
            else:
                tfidf_names.append(f"tfidf_{i}")
        combined_names = tfidf_names + (dense_names or [])
        # reduce to topk contributors
        idxs = np.argsort(np.abs(arr))[-topk:][::-1]
        top_names = [combined_names[i] for i in idxs]
        top_vals = arr[idxs]
        # plot horizontal bars
        fig, ax = plt.subplots(figsize=(8, min(0.35*len(top_names)+1.5, 10)))
        colors = ['green' if v>0 else 'red' for v in top_vals]
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_vals[::-1], color=colors[::-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("SHAP value")
        ax.set_title("Top feature contributions (SHAP-like)")
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150)
        plt.close(fig)
        return tmp.name
    except Exception as e:
        print("SHAP explanation error:", e)
        return None

# -------------------------
# Gradio predict function
# -------------------------
def predict_single(text, model_name):
    # basic checks
    if model_name is None or model_name == "":
        return {"label": "No model", "prob": 0.0, "explain_path": None, "error": "Select a model from the dropdown."}

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return {"label": "Missing model", "prob": 0.0, "explain_path": None, "error": f"Model file {model_name} not found in {MODELS_DIR}"}

    # load model (cache might help, but keep simple)
    model = joblib.load(model_path)

    # create dense and tfidf combined features
    dense = make_dense_from_text(text)
    dense_scaled = SCALER.transform(dense)
    X = combine_tfidf_and_dense([text], dense_scaled)

    # predictions
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
        else:
            # fallback: decision_function -> sigmoid
            try:
                score = model.decision_function(X)[0]
                prob = float(1 / (1 + np.exp(-score)))
            except Exception:
                pred = model.predict(X)[0]
                prob = float(pred)
    except Exception as e:
        return {"label": "Error", "prob": 0.0, "explain_path": None, "error": f"Prediction failed: {e}"}

    label = int(prob >= 0.5)

    # SHAP explain (best effort)
    explain_path = None
    dense_names = FEATURE_META.get("dense_feature_names", ['text_len','num_exclaims','num_questions','num_words','product_freq_enc','company_mapped_freq','state_idx'])
    if SHAP_AVAILABLE:
        try:
            explain_path = shap_explain_for_single(model, X, vec=TFIDF, dense_names=dense_names, topk=20)
        except Exception as e:
            print("SHAP generation failed:", e)
            explain_path = None

    return {"label": str(label), "prob": prob, "explain_path": explain_path, "error": ""}

# -------------------------
# Gradio UI components
# -------------------------
def run_ui(host="127.0.0.1", port=7860):
    # Dropdown options
    models = AVAILABLE_MODELS
    if not models:
        models = ["(no models found in outputs/models/)"]

    with gr.Blocks(title="Consumer Complaints — Timely Response Predictor") as demo:
        gr.Markdown("## Consumer Complaints — Timely response prediction")
        gr.Markdown("Paste a complaint text below or upload a small CSV. Select a trained model and click *Predict*. SHAP explanations will be shown when available.")

        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(lines=8, placeholder="Paste complaint text here...", label="Complaint text")
                model_dd = gr.Dropdown(choices=models, value=models[0], label="Choose model (.joblib in outputs/models/)")
                predict_btn = gr.Button("Predict")
                csv_in = gr.File(
                    label="Optional: upload CSV (first column must be text) - will ignore if textbox has text",
                    file_count="single",
                    type="filepath"
                )
                csv_col = gr.Textbox(label="CSV text column name (if CSV uploaded)", placeholder="raw_text or text_clean (default: first column)")
                batch_btn = gr.Button("Run batch (CSV)")
            with gr.Column(scale=1):
                out_label = gr.Label(label="Predicted label (timely_response?)")
                out_prob = gr.Number(label="Predicted probability (class=1)", precision=4)
                out_error = gr.Textbox(label="Error / Notes", interactive=False)
                out_image = gr.Image(label="SHAP explanation (if available)", type="filepath")

        # single predict action
        def on_predict(text, model_name):
            if (text is None or str(text).strip() == ""):
                return {"label": "No input", "prob": 0.0, "explain_path": None, "error": "Provide complaint text or upload CSV."}
            return predict_single(text, model_name)

        predict_btn.click(on_predict, inputs=[text_in, model_dd], outputs=[out_label, out_prob, out_image, out_error])

        # batch predict (CSV)
        def on_batch(file_obj, colname, model_name):
            if file_obj is None:
                return "No file uploaded", None
            # load csv
            try:
                df = pd.read_csv(file_obj)

            except Exception as e:
                return f"CSV read error: {e}", None
            if colname and colname in df.columns:
                txt_col = colname
            else:
                # fallback to first object column
                txt_cols = [c for c in df.columns if df[c].dtype == object]
                if not txt_cols:
                    return "No text-like column found in CSV", None
                txt_col = txt_cols[0]
            texts = df[txt_col].astype(str).tolist()
            # load model
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                return f"Model not found: {model_name}", None
            model = joblib.load(model_path)
            # prepare features
            dense_list = [make_dense_from_text(t) for t in texts]
            dense_np = np.vstack(dense_list)
            dense_scaled = SCALER.transform(dense_np)
            X = combine_tfidf_and_dense(texts, dense_scaled)
            # predict probabilities (if available)
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:,1]
                else:
                    try:
                        scores = model.decision_function(X)
                        probs = 1/(1+np.exp(-scores))
                    except Exception:
                        probs = model.predict(X)
                preds = (probs >= 0.5).astype(int)
            except Exception as e:
                return f"Batch prediction failed: {e}", None
            # attach to df and provide download
            df["_pred_prob_"] = probs
            df["_pred_label_"] = preds
            out_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            df.to_csv(out_csv.name, index=False)
            return f"Batch done. {len(df)} rows processed. Download below.", out_csv.name

        batch_btn.click(on_batch, inputs=[csv_in, csv_col, model_dd], outputs=[out_error, out_image])

        # small footer
        gr.Markdown("App created for demo. Ensure you trained and saved at least one model in `outputs/models/` and built features using `src/features.py`.")

    demo.launch(server_name=host, server_port=port, debug=False, share=False)

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    # ensure global names exist in closure
    SCALER = SCALER
    FEATURE_META = FEATURE_META
    TFIDF = TFIDF
    MODELS_DIR = MODELS_DIR
    run_ui()
