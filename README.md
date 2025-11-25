# 📘 **Predicting Timely Response to Consumer Finance Complaints**

### *Final Year CSE Project – Machine Learning + NLP + Data Analytics *

---

## 🧑‍🎓 **Team Details**

**Branch:** Computer Science & Engineering (CSE)

**Members:**

* **22CS3065 - Utsav Shingala**
* **22CS3020 - Ayush Mittal**
* **22CS3047 - Rishabh Kumar**
* **22CS3001 - Aakarsh Verma**
---

# 📌 **Project Overview**

This project builds a complete **machine learning pipeline** to predict whether a consumer complaint submitted to a US financial institution receives a **timely response**.
The complaint narratives are long, noisy, and extremely imbalanced (~98% “timely”), making this a real-world NLP classification challenge.

The pipeline includes:

* Data cleaning & preprocessing
* TF-IDF + engineered dense features
* Handling class imbalance
* Multiple ML models (LR, RF, XGBoost)
* Threshold tuning
* Evaluation & interpretability
* Gradio-based interactive web UI

This repository is structured for academic evaluation and real execution.

---

# 📂 **Repository Structure**

```
final_year_finance_project/
│
├── README.md                 # Project overview (this file)
├── report/  
│   └── report.md             # Full academic-style project report
│
├── data/
│   ├── raw/                  # Kaggle dataset (user-provided)
│   ├── processed/            # Cleaned + split CSVs
│   └── fetch_data.py         # Optional downloader (unused for Kaggle)
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── retrain_class_weighted.py
│   ├── retrain_rf_only.py
│   ├── tune_threshold_class0.py
│   ├── infer_and_explain.py
│   └── gradio_app.py         # Web UI
│
├── outputs/
│   ├── features/             # TF-IDF vectors, scalers
│   ├── models/               # Saved models + metrics + confusion matrices
│   └── figures/              # Plots generated during analysis
│
└── presentation/
    └── slides_template.md
```

---

# 📊 **Dataset**

**Source:** Kaggle – *US Consumer Finance Complaints*
**File:** `consumer_complaints.csv`

Key fields:

* Narrative text
* Product & Company metadata
* State
* Target: `Timely response? (Yes/No)`

**Class imbalance:**

* Timely: **~97.6%**
* Not timely: **~2.4%**

This imbalance shapes the whole modeling approach.

---

# 🧹 **Data Cleaning & Preprocessing**

Implemented in:
`src/preprocessing.py`

Key steps:

* Clean narrative text (lowercase, remove emails, numbers, symbols)
* Handle missing values
* Map target Yes/No → 1/0
* Stratified train/val/test split
* Save cleaned datasets

---

# 🛠️ **Feature Engineering**

Implemented in:
`src/features.py`

### **1. TF-IDF Representation**

* ~20,000 vocabulary terms
* character-level + word-level mix
* optimized for large sparse text

### **2. Dense Features**

* text length
* punctuation counts
* word count
* product frequency encoding
* company frequency encoding
* state encoding

### **3. Scaling**

* Dense features scaled using `StandardScaler`.

---

# 🤖 **Models Used**

Implemented in:
`src/models.py` and `src/retrain_class_weighted.py`

### **Baseline Models**

* Logistic Regression (unbalanced)
* Random Forest
* XGBoost

### **Balanced Models**

To address extreme imbalance:

* LR with `class_weight='balanced'`
* RF with `class_weight='balanced'`
* XGBoost with `scale_pos_weight`

### **Threshold Tuning**

Script:
`src/tune_threshold_class0.py`

Optimizes F1-score for minority class (0 = “Not Timely”).

---

# 📈 **Final Model Results (After Threshold Tuning)**

Using tuned threshold **0.8198** on `score0 = 1 - prob(class=1)`:

| Metric (Class 0)       | Test Value |
| ---------------------- | ---------- |
| **Precision**          | 0.136      |
| **Recall**             | 0.648      |
| **F1-score**           | 0.225      |
| **Accuracy (overall)** | 0.97       |

**Confusion Matrix (Test):**

```
[[1540,  837],      # class 0 actual
 [9758, 71259]]     # class 1 actual
```

### Interpretation:

* Predicts **most “Not Timely” cases** (good recall).
* Still produces some false positives (low precision).
* Suitable for a **screening system** where recall is more important.

---

# 🧪 **Validation & Performance Evaluation**

* Stratified splitting
* Precision/Recall/F1 used for class-specific evaluation
* ROC-AUC + PR-AUC computed
* Threshold tuned for optimal F1 minority class performance
* Confusion matrices and metrics auto-saved in `outputs/models`

---

# 🌐 **Interactive Demo (Gradio UI)**

Run:

```bash
uv run python src/gradio_app.py
```

Then open:

```
http://127.0.0.1:7860
```

Features:

* Live prediction from text
* SHAP explanation image
* Probability score
* Threshold-adjusted output

---

# ▶️ **How to Run the Entire Pipeline**

```bash
# 1. Install dependencies
uv sync
# or
pip install -r requirements.txt

# 2. Preprocess raw Kaggle dataset
uv run python src/preprocessing.py data/raw/consumer_complaints.csv data/processed/

# 3. Generate features
uv run python src/features.py --processed-dir data/processed --vectorizer outputs/models/tfidf_vectorizer.joblib --out-dir outputs/features

# 4. Train models
uv run python src/models.py --train all

# 5. Balanced training
uv run python src/retrain_class_weighted.py

# 6. Threshold tuning
uv run python src/tune_threshold_class0.py --model outputs/models/lr_balanced.joblib
```

---

# 📄 **Full Report**

The complete project report is available at:

📁 Report: https://docs.google.com/document/d/1OeivPbC8R_ApO0tbZDldAekHLIOr06lsByvWq9lm0Oo/edit?usp=sharing

---

# 📌 **Future Improvements**

* Use transformer models (DistilBERT, BERT)
* SMOTE + dimensionality reduction (SVD)
* Two-stage classifier (screening + precision filter)
* Deploy model as a cloud API


> NOTE: Large dataset and model artifacts are excluded from this GitHub repository.
> To reproduce results:
> 1. Download `consumer_complaints.csv` from Kaggle: https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints
> 2. Place the file at `data/raw/consumer_complaints.csv`
> 3. Run preprocessing, feature creation and training as described in the README.
