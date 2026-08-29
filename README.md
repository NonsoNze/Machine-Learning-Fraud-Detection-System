# Fraud Detection — Two Implementations

This repository contains two independent fraud detection projects, each demonstrating a
different architecture and approach to the same problem. They are designed to be read
together: the first establishes the core concepts and hybrid architecture; the second
applies a production-scale pipeline to real, competition-grade data.

---

## Projects

### 1. `synthetic/` — Hybrid Ensemble on Synthetic Data

A fraud detection system combining an unsupervised Isolation Forest with a supervised
Gradient Boosting classifier in a weighted ensemble (80% supervised, 20% anomaly signal).

The hybrid architecture addresses a fundamental limitation of purely supervised fraud
detection: a supervised model can only recognise fraud patterns it has seen labeled examples
of. The Isolation Forest provides a second signal that flags statistically unusual behaviour
regardless of whether it matches any known fraud type, making the system more robust to
novel tactics.

Built as a Python script with a real-time transaction scoring interface that returns a fraud
probability, anomaly flag, ensemble score, and a tiered ALLOW/BLOCK decision. Includes a
custom scikit-learn-compatible feature transformer and model persistence via joblib.

**Key design decisions:**
- Unsupervised + supervised ensemble to balance known-pattern recognition with novel anomaly detection
- Seven domain-driven engineered features including transaction velocity ratios and composite risk scoring
- Five-fold stratified cross-validation on a realistically imbalanced dataset (2.5% fraud rate)
- Real-time scoring interface designed for deployment readiness

**Stack:** Python, scikit-learn (Isolation Forest, Gradient Boosting), pandas, NumPy, joblib

---

### 2. `ieee-cis/` — Hybrid Ensemble on Real Transaction Data

A production-style fraud detection pipeline built on real anonymised transaction data from
the IEEE-CIS Fraud Detection Kaggle competition, provided by Vesta Corporation. This project
applies the same hybrid ensemble architecture as the synthetic project by combining a supervised
LightGBM classifier with an unsupervised Isolation Forest, but adapted for the scale and
complexity of real-world fraud data i.e. genuine class imbalance, high-dimensional sparse features,
and 590,000 transactions.

A key distinction from the synthetic project is the use of the unlabeled Kaggle test set.
Because the Isolation Forest requires no ground truth labels, it is applied to the test set
directly, scoring genuinely unseen transactions and blending the result with the LightGBM
supervised signal in the same 80/20 ensemble. This mirrors real production deployment where
transactions arrive unlabeled and must be assessed in real time.

Built as a Jupyter notebook with end-to-end documentation of every decision from data
loading through to a leakage audit and a real-time hybrid scoring interface.

**Key design decisions:**
- Hybrid ensemble: LightGBM (supervised, trained on labeled data) + Isolation Forest (unsupervised, applied to unlabeled test set) blended 80/20
- LightGBM selected for efficiency at scale, native missing value handling, and strong performance on tabular fraud data
- Isolation Forest applied to test set to demonstrate production-style scoring on genuinely unseen transactions
- Explicit class imbalance handling via class weighting and precision-recall threshold tuning
- Stratified evaluation across transaction amount ranges to understand where model reliability holds up and where it does not
- Leakage audit documenting data preparation risks and anomaly score normalisation limitation

**Stack:** Python, LightGBM, scikit-learn (Isolation Forest), pandas, NumPy, matplotlib, seaborn

---

## Why Two Projects

Both projects use the same hybrid ensemble architecture, supervised classifier combined
with unsupervised Isolation Forest in an 80/20 blend. The difference lies in the data, the scale,
the models chosen within that architecture, and how the unlabeled test set is used.

| | Synthetic | IEEE-CIS |
|---|---|---|
| Data | Synthetic, controlled | Real, anonymised |
| Rows | 20,000 | 590,540 training + 506,691 test |
| Fraud rate | 2.5% | 3.5% |
| Features | 14 raw, 7 engineered | 434 post-merge |
| Architecture | Hybrid ensemble (unsupervised + supervised) | Hybrid ensemble (unsupervised + supervised) |
| Supervised model | Gradient Boosting (sklearn) | LightGBM |
| Unsupervised model | Isolation Forest (on training data) | Isolation Forest (applied to unlabeled test set) |
| Imbalance handling | Ensemble anomaly signal | Class weighting + threshold tuning |
| Test set usage | N/A (synthetic) | Unsupervised scoring on genuinely unseen transactions |
| Format | Python script | Jupyter notebook |

The synthetic project demonstrates the architecture cleanly on controlled data, with a
custom scikit-learn transformer and deployment-ready model persistence. The IEEE project
demonstrates what the same architecture looks like at production scale, on messy real data,
with the Isolation Forest applied to the genuinely unlabeled Kaggle test set — the closest
thing to real deployment conditions available without a live transaction stream.

---

## Repository Structure

```
fraud-detection/
    synthetic/
        fraud_detection.py       # Full pipeline and real-time scorer
        README.md
    ieee-cis/
        IEEE_Fraud_Detection.ipynb   # End-to-end notebook
        README.md
    README.md                    # This file
```

---

## Requirements

Both projects use Python 3.9+. Install dependencies for each project separately:

**Synthetic:**
```bash
pip install scikit-learn pandas numpy joblib
```

**IEEE-CIS:**
```bash
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn
```

---

## Data

The synthetic project generates its own data and requires no external files.

The IEEE-CIS project requires downloading all four competition files from
[Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) (free, requires joining
the competition). Place `train_transaction.csv`, `train_identity.csv`,
`test_transaction.csv`, and `test_identity.csv` in the `ieee-cis/` directory before
running the notebook. The training files are used for model development and evaluation;
the test files are used for the unsupervised Isolation Forest scoring in Section 12.5.
