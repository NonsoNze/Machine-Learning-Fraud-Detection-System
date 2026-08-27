# Machine-Learning-Fraud-Detection-System



An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, combining unsupervised anomaly detection with supervised classification in an ensemble architecture.

## Overview

This project simulates a realistic fraud detection workflow: generating transaction data, engineering domain-informed features, training a hybrid (unsupervised + supervised) model, evaluating it against industry-relevant metrics, and scoring transactions in real time with explainable risk decisions.

It's designed to reflect how fraud detection systems work in production — not just a single classifier, but a layered system that balances catching known fraud patterns with flagging novel, never-seen-before anomalies.

## Why a Hybrid Approach?

Fraud detection has a unique challenge: fraud patterns constantly evolve, and labeled fraud examples are rare and lagging (you only know it was fraud after the fact, often after a chargeback). Relying on a single supervised model risks missing new fraud tactics it's never seen labeled examples of.

This system blends two paradigms:

| Model | Type | Catches |
|---|---|---|
| **Isolation Forest** | Unsupervised | Novel, never-seen anomalies — doesn't need fraud labels |
| **Gradient Boosting** | Supervised | Known fraud patterns learned from historical labeled data |

The two scores are blended into a single ensemble score (80% supervised, 20% anomaly), giving the system both pattern-recognition power and resilience to unseen fraud tactics.

## Features

- **Synthetic data generator** — produces realistic transaction data with configurable fraud rate and embedded fraud signals (odd hours, high-risk countries, velocity spikes)
- **Custom feature engineering pipeline** — sklearn-compatible transformer producing 18 model-ready features from 14 raw fields
- **Ensemble fraud scoring** — Isolation Forest + Gradient Boosting blended into a single risk score
- **Real-time transaction scoring** — pass a single transaction, get back a structured decision
- **Risk tiering** — transactions are bucketed into `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` risk levels
- **Model evaluation suite** — classification report, ROC-AUC, average precision, confusion matrix, feature importance ranking
- **Model persistence** — save/load trained models via `joblib`

## Project Structure

```
fraud_detection.py
├── generate_transaction_data()       # Synthetic dataset generator
├── FraudFeatureEngineer              # Feature engineering transformer
├── AnomalyDetector                   # Isolation Forest wrapper
├── SupervisedFraudDetector           # Gradient Boosting wrapper
├── FraudDetectionSystem              # Orchestrates training, evaluation, scoring
└── main()                            # Trains, evaluates, and demos the system
```

## Installation

```bash
pip install scikit-learn pandas numpy joblib
```

## Usage

### Run the full pipeline (train, evaluate, demo)

```bash
python fraud_detection.py
```

This will:
1. Generate 20,000 synthetic transactions (2.5% fraud rate)
2. Train the feature pipeline, anomaly detector, and supervised model
3. Print evaluation metrics (classification report, ROC-AUC, confusion matrix, feature importances)
4. Score two example transactions (one legitimate, one fraudulent) in real time
5. Save the trained model to `fraud_detection_model.joblib`

### Use the trained system in your own code

```python
from fraud_detection import FraudDetectionSystem

# Load a previously trained model
system = FraudDetectionSystem.load("fraud_detection_model.joblib")

transaction = {
    "transaction_id": "TX_001",
    "amount": 1200.00,
    "merchant_type": "online",
    "country": "NG",
    "hour_of_day": 3,
    "day_of_week": 6,
    "card_present": 0,
    "transactions_last_1h": 5,
    "transactions_last_24h": 18,
    "avg_transaction_amount": 60.0,
    "distance_from_home_km": 8200.0,
    "new_merchant": 1,
    "declined_last_24h": 2,
    "is_international": 1,
}

result = system.score_transaction(transaction)
print(result)
# {
#   'transaction_id': 'TX_001',
#   'fraud_probability': 0.91,
#   'anomaly_flag': True,
#   'ensemble_score': 0.93,
#   'risk_level': 'CRITICAL',
#   'decision': 'BLOCK',
#   ...
# }
```

## Feature Engineering

Raw transaction fields alone don't tell the full story — a $1,200 charge means nothing without context. The feature engineering layer derives signal-rich features:

| Engineered Feature | What It Captures |
|---|---|
| `amount_vs_avg_ratio` | How unusual this amount is relative to the cardholder's typical spend |
| `tx_velocity_ratio` | Transaction bursts (many transactions in a short window) |
| `is_night` | Off-hours activity (00:00–05:00), a common fraud window |
| `is_weekend` | Weekend transaction flag |
| `risk_score` | Composite score combining international, new merchant, prior declines, night activity, and card-not-present signals |

See [feature engineering details] for the full breakdown of how each feature is derived and why it matters.

## Model Evaluation

The system reports:

- **Classification report** (precision, recall, F1 for fraud vs. legitimate classes)
- **ROC-AUC** and **Average Precision** — more informative than accuracy given severe class imbalance
- **Confusion matrix** — broken down into true/false positives and negatives with fraud-specific framing
- **Feature importance** — ranked list of which features drive the supervised model's decisions

Fraud detection is a textbook imbalanced classification problem (~2.5% positive class here), so accuracy alone is not used as a success metric — precision/recall tradeoffs and ROC-AUC are prioritized instead.

## Risk Tiers

Each scored transaction is bucketed into a risk tier for downstream decisioning:

| Tier | Score Range | Typical Action |
|---|---|---|
| `LOW` | 0.00 – 0.30 | Allow |
| `MEDIUM` | 0.30 – 0.60 | Allow, monitor |
| `HIGH` | 0.60 – 0.85 | Flag for review |
| `CRITICAL` | 0.85 – 1.00 | Block / require verification |

## Limitations & Future Improvements

This project uses synthetic data and is intended as a portfolio/learning demonstration, not a production-ready fraud system. Notable areas for extension:

- [ ] Explicit class-imbalance handling (SMOTE, class weighting) rather than relying solely on threshold tuning
- [ ] REST API layer (e.g., FastAPI) for real-time deployment
- [ ] Model monitoring for data/concept drift as fraud patterns evolve over time
- [ ] Hyperparameter tuning via grid/Bayesian search
- [ ] Integration with real-world, anonymized transaction datasets (e.g., Kaggle's IEEE-CIS or ULB datasets)
- [ ] SHAP-based explainability for individual transaction decisions

## Tech Stack

- **Python 3.9+**
- **scikit-learn** — Isolation Forest, Gradient Boosting, preprocessing, cross-validation
- **Pandas / NumPy** — data manipulation
- **joblib** — model persistence

## License

This project is open for personal and educational use.
