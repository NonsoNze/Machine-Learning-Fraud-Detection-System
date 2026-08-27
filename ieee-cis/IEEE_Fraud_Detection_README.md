# IEEE-CIS Fraud Detection — Hybrid Ensemble Pipeline

A production-style fraud detection pipeline built on real anonymised transaction data from
the [IEEE-CIS Fraud Detection Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection),
provided by Vesta Corporation.

The pipeline combines a supervised LightGBM classifier with an unsupervised Isolation Forest
in a weighted ensemble, applying the same hybrid architecture as the synthetic fraud detection
project but adapted for the scale and complexity of real competition-grade data. The unlabeled
Kaggle test set is used for the unsupervised component, demonstrating how anomaly detection
can be applied to genuinely unseen transactions without requiring ground truth labels.

## Dataset

| Property | Value |
|---|---|
| Source | Vesta Corporation via Kaggle (IEEE-CIS Fraud Detection) |
| Training transactions | 590,540 |
| Training with identity data | 144,233 (24.4%) |
| Fraud rate | 3.50% |
| Transaction features | 394 |
| Identity features | 41 |
| Total features (merged) | 434 |

The dataset consists of two files joined on `TransactionID`:
- `train_transaction.csv` — transaction-level features (amounts, card info, counts, V-columns)
- `train_identity.csv` — device and network identity features (sparse, 24.4% coverage)

## Pipeline Structure

| Section | Contents |
|---|---|
| 1. Setup & Data Loading | Load and merge transaction + identity tables |
| 2. EDA | Target distribution, amount analysis, time patterns, missing value audit, categorical fraud rates |
| 3. Data Cleaning | Drop >90% missing columns, median impute numerics, fill categorical unknowns |
| 4. Feature Engineering | Time features, log-transformed amount, card-level velocity and deviation |
| 5. Preprocessing | Label encoding, standard scaling (continuous only), stratified train/val split |
| 6. Baseline | Majority class accuracy and no-skill AUC — establishes why accuracy is not a useful metric here |
| 7. Model Comparison | Logistic Regression vs LightGBM via 3-fold stratified cross-validation |
| 8. Imbalance Handling | Class weighting + precision-recall threshold tuning |
| 9. Hyperparameter Tuning | RandomizedSearchCV (n_iter=30) over LightGBM structural parameters |
| 10. Final Evaluation | ROC-AUC, Average Precision, classification report, confusion matrix, stratified evaluation by transaction amount |
| 11. Feature Importance | LightGBM native importance + permutation importance on validation set |
| 12. Real-Time Scoring | Single-transaction scoring function returning fraud probability, risk tier, and ALLOW/BLOCK decision |
| 12.5 Hybrid Ensemble on Test Data | Isolation Forest fitted on training data and applied to unlabeled Kaggle test set; blended 80/20 with LightGBM into a hybrid ensemble score with risk tier distribution and agreement analysis |
| 13. Leakage Audit | Explicit check for post-outcome information, data preparation leakage, and anomaly score normalisation risk |
| 14. Limitations | Honest documentation of methodology constraints and recommended next steps |

## Key Design Decisions

**Why a hybrid ensemble?**
A purely supervised model can only recognise fraud patterns it has seen labeled examples of.
The Isolation Forest provides a second, unsupervised signal that flags statistically unusual
behaviour regardless of whether it matches any known fraud type. This makes the system more
robust to novel tactics. The ensemble blends the two signals 80% supervised and 20% anomaly,
the same weighting used in the synthetic project, applied here to real data at production scale.

**Why apply the Isolation Forest to the test set?**
The Kaggle test set has no labels, but unsupervised methods do not need them. Fitting the
Isolation Forest on training data and scoring the test set mirrors genuine production deployment,
where new transactions arrive unlabeled and must be assessed in real time. This is the only
legitimate use of the test set outside of a Kaggle submission.

**Why LightGBM as the supervised component?**
LightGBM handles high-dimensional tabular data efficiently, supports native missing value
treatment (important given the V-column sparsity), and scales to 500K+ rows without the
memory issues of Random Forest at this size. It consistently outperforms sklearn's
HistGradientBoostingClassifier on competition-grade fraud data.

**Why ROC-AUC and Average Precision over accuracy?**
With a 3.5% fraud rate, a model that always predicts "not fraud" achieves 96.5% accuracy
while catching zero fraud. ROC-AUC measures ranking quality across all thresholds.
Average Precision (PR-AUC) is even more informative for imbalanced problems — it focuses
on the minority class directly and is more sensitive to the precision-recall tradeoff that
matters in fraud detection.

**Why threshold tuning?**
The default 0.5 threshold is rarely optimal for imbalanced classification. The threshold
is tuned to maximise F1 on the fraud class, reflecting the business tradeoff: missing fraud
is costly, but excessive false positives (blocking legitimate transactions) damages customer
experience. The optimal threshold is determined on the validation set.

**Why class weighting?**
Class weighting penalises the model more heavily for misclassifying fraud during training,
pushing it to learn from the minority class rather than optimising for the majority.
Combined with threshold tuning, this substantially improves recall on the fraud class.

## Performance (50K row validation run)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9414 |
| Average Precision | 0.7769 |

*On the full 590K dataset with hyperparameter tuning, expect ROC-AUC ~0.96–0.98.
Competition top scores were approximately 0.96.*

## Feature Engineering

Seven features are engineered from the raw dataset:

| Feature | Description | Rationale |
|---|---|---|
| `hour` | Hour of day derived from `TransactionDT` | Fraud disproportionately occurs at night |
| `day` | Day of week proxy from `TransactionDT` | Captures weekly fraud patterns |
| `TransactionAmt_log` | Log-transformed transaction amount | Compresses heavy right skew |
| `amt_vs_card_median` | Amount divided by card's median spend | Flags unusually large transactions relative to a card's history |
| `card1_tx_count` | Total transactions per card | High frequency can indicate card testing or account takeover |
| `has_identity` | Binary flag for identity data availability | Presence of identity data is itself a signal |
| `email_match` | Whether purchaser and recipient email domains match | Mismatch is a known fraud indicator |

## Leakage Audit

Two engineered features (`card1_tx_count`, `amt_vs_card_median`) are computed across the
full dataset before splitting, meaning validation rows contribute to the statistics used
to transform training rows. This is a mild but real form of data preparation leakage.

In production, these would be computed from a rolling historical window per card, using
only data from before each transaction. The notebook documents this explicitly rather than
ignoring it.

## Requirements

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

`scikit-learn` is required for the Isolation Forest, preprocessing, and evaluation utilities.
`lightgbm` is required for the supervised ensemble component.

## Running the Notebook

1. Download the data from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
   (requires joining the competition — free)
2. Place all four files in the same directory as the notebook:
   `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`
3. Run all cells top to bottom

The training files are used for model development and evaluation. The test files are used
in Section 12.5 for the unsupervised Isolation Forest scoring on unlabeled transactions.

The full pipeline on the complete dataset takes approximately 15–30 minutes depending on hardware,
driven primarily by the RandomizedSearchCV step. To reduce runtime, lower `n_iter` in Section 9
or reduce `n_estimators`.

## Limitations and Honest Caveats

- **Velocity features computed globally** — `card1_tx_count` and `amt_vs_card_median` should be
  computed from historical data per card in a production system
- **Random rather than time-based split** — chronological splitting would give a more realistic
  performance estimate by preventing future data from informing past predictions
- **V-column anonymisation** — 339 features have unknown physical meaning, limiting domain-driven
  feature selection
- **No monitoring framework** — fraud patterns evolve; this notebook does not implement drift
  detection or retraining triggers
- **Identity data sparsity** — only 24.4% of transactions have identity data, limiting the
  reliability of identity-based features
- **Isolation Forest normalisation uses test-set bounds** — the anomaly score rescaling in
  Section 12.5 is computed from the full test score distribution; a production system would
  fit normalisation bounds on training scores only and apply them fixed to incoming transactions
- **Ensemble weighting is fixed at 80/20** — the supervised/unsupervised blend ratio was
  carried over from the synthetic project without being optimised for this dataset; the optimal
  weighting could be tuned empirically on the validation set

## Project Context

This project is part of a portfolio demonstrating applied data science methodology on real,
competition-grade data. It follows the same principles applied in the CMAPSS Predictive Maintenance
project: rigorous evaluation, transparent documentation of limitations, and honest reporting of
where the methodology holds up and where it does not.

## References

- Vesta Corporation / IEEE-CIS Fraud Detection Competition, Kaggle (2019)
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS.
