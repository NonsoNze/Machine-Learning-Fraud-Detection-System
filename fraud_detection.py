"""
Fraud Detection System
======================
A machine learning-based fraud detection pipeline using:
- Feature engineering
- Isolation Forest (unsupervised anomaly detection)
- Random Forest Classifier (supervised detection)
- Real-time scoring with explainability
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


# ─────────────────────────────────────────────
# 1.  DATA GENERATION (simulates real tx data)
# ─────────────────────────────────────────────

def generate_transaction_data(n_samples: int = 10_000, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Generate synthetic credit-card transaction data."""
    random.seed(42)
    np.random.seed(42)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    merchants = ["retail", "restaurant", "online", "travel", "grocery", "gas", "entertainment"]
    countries  = ["US", "UK", "CA", "AU", "DE", "FR", "CN", "RU", "NG", "BR"]
    high_risk  = {"CN", "RU", "NG"}

    def make_rows(n, is_fraud):
        rows = []
        for _ in range(n):
            country = random.choice(list(high_risk) if is_fraud and random.random() < 0.4 else countries)
            rows.append({
                "transaction_id":  str(random.randint(10**9, 10**10 - 1)),
                "amount":          round(np.random.exponential(500 if is_fraud else 80) + 1, 2),
                "merchant_type":   random.choice(merchants),
                "country":         country,
                "hour_of_day":     random.randint(0, 3) if (is_fraud and random.random() < 0.5) else random.randint(0, 23),
                "day_of_week":     random.randint(0, 6),
                "card_present":    0 if (is_fraud and random.random() < 0.7) else random.randint(0, 1),
                "transactions_last_1h":  random.randint(3, 15) if is_fraud else random.randint(0, 3),
                "transactions_last_24h": random.randint(10, 50) if is_fraud else random.randint(1, 10),
                "avg_transaction_amount": round(random.uniform(10, 200), 2),
                "distance_from_home_km":  round(np.random.exponential(800 if is_fraud else 20), 2),
                "new_merchant":    1 if (is_fraud and random.random() < 0.6) else random.randint(0, 1),
                "declined_last_24h": random.randint(0, 5) if is_fraud else 0,
                "is_international": 1 if country not in {"US", "CA"} else 0,
                "is_fraud":        int(is_fraud),
            })
        return rows

    df = pd.DataFrame(make_rows(n_legit, False) + make_rows(n_fraud, True))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dataset: {len(df):,} transactions | {df['is_fraud'].sum():,} fraud ({df['is_fraud'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature engineer for fraud detection."""

    def __init__(self):
        self.merchant_encoder = LabelEncoder()
        self.country_encoder  = LabelEncoder()
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        self.merchant_encoder.fit(X["merchant_type"])
        self.country_encoder.fit(X["country"])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Encode categoricals
        df["merchant_encoded"] = self.merchant_encoder.transform(
            df["merchant_type"].map(
                lambda v: v if v in self.merchant_encoder.classes_ else self.merchant_encoder.classes_[0]
            )
        )
        df["country_encoded"] = self.country_encoder.transform(
            df["country"].map(
                lambda v: v if v in self.country_encoder.classes_ else self.country_encoder.classes_[0]
            )
        )

        # Derived features
        df["amount_vs_avg_ratio"] = df["amount"] / (df["avg_transaction_amount"] + 1e-9)
        df["tx_velocity_ratio"]   = df["transactions_last_1h"] / (df["transactions_last_24h"] + 1e-9)
        df["is_night"]            = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)
        df["is_weekend"]          = (df["day_of_week"] >= 5).astype(int)
        df["risk_score"]          = (
            df["is_international"] * 2
            + df["new_merchant"]
            + df["declined_last_24h"]
            + df["is_night"]
            + (1 - df["card_present"]) * 2
        )

        return df[self._feature_cols()]

    def _feature_cols(self):
        return [
            "amount", "hour_of_day", "day_of_week", "card_present",
            "transactions_last_1h", "transactions_last_24h",
            "avg_transaction_amount", "distance_from_home_km",
            "new_merchant", "declined_last_24h", "is_international",
            "merchant_encoded", "country_encoded",
            "amount_vs_avg_ratio", "tx_velocity_ratio",
            "is_night", "is_weekend", "risk_score",
        ]


# ─────────────────────────────────────────────
# 3.  MODELS
# ─────────────────────────────────────────────

class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest."""

    def __init__(self, contamination: float = 0.02):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 (fraud / anomaly) or 0 (normal)."""
        X_scaled = self.scaler.transform(X)
        raw = self.model.predict(X_scaled)   # -1 = anomaly, 1 = normal
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Lower score → more anomalous."""
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


class SupervisedFraudDetector:
    """Supervised fraud classifier with cross-validation and explainability."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        self.scaler       = StandardScaler()
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"\n[Supervised] 5-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importances(self) -> pd.DataFrame:
        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp


# ─────────────────────────────────────────────
# 4.  ENSEMBLE + REAL-TIME SCORER
# ─────────────────────────────────────────────

class FraudDetectionSystem:
    """
    End-to-end fraud detection system combining:
    - Feature engineering
    - Unsupervised anomaly detection
    - Supervised gradient boosting
    - Ensemble scoring with configurable threshold
    """

    RISK_THRESHOLDS = {
        "LOW":    (0.00, 0.30),
        "MEDIUM": (0.30, 0.60),
        "HIGH":   (0.60, 0.85),
        "CRITICAL":(0.85, 1.01),
    }

    def __init__(self, fraud_threshold: float = 0.5):
        self.engineer   = FraudFeatureEngineer()
        self.anomaly    = AnomalyDetector()
        self.supervised = SupervisedFraudDetector()
        self.threshold  = fraud_threshold
        self._trained   = False

    # ── Training ──────────────────────────────

    def fit(self, df: pd.DataFrame):
        print("\n══ Training Fraud Detection System ══")
        y = df["is_fraud"]
        X_raw = df.drop(columns=["transaction_id", "is_fraud"])

        print("  → Engineering features …")
        self.engineer.fit(X_raw)
        X_feat = self.engineer.transform(X_raw)

        print("  → Training anomaly detector …")
        self.anomaly.fit(X_feat.values)

        print("  → Training supervised model …")
        self.supervised.fit(X_feat, y)

        self._trained = True
        print("\n✓ System trained successfully.\n")
        return self

    # ── Evaluation ────────────────────────────

    def evaluate(self, df: pd.DataFrame):
        assert self._trained, "Call fit() first."
        y_true = df["is_fraud"]
        X_raw  = df.drop(columns=["transaction_id", "is_fraud"])
        X_feat = self.engineer.transform(X_raw)

        fraud_proba  = self.supervised.predict_proba(X_feat)
        anomaly_flag = self.anomaly.predict(X_feat.values)
        ensemble     = self._ensemble_score(fraud_proba, anomaly_flag)
        y_pred       = (ensemble >= self.threshold).astype(int)

        print("\n══ Evaluation Report ══")
        print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
        print(f"ROC-AUC:          {roc_auc_score(y_true, ensemble):.4f}")
        print(f"Avg Precision:    {average_precision_score(y_true, ensemble):.4f}")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives  (correctly flagged legit):  {tn:>6,}")
        print(f"  False Positives (legit flagged as fraud):   {fp:>6,}")
        print(f"  False Negatives (fraud missed):             {fn:>6,}")
        print(f"  True Positives  (fraud caught):             {tp:>6,}")

        # Feature importances
        print("\nTop 10 Features by Importance:")
        fi = self.supervised.feature_importances().head(10)
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] * 300)
            print(f"  {row['feature']:<28} {row['importance']:.4f}  {bar}")

    # ── Real-time Scoring ─────────────────────

    def score_transaction(self, tx: dict) -> dict:
        """Score a single transaction and return a structured result."""
        assert self._trained, "Call fit() first."
        df_tx  = pd.DataFrame([tx])
        X_feat = self.engineer.transform(df_tx)

        fraud_proba  = self.supervised.predict_proba(X_feat)[0]
        anomaly_flag = self.anomaly.predict(X_feat.values)[0]
        score        = self._ensemble_score(
            np.array([fraud_proba]), np.array([anomaly_flag])
        )[0]

        risk_level = next(
            k for k, (lo, hi) in self.RISK_THRESHOLDS.items() if lo <= score < hi
        )
        decision = "BLOCK" if score >= self.threshold else "ALLOW"

        return {
            "transaction_id":  tx.get("transaction_id", "N/A"),
            "fraud_probability": round(float(fraud_proba), 4),
            "anomaly_flag":    bool(anomaly_flag),
            "ensemble_score":  round(float(score), 4),
            "risk_level":      risk_level,
            "decision":        decision,
            "features_used":   X_feat.to_dict(orient="records")[0],
        }

    # ── Persistence ───────────────────────────

    def save(self, path: str = "fraud_model.joblib"):
        joblib.dump(self, path)
        print(f"Model saved → {path}")

    @staticmethod
    def load(path: str = "fraud_model.joblib") -> "FraudDetectionSystem":
        return joblib.load(path)

    # ── Internals ─────────────────────────────

    @staticmethod
    def _ensemble_score(sup_proba: np.ndarray, anomaly_flags: np.ndarray) -> np.ndarray:
        """Weighted blend: 80% supervised + 20% anomaly signal."""
        return 0.80 * sup_proba + 0.20 * anomaly_flags


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

def main():
    # Generate data
    df = generate_transaction_data(n_samples=20_000, fraud_rate=0.025)

    # Train / test split (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_fraud"]
    )

    # Build and train system
    system = FraudDetectionSystem(fraud_threshold=0.50)
    system.fit(train_df)

    # Evaluate on held-out test set
    system.evaluate(test_df)

    # ── Demo: real-time scoring ──
    print("\n══ Real-Time Scoring Demo ══")

    sample_legitimate = {
        "transaction_id": "TX_LEGIT_001",
        "amount": 42.50,
        "merchant_type": "grocery",
        "country": "US",
        "hour_of_day": 14,
        "day_of_week": 2,
        "card_present": 1,
        "transactions_last_1h": 0,
        "transactions_last_24h": 3,
        "avg_transaction_amount": 55.0,
        "distance_from_home_km": 2.5,
        "new_merchant": 0,
        "declined_last_24h": 0,
        "is_international": 0,
    }

    sample_fraud = {
        "transaction_id": "TX_FRAUD_001",
        "amount": 2850.00,
        "merchant_type": "online",
        "country": "NG",
        "hour_of_day": 2,
        "day_of_week": 6,
        "card_present": 0,
        "transactions_last_1h": 8,
        "transactions_last_24h": 22,
        "avg_transaction_amount": 55.0,
        "distance_from_home_km": 9400.0,
        "new_merchant": 1,
        "declined_last_24h": 3,
        "is_international": 1,
    }

    for tx in [sample_legitimate, sample_fraud]:
        result = system.score_transaction(tx)
        print(f"\n  Transaction : {result['transaction_id']}")
        print(f"  Amount      : ${tx['amount']:,.2f}")
        print(f"  Fraud Prob  : {result['fraud_probability']:.4f}")
        print(f"  Anomaly     : {result['anomaly_flag']}")
        print(f"  Score       : {result['ensemble_score']:.4f}")
        print(f"  Risk Level  : {result['risk_level']}")
        print(f"  Decision    : {'🚫 ' if result['decision']=='BLOCK' else '✅ '}{result['decision']}")

    # Save model
    system.save("fraud_detection_model.joblib")


if __name__ == "__main__":
    main()
