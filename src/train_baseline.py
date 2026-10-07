"""
Time-Based Validation & Baseline Model Comparison
--------------------------------------------------
Compares:
1. Logistic Regression (Linear baseline with OneHotEncoding & StandardScaling)
2. Random Forest Classifier
3. LightGBM Classifier (Gradient Boosted Trees with native categorical handling)

Validation Protocol:
- Time-based holdout:
  - Train: 2025-04-01 to 2026-03-31 (12 months, 8,378 orders)
  - Validation: 2026-04-01 to 2026-06-30 (3 months, 2,126 orders)
  Matches exact horizon of unlabelled test set (2026-07-01 to 2026-09-30, 2,096 orders).
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb


def run_experiment():
    df = pd.read_parquet("data/train_cleaned.parquet")

    # Time-based split
    split_date = "2026-04-01"
    train_mask = df["order_placed_at"] < split_date
    val_mask = df["order_placed_at"] >= split_date

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Train split (< {split_date}): {len(train_df)} rows, return rate: {train_df['returned'].mean():.4f}")
    print(f"Val split   (>= {split_date}): {len(val_df)} rows, return rate: {val_df['returned'].mean():.4f}")

    num_cols = [
        "order_value_inr", "qty", "discount_pct", "promised_delivery_days",
        "customer_prior_orders", "customer_prior_returns", "prior_return_ratio",
        "customer_tenure_days", "product_age_days", "unit_price",
        "order_hour", "order_dayofweek", "is_weekend",
        "is_shield", "is_gift_flag", "has_delivery_note", "is_walkin_pincode",
        "warranty_months", "list_price_inr"
    ]

    cat_cols = ["sales_channel", "payment_mode", "family", "sku", "state", "city"]
    target = "returned"

    X_train = train_df[num_cols + cat_cols]
    y_train = train_df[target]
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df[target]

    # Preprocessor for Sklearn models
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    models = {}

    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_prob = lr_pipe.predict_proba(X_val)[:, 1]
    models["Logistic Regression"] = lr_prob

    # 2. Random Forest
    print("Training Random Forest...")
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_prob = rf_pipe.predict_proba(X_val)[:, 1]
    models["Random Forest"] = rf_prob

    # 3. LightGBM
    print("Training LightGBM...")
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    for c in cat_cols:
        X_train_lgb[c] = X_train_lgb[c].astype("category")
        X_val_lgb[c] = X_val_lgb[c].astype("category")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train)
    lgb_prob = lgb_model.predict_proba(X_val_lgb)[:, 1]
    models["LightGBM"] = lgb_prob

    # Benchmark comparison
    print("\n" + "=" * 65)
    print(f"{'Model':<22} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8} | {'LogLoss':<8}")
    print("=" * 65)
    for name, probs in models.items():
        roc = roc_auc_score(y_val, probs)
        pr = average_precision_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        ll = log_loss(y_val, probs)
        print(f"{name:<22} | {roc:<8.4f} | {pr:<8.4f} | {brier:<8.4f} | {ll:<8.4f}")
    print("=" * 65)

    # Feature Importance for LightGBM
    feat_imp = pd.Series(lgb_model.feature_importances_, index=num_cols + cat_cols).sort_values(ascending=False)
    print("\nTop 12 Most Important Features (LightGBM):")
    print(feat_imp.head(12))

    return models, lgb_model


if __name__ == "__main__":
    run_experiment()
