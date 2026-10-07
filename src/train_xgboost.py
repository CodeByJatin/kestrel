"""
XGBoost Benchmark on Out-of-Time Validation Split
-------------------------------------------------
Evaluates XGBoost performance against Logistic Regression and LightGBM.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

# Load clean parquet
df = pd.read_parquet("data/train_cleaned.parquet")

split_date = "2026-04-01"
train_df = df[df["order_placed_at"] < split_date].copy()
val_df = df[df["order_placed_at"] >= split_date].copy()

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

X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df[target]
X_val = val_df[num_cols + cat_cols].copy()
y_val = val_df[target]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}")

# Test a few XGBoost configurations
configs = [
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, "scale_pos_weight": 1},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 200, "scale_pos_weight": 1},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, "scale_pos_weight": 3},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, "scale_pos_weight": 7.7},
]

print("\n" + "=" * 70)
print(f"{'Config':<35} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}")
print("=" * 70)

for cfg in configs:
    clf = xgb.XGBClassifier(
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        scale_pos_weight=cfg["scale_pos_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, probs)
    pr = average_precision_score(y_val, probs)
    brier = brier_score_loss(y_val, probs)

    cfg_str = f"depth={cfg['max_depth']}, w={cfg['scale_pos_weight']}"
    print(f"{cfg_str:<35} | {roc:<8.4f} | {pr:<8.4f} | {brier:<8.4f}")

print("=" * 70)
