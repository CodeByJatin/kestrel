"""
Financial & Operational Cost Matrix Evaluation
-----------------------------------------------
Evaluates business value using exact numbers from ops-policy.pdf:
- Cost of a return: Rs 1,150
- Pre-dispatch confirmation call: Rs 45
- Return prevention rate of call: 35%
- Blunt Hold cancellation penalty: 12% customer cancellation on held orders
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, average_precision_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def evaluate_thresholds():
    df = pd.read_parquet("data/train_cleaned.parquet")
    train_df = df[df["order_placed_at"] < "2026-04-01"]
    val_df = df[df["order_placed_at"] >= "2026-04-01"].copy()

    num_cols = [
        "order_value_inr", "qty", "discount_pct", "promised_delivery_days",
        "customer_prior_orders", "customer_prior_returns", "prior_return_ratio",
        "customer_tenure_days", "product_age_days", "unit_price",
        "order_hour", "order_dayofweek", "is_weekend",
        "is_shield", "is_gift_flag", "has_delivery_note", "is_walkin_pincode",
        "warranty_months", "list_price_inr"
    ]
    cat_cols = ["sales_channel", "payment_mode", "family", "sku", "state", "city"]

    X_train = train_df[num_cols + cat_cols]
    y_train = train_df["returned"]
    X_val = val_df[num_cols + cat_cols]
    y_val = val_df["returned"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    lr.fit(X_train, y_train)
    val_df["score"] = lr.predict_proba(X_val)[:, 1]

    total_orders = len(val_df)
    total_actual_returns = val_df["returned"].sum()
    unmanaged_return_cost = total_actual_returns * 1150  # Rs 1,150 per return

    print(f"Validation Orders: {total_orders}")
    print(f"Total Actual Returns: {total_actual_returns} ({total_actual_returns/total_orders*100:.2f}%)")
    print(f"Unmanaged Status-Quo Return Cost: Rs {unmanaged_return_cost:,.0f} (over 3 months)\n")

    print("--- POLICY INTERVENTION: CONFIRMATION CALL (Rs 45/call, 35% return prevention) ---")
    print(f"{'Threshold':<10} | {'Flagged':<8} | {'True Pos':<9} | {'False Pos':<9} | {'Precision':<9} | {'Recall':<8} | {'Call Cost':<10} | {'Saved Return':<12} | {'Net Savings (Rs)':<16}")
    print("-" * 105)

    best_threshold = None
    max_savings = -float("inf")

    for t in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]:
        flagged = (val_df["score"] >= t)
        tp = ((val_df["score"] >= t) & (val_df["returned"] == 1)).sum()
        fp = ((val_df["score"] >= t) & (val_df["returned"] == 0)).sum()
        prec = tp / flagged.sum() if flagged.sum() > 0 else 0
        rec = tp / total_actual_returns

        call_cost = flagged.sum() * 45
        returns_prevented = tp * 0.35
        saved_return_cost = returns_prevented * 1150
        net_savings = saved_return_cost - call_cost

        if net_savings > max_savings:
            max_savings = net_savings
            best_threshold = t

        print(f"{t:<10.2f} | {flagged.sum():<8} | {tp:<9} | {fp:<9} | {prec:<9.3f} | {rec:<8.3f} | Rs {call_cost:<7,.0f} | Rs {saved_return_cost:<9,.0f} | Rs {net_savings:<13,.0f}")

    print("-" * 105)
    print(f"Optimal Threshold for Confirmation Calls: {best_threshold:.2f} yielding Net Savings of Rs {max_savings:,.0f} over 3 months (~Rs {max_savings/3:,.0f}/month)\n")


if __name__ == "__main__":
    evaluate_thresholds()
