"""
Final Test Prediction Pipeline
------------------------------
Trains the optimal production model on the entire historical 15-month cleaned dataset
(10,504 deduplicated orders) and generates calibrated return risk scores for test_unlabelled.csv.

Outputs:
- predictions.csv (matching sample_submission.csv schema: order_id,score)
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb


def generate_test_predictions():
    print("Loading cleaned datasets...")
    train = pd.read_parquet("data/train_cleaned.parquet")
    test = pd.read_parquet("data/test_cleaned.parquet")

    print(f"Full Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

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

    X_train = train[num_cols + cat_cols].copy()
    y_train = train[target]
    X_test = test[num_cols + cat_cols].copy()

    # 1. Fit Regularized Logistic Regression Pipeline
    print("Training regularized Logistic Regression on 10,504 historical orders...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    lr_model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ])
    lr_model.fit(X_train, y_train)
    lr_test_probs = lr_model.predict_proba(X_test)[:, 1]

    # 2. Fit Calibrated LightGBM
    print("Training regularized LightGBM on 10,504 historical orders...")
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_train_lgb[c] = X_train_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train)
    lgb_test_probs = lgb_model.predict_proba(X_test_lgb)[:, 1]

    # Ensemble Blend (50/50 - smooth linear generalizability + gradient boosted tree tuning)
    final_scores = (0.5 * lr_test_probs + 0.5 * lgb_test_probs).round(4)

    # Format submission
    sub = pd.DataFrame({
        "order_id": test["order_id"],
        "score": final_scores
    })

    # Validate against sample_submission.csv
    sample_sub = pd.read_csv("sample_submission.csv")
    assert sub.shape == sample_sub.shape, f"Shape mismatch: {sub.shape} vs {sample_sub.shape}"
    assert (sub["order_id"] == sample_sub["order_id"]).all(), "Order IDs do not match sample submission!"
    assert not sub["score"].isna().any(), "Found NaN in predictions!"

    # Save output
    output_path = "predictions.csv"
    sub.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated {output_path} ({len(sub)} rows).")

    print("\n--- Predictions Summary Statistics ---")
    print(f"Min Score:    {sub['score'].min():.4f}")
    print(f"25th Pct:     {sub['score'].quantile(0.25):.4f}")
    print(f"Median Score: {sub['score'].median():.4f}")
    print(f"Mean Score:   {sub['score'].mean():.4f}")
    print(f"75th Pct:     {sub['score'].quantile(0.75):.4f}")
    print(f"Max Score:    {sub['score'].max():.4f}")
    print(f"Orders >= 0.12 (Call Trigger): {(sub['score'] >= 0.12).sum()} ({(sub['score'] >= 0.12).mean()*100:.1f}%)")
    print(f"Orders >= 0.50 (Extreme Risk): {(sub['score'] >= 0.50).sum()}")

    # Save artifacts for the API service
    import joblib
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(lr_model, "artifacts/lr_model.joblib")
    joblib.dump(lgb_model, "artifacts/lgb_model.joblib")
    print("Saved model artifacts to artifacts/ directory for the API service.")


if __name__ == "__main__":
    generate_test_predictions()
