"""
Data Preprocessing & Cleaning Pipeline for Kestrel Returns Risk
----------------------------------------------------------------
Handles:
1. Deduplication of partner_feed re-imports (651 duplicates).
2. Elimination of post-dispatch target leakage (pickup_scheduled_at, last_service_event_type).
3. Correction of October 2025 payment gateway currency bug (100x paise inflation).
4. Missing value imputation (delivery_note, pincodes).
5. Merging customer and product metadata.
6. Dispatch-time feature engineering.
"""

import os
import pandas as pd
import numpy as np


def load_and_clean_data(base_path: str = "."):
    # 1. Load raw files
    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test_unlabelled.csv")
    customers_path = os.path.join(base_path, "customers.csv")
    products_path = os.path.join(base_path, "products.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)

    # 2. Deduplicate Train on order_id (Drop 651 partner_feed duplicates, keep CRM)
    train = train.sort_values(by=["order_id", "source"], ascending=[True, False])
    train = train.drop_duplicates(subset=["order_id"], keep="first").reset_index(drop=True)

    # 3. Merge Customer and Product metadata
    train = train.merge(customers, on="customer_id", how="left")
    test = test.merge(customers, on="customer_id", how="left")

    train = train.merge(products, on="sku", how="left")
    test = test.merge(products, on="sku", how="left")

    # 4. Remove Post-Dispatch Target Leakage Columns
    leakage_cols = ["pickup_scheduled_at", "last_service_event_type"]
    train = train.drop(columns=[c for c in leakage_cols if c in train.columns])
    test = test.drop(columns=[c for c in leakage_cols if c in test.columns])

    # 5. Fix October 2025 Gateway 100x Value Bug & Missing Values & Feature Engineering
    for df in [train, test]:
        # Correct order value (recalculated from unit price, quantity, and discount)
        expected_value = (df["qty"] * df["list_price_inr"] * (1 - df["discount_pct"] / 100.0)).round(2)
        df["order_value_inr"] = expected_value

        # Handle missing notes
        df["has_delivery_note"] = df["delivery_note"].notna().astype(int)
        df["delivery_note"] = df["delivery_note"].fillna("NONE")

        # Pincode default flag (000000 = walk-in / partner orders with no address)
        df["is_walkin_pincode"] = (df["delivery_pincode"].astype(str).str.zfill(6) == "000000").astype(int)

        # Datetime parsing
        df["order_placed_at"] = pd.to_datetime(df["order_placed_at"])
        df["signup_date"] = pd.to_datetime(df["signup_date"])
        df["launch_date"] = pd.to_datetime(df["launch_date"])

        # Temporal tenure features
        df["customer_tenure_days"] = ((df["order_placed_at"] - df["signup_date"]).dt.total_seconds() / 86400.0).clip(lower=0)
        df["product_age_days"] = ((df["order_placed_at"] - df["launch_date"]).dt.total_seconds() / 86400.0).clip(lower=0)

        # Calendar features
        df["order_hour"] = df["order_placed_at"].dt.hour
        df["order_dayofweek"] = df["order_placed_at"].dt.dayofweek
        df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
        df["order_month"] = df["order_placed_at"].dt.month

        # Customer behavioral signals
        df["prior_return_ratio"] = df["customer_prior_returns"] / (df["customer_prior_orders"] + 1.0)
        df["unit_price"] = df["order_value_inr"] / df["qty"]

        # Indicator flags
        df["is_shield"] = (df["shield_member"] == "Y").astype(int)
        df["is_gift_flag"] = (df["is_gift"] == "Y").astype(int)

    return train, test


if __name__ == "__main__":
    train_clean, test_clean = load_and_clean_data(".")
    print(f"Cleaned Train shape: {train_clean.shape}, Nulls: {train_clean.isna().sum().sum()}")
    print(f"Cleaned Test shape: {test_clean.shape}, Nulls: {test_clean.isna().sum().sum()}")
    print(f"Cleaned Train return rate: {train_clean['returned'].mean():.4f}")
    
    # Save cleaned files to data/ for rapid training
    os.makedirs("data", exist_ok=True)
    train_clean.to_parquet("data/train_cleaned.parquet", index=False)
    test_clean.to_parquet("data/test_cleaned.parquet", index=False)
    print("Saved clean parquet files to data/ directory.")
