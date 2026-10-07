"""
Kestrel Returns Risk - Core Inference & Reason Engine
-----------------------------------------------------
Loads trained model pipeline and generates:
- Calibrated probability score
- Operational risk categorization
- Business action recommendation
- Plain English reasons based on model features
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CUSTOMERS_CSV = os.path.join(BASE_DIR, "customers.csv")
PRODUCTS_CSV = os.path.join(BASE_DIR, "products.csv")

# Global cached artifacts
_lr_model = None
_lgb_model = None
_customers_df = None
_products_df = None


def load_resources():
    global _lr_model, _lgb_model, _customers_df, _products_df
    if _lr_model is None:
        lr_path = os.path.join(ARTIFACTS_DIR, "lr_model.joblib")
        if os.path.exists(lr_path):
            _lr_model = joblib.load(lr_path)
    if _lgb_model is None:
        lgb_path = os.path.join(ARTIFACTS_DIR, "lgb_model.joblib")
        if os.path.exists(lgb_path):
            _lgb_model = joblib.load(lgb_path)
    if _customers_df is None and os.path.exists(CUSTOMERS_CSV):
        _customers_df = pd.read_csv(CUSTOMERS_CSV).set_index("customer_id")
    if _products_df is None and os.path.exists(PRODUCTS_CSV):
        _products_df = pd.read_csv(PRODUCTS_CSV).set_index("sku")


def enrich_order(raw: dict) -> pd.DataFrame:
    load_resources()
    data = raw.copy()

    # Look up customer
    cid = data.get("customer_id", "")
    if _customers_df is not None and cid in _customers_df.index:
        cust = _customers_df.loc[cid]
        data.setdefault("city", str(cust["city"]))
        data.setdefault("state", str(cust["state"]))
        data.setdefault("signup_date", str(cust["signup_date"]))
        data.setdefault("shield_member", str(cust["shield_member"]))
    else:
        data.setdefault("city", "Pune")
        data.setdefault("state", "MH")
        data.setdefault("signup_date", "2024-01-01")
        data.setdefault("shield_member", "N")

    # Look up product
    sku = data.get("sku", "")
    if _products_df is not None and sku in _products_df.index:
        prod = _products_df.loc[sku]
        data.setdefault("family", str(prod["family"]))
        data.setdefault("model_name", str(prod["model_name"]))
        data.setdefault("list_price_inr", int(prod["list_price_inr"]))
        data.setdefault("warranty_months", int(prod["warranty_months"]))
        data.setdefault("launch_date", str(prod["launch_date"]))
    else:
        data.setdefault("family", "Air Fryer")
        data.setdefault("model_name", "Unknown")
        data.setdefault("list_price_inr", 6999)
        data.setdefault("warranty_months", 12)
        data.setdefault("launch_date", "2024-01-01")

    # Fill defaults
    data.setdefault("sales_channel", "app")
    data.setdefault("payment_mode", "cod")
    data.setdefault("discount_pct", 10)
    data.setdefault("qty", 1)
    data.setdefault("promised_delivery_days", 4)
    data.setdefault("delivery_pincode", 411001)
    data.setdefault("is_gift", "N")
    data.setdefault("customer_prior_orders", 0)
    data.setdefault("customer_prior_returns", 0)
    data.setdefault("delivery_note", "NONE")
    data.setdefault("order_placed_at", "2026-08-01 14:00:00")

    # Computed fields
    qty = int(data["qty"])
    list_price = float(data["list_price_inr"])
    disc = float(data["discount_pct"])
    order_val = round(qty * list_price * (1 - disc / 100.0), 2)
    data["order_value_inr"] = order_val

    # Dates
    order_dt = pd.to_datetime(data["order_placed_at"])
    signup_dt = pd.to_datetime(data["signup_date"])
    launch_dt = pd.to_datetime(data["launch_date"])

    data["has_delivery_note"] = 0 if str(data["delivery_note"]).upper() in ["NONE", "", "NAN"] else 1
    data["is_walkin_pincode"] = 1 if str(data["delivery_pincode"]).zfill(6) == "000000" else 0
    data["customer_tenure_days"] = max(0.0, (order_dt - signup_dt).total_seconds() / 86400.0)
    data["product_age_days"] = max(0.0, (order_dt - launch_dt).total_seconds() / 86400.0)
    data["order_hour"] = order_dt.hour
    data["order_dayofweek"] = order_dt.dayofweek
    data["is_weekend"] = 1 if order_dt.dayofweek >= 5 else 0
    data["prior_return_ratio"] = float(data["customer_prior_returns"]) / (float(data["customer_prior_orders"]) + 1.0)
    data["unit_price"] = order_val / max(1, qty)
    data["is_shield"] = 1 if data["shield_member"] == "Y" else 0
    data["is_gift_flag"] = 1 if data["is_gift"] == "Y" else 0

    return pd.DataFrame([data]), data


def generate_reasons(row: dict, score: float) -> list:
    reasons = []

    # 1. Customer return history
    prior_orders = int(row.get("customer_prior_orders", 0))
    prior_returns = int(row.get("customer_prior_returns", 0))
    if prior_returns >= 2:
        reasons.append(f"High historical return frequency: Customer returned {prior_returns} out of {prior_orders} previous orders.")
    elif prior_returns == 1:
        reasons.append(f"Customer has 1 prior return in record ({prior_orders} previous orders).")
    elif prior_orders == 0:
        reasons.append("First-time customer: No previous purchase history to baseline return likelihood.")

    # 2. Payment mode
    pm = str(row.get("payment_mode", "")).lower()
    if pm == "cod":
        reasons.append("Payment Mode is Cash on Delivery (COD): Carries a 2.4x higher historical return rate (18.8%) compared to prepaid UPI (7.7%).")
    elif pm == "prepaid_upi":
        reasons.append("Prepaid UPI payment: Indicates higher customer commitment (lowest historical return segment).")

    # 3. Product family
    family = str(row.get("family", ""))
    if family == "Robot Vacuum":
        reasons.append("Product Family 'Robot Vacuum' has elevated historical return rates (19.5% category average) due to home suitability checks.")
    elif family == "Water Purifier":
        reasons.append("Product Family 'Water Purifier' has elevated returns (14.8%) frequently tied to plumbing compatibility issues.")

    # 4. Delivery promise
    days = int(row.get("promised_delivery_days", 4))
    if days >= 7:
        reasons.append(f"Extended delivery lead time ({days} promised days): Shipments with 7+ days promise face elevated buyer cancellation & remorse risk.")

    # 5. Shield member policy nuance
    is_shield = row.get("shield_member") == "Y" or row.get("is_shield") == 1
    if is_shield:
        reasons.append("Kestrel Shield Member: Eligible for free 30-day returns. Recommended for white-glove confirmation call rather than shipment hold to safeguard high customer lifetime value.")

    # 6. Walk-in / Pincode
    if int(row.get("is_walkin_pincode", 0)) == 1:
        reasons.append("Default pincode (000000) recorded: Likely a partner walk-in order without captured delivery address.")

    # 7. Gift order
    if row.get("is_gift") == "Y":
        reasons.append("Marked as Gift order: Gift recipients show slightly higher return rates due to preference mismatch.")

    # Fallback if few triggered
    if len(reasons) < 2:
        if score > 0.12:
            reasons.append("Combined feature interaction indicates above-average return propensity at dispatch.")
        else:
            reasons.append("Order parameters align with low-risk operational baseline; standard dispatch approved.")

    return reasons[:4]


def predict_order(raw: dict) -> dict:
    load_resources()
    df, meta = enrich_order(raw)

    num_cols = [
        "order_value_inr", "qty", "discount_pct", "promised_delivery_days",
        "customer_prior_orders", "customer_prior_returns", "prior_return_ratio",
        "customer_tenure_days", "product_age_days", "unit_price",
        "order_hour", "order_dayofweek", "is_weekend",
        "is_shield", "is_gift_flag", "has_delivery_note", "is_walkin_pincode",
        "warranty_months", "list_price_inr"
    ]
    cat_cols = ["sales_channel", "payment_mode", "family", "sku", "state", "city"]

    features_df = df[num_cols + cat_cols]

    # Predict with LR
    lr_score = float(_lr_model.predict_proba(features_df)[:, 1][0]) if _lr_model else 0.114

    # Predict with LightGBM if available
    if _lgb_model is not None:
        lgb_df = features_df.copy()
        for c in cat_cols:
            lgb_df[c] = lgb_df[c].astype("category")
        lgb_score = float(_lgb_model.predict_proba(lgb_df)[:, 1][0])
        final_score = round(0.5 * lr_score + 0.5 * lgb_score, 4)
    else:
        final_score = round(lr_score, 4)

    # Risk level & action
    is_shield = meta.get("shield_member") == "Y" or meta.get("is_shield") == 1
    if final_score >= 0.30:
        risk_level = "VERY HIGH"
        action = "PRIORITY PRE-DISPATCH CONFIRMATION CALL (Rs 45)"
    elif final_score >= 0.12:
        risk_level = "ELEVATED"
        if is_shield:
            action = "WHITE-GLOVE VIP CONFIRMATION CALL (Rs 45) [Do Not Hold]"
        else:
            action = "PRE-DISPATCH CONFIRMATION CALL (Rs 45)"
    else:
        risk_level = "LOW"
        action = "STANDARD DISPATCH (No Intervention Required)"

    reasons = generate_reasons(meta, final_score)

    return {
        "order_id": raw.get("order_id", "ORDER-TEMP"),
        "return_risk_score": final_score,
        "risk_percentage": f"{final_score * 100:.1f}%",
        "risk_level": risk_level,
        "recommended_action": action,
        "shield_member": is_shield,
        "order_value_inr": meta.get("order_value_inr"),
        "reasons": reasons,
        "metadata": {
            "customer_id": meta.get("customer_id"),
            "sku": meta.get("sku"),
            "family": meta.get("family"),
            "payment_mode": meta.get("payment_mode"),
            "promised_delivery_days": meta.get("promised_delivery_days"),
        }
    }
