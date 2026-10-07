import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb

df = pd.read_parquet("data/train_cleaned.parquet")
train_df = df[df["order_placed_at"] < "2026-04-01"]
val_df = df[df["order_placed_at"] >= "2026-04-01"]

num_cols = [
    "order_value_inr", "qty", "discount_pct", "promised_delivery_days",
    "customer_prior_orders", "customer_prior_returns", "prior_return_ratio",
    "customer_tenure_days", "product_age_days", "unit_price",
    "order_hour", "order_dayofweek", "is_weekend",
    "is_shield", "is_gift_flag", "has_delivery_note", "is_walkin_pincode",
    "warranty_months", "list_price_inr"
]
cat_cols = ["sales_channel", "payment_mode", "family", "sku", "state", "city"]

X_train = train_df[num_cols + cat_cols].copy()
y_train = train_df["returned"]
X_val = val_df[num_cols + cat_cols].copy()
y_val = val_df["returned"]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")

configs = [
    {"scale_pos_weight": 1, "max_depth": 4, "num_leaves": 15, "lr": 0.03, "min_child_samples": 30},
    {"scale_pos_weight": 3, "max_depth": 4, "num_leaves": 15, "lr": 0.03, "min_child_samples": 30},
    {"scale_pos_weight": 7.7, "max_depth": 4, "num_leaves": 15, "lr": 0.03, "min_child_samples": 30},
    {"scale_pos_weight": 1, "max_depth": 3, "num_leaves": 7, "lr": 0.02, "min_child_samples": 50},
    {"scale_pos_weight": 3, "max_depth": 3, "num_leaves": 7, "lr": 0.02, "min_child_samples": 50},
    {"scale_pos_weight": 7.7, "max_depth": 3, "num_leaves": 7, "lr": 0.02, "min_child_samples": 50},
]

for cfg in configs:
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=cfg["lr"],
        max_depth=cfg["max_depth"],
        num_leaves=cfg["num_leaves"],
        scale_pos_weight=cfg["scale_pos_weight"],
        min_child_samples=cfg["min_child_samples"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, probs)
    pr = average_precision_score(y_val, probs)
    brier = brier_score_loss(y_val, probs)
    print(f"w={cfg['scale_pos_weight']}, d={cfg['max_depth']}, leaves={cfg['num_leaves']}: ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}, Brier={brier:.4f}")
