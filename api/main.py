"""
FastAPI Server for Kestrel Returns Risk Service
-----------------------------------------------
Endpoints:
- POST /predict: Scores a single order JSON and returns score + employee reasons.
- GET /sample-orders: Returns curated real-world orders for 1-click testing in UI.
- GET /health: Health status and model load verification.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from api.service import predict_order, load_resources

app = FastAPI(
    title="Kestrel Returns Risk API",
    description="Pre-dispatch order return risk scoring & explanation engine",
    version="1.0.0"
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrderRequest(BaseModel):
    order_id: Optional[str] = "KO2610504"
    customer_id: Optional[str] = "KC101428"
    sku: Optional[str] = "KH-RV-01"
    sales_channel: Optional[str] = "marketplace"
    payment_mode: Optional[str] = "cod"
    discount_pct: Optional[float] = 10.0
    qty: Optional[int] = 1
    promised_delivery_days: Optional[int] = 8
    delivery_pincode: Optional[int] = 411001
    is_gift: Optional[str] = "N"
    customer_prior_orders: Optional[int] = 3
    customer_prior_returns: Optional[int] = 2
    delivery_note: Optional[str] = "Call before delivery"
    order_placed_at: Optional[str] = "2026-08-01 14:00:00"


@app.on_event("startup")
def startup_event():
    load_resources()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "kestrel-returns-risk",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict_endpoint(order: OrderRequest):
    try:
        result = predict_order(order.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample-orders")
def get_sample_orders():
    """Curated order scenarios for easy one-click testing in the frontend"""
    return [
        {
            "label": "High Risk - COD Robot Vacuum with Prior Returns",
            "data": {
                "order_id": "ORD-TEST-001",
                "customer_id": "KC100015",
                "sku": "KH-RV-01",
                "sales_channel": "marketplace",
                "payment_mode": "cod",
                "discount_pct": 15,
                "qty": 1,
                "promised_delivery_days": 8,
                "delivery_pincode": 411001,
                "is_gift": "N",
                "customer_prior_orders": 3,
                "customer_prior_returns": 2,
                "delivery_note": "Ring twice, call before delivery"
            }
        },
        {
            "label": "Shield VIP Member - Air Fryer (Do Not Hold)",
            "data": {
                "order_id": "ORD-TEST-002",
                "customer_id": "KC100021",
                "sku": "KH-AF-02",
                "sales_channel": "app",
                "payment_mode": "prepaid_upi",
                "discount_pct": 5,
                "qty": 1,
                "promised_delivery_days": 4,
                "delivery_pincode": 400001,
                "is_gift": "N",
                "customer_prior_orders": 5,
                "customer_prior_returns": 1,
                "delivery_note": "Leave with security"
            }
        },
        {
            "label": "Low Risk - Prepaid Mixer Grinder (Quick Dispatch)",
            "data": {
                "order_id": "ORD-TEST-003",
                "customer_id": "KC100003",
                "sku": "KH-MG-01",
                "sales_channel": "web",
                "payment_mode": "prepaid_upi",
                "discount_pct": 10,
                "qty": 1,
                "promised_delivery_days": 2,
                "delivery_pincode": 560001,
                "is_gift": "N",
                "customer_prior_orders": 2,
                "customer_prior_returns": 0,
                "delivery_note": "Office address, weekdays only"
            }
        },
        {
            "label": "Partner Outlet Walk-in - Water Purifier (Pincode 000000)",
            "data": {
                "order_id": "ORD-TEST-004",
                "customer_id": "KC100100",
                "sku": "KH-WP-01",
                "sales_channel": "partner_outlet",
                "payment_mode": "prepaid_card",
                "discount_pct": 0,
                "qty": 1,
                "promised_delivery_days": 6,
                "delivery_pincode": 0,
                "is_gift": "N",
                "customer_prior_orders": 0,
                "customer_prior_returns": 0,
                "delivery_note": "NONE"
            }
        }
    ]


# Mount frontend dist if built
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = os.path.join(dist_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
