# Kestrel Home — Pre-Dispatch Return Risk & Intervention Engine

Production machine learning pipeline, explainable API, and TypeScript interface designed to identify high-risk appliance orders prior to shipment and maximize net rupee savings under Kestrel Operations Policy v4.1.

---

## 🚀 Quickstart (Run on a Clean Machine)

The service starts locally **without any paid API keys** or external network requirements.

### 1. Requirements
* Python 3.10+
* (Optional for frontend development: Node.js 18+)

### 2. Install Python Dependencies
```bash
pip install fastapi uvicorn scikit-learn lightgbm pandas numpy
```

### 3. Launch the Service
```bash
python run_service.py
```
* **Frontend UI (TypeScript / React):** Open [http://localhost:8000](http://localhost:8000)
* **Interactive API Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Project Deliverables

| Deliverable | File / Location | Description |
|---|---|---|
| **Test Predictions** | [`predictions.csv`](predictions.csv) | Calibrated return risk probabilities for all 2,096 test orders. |
| **Executive Memo** | [`memo-for-ritu.md`](memo-for-ritu.md) | 1-page non-technical decision memo for Head of Operations Ritu Deshpande. |
| **Submission Form** | [`submission-form.md`](submission-form.md) | Exhaustive answers covering metrics, data critique, and rupee arithmetic. |
| **Working Service** | [`run_service.py`](run_service.py), [`api/`](api/) | FastAPI REST service with pre-dispatch explainability engine. |
| **Frontend UI** | [`frontend/`](frontend/) | Responsive TypeScript / React warehouse packaging station interface. |
| **Data Cleaning** | [`src/preprocess.py`](src/preprocess.py) | Automated pipeline fixing duplicates, 100x currency bug, and leakage. |
| **Validation Evidence** | [`src/train_baseline.py`](src/train_baseline.py) | Out-of-time chronological validation benchmark. |

---

## 💡 Key Business & Technical Discoveries

1. **Pushing Back on "95% Accuracy":**
   With an 11.4% historical return rate, a trivial model predicting "no return" achieves **88.6% accuracy** while saving ₹0. We optimized for **PR-AUC (0.413)** and **Net Rupee Savings**.
2. **Replacing "Dispatch Holds" with "₹45 Confirmation Calls":**
   Holding orders > 24 hours triggers a **12% customer cancellation rate** (Policy §7), destroying **>₹1.4 lakh in profit margin**. Routing high-risk shipments to a ₹45 pre-dispatch verification call resolves issues with **0 cancellations**, netting **+₹37,625 per quarter**.
3. **Protecting Shield VIPs:**
   Shield members return 18.6% of orders because returns are free (§6), but have the highest Lifetime Value (3 appliances/year). The system tags them for white-glove care, preventing shipment holds that cause churn.
4. **Target Leakage Purged:**
   Eliminated post-dispatch columns (`pickup_scheduled_at` and `last_service_event_type`) that only exist weeks after delivery in historical data.

---

## 🛠️ Reproducing the ML Pipeline

1. **Data Preprocessing & Sanitization:**
   ```bash
   python src/preprocess.py
   ```
2. **Run Time-Based Validation Benchmark:**
   ```bash
   python src/train_baseline.py
   ```
3. **Evaluate Rupee Threshold Matrix:**
   ```bash
   python src/evaluate_business_cost.py
   ```
4. **Generate Final Predictions:**
   ```bash
   python src/generate_predictions.py
   ```

---

## 💻 Frontend Development (TypeScript)

To modify or rebuild the frontend UI:
```bash
cd frontend
npm install
npm run build    # Compiles TypeScript and bundles to dist/
```
The compiled bundle in `frontend/dist` is automatically served by FastAPI on port 8000.
