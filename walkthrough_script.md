# 3-Minute Screen Recording Walkthrough Script

> **Time Limit:** Exactly 3 minutes or less.  
> **Rule:** No slides. Show your screen (VS Code / Terminal / Browser UI).  
> **Key Sections:** 
> 1. What you tried (0:00 – 1:00)
> 2. What you changed (1:00 – 2:00)
> 3. What you threw away (2:00 – 3:00)

---

### Minute 1: What We Tried (0:00 – 1:00)
* **Screen:** Show your code editor (`src/train_baseline.py` and `src/train_xgboost.py`).
* **Say:**
  > "Hi, here is our 3-minute walkthrough for the Kestrel Home Returns Risk system. 
  > We started by auditing the 15 months of historical orders. Because test data represents future orders, we rejected random cross-validation and set up a strict chronological holdout: 12 months for training and the latest 3 months (2,126 orders) for validation.
  > We benchmarked four distinct architectures on this time split:
  > - Logistic Regression with L2 regularization
  > - Random Forest
  > - LightGBM
  > - And XGBoost 3.2.0.
  > Surprisingly, the regularized linear model achieved the highest out-of-time ROC-AUC of 0.7876 and PR-AUC of 0.4130, compared to 0.7686 for LightGBM and 0.7656 for XGBoost. This is because key return drivers—Cash on Delivery, past customer return ratio, and promised delivery lead times—act as strong additive factors where deep decision trees overfit to temporal noise."

---

### Minute 2: What We Changed (1:00 – 2:00)
* **Screen:** Show `memo-for-ritu.md` and the browser UI at `http://localhost:8000`.
* **Say:**
  > "Next, we fundamentally pushed back on the client's business ask. 
  > Ritu wanted a model that holds dispatch with '95% accuracy'. 
  > First, with an 11.4% return base rate, naive accuracy is meaningless—a model predicting zero returns achieves 88.6% accuracy while saving nothing.
  > Second, operations policy Section 7 states that holding orders for more than 24 hours causes a 12% customer cancellation rate. Freezing 600 orders would destroy over 1.4 lakh rupees in profit margin!
  > Furthermore, Shield members return 18.6% of orders because returns are free, but they have the highest lifetime value. Holding their orders would alienate our best VIPs.
  > So we changed the operational response: instead of holding orders, we route high-risk orders (risk >= 12%) to the Section 7 pre-dispatch confirmation call pilot. At 45 rupees per call, it prevents 35% of returns with zero cancellations, netting over 37,000 rupees in quarterly profit."

---

### Minute 3: What We Threw Away & Demo (2:00 – 3:00)
* **Screen:** Show the React / TypeScript UI at `http://localhost:8000`, click **"🚨 High Risk"** and **"⭐ Shield VIP Member"**.
* **Say:**
  > "Finally, here is what we threw away:
  > 1. We threw away the post-dispatch leakage columns: `pickup_scheduled_at` and `last_service_event_type`. In the training data, they gave 99% accuracy because they recorded post-delivery reverse pickups, but in the pre-dispatch test set, they don't exist.
  > 2. We threw away the October 2025 order values, where the payment gateway had inflated values by 100x into paise, and recalculated true rupees.
  > 3. We threw away paid LLM APIs for live scoring. Our working service runs 100% locally with zero paid keys on FastAPI and React TypeScript.
  > 
  > In the UI, packaging operators can click one button to evaluate risk. When an order is flagged, it gives plain-English reasons—like 2 prior returns on COD—and specific operational guidance to call rather than hold. 
  > Everything runs with one command: `python run_service.py`. Thank you!"
