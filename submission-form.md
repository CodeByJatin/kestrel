# Submission Form — Task 2 V1: Kestrel Home — Returns Risk (Variant A)

---

### Github Repo URL*
https://github.com/CodeByJatin/kestrel

---

### What did you build, and what business decision does it support? State the number and the rupees.*

We built a pre-dispatch risk assessment engine comprising:
1. A calibrated machine learning prediction pipeline that scores order return risk before shipment.
2. A local FastAPI inference service and React/TypeScript warehouse interface that surfaces explainable reasons for packaging staff.
3. An operational intervention protocol grounded in Kestrel's operations policy v4.1.

**The Business Decision:**
We replaced Ritu Deshpande’s proposed blunt "dispatch hold" with a targeted **₹45 pre-dispatch confirmation call** for orders with predicted return risk $\ge 0.12$. 

**The Number and the Rupees:**
* At Kestrel's quarterly volume (~2,100 orders, or ~700/month), the system flags **~198 orders per month** (30% of total volume).
* Holding orders causes a **12% customer cancellation rate** (Policy §7), destroying **~₹1.4 lakh in gross contribution profit** quarterly.
* Instead, conducting a ₹45 pre-dispatch confirmation call prevents **35% of would-be returns** (Policy §7) with **0 cancellations**:
  - **Quarterly Call Cost:** 595 calls × ₹45 = **₹26,775** (~₹8,925/month).
  - **Quarterly Return Expenses Avoided:** 56 returns prevented × ₹1,150 policy cost = **₹64,400** (~₹21,466/month).
  - **Net Rupee Profit Generated:** **+₹37,625 per quarter (~₹12,542/month in net savings)**.

---

### What score do you expect `predictions.csv` to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it. We compare this with the real score.*

* **Primary Metric:** **ROC-AUC** and **PR-AUC (Average Precision)**.
* **Expected Score:** 
  - **ROC-AUC:** **0.77 – 0.79** (Estimated: **0.785**).
  - **PR-AUC:** **0.39 – 0.42** (Estimated: **0.408**).
  - **Brier Score (Calibration):** **0.087 – 0.089**.
* **Why this metric over accuracy:**
  The natural return rate is only **11.4%**. Naive classification accuracy is misleading: a trivial dummy model predicting "no return" on every order achieves **88.6% accuracy** while catching 0 returns and delivering ₹0 in value. ROC-AUC and PR-AUC measure true discrimination and ranking ability across the entire volume, ensuring warehouse staff prioritize the riskiest orders first.
* **How we estimated it:**
  We estimated this using a strict **out-of-time chronological validation split** (training on 12 months from April 2025 – March 2026, and testing on the subsequent 3 months from April 2026 – June 2026; 2,126 orders). On this holdout, our regularized linear model achieved an ROC-AUC of **0.7876** and PR-AUC of **0.4130**, while the ensemble blend scored **0.7816 ROC-AUC** and **0.4043 PR-AUC**.

---

### How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.*

* **Validation Strategy:**
  We used an **out-of-time temporal holdout split**, mirroring the production deployment horizon:
  - **Train Period:** 2025-04-01 to 2026-03-31 (12 months; 8,378 orders; return rate 11.40%).
  - **Validation Period:** 2026-04-01 to 2026-06-30 (3 months; 2,126 orders; return rate 11.52%).
* **Benchmarking Results:**
  - **Regularized Logistic Regression (C=0.1):** ROC-AUC **0.7876** | PR-AUC **0.4130** | Brier **0.0872**
  - **Ensemble Blend (LR + LightGBM):** ROC-AUC **0.7816** | PR-AUC **0.4043** | Brier **0.0880**
  - **LightGBM (depth=3, leaves=7):** ROC-AUC **0.7686** | PR-AUC **0.3746** | Brier **0.0881**
  - **XGBoost 3.2.0 (depth=3):** ROC-AUC **0.7656** | PR-AUC **0.3671** | Brier **0.0878**
  - **Random Forest (200 trees, depth=10):** ROC-AUC **0.7365** | PR-AUC **0.3513** | Brier **0.0910**
* **Error Rate & Operational Confusion Matrix (at threshold 0.12 on 2,126 validation orders):**
  - True Positives (Caught returns): **160** (Recall: 65.3%)
  - False Negatives (Missed returns): **85**
  - False Positives (Called orders that wouldn't have returned): **435**
  - True Negatives (Correctly passed): **1,446**
* **Cases the Model Gets Wrong:**
  1. *First-time customers buying standard appliances (Air Fryer/Mixer) via COD with zero past purchase history:* Without a historical return ratio, the model under-predicts remorse risk.
  2. *Defective manufacturing batches:* Unpredictable technical hardware failures that occur weeks post-delivery cannot be captured from checkout metadata.
  3. *Newly launched SKUs:* Products launched in late 2025 have sparse return baselines.

---

### Did you change, narrow, or push back on the client's ask? What, when, and why. [can only raise your score]*

**Yes, we pushed back on two central pillars of Ritu’s request:**

1. **Pushing Back on "95% Accuracy":**
   - *Why:* With an 11.4% return prevalence, a model that classifies everything as "non-return" scores 88.6% accuracy while providing zero business utility. Reaching 95% accuracy naively requires suppressing positive classifications, resulting in <15% recall and leaving >₹2 lakh in return losses on the table. We shifted the objective to **PR-AUC and Net Rupee Optimization**.
2. **Pushing Back on "Hold Dispatch on Anything It Flags":**
   - *Why:* Policy §7 states that holding an order > 24 hours causes **12% customer cancellation**. Freezing 595 orders would forfeit ~71 orders, losing **>₹1.4 lakh in profit margin**.
   - *What we changed:* We rerouted high-risk orders to the **pre-dispatch confirmation call pilot** (Policy §7). At ₹45 per call, it prevents 35% of returns, triggers zero cancellations, and nets **+₹37,625 quarterly profit**.
3. **Protecting Kestrel Shield Members:**
   - Ritu said: *"We'll deal with Shield later."*
   - *Why we refused to ignore Shield:* Shield members return 18.6% of orders (2x non-Shield) because returns are free (§6). However, they buy 3 appliances/year and have the highest LTV. Holding their orders would destroy high-value customer goodwill (confirming Service Desk Manager Meenal Joshi's concern). We programmed the system to route Shield orders to a **"White-Glove VIP Confirmation Call"** rather than a dispatch freeze.

---

### What is wrong with what you are handing us, or with the data we handed you? Be specific: bugs, shortcuts, columns you did not trust, rows that looked wrong. [can only raise your score]*

**1. Severe Post-Dispatch Target Leakage in `train.csv`:**
* `pickup_scheduled_at`: Populated in 1,231 training rows (91.8% return rate), but **100% empty (0 rows)** in `test_unlabelled.csv`.
* `last_service_event_type`: In `train.csv`, contains post-delivery statuses (`REVERSE_PICKUP` = 100% returns; `DEMO_DONE`/`INSTALL_DONE` = 0% returns). In `test_unlabelled.csv`, only pre-dispatch states exist (`NONE` and `INSTALL_BOOKED`).
* *Action:* Both columns were strictly purged from feature training.

**2. October 2025 Payment Gateway 100x Paise Bug:**
* Tanmay noted: *"October festive orders went through the new payment gateway, I haven't checked those."*
* Every single order in October 2025 had `order_value_inr` recorded in **paise instead of rupees** (multiplied by exactly 100, e.g. ₹7,838 recorded as ₹783,797).
* *Action:* Recomputed true order values from catalog prices, quantities, and discounts.

**3. Ingestion Duplicates:**
* 651 order pairs (1,302 rows) in `train.csv` appeared twice due to feed re-imports (`crm` vs `partner_feed`).
* *Action:* Deduplicated on `order_id` to establish 10,504 unique historical records.

**4. Default Walk-in Pincodes:**
* 848 orders in train and 176 in test had `delivery_pincode == 000000` (walk-in outlet orders with no address).
* *Action:* Explicitly engineered an `is_walkin_pincode` flag rather than treating it as numeric geographic distance.

---

### What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)? Show the arithmetic. If you used no paid calls, say so.*

* **Paid API Calls:** **₹0.00**. We used **zero paid API calls**. All inference runs locally on the host machine using scikit-learn / LightGBM.
* **Compute Cost per Prediction:**
  - Inference Latency: < 1.8 milliseconds per order on standard CPU.
  - Server Spec: Modest container (e.g. 0.5 vCPU, 512MB RAM on AWS/GCP or local warehouse terminal ~₹400–600/month if hosted on cloud; ₹0 if run locally on existing packaging station PCs).
* **Monthly Arithmetic:**
  - Volume: 700 orders/month.
  - LLM/Model API Fees: **₹0.00 / month**.
  - Total incremental model runtime cost: **₹0.00 / month**.

---

### What did you deliberately leave out, and why that rather than something else?*

1. **Deliberately Left Out Deep Learning / Neural Networks:** Tabular datasets with categorical cardinality and small volume (~10k rows) perform significantly better and train 100x faster on regularized linear models and gradient boosted trees.
2. **Deliberately Left Out Paid LLM API Dependencies for Live Scoring:** The brief required zero paid keys on a clean machine. Generating real-time explanations using model coefficient attribution guarantees sub-2ms latency, 100% offline uptime, and ₹0 runtime cost.
3. **Deliberately Left Out Automatic Order Cancellations:** Cancelling orders autonomously creates unacceptable customer churn. Human intervention via phone confirmation retains 65% of flagged orders as completed sales.
4. **Deliberately Left Out Discarded XGBoost in Live Deployment:** Benchmarking showed XGBoost (ROC-AUC 0.7656) did not beat regularized Logistic Regression (0.7876) or LightGBM (0.7686), yet added a ~102 MB package dependency.

---

### Anything you built or found that nobody asked for?*

1. **Interactive 1-Click Operational Scenario Tester in the UI:** Built instant pre-set buttons in the React/TypeScript frontend (High Risk COD, Shield VIP Member, Low Risk Prepaid, Partner Walk-in) allowing operators to test edge cases with zero typing.
2. **The "Shield VIP Member" Guidance Protocol:** Hardcoded a policy safeguard in the UI alerting packaging staff that Shield members must receive white-glove calls rather than dispatch holds.
3. **Financial Impact Box in the UI:** Shows packaging operators the exact rupee economics for every order (e.g., "Rs 45 Call vs Rs 1,150 Return Saved").

---

### What did you use AI for? Which tools and models, where they helped, where they misled you, what you threw away. Link your three-minute screen recording here*

* **Tools & Models Used:**
  - Coding assistance for scaffolding the TypeScript React application and exploratory data analysis scripts.
* **Where AI Helped:**
  - Rapid generation of TypeScript boilerplate and CSS styling.
  - Accelerated financial matrix computations mapping policy costs.
* **Where AI Misled Us:**
  - Initially proposed using LLM APIs to generate dynamic customer call scripts, which would have violated the zero-paid-API-key requirement and introduced external latency. We discarded this in favor of deterministic model-weight reason attribution.
  - Initially suggested standard random k-fold cross validation, which would have leaked future temporal data into past training folds. We discarded this in favor of a strict chronological out-of-time split.
* **What We Threw Away:**
  - Discarded complex XGBoost configurations in favor of a lightweight, highly generalizable regularized linear pipeline.
  - Purged post-dispatch leakage features (`pickup_scheduled_at`, `last_service_event_type`).
* **Your Public Google Drive Link:**
  https://drive.google.com/drive/u/0/folders/1lXUYsWBVcC3Z7gzdUodJgAJvJf6i8xqk

---

### Someone picks this up on Monday and you are unreachable. The three things they need to know.*

1. **How to run the service in 1 command:**
   Run `python run_service.py`. The API and compiled TypeScript interface will immediately launch at `http://localhost:8000` with zero API keys required.
2. **Never hold a Shield member's order:**
   Shield members represent 20% of orders and have the highest LTV. When their orders are flagged, initiate a courteous confirmation call—never place their orders on shipment hold.
3. **Never train on `pickup_scheduled_at` or `last_service_event_type`:**
   These columns are post-delivery outcomes that do not exist at pre-dispatch. Including them in retrained models will destroy real-world predictive validity.
