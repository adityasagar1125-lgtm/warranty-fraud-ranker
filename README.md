# Kestrel Claim Triage: A Capacity-Aware Warranty Fraud Ranker

**Author:** Kabir Nanda (Account Lead) & Warranty Intelligence Team  
**Policy Baseline:** Kestrel Operations Policy v4.1  
**Status:** Production Ready (v1.0) · Fully Local (Zero Paid API Keys)

---

## 1. Project Overview

The **Kestrel Claim Triage** system is a capacity-aware warranty fraud ranker engineered for Kestrel Home Appliances. 

Rather than chasing an uninformative 97% overall accuracy target—which an "Approve All" baseline easily achieves on ~1.2% fraud data while recovering ₹0—this system is built around operational realities:
* **Fixed Desk Capacity:** 40 claims reviewed per month (Policy §5).
* **Asymmetric Costs:** Holding a genuine claim costs **₹380** in customer goodwill; paying a fraudulent claim costs the **full claim amount** (Policy §4).
* **Policy Loophole Mitigation:** Captures the 6.4x spike in small-claim fraud triggered by the May 1, 2026 auto-approval rule (< ₹2,000).
* **Fairness via Empirical-Bayes Shrinkage:** Protects innocent Tier-2/Tier-3 partners (0.1% fraud in clean cohorts) while isolating rogue outlets.

---

## 2. Quickstart (Clean Machine Setup)

This system requires **no paid external API keys** and runs completely offline.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Web Dashboard & REST API
```bash
python app.py
```
*Or using uvicorn directly:*
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### Step 3: Open the Interactive UI
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 3. Retraining & Re-Evaluating the Pipeline

To re-run the full time-based split validation, benchmark comparisons, and generate `predictions.csv`:
```bash
python train_and_evaluate.py
```

Outputs generated:
* `predictions.csv`: Exactly 2,252 rows matching `sample_submission.csv` (`claim_id,score`).
* `models/kestrel_pipeline.joblib`: Serialized production pipeline artifact.
* `triage_test.json` & `triage_test.csv`: Enriched monthly triage queues for the desk.
* `evaluation_metrics.json`: Validation numbers and feature importance rankings.

---

## 4. System Architecture

The solution integrates a 3-layer architecture:

```
[ Raw Claims & Serials ] ──► [ Layer 1: Calibrated Gradient Booster ]
                                          │
                             [ Layer 2: Entity Risk & EB Shrinkage ]
                                          │
                             [ Layer 3: Capacity Rupee Ranker ] ──► Top 40 Review Queue
```

### Layer 1: Calibrated Gradient-Boosted Classifier
* HistGradientBoosting + XGBoost ensemble trained with recency weighting and legacy Zoho noise down-weighting ($w=0.75$).
* Tabular features: Amount to list price ratio, purchase-to-warranty duration, customer prior claims non-linear curve, text TF-IDF + SVD embeddings, and keyword triggers.

### Layer 2: Entity-Risk Layer & Empirical-Bayes Shrinkage
* Shrunk outlet risk: $p_{\text{shrunk}} = \frac{k_{\text{outlet}} + m \cdot p_{\text{cohort}}}{n_{\text{outlet}} + m}$ with $m=12$.
* Outlets with 3 claims are pulled to the cohort prior (~1.2%), while rogue outlets like `SP3160` (10 fraud in 39 claims) receive an empirical risk score of **36.8%**.
* Serial reuse graph tracks cross-partner and multi-SKU serial sharing rings.

### Layer 3: Capacity-Aware Rupee Decision Engine
* Decision rule: $\text{Expected Net Saving} = p \times \text{Claim Amount} - (1 - p) \times 380$.
* Ranks claims by expected rupees saved and routes the top 40 each month to the investigation desk.
* Generates deterministic, verifiable plain-English reasons (never hallucinates).

---

## 5. REST API Documentation

### `POST /score`
Scores a single claim payload.

**Request Body:**
```json
{
  "claim_id": "WC720101",
  "partner_id": "SP3160",
  "sku": "KH-MG-02",
  "product_serial": "KH468276030",
  "claim_amount_inr": 1950,
  "days_since_purchase": 85,
  "photo_attached": "N",
  "partner_inspected": "N",
  "claim_description": "blade jammed",
  "inspector_note": "",
  "customer_prior_claims": 0
}
```

**Response:**
```json
{
  "claim_id": "WC720101",
  "score": 0.999,
  "calibrated_probability": 0.999,
  "expected_saving_inr": 1947.67,
  "recommendation": "CHECK",
  "reasons": [
    "Outlet SP3160: 10 of 39 historical claims were verified fraud (shrunk outlet risk: 19.8%).",
    "Threshold gaming pattern: Claim amount Rs 1,950 sits immediately below the Rs 2,000 auto-approval cap.",
    "Serial reuse alert: Serial KH468276030 was involved in 2 previous fraudulent claim(s).",
    "Missing physical evidence: No photo attached to claim submission."
  ],
  "confidence_note": "High confidence: Partner SP3160 has 39 historical records in system.",
  "data_warnings": [
    "Threshold gaming alert: Rs 1,950 sits immediately below the Rs 2,000 auto-approval cap."
  ]
}
```

### `GET /health`
Returns service health, model loading state, and policy parameters.

### `GET /api/top40?month=2026-07`
Returns the 40 highest-saving claims for the desk for July, August, or September 2026.

### `GET /predictions.csv`
Direct file download for the official submission file.

---

## 6. Project Artifacts & Deliverables

| File | Description |
|---|---|
| [`predictions.csv`](file:///d:/internship%20project%202/predictions.csv) | Final 2,252 scored test claims matching `sample_submission.csv`. |
| [`memo_to_ritu.md`](file:///d:/internship%20project%202/memo_to_ritu.md) | Non-technical executive memo for Ritu Deshpande. |
| [`evidence_report.md`](file:///d:/internship%20project%202/evidence_report.md) | Comprehensive validation, benchmark comparisons, and ROI report. |
| [`submission-form.md`](file:///d:/internship%20project%202/submission-form.md) | Completed submission form detailing tools, discarded approaches, and decisions. |
| [`train_and_evaluate.py`](file:///d:/internship%20project%202/train_and_evaluate.py) | Standalone training and evaluation script. |
| [`app.py`](file:///d:/internship%20project%202/app.py) | FastAPI service and backend endpoints. |
| [`kestrel/`](file:///d:/internship%20project%202/kestrel) | Modular Python package for cleaner, feature pipeline, model, and decision engine. |
| [`static/`](file:///d:/internship%20project%202/static) | Polished single-screen web UI with live evaluator, scenario loader, and Top 40 desk. |

---
*Developed for Kestrel Home Appliances Operations Team.*
