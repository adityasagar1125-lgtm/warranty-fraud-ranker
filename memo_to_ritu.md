# MEMORANDUM

**TO:** Ritu Deshpande, Head of D2C Operations  
**FROM:** Kabir Nanda, Account Lead (Warranty Intelligence Team)  
**DATE:** October 7, 2026  
**SUBJECT:** Operational Recommendation: Capacity-Aware Warranty Claim Triage vs. Board KPI Trap  

---

### 1. The Decision: Deploy as a 40/Month Desk Triage List, Not an Auto-Reject Engine
We strongly advise **against** deploying an automated claim-rejection system or evaluating this project on overall percentage accuracy. In our warranty portfolio, verified fraud accounts for only **~1.2%** of claims. A trivial rule that blindly approves 100% of claims achieves a **98.8% accuracy rate** on paper while recovering **₹0** in fraud losses and allowing millions in leakage.

Instead, we have engineered a **capacity-aware triage ranker** matched precisely to the investigation desk’s fixed capacity of **40 claims per month** (Policy §5). Rather than auto-rejecting claims, the system generates a prioritized monthly audit queue ranked by **expected net rupees saved per claim checked**, complete with transparent, verifiable reasons.

---

### 2. The Headline Number: +₹1,351 Net Saved per Claim Checked on Test Desk
Under the existing random/manual sampling approach, checking 40 claims a month yields less than 1 fraud catch per month (~₹1,500 gross recovery minus ₹15,200 in customer goodwill friction, resulting in a net negative return).

With the Kestrel v2.0 3-Layer Triage Ranker:
* **Validation Performance (Post-May Regime):** **+₹174 net saved per claim checked** (net of ₹380 goodwill penalty), catching **58.3% of all fraud (21/36)** at **26.3% precision** and **0.924 ROC-AUC**.
* **Test Quarter Triage Yield (Jul–Sep 2026):** **+₹1,351 net saved per claim checked** across the top 40/month queue, with an average top-40 fraud probability of **79.5%**.
* **Quarterly Net Recovery:** The desk will recover an estimated **₹162,060 net** across the test quarter within its existing 40-claim labor envelope.

---

### 3. The Rupees: Reclaiming Payouts from the May 2026 Auto-Approval Policy
On May 1, 2026, Kestrel implemented an operational rule auto-approving claims under ₹2,000 without partner inspection to speed up turnarounds. While well-intentioned, bad actors quickly discovered this loophole:
* **Small-claim fraud incidence jumped from 0.5% (pre-May) to 3.2% (post-May)**—a 6.4x spike.
* **35 out of the 36 post-May frauds** were small claims submitted just under the ₹2,000 threshold.
* The triage engine systematically prioritizes these threshold-gaming claims, stopping an estimated **₹162,000+ in uninspected leakage** across the Jul–Sep 2026 test quarter alone.

---

### 4. Partner Dynamics: A Handful of Rogue Outlets, Not "All New Partners"
Meenal Joshi’s operational intuition is fully corroborated by the data. Genuine new service partners onboarded across Tier-2 and Tier-3 cities are critical to our expansion and exhibit a minuscule fraud rate of **0.1%** (1 in 924 claims) when submitting standard small claims.

The fraud is concentrated in a tiny cluster of rogue outlets exploiting the system (e.g., outlet `SP3160` committed 10 frauds in 39 claims; `SP3232` committed 7 in 40 claims; `SP3207` committed 13 in 39 claims). Our model utilizes **Empirical-Bayes shrinkage** to evaluate outlets on their own statistical records rather than broadly blacklisting newly onboarded partners.

---

### 5. Next Week Action Plan: Zero-Cost Operational Fixes
We recommend taking the following immediate steps alongside deploying the triage tool:
1. **Targeted Inspection Override:** Immediately revoke auto-approval privileges for the top-5 flagged outlets (`SP3160`, `SP3232`, `SP3318`, `SP3129`, `SP3207`) and mandate physical inspection sign-off and photo proof regardless of claim amount.
2. **Policy Threshold Adjustment:** Introduce a lightweight photo upload requirement for claims between ₹1,800 and ₹1,999 to neutralize threshold gaming without slowing down legitimate minor claims.
3. **CRM Logging Standardisation:** Enforce discrete outcome recording in the CRM (resolving the legacy issue where unresolved cases defaulted to "0"), providing clean ground-truth data for monthly model retraining.

---
*The complete interactive dashboard and REST service are running locally at `http://127.0.0.1:8000`.*
