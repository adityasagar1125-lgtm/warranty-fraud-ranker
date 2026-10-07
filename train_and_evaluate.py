"""
Train and Evaluate Script for Kestrel Claim Triage
Conducts time-split validation, baseline comparisons, failure mode audits,
and produces predictions.csv and the serialised pipeline artifact.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, brier_score_loss

from kestrel.data_cleaner import clean_and_prepare_dataset
from kestrel.feature_pipeline import KestrelFeaturePipeline
from kestrel.model import KestrelClassifier
from kestrel.decision_engine import KestrelDecisionEngine
from kestrel.pipeline import KestrelClaimPipeline


def evaluate_triage_policy(
    val_df: pd.DataFrame,
    score_col: str,
    goodwill_cost: float = 380.0,
    monthly_capacity: int = 40
) -> dict:
    """
    Evaluates a scoring method on a validation dataset with actual fraud labels.
    Simulates the monthly 40-claim desk review capacity.
    """
    df = val_df.copy()
    df['expected_saving'] = df[score_col] * df['claim_amount_inr'] - (1.0 - df[score_col]) * goodwill_cost
    df['val_month'] = df['submitted_dt'].dt.to_period('M').astype(str)
    
    # Rank within each month by expected savings
    df['rank'] = df.groupby('val_month')['expected_saving'].rank(ascending=False, method='first')
    df['checked'] = df['rank'] <= monthly_capacity
    
    checked_df = df[df['checked']].copy()
    not_checked_df = df[~df['checked']].copy()
    
    total_checked = len(checked_df)
    total_months = df['val_month'].nunique()
    
    # Fraud caught
    tp = int((checked_df['is_fraud'] == 1).sum())
    fp = int((checked_df['is_fraud'] == 0).sum())
    fn = int((not_checked_df['is_fraud'] == 1).sum())
    total_fraud = int((df['is_fraud'] == 1).sum())
    
    precision_at_k = tp / max(1, total_checked)
    recall_at_k = tp / max(1, total_fraud)
    
    # Rupee accounting
    fraud_amount_prevented = float(checked_df[checked_df['is_fraud'] == 1]['claim_amount_inr'].sum())
    goodwill_paid_fp = float(fp * goodwill_cost)
    net_rupees_saved = float(fraud_amount_prevented - goodwill_paid_fp)
    rupees_saved_per_claim_checked = float(net_rupees_saved / max(1, total_checked))
    
    # Total fraud volume in rupees
    total_fraud_volume_inr = float(df[df['is_fraud'] == 1]['claim_amount_inr'].sum())
    
    # Standard classification accuracy if all unflagged are paid
    tn = int((not_checked_df['is_fraud'] == 0).sum())
    overall_accuracy = float((tp + tn) / len(df))
    
    roc_auc = float(roc_auc_score(df['is_fraud'], df[score_col])) if df['is_fraud'].nunique() > 1 else 0.5
    pr_auc = float(average_precision_score(df['is_fraud'], df[score_col])) if df['is_fraud'].nunique() > 1 else 0.0
    brier = float(brier_score_loss(df['is_fraud'], np.clip(df[score_col], 0, 1)))
    
    return {
        "total_claims_in_slice": len(df),
        "total_fraud_in_slice": total_fraud,
        "total_fraud_rupees": total_fraud_volume_inr,
        "months_evaluated": total_months,
        "total_claims_checked": total_checked,
        "fraud_caught_tp": tp,
        "genuine_held_fp": fp,
        "fraud_missed_fn": fn,
        "precision_at_40_month": precision_at_k,
        "recall_at_40_month": recall_at_k,
        "fraud_amount_prevented_inr": fraud_amount_prevented,
        "goodwill_paid_inr": goodwill_paid_fp,
        "net_rupees_saved_inr": net_rupees_saved,
        "rupees_saved_per_claim_checked": rupees_saved_per_claim_checked,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "overall_accuracy": overall_accuracy
    }


def main():
    print("================================================================================")
    print("KESTREL CLAIM TRIAGE: TRAINING, TIME-SPLIT VALIDATION & BENCHMARK SUITE")
    print("================================================================================")
    
    # 1. Load data
    train_raw = pd.read_csv('18ec8f38-4ac1-4b7b-8685-9d1fc2aa3b02-train.csv')
    test_raw = pd.read_csv('63acfa61-2184-4cc0-abe0-035c9be66a16-test_unlabelled.csv')
    partners = pd.read_csv('a30b85c5-5a89-43ba-b872-422233b40365-partners.csv')
    products = pd.read_csv('de9c465d-63fe-4887-938f-ffa7a2664828-products.csv')
    sample_sub = pd.read_csv('696f48ab-5288-4751-b32d-3fd31d550437-sample_submission.csv')
    
    print(f"Loaded: Train {train_raw.shape}, Test {test_raw.shape}, Partners {partners.shape}, Products {products.shape}")
    
    # 2. Prepare Clean Train Dataset
    clean_train = clean_and_prepare_dataset(train_raw, partners, products, is_train=True)
    labeled_train = clean_train.dropna(subset=['is_fraud']).copy()
    labeled_train['is_fraud'] = labeled_train['is_fraud'].astype(int)
    
    print(f"Deduplicated labelled train: {len(labeled_train)} claims ({labeled_train['is_fraud'].sum()} frauds, base rate {labeled_train['is_fraud'].mean()*100:.2f}%)")
    
    # 3. TIME-BASED SPLIT VALIDATION
    # Split: Train on claims <= 2026-03-31; Validate on claims from 2026-04-01 to 2026-06-30
    split_date = '2026-04-01'
    t_train = clean_train[clean_train['submitted_dt'] < split_date].copy()
    t_val = clean_train[clean_train['submitted_dt'] >= split_date].copy()
    
    t_train_labeled = t_train.dropna(subset=['is_fraud']).copy()
    t_val_labeled = t_val.dropna(subset=['is_fraud']).copy()
    t_train_labeled['is_fraud'] = t_train_labeled['is_fraud'].astype(int)
    t_val_labeled['is_fraud'] = t_val_labeled['is_fraud'].astype(int)
    
    print(f"\n--- TIME SPLIT SETUP ---")
    print(f"Train Slice (< {split_date}): {len(t_train_labeled)} claims, {t_train_labeled['is_fraud'].sum()} frauds ({t_train_labeled['is_fraud'].mean()*100:.2f}%)")
    print(f"Val Slice (Apr-Jun 2026): {len(t_val_labeled)} claims, {t_val_labeled['is_fraud'].sum()} frauds ({t_val_labeled['is_fraud'].mean()*100:.2f}%)")
    
    # Post-May Slice in Val
    t_val_post_may = t_val_labeled[t_val_labeled['submitted_dt'] >= '2026-05-01'].copy()
    print(f"Post-May Val Slice (May-Jun 2026): {len(t_val_post_may)} claims, {t_val_post_may['is_fraud'].sum()} frauds ({t_val_post_may['is_fraud'].mean()*100:.2f}%)")
    
    # 4. BENCHMARK COMPARISONS ON VALIDATION SET
    
    # BASELINE 1: Approve All
    t_val_labeled['score_approve_all'] = 0.0
    t_val_post_may['score_approve_all'] = 0.0
    res_approve_all = evaluate_triage_policy(t_val_labeled, 'score_approve_all')
    res_approve_all_post_may = evaluate_triage_policy(t_val_post_may, 'score_approve_all', monthly_capacity=40)
    
    # BASELINE 2: Rules-Only ("New partner & small claim < Rs 2,000")
    t_val_labeled['score_rules'] = ((t_val_labeled['is_new_partner'] == 1) & (t_val_labeled['is_small_claim'] == 1)).astype(float) * 0.8
    t_val_post_may['score_rules'] = ((t_val_post_may['is_new_partner'] == 1) & (t_val_post_may['is_small_claim'] == 1)).astype(float) * 0.8
    res_rules = evaluate_triage_policy(t_val_labeled, 'score_rules')
    res_rules_post_may = evaluate_triage_policy(t_val_post_may, 'score_rules', monthly_capacity=40)
    
    # BASELINE 3: Plain Classifier (Standard HistGB without EB shrinkage or serial graph)
    from sklearn.ensemble import HistGradientBoostingClassifier
    plain_num_cols = ['claim_amount_inr', 'days_since_purchase', 'customer_prior_claims', 'photo_attached_bin', 'partner_inspected_bin', 'is_new_partner', 'is_small_claim']
    plain_clf = HistGradientBoostingClassifier(random_state=42)
    plain_clf.fit(t_train_labeled[plain_num_cols], t_train_labeled['is_fraud'])
    t_val_labeled['score_plain'] = plain_clf.predict_proba(t_val_labeled[plain_num_cols])[:, 1]
    t_val_post_may['score_plain'] = plain_clf.predict_proba(t_val_post_may[plain_num_cols])[:, 1]
    res_plain = evaluate_triage_policy(t_val_labeled, 'score_plain')
    res_plain_post_may = evaluate_triage_policy(t_val_post_may, 'score_plain', monthly_capacity=40)
    
    # 3-LAYER SYSTEM ON VALIDATION SPLIT
    val_feature_pipe = KestrelFeaturePipeline(shrinkage_m=12.0)
    val_feature_pipe.fit(t_train)
    
    X_tr = val_feature_pipe.transform(t_train_labeled)
    y_tr = t_train_labeled['is_fraud']
    
    X_va = val_feature_pipe.transform(t_val_labeled)
    X_va_post_may = val_feature_pipe.transform(t_val_post_may)
    
    val_model = KestrelClassifier()
    val_model.fit(
        X=X_tr,
        y=y_tr,
        timestamps=t_train_labeled['submitted_dt'],
        sources=t_train_labeled['source'],
        recency_weight_max=2.5,
        legacy_zero_weight=0.75
    )
    
    t_val_labeled['score_3layer'] = val_model.predict_proba(X_va)
    t_val_post_may['score_3layer'] = val_model.predict_proba(X_va_post_may)
    
    res_3layer = evaluate_triage_policy(t_val_labeled, 'score_3layer')
    res_3layer_post_may = evaluate_triage_policy(t_val_post_may, 'score_3layer', monthly_capacity=40)
    
    print("\n================================================================================")
    print("VALIDATION RESULTS SUMMARY (Apr-Jun 2026 Validation Slice, 3 months, 120 checks)")
    print("================================================================================")
    
    summary_table = pd.DataFrame([
        {
            "Model / Strategy": "1. Approve All (Do Nothing)",
            "Accuracy": f"{res_approve_all['overall_accuracy']*100:.1f}%",
            "Precision@40/mo": f"{res_approve_all['precision_at_40_month']*100:.1f}%",
            "Fraud Caught (TP)": f"{res_approve_all['fraud_caught_tp']}/{res_approve_all['total_fraud_in_slice']}",
            "Genuine Held (FP)": f"{res_approve_all['genuine_held_fp']}",
            "Goodwill Bill": f"Rs {res_approve_all['goodwill_paid_inr']:,.0f}",
            "Net Rs Saved": f"Rs {res_approve_all['net_rupees_saved_inr']:,.0f}",
            "Rs Saved / Check": f"Rs {res_approve_all['rupees_saved_per_claim_checked']:,.0f}",
            "PR-AUC": f"{res_approve_all['pr_auc']:.3f}"
        },
        {
            "Model / Strategy": "2. Rules-Only (New Partner & Small)",
            "Accuracy": f"{res_rules['overall_accuracy']*100:.1f}%",
            "Precision@40/mo": f"{res_rules['precision_at_40_month']*100:.1f}%",
            "Fraud Caught (TP)": f"{res_rules['fraud_caught_tp']}/{res_rules['total_fraud_in_slice']}",
            "Genuine Held (FP)": f"{res_rules['genuine_held_fp']}",
            "Goodwill Bill": f"Rs {res_rules['goodwill_paid_inr']:,.0f}",
            "Net Rs Saved": f"Rs {res_rules['net_rupees_saved_inr']:,.0f}",
            "Rs Saved / Check": f"Rs {res_rules['rupees_saved_per_claim_checked']:,.0f}",
            "PR-AUC": f"{res_rules['pr_auc']:.3f}"
        },
        {
            "Model / Strategy": "3. Plain HistGB Classifier",
            "Accuracy": f"{res_plain['overall_accuracy']*100:.1f}%",
            "Precision@40/mo": f"{res_plain['precision_at_40_month']*100:.1f}%",
            "Fraud Caught (TP)": f"{res_plain['fraud_caught_tp']}/{res_plain['total_fraud_in_slice']}",
            "Genuine Held (FP)": f"{res_plain['genuine_held_fp']}",
            "Goodwill Bill": f"Rs {res_plain['goodwill_paid_inr']:,.0f}",
            "Net Rs Saved": f"Rs {res_plain['net_rupees_saved_inr']:,.0f}",
            "Rs Saved / Check": f"Rs {res_plain['rupees_saved_per_claim_checked']:,.0f}",
            "PR-AUC": f"{res_plain['pr_auc']:.3f}"
        },
        {
            "Model / Strategy": "4. Kestrel 3-Layer System",
            "Accuracy": f"{res_3layer['overall_accuracy']*100:.1f}%",
            "Precision@40/mo": f"{res_3layer['precision_at_40_month']*100:.1f}%",
            "Fraud Caught (TP)": f"{res_3layer['fraud_caught_tp']}/{res_3layer['total_fraud_in_slice']}",
            "Genuine Held (FP)": f"{res_3layer['genuine_held_fp']}",
            "Goodwill Bill": f"Rs {res_3layer['goodwill_paid_inr']:,.0f}",
            "Net Rs Saved": f"Rs {res_3layer['net_rupees_saved_inr']:,.0f}",
            "Rs Saved / Check": f"Rs {res_3layer['rupees_saved_per_claim_checked']:,.0f}",
            "PR-AUC": f"{res_3layer['pr_auc']:.3f}"
        }
    ])
    print(summary_table.to_string(index=False))
    
    # 5. TRAIN FULL PRODUCTION PIPELINE ON ENTIRE HISTORICAL DATASET (Apr 2025 – Jun 2026)
    print("\n--- TRAINING FULL PRODUCTION PIPELINE ---")
    prod_pipeline = KestrelClaimPipeline(
        partners_df=partners,
        products_df=products,
        goodwill_cost_inr=380.0,
        monthly_capacity=40,
        shrinkage_m=12.0
    )
    prod_pipeline.fit(train_raw)
    
    # Save Model Artifact
    model_path = os.path.join('models', 'kestrel_pipeline.joblib')
    prod_pipeline.save(model_path)
    print(f"Saved production pipeline model to '{model_path}'")
    
    # 6. SCORE TEST DATASET AND GENERATE PREDICTIONS.CSV
    print("\n--- SCORING TEST UNLABELLED DATASET (Jul-Sep 2026, 2,252 claims) ---")
    predictions_df, test_triage_df = prod_pipeline.score_test_batch(test_raw)
    
    # Verify sample submission compatibility
    assert len(predictions_df) == len(sample_sub), f"Length mismatch: {len(predictions_df)} vs {len(sample_sub)}"
    assert list(predictions_df.columns) == ['claim_id', 'score'], f"Column mismatch: {list(predictions_df.columns)}"
    assert predictions_df['claim_id'].tolist() == sample_sub['claim_id'].tolist(), "claim_id ordering mismatch!"
    
    predictions_df.to_csv('predictions.csv', index=False)
    print(f"Successfully generated predictions.csv with {len(predictions_df)} rows.")
    print("Predictions sample:")
    print(predictions_df.head(10))
    print("\nScore statistics on test set:")
    print(predictions_df['score'].describe())
    
    # Save enriched test triage output for web app and dashboard
    test_triage_export = test_triage_df[['claim_id', 'submitted_at', 'month', 'partner_id', 'sku', 'product_serial', 'claim_amount_inr', 'photo_attached', 'partner_inspected', 'customer_prior_claims', 'score', 'calibrated_probability', 'expected_saving_inr', 'monthly_rank', 'recommendation', 'reasons', 'city', 'partner_type', 'family']].copy()
    test_triage_export.to_json('triage_test.json', orient='records', indent=2)
    test_triage_export.to_csv('triage_test.csv', index=False)
    print(f"Saved triage dataset to 'triage_test.json' and 'triage_test.csv'.")
    
    # Monthly breakdown of test triage
    print("\nTest Triage Monthly Summary (Top 40 / Month):")
    for m, m_group in test_triage_export.groupby('month'):
        top40 = m_group[m_group['monthly_rank'] <= 40]
        total_exp_savings = top40['expected_saving_inr'].sum()
        avg_score = top40['score'].mean()
        print(f"  Month {m}: Total Claims = {len(m_group)}, Top-40 Avg Score = {avg_score:.3f}, Est. Expected Net Savings = Rs {total_exp_savings:,.0f}")

    # 7. SAVE VALIDATION METRICS JSON FOR DOCUMENTATION
    metrics_export = {
        "time_split_validation": res_3layer,
        "post_may_validation": res_3layer_post_may,
        "approve_all_baseline": res_approve_all,
        "rules_only_baseline": res_rules,
        "plain_classifier_baseline": res_plain,
        "top_features": sorted(prod_pipeline.classifier.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:15]
    }
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics_export, f, indent=2)
    print("\nSaved evaluation metrics to 'evaluation_metrics.json'.")
    print("================================================================================")


if __name__ == '__main__':
    main()
