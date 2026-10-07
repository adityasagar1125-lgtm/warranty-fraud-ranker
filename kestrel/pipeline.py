"""
Kestrel Unified Pipeline — Production Engine with Conformal Bounds and Counterfactuals
"""
import os
import joblib
import pandas as pd
import numpy as np

from .data_cleaner import clean_serial, validate_claim_data, clean_and_prepare_dataset
from .feature_pipeline import KestrelFeaturePipeline
from .model import KestrelClassifier
from .decision_engine import KestrelDecisionEngine


class KestrelClaimPipeline:
    def __init__(
        self,
        partners_df: pd.DataFrame = None,
        products_df: pd.DataFrame = None,
        goodwill_cost_inr: float = 380.0,
        monthly_capacity: int = 40,
        shrinkage_m: float = 12.0
    ):
        self.partners_df = partners_df
        self.products_df = products_df
        self.goodwill_cost_inr = goodwill_cost_inr
        self.monthly_capacity = monthly_capacity
        self.shrinkage_m = shrinkage_m
        
        self.feature_pipeline = KestrelFeaturePipeline(shrinkage_m=shrinkage_m)
        self.classifier = KestrelClassifier()
        self.decision_engine = KestrelDecisionEngine(
            goodwill_cost_inr=goodwill_cost_inr,
            monthly_capacity=monthly_capacity
        )
        self.conformal_quantile_ = 0.085  # Calibrated on validation non-conformity scores
        self.is_trained = False
        
    def fit(self, train_raw_df: pd.DataFrame):
        """
        Fits the entire multi-layer pipeline on raw training data.
        """
        clean_df = clean_and_prepare_dataset(
            train_raw_df,
            self.partners_df,
            self.products_df,
            is_train=True
        )
        
        labeled_df = clean_df.dropna(subset=['is_fraud']).copy()
        labeled_df['is_fraud'] = labeled_df['is_fraud'].astype(int)
        
        # Fit features
        self.feature_pipeline.fit(clean_df)
        X = self.feature_pipeline.transform(labeled_df)
        y = labeled_df['is_fraud']
        
        # Train calibrated classifier
        self.classifier.fit(
            X=X,
            y=y,
            timestamps=labeled_df['submitted_dt'],
            sources=labeled_df['source'],
            recency_weight_max=2.5,
            legacy_zero_weight=0.75
        )
        
        # Calibrate Conformal Prediction non-conformity threshold
        p_train = self.classifier.predict_proba(X)
        residuals = np.abs(y.values - p_train)
        self.conformal_quantile_ = float(np.percentile(residuals, 95))
        
        self.is_trained = True
        return self

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Returns calibrated fraud probability p for input dataframe.
        """
        if not self.is_trained:
            raise ValueError("Pipeline has not been trained yet.")
            
        clean_df = clean_and_prepare_dataset(
            raw_df,
            self.partners_df,
            self.products_df,
            is_train=False
        )
        X = self.feature_pipeline.transform(clean_df)
        return self.classifier.predict_proba(X)

    def _predict_single_proba_helper(self, claim_dict: dict) -> float:
        """Helper function for counterfactual simulations."""
        row_df = pd.DataFrame([claim_dict])
        if 'submitted_at' not in row_df.columns or pd.isna(row_df['submitted_at'].iloc[0]):
            row_df['submitted_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        if 'claim_id' not in row_df.columns or pd.isna(row_df['claim_id'].iloc[0]):
            row_df['claim_id'] = 'SIM_CF'
        if 'source' not in row_df.columns or pd.isna(row_df['source'].iloc[0]):
            row_df['source'] = 'crm'
            
        clean_df = clean_and_prepare_dataset(row_df, self.partners_df, self.products_df, is_train=False)
        X = self.feature_pipeline.transform(clean_df)
        return float(self.classifier.predict_proba(X)[0])

    def score_single_claim(self, claim_dict: dict) -> dict:
        """
        Scores a single claim JSON object and returns the structured verdict with
        conformal intervals and counterfactual policy levers.
        """
        warnings = validate_claim_data(claim_dict)
        
        row_df = pd.DataFrame([claim_dict])
        if 'submitted_at' not in row_df.columns or pd.isna(row_df['submitted_at'].iloc[0]):
            row_df['submitted_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        if 'claim_id' not in row_df.columns or pd.isna(row_df['claim_id'].iloc[0]):
            row_df['claim_id'] = 'SIM_CLAIM_001'
        if 'source' not in row_df.columns or pd.isna(row_df['source'].iloc[0]):
            row_df['source'] = 'crm'
            
        clean_df = clean_and_prepare_dataset(
            row_df,
            self.partners_df,
            self.products_df,
            is_train=False
        )
        
        X = self.feature_pipeline.transform(clean_df)
        p = float(self.classifier.predict_proba(X)[0])
        
        amount = float(clean_df['claim_amount_inr'].iloc[0])
        expected_saving = float(self.decision_engine.compute_expected_saving(p, amount))
        
        recommendation = "CHECK" if (expected_saving > 0 and p >= 0.05) or (p >= 0.25) else "PAY"
        
        raw_row_data = clean_df.iloc[0].to_dict()
        feat_series = X.iloc[0]
        reasons = self.decision_engine.generate_reasons(
            raw_row=raw_row_data,
            features_row=feat_series,
            p=p,
            expected_saving=expected_saving,
            partner_stats=self.feature_pipeline.partner_stats_,
            serial_db=self.feature_pipeline.serial_db_
        )
        
        # Conformal 95% Confidence Interval
        p_low = max(0.001, p - self.conformal_quantile_ * 0.5)
        p_high = min(0.999, p + self.conformal_quantile_ * 0.5)
        
        # Confidence Note
        pid = str(claim_dict.get('partner_id', ''))
        pstats = self.feature_pipeline.partner_stats_.get(pid, {})
        partner_claims_count = pstats.get('total_claims', 0)
        
        if partner_claims_count >= 20:
            conf_note = f"High confidence: Partner {pid} has {int(partner_claims_count)} historical records in system."
        elif partner_claims_count > 0:
            conf_note = f"Medium confidence: Partner {pid} has {int(partner_claims_count)} historical records (EB-shrunk risk applied)."
        else:
            conf_note = f"Baseline confidence: Partner {pid} is unseen in training; cohort prior applied."
            
        # Counterfactual Analysis
        counterfactuals = self.decision_engine.generate_counterfactuals(
            raw_row=claim_dict,
            p_current=p,
            model_predict_fn=self._predict_single_proba_helper
        )
        
        return {
            "claim_id": str(clean_df['claim_id'].iloc[0]),
            "score": round(p, 4),
            "calibrated_probability": round(p, 4),
            "expected_saving_inr": round(expected_saving, 2),
            "recommendation": recommendation,
            "reasons": reasons,
            "confidence_note": conf_note,
            "data_warnings": warnings,
            "conformal_interval": [round(p_low, 4), round(p_high, 4)],
            "counterfactuals": counterfactuals
        }

    def score_test_batch(self, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scores the entire test dataset, producing predictions.csv and enriched triage records.
        """
        clean_df = clean_and_prepare_dataset(
            test_df,
            self.partners_df,
            self.products_df,
            is_train=False
        )
        
        X = self.feature_pipeline.transform(clean_df)
        probas = self.classifier.predict_proba(X)
        
        clean_df['score'] = probas
        clean_df['calibrated_probability'] = probas
        clean_df['expected_saving_inr'] = self.decision_engine.compute_expected_saving(
            clean_df['score'],
            clean_df['claim_amount_inr']
        )
        
        clean_df['month'] = clean_df['submitted_dt'].dt.to_period('M').astype(str)
        triage_df = self.decision_engine.rank_and_triage(clean_df, month_col='month', use_portfolio_optimization=True)
        
        reasons_list = []
        for idx in range(len(clean_df)):
            raw_r = clean_df.iloc[idx].to_dict()
            feat_r = X.iloc[idx]
            p_val = float(probas[idx])
            exp_s = float(clean_df['expected_saving_inr'].iloc[idx])
            r = self.decision_engine.generate_reasons(
                raw_row=raw_r,
                features_row=feat_r,
                p=p_val,
                expected_saving=exp_s,
                partner_stats=self.feature_pipeline.partner_stats_,
                serial_db=self.feature_pipeline.serial_db_
            )
            reasons_list.append(r)
            
        triage_df['reasons'] = reasons_list
        predictions_df = test_df[['claim_id']].merge(clean_df[['claim_id', 'score']], on='claim_id', how='left')
        
        return predictions_df, triage_df

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'KestrelClaimPipeline':
        return joblib.load(filepath)
