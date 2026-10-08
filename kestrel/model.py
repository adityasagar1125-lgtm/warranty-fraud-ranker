"""
Kestrel Model Engine v2.0 — Calibrated Gradient Boosted Ensemble
Enhancements:
1. Monotonic constraints on known-directional features
2. Platt Scaling (Logistic Sigmoid Calibration)
3. Recency weighting + legacy Zoho noise down-weighting
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# Features where higher value should monotonically increase fraud probability
MONOTONIC_INCREASE = {
    'partner_shrunk_risk', 'partner_ewma_fraud_rate', 'partner_raw_fraud_rate',
    'uninspected_small_claim', 'uninspected_new_partner', 'is_threshold_gaming',
    'serial_cross_partner_count', 'serial_is_reused', 'serial_multi_partner',
    'rules_score', 'customer_prior_claims', 'prior_claims_squared',
    'high_prior_claims', 'partner_risk_x_small_claim', 'partner_risk_x_uninspected',
    'ewma_risk_x_small_claim', 'ewma_risk_x_uninspected',
    'new_partner_small_post_may', 'partner_small_uninspected_share',
    'partner_thresh_gaming_share',
}

# Features where higher value should monotonically decrease fraud probability
MONOTONIC_DECREASE = {
    'photo_attached', 'partner_inspected', 'has_inspector_note',
}


class KestrelClassifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.base_hgb = None
        self.base_xgb = None
        self.calibrator = None
        self.feature_names_ = []
        self.feature_importances_ = {}

    def _build_monotonic_constraints(self, feature_names: list[str]) -> list[int]:
        """Build monotonic constraint vector from feature names."""
        constraints = []
        for name in feature_names:
            if name in MONOTONIC_INCREASE:
                constraints.append(1)
            elif name in MONOTONIC_DECREASE:
                constraints.append(-1)
            else:
                constraints.append(0)
        return constraints

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamps: pd.Series,
        sources: pd.Series,
        recency_weight_max: float = 2.5,
        legacy_zero_weight: float = 0.75
    ):
        """
        Fits the gradient boosting model with:
        1. Recency weighting (exponential/linear time growth)
        2. Down-weighting of legacy Zoho zeros (Tanmay's data trap #2)
        3. Monotonic constraints on known-directional features
        4. Smooth Platt sigmoid calibration
        """
        self.feature_names_ = list(X.columns)
        monotonic_cst = self._build_monotonic_constraints(self.feature_names_)

        # Calculate sample weights
        weights = np.ones(len(X), dtype=float)

        # Down-weight legacy Zoho 0s
        if sources is not None:
            is_legacy_zero = (sources == 'legacy_zoho') & (y == 0)
            weights[is_legacy_zero] = legacy_zero_weight

        # Recency weighting
        if timestamps is not None:
            t_min = timestamps.min()
            t_max = timestamps.max()
            span_days = max(1.0, (t_max - t_min).total_seconds() / 86400.0)
            elapsed_days = (timestamps - t_min).dt.total_seconds() / 86400.0
            recency_factors = 1.0 + (recency_weight_max - 1.0) * (elapsed_days / span_days)
            weights *= recency_factors

        fraud_multiplier = 2.0
        weights[y == 1] = weights[y == 1] * fraud_multiplier

        # Base HistGB with monotonic constraints
        self.base_hgb = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.045,
            max_leaf_nodes=31,
            min_samples_leaf=12,
            l2_regularization=1.5,
            monotonic_cst=monotonic_cst,
            random_state=self.random_state
        )
        self.base_hgb.fit(X, y, sample_weight=weights)

        # Base XGBoost (monotonic constraints as tuple string for xgb)
        xgb_mono = tuple(monotonic_cst)
        pos_weight = (len(y) - y.sum()) / max(1.0, y.sum()) * 0.35
        self.base_xgb = xgb.XGBClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=pos_weight,
            monotone_constraints=xgb_mono,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.base_xgb.fit(X, y, sample_weight=weights)

        # Platt Sigmoid Calibration on ensemble log-odds
        p_hgb = self.base_hgb.predict_proba(X)[:, 1]
        p_xgb = self.base_xgb.predict_proba(X)[:, 1]
        p_raw = 0.55 * p_hgb + 0.45 * p_xgb
        p_raw = np.clip(p_raw, 1e-6, 1.0 - 1e-6)

        log_odds = np.log(p_raw / (1.0 - p_raw))
        self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        self.calibrator.fit(log_odds.reshape(-1, 1), y)

        # Feature importances (blend both models)
        hgb_imp = np.zeros(len(self.feature_names_))
        # HistGB doesn't expose feature_importances_ directly in all versions;
        # use permutation-based from XGB as primary
        xgb_importances = self.base_xgb.feature_importances_
        for name, imp in zip(self.feature_names_, xgb_importances):
            self.feature_importances_[name] = float(imp)

        return self

    def predict_raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_hgb = self.base_hgb.predict_proba(X)[:, 1]
        p_xgb = self.base_xgb.predict_proba(X)[:, 1]
        return 0.55 * p_hgb + 0.45 * p_xgb

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns well-calibrated posterior probability p = P(fraud | features).
        """
        p_raw = self.predict_raw_proba(X)
        p_raw = np.clip(p_raw, 1e-6, 1.0 - 1e-6)
        log_odds = np.log(p_raw / (1.0 - p_raw))
        p_cal = self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        return np.clip(p_cal, 0.001, 0.999)
