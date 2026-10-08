"""
Kestrel Feature Pipeline v2.0 — Enhanced Forensic Intelligence
Implements:
1. Layer 1: Tabular, Text (TF-IDF + SVD), Non-linear Ratios, Temporal Patterns
2. Layer 2: Time-Decayed Empirical-Bayes Outlet Shrinkage, Behavioral Fingerprints
   (Simpson Concentration, Inter-Arrival Burstiness, Off-Hours Ratio),
   Fuzzy Serial Graph (Levenshtein Collisions), Benford's Law Digit Divergence,
   Family-Relative Amount Anomaly, Hybrid Rules Baseline Signal
"""
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def calculate_benford_divergence(amounts: list[float]) -> float:
    """
    Computes Kullback-Leibler divergence between first-digit distribution
    of claim amounts and theoretical Benford's Law.
    Benford P(d) = log10(1 + 1/d) for d in 1..9.
    """
    if not amounts or len(amounts) < 5:
        return 0.0

    first_digits = []
    for a in amounts:
        if a > 0:
            s = str(int(a)).lstrip('0')
            if s:
                first_digits.append(int(s[0]))

    if len(first_digits) < 5:
        return 0.0

    counts = pd.Series(first_digits).value_counts()
    n = len(first_digits)

    kl_div = 0.0
    for d in range(1, 10):
        p_actual = (counts.get(d, 0) + 0.5) / (n + 4.5)  # Laplace smoothed
        p_benford = math.log10(1.0 + 1.0 / d)
        kl_div += p_actual * math.log(p_actual / p_benford)

    return max(0.0, float(kl_div))


def get_serial_prefix_hash(serial: str) -> str:
    """Prefix hash for blocking fuzzy serial collisions."""
    s = str(serial).strip().upper()
    return s[:6] if len(s) >= 6 else s


def is_levenshtein_1(s1: str, s2: str) -> bool:
    """Fast check if two strings have Levenshtein edit distance == 1."""
    if s1 == s2:
        return False
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 1:
        return False
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    i = 0
    j = 0
    diff = 0
    while i < len1 and j < len2:
        if s1[i] != s2[j]:
            diff += 1
            if diff > 1:
                return False
            if len1 == len2:
                i += 1
                j += 1
            else:
                j += 1
        else:
            i += 1
            j += 1
    return True


def compute_simpson_concentration(values: list) -> float:
    """
    Simpson Concentration Index: sum of (n_i / N)^2.
    Higher value = more concentrated on fewer categories (suspicious for an outlet).
    A legit service centre repairs diverse products; a fraudster repeats the same SKU.
    """
    if not values or len(values) < 2:
        return 1.0
    counts = pd.Series(values).value_counts()
    n = len(values)
    return float((counts / n).pow(2).sum())


def compute_burstiness(timestamps: list) -> float:
    """
    Inter-arrival time burstiness index.
    B = (σ_iat - μ_iat) / (σ_iat + μ_iat)
    B → +1 means bursty (suspicious), B → 0 means periodic (normal), B → -1 means regular.
    """
    if len(timestamps) < 3:
        return 0.0
    ts = sorted(pd.to_datetime(timestamps))
    iats = [(ts[i + 1] - ts[i]).total_seconds() / 3600.0 for i in range(len(ts) - 1)]
    if not iats:
        return 0.0
    mu = np.mean(iats)
    sigma = np.std(iats)
    denom = sigma + mu
    if denom == 0:
        return 0.0
    return float(np.clip((sigma - mu) / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Main Feature Pipeline
# ---------------------------------------------------------------------------

class KestrelFeaturePipeline:
    def __init__(self, shrinkage_m: float = 12.0, ewma_decay_per_day: float = 0.03):
        self.shrinkage_m = shrinkage_m
        self.ewma_decay_per_day = ewma_decay_per_day
        self.global_prior_ = 0.0123
        self.cohort_prior_ = {}
        self.partner_stats_ = {}
        self.serial_db_ = {}
        self.serial_prefix_map_ = {}
        self.family_stats_ = {}          # NEW v2: per-family amount distribution
        self.tfidf_desc_ = None
        self.svd_desc_ = None
        self.is_fitted = False

        self.desc_keywords = [
            'blade jammed', 'remote not working', 'water leaking', 'display not working',
            'unit not heating', 'filter indicator stuck', 'loud noise', 'burning smell',
            'tripping mcb', 'not charging', 'motor not running', 'power button'
        ]

        self.note_keywords = [
            'customer has bill', 'pcb replaced', 'photos match', 'minor fault',
            'motor winding', 'unit inspected', 'heating element'
        ]

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the feature extractors on historical training claims.
        v2.0: Adds EWMA decay, Simpson concentration, burstiness, weekend ratio,
              and per-family amount distributions.
        """
        df = train_df.copy()
        labeled_df = df.dropna(subset=['is_fraud']).copy()

        # 1. Global prior
        self.global_prior_ = float(labeled_df['is_fraud'].mean())
        if np.isnan(self.global_prior_) or self.global_prior_ == 0:
            self.global_prior_ = 0.0123

        # 2. Partner Type Cohort Prior
        type_groups = labeled_df.groupby('partner_type')['is_fraud']
        for ptype, vals in type_groups:
            self.cohort_prior_[ptype] = float(vals.mean())

        # 3. Per-Family Claim Amount Distribution (for z-score anomaly)
        family_groups = df.groupby('family')['claim_amount_inr']
        for fam, amounts in family_groups:
            fam_mean = float(amounts.mean())
            fam_std = float(amounts.std())
            if np.isnan(fam_std) or fam_std < 1.0:
                fam_std = max(1.0, fam_mean * 0.3)
            self.family_stats_[fam] = {'mean': fam_mean, 'std': fam_std}

        # 4. Aggregate partner-level stats
        partner_labeled = labeled_df.groupby('partner_id').agg(
            total_claims=('is_fraud', 'count'),
            fraud_claims=('is_fraud', 'sum'),
            partner_type=('partner_type', 'first'),
            city=('city', 'first')
        ).reset_index()

        partner_all = df.groupby('partner_id').agg(
            all_claims=('claim_id', 'count'),
            small_uninspected=('uninspected_small_claim', 'sum'),
            threshold_gaming=('is_threshold_gaming', 'sum'),
            recent_post_may=('post_may', 'sum')
        ).reset_index()

        # Partner-specific claim amount history for Benford calculation
        partner_amounts = df.groupby('partner_id')['claim_amount_inr'].apply(list).to_dict()

        # 5. Per-partner behavioral features (NEW v2)
        # SKU concentration (Simpson), burstiness, weekend ratio, EWMA fraud rate
        partner_skus = df.groupby('partner_id')['sku'].apply(list).to_dict()
        partner_timestamps = df.groupby('partner_id')['submitted_dt'].apply(list).to_dict()

        # Weekend / off-hours ratio per partner
        partner_weekend_ratio = {}
        for pid, ts_list in partner_timestamps.items():
            ts_parsed = pd.to_datetime(ts_list)
            weekend_count = (ts_parsed.dayofweek >= 5).sum()
            partner_weekend_ratio[pid] = float(weekend_count / max(1, len(ts_parsed)))

        # EWMA temporal-decay fraud rate per partner
        partner_ewma = {}
        reference_date = df['submitted_dt'].max()
        for pid in labeled_df['partner_id'].unique():
            p_claims = labeled_df[labeled_df['partner_id'] == pid].sort_values('submitted_dt')
            weights = []
            fraud_labels = []
            for _, row in p_claims.iterrows():
                days_ago = max(0, (reference_date - row['submitted_dt']).days)
                w = math.exp(-self.ewma_decay_per_day * days_ago)
                weights.append(w)
                fraud_labels.append(float(row['is_fraud']))
            if weights and sum(weights) > 0:
                partner_ewma[pid] = sum(w * f for w, f in zip(weights, fraud_labels)) / sum(weights)
            else:
                partner_ewma[pid] = self.global_prior_

        p_merged = partner_labeled.merge(partner_all, on='partner_id', how='outer').fillna(0)

        for _, row in p_merged.iterrows():
            pid = row['partner_id']
            ptype = row.get('partner_type', 'authorised_service_centre')
            if isinstance(ptype, float) or ptype == 0:
                ptype = 'authorised_service_centre'
            c_prior = self.cohort_prior_.get(ptype, self.global_prior_)

            n = float(row.get('total_claims', 0))
            k = float(row.get('fraud_claims', 0))
            all_c = float(row.get('all_claims', 1))
            small_uninsp = float(row.get('small_uninspected', 0))
            thresh_game = float(row.get('threshold_gaming', 0))

            # Static Empirical Bayes Shrunk Risk
            shrunk_risk = (k + self.shrinkage_m * c_prior) / (n + self.shrinkage_m)

            # EWMA Temporal-Decay Fraud Rate (NEW v2)
            ewma_rate = partner_ewma.get(pid, c_prior)

            # Benford divergence on outlet claim amounts
            amt_hist = partner_amounts.get(pid, [])
            benford_div = calculate_benford_divergence(amt_hist)

            # Post-May burst ratio
            recent_c = float(row.get('recent_post_may', 0))
            burst_velocity = (recent_c / max(1.0, all_c - recent_c)) if (all_c - recent_c) > 0 else 1.0

            # Simpson SKU Concentration (NEW v2)
            skus = partner_skus.get(pid, [])
            simpson_idx = compute_simpson_concentration(skus)

            # Inter-arrival Burstiness (NEW v2)
            ts_list = partner_timestamps.get(pid, [])
            burst_idx = compute_burstiness(ts_list)

            # Weekend / off-hours ratio (NEW v2)
            wkend_ratio = partner_weekend_ratio.get(pid, 0.0)

            self.partner_stats_[pid] = {
                'shrunk_risk': shrunk_risk,
                'ewma_fraud_rate': ewma_rate,               # NEW v2
                'raw_fraud_rate': k / max(1.0, n),
                'total_claims': n,
                'fraud_claims': k,
                'small_uninspected_share': small_uninsp / max(1.0, all_c),
                'threshold_gaming_share': thresh_game / max(1.0, all_c),
                'benford_divergence': benford_div,
                'burst_velocity': burst_velocity,
                'simpson_concentration': simpson_idx,         # NEW v2
                'inter_arrival_burstiness': burst_idx,        # NEW v2
                'weekend_ratio': wkend_ratio,                 # NEW v2
                'partner_type': ptype
            }

        # 6. Build Serial Database & Fuzzy Prefix Hash Map
        # NOTE v2: Still store fraud_count for explanation reasons, but
        # DO NOT use it as a ML feature (prevents target leakage)
        for _, row in df.iterrows():
            serial = row.get('norm_serial', '')
            if serial:
                if serial not in self.serial_db_:
                    self.serial_db_[serial] = {
                        'claim_count': 0,
                        'partners': set(),
                        'skus': set(),
                        'fraud_count': 0  # Kept for reason generation only
                    }
                self.serial_db_[serial]['claim_count'] += 1
                self.serial_db_[serial]['partners'].add(row.get('partner_id', ''))
                self.serial_db_[serial]['skus'].add(row.get('sku', ''))
                if not pd.isna(row.get('is_fraud')) and row.get('is_fraud') == 1:
                    self.serial_db_[serial]['fraud_count'] += 1

                prefix = get_serial_prefix_hash(serial)
                if prefix not in self.serial_prefix_map_:
                    self.serial_prefix_map_[prefix] = set()
                self.serial_prefix_map_[prefix].add(serial)

        # 7. Fit Text TF-IDF + SVD on claim descriptions
        clean_descriptions = df['claim_description'].fillna('').astype(str).tolist()
        self.tfidf_desc_ = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        tfidf_mat = self.tfidf_desc_.fit_transform(clean_descriptions)
        self.svd_desc_ = TruncatedSVD(n_components=6, random_state=42)
        self.svd_desc_.fit(tfidf_mat)

        self.is_fitted = True
        return self

    def compute_fuzzy_serial_collisions(self, serial: str) -> int:
        """Counts how many historical serials exist with edit distance == 1."""
        if not serial:
            return 0
        prefix = get_serial_prefix_hash(serial)
        candidates = self.serial_prefix_map_.get(prefix, set())
        collisions = 0
        for cand in candidates:
            if is_levenshtein_1(serial, cand):
                collisions += 1
        return collisions

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms dataset into the enhanced multi-layer feature matrix.
        v2.0: Removes serial_fraud_hist (leakage fix), adds temporal features,
              behavioral fingerprints, family anomaly z-scores, and rules signal.
        """
        feats = pd.DataFrame(index=df.index)

        # --- Layer 1: Tabular Core ---
        feats['claim_amount_inr'] = df['claim_amount_inr'].astype(float)
        feats['log_claim_amount'] = np.log1p(df['claim_amount_inr'].clip(lower=0))
        feats['list_price_inr'] = df['list_price_inr'].fillna(5000).astype(float)
        feats['amount_to_list_ratio'] = (df['claim_amount_inr'] / feats['list_price_inr']).clip(0, 3.0)
        feats['days_since_purchase'] = df['days_since_purchase'].astype(float)
        feats['warranty_months'] = df['warranty_months'].fillna(12).astype(float)
        feats['purchase_to_warranty_ratio'] = df['purchase_to_warranty_ratio'].fillna(0.5).astype(float)
        feats['customer_prior_claims'] = df['customer_prior_claims'].astype(float)
        feats['photo_attached'] = df['photo_attached_bin'].astype(float)
        feats['partner_inspected'] = df['partner_inspected_bin'].astype(float)
        feats['is_resubmission'] = df['is_resubmission'].astype(float)

        # --- Policy Signals & Interaction Exploits ---
        feats['is_small_claim'] = (df['claim_amount_inr'] < 2000).astype(float)
        feats['is_threshold_gaming'] = ((df['claim_amount_inr'] >= 1900) & (df['claim_amount_inr'] < 2000)).astype(float)
        feats['uninspected_small_claim'] = ((feats['is_small_claim'] == 1) & (feats['partner_inspected'] == 0)).astype(float)
        feats['post_may'] = df['post_may'].astype(float)
        feats['partner_age_days'] = df['partner_age_days'].fillna(365).astype(float)
        feats['is_new_partner'] = (feats['partner_age_days'] <= 365).astype(float)
        feats['new_partner_small_post_may'] = (feats['is_new_partner'] * feats['is_small_claim'] * feats['post_may']).astype(float)
        feats['uninspected_new_partner'] = (feats['is_new_partner'] * (1.0 - feats['partner_inspected'])).astype(float)

        feats['high_prior_claims'] = (feats['customer_prior_claims'] >= 2).astype(float)
        feats['prior_claims_squared'] = feats['customer_prior_claims'] ** 2

        # --- NEW v2: Temporal Pattern Features ---
        if 'submitted_dt' in df.columns:
            dt_col = pd.to_datetime(df['submitted_dt'])
            feats['day_of_week'] = dt_col.dt.dayofweek.astype(float)
            feats['is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(float)
            feats['hour_of_day'] = dt_col.dt.hour.astype(float)
            feats['is_off_hours'] = ((dt_col.dt.hour < 8) | (dt_col.dt.hour >= 20)).astype(float)
            feats['day_of_month'] = dt_col.dt.day.astype(float)
            feats['days_to_month_end'] = (dt_col.dt.days_in_month - dt_col.dt.day).astype(float)

        # --- NEW v2: Family-Relative Amount Anomaly (z-score) ---
        family_z_scores = []
        family_pct_ranks = []
        for _, row in df.iterrows():
            fam = row.get('family', 'Other')
            amount = float(row.get('claim_amount_inr', 0))
            if fam in self.family_stats_:
                f_mean = self.family_stats_[fam]['mean']
                f_std = self.family_stats_[fam]['std']
                z = (amount - f_mean) / max(1.0, f_std)
                family_z_scores.append(float(np.clip(z, -5.0, 5.0)))
            else:
                family_z_scores.append(0.0)
            family_pct_ranks.append(0.0)  # Placeholder, can add ECDF later
        feats['family_amount_zscore'] = family_z_scores

        # --- Layer 2: Entity Risk, EB Shrinkage, Behavioral Fingerprints ---
        shrunk_risks = []
        ewma_fraud_rates = []
        raw_fraud_rates = []
        partner_total_claims = []
        partner_small_uninspected_shares = []
        partner_thresh_gaming_shares = []
        benford_scores = []
        burst_velocities = []
        simpson_concentrations = []
        burstiness_indices = []
        weekend_ratios = []

        for _, row in df.iterrows():
            pid = row.get('partner_id', '')
            ptype = row.get('partner_type', 'authorised_service_centre')
            c_prior = self.cohort_prior_.get(ptype, self.global_prior_)

            if pid in self.partner_stats_:
                pstats = self.partner_stats_[pid]
                shrunk_risks.append(pstats['shrunk_risk'])
                ewma_fraud_rates.append(pstats['ewma_fraud_rate'])
                raw_fraud_rates.append(pstats['raw_fraud_rate'])
                partner_total_claims.append(pstats['total_claims'])
                partner_small_uninspected_shares.append(pstats['small_uninspected_share'])
                partner_thresh_gaming_shares.append(pstats['threshold_gaming_share'])
                benford_scores.append(pstats['benford_divergence'])
                burst_velocities.append(pstats['burst_velocity'])
                simpson_concentrations.append(pstats['simpson_concentration'])
                burstiness_indices.append(pstats['inter_arrival_burstiness'])
                weekend_ratios.append(pstats['weekend_ratio'])
            else:
                shrunk_risks.append(c_prior)
                ewma_fraud_rates.append(c_prior)
                raw_fraud_rates.append(c_prior)
                partner_total_claims.append(0.0)
                partner_small_uninspected_shares.append(0.0)
                partner_thresh_gaming_shares.append(0.0)
                benford_scores.append(0.0)
                burst_velocities.append(1.0)
                simpson_concentrations.append(1.0)
                burstiness_indices.append(0.0)
                weekend_ratios.append(0.0)

        feats['partner_shrunk_risk'] = shrunk_risks
        feats['partner_ewma_fraud_rate'] = ewma_fraud_rates           # NEW v2
        feats['partner_raw_fraud_rate'] = raw_fraud_rates
        feats['partner_total_claims'] = partner_total_claims
        feats['partner_small_uninspected_share'] = partner_small_uninspected_shares
        feats['partner_thresh_gaming_share'] = partner_thresh_gaming_shares
        feats['outlet_benford_divergence'] = benford_scores
        feats['outlet_burst_velocity'] = burst_velocities
        feats['outlet_simpson_concentration'] = simpson_concentrations  # NEW v2
        feats['outlet_burstiness_index'] = burstiness_indices          # NEW v2
        feats['outlet_weekend_ratio'] = weekend_ratios                 # NEW v2

        # Interaction features
        feats['partner_risk_x_small_claim'] = feats['partner_shrunk_risk'] * feats['is_small_claim']
        feats['partner_risk_x_uninspected'] = feats['partner_shrunk_risk'] * (1.0 - feats['partner_inspected'])
        feats['ewma_risk_x_small_claim'] = feats['partner_ewma_fraud_rate'] * feats['is_small_claim']  # NEW v2
        feats['ewma_risk_x_uninspected'] = feats['partner_ewma_fraud_rate'] * (1.0 - feats['partner_inspected'])  # NEW v2

        # --- Layer 2: Serial Graph (LEAKAGE-FREE in v2) ---
        # NOTE v2: serial_fraud_hist is REMOVED from ML features.
        # It was 78% of feature importance but is pure target leakage.
        serial_claim_counts = []
        serial_cross_partners = []
        serial_cross_skus = []
        fuzzy_collisions = []

        for _, row in df.iterrows():
            serial = row.get('norm_serial', '')
            if serial in self.serial_db_:
                s_entry = self.serial_db_[serial]
                serial_claim_counts.append(float(s_entry['claim_count']))
                serial_cross_partners.append(float(len(s_entry['partners'])))
                serial_cross_skus.append(float(len(s_entry['skus'])))
            else:
                serial_claim_counts.append(1.0)
                serial_cross_partners.append(1.0)
                serial_cross_skus.append(1.0)

            # Fuzzy Levenshtein collisions
            f_coll = self.compute_fuzzy_serial_collisions(serial)
            fuzzy_collisions.append(float(f_coll))

        feats['serial_claim_count'] = serial_claim_counts
        feats['serial_cross_partner_count'] = serial_cross_partners
        feats['serial_cross_sku_count'] = serial_cross_skus
        # v2: REMOVED serial_fraud_hist — this was target leakage (78% importance)
        feats['serial_is_reused'] = (feats['serial_claim_count'] > 1).astype(float)
        feats['serial_multi_partner'] = (feats['serial_cross_partner_count'] > 1).astype(float)
        feats['serial_fuzzy_collision_count'] = fuzzy_collisions
        feats['has_fuzzy_serial_mutation'] = (feats['serial_fuzzy_collision_count'] > 0).astype(float)

        # --- NEW v2: Hybrid Rules Baseline Signal ---
        # The rules-only baseline (ROC-AUC 0.83) outperforms ML alone.
        # Feed it as a feature so the ML model can blend domain knowledge.
        feats['rules_score'] = (
            (feats['is_new_partner'] == 1) & (feats['is_small_claim'] == 1)
        ).astype(float) * 0.8

        # --- Categorical Encodings ---
        ptype_series = df['partner_type'].fillna('authorised_service_centre')
        feats['partner_type_asc'] = (ptype_series == 'authorised_service_centre').astype(float)
        feats['partner_type_franchise'] = (ptype_series == 'franchise').astype(float)
        feats['partner_type_freelance'] = (ptype_series == 'freelance_technician').astype(float)

        pfamily_series = df['family'].fillna('Other')
        for fam in ['Air Fryer', 'Air Purifier', 'Mixer Grinder', 'Robot Vacuum', 'Water Purifier']:
            feats[f'family_{fam.lower().replace(" ", "_")}'] = (pfamily_series == fam).astype(float)

        # --- Text Features ---
        desc_text = df['claim_description'].fillna('').astype(str).str.lower()
        note_text = df['inspector_note'].fillna('').astype(str).str.lower()

        for kw in self.desc_keywords:
            clean_col = 'kw_desc_' + re.sub(r'[^a-z0-9]', '_', kw)
            feats[clean_col] = desc_text.apply(lambda t: 1.0 if kw in t else 0.0)

        for kw in self.note_keywords:
            clean_col = 'kw_note_' + re.sub(r'[^a-z0-9]', '_', kw)
            feats[clean_col] = note_text.apply(lambda t: 1.0 if kw in t else 0.0)

        feats['has_inspector_note'] = (note_text.str.len() > 3).astype(float)
        feats['desc_char_length'] = desc_text.str.len().astype(float)

        if self.tfidf_desc_ is not None and self.svd_desc_ is not None:
            tfidf_desc_mat = self.tfidf_desc_.transform(desc_text.tolist())
            svd_desc_feats = self.svd_desc_.transform(tfidf_desc_mat)
            for i in range(svd_desc_feats.shape[1]):
                feats[f'desc_svd_{i}'] = svd_desc_feats[:, i]

        return feats
