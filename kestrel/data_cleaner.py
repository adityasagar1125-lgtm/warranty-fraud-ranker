"""
Kestrel Data Cleaner
Handles deduplication, serial normalisation, timestamp parsing, and data validation.
Preserves original input ordering for testing and submissions.
"""
import re
import pandas as pd
import numpy as np


def clean_serial(serial_str: str) -> str:
    """
    Normalise product serial number:
    - Strip whitespace
    - Uppercase
    - Remove hyphens, spaces, and special characters
    """
    if pd.isna(serial_str):
        return ""
    s = str(serial_str).strip().upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def validate_claim_data(row: dict) -> list[str]:
    """
    Inspect input claim row and produce polite data quality warnings.
    """
    warnings = []
    
    # Serial validation
    raw_serial = str(row.get('product_serial', ''))
    clean = clean_serial(raw_serial)
    if not clean:
        warnings.append("Missing product serial number.")
    elif len(clean) not in [10, 11, 12]:
        warnings.append(f"Irregular serial length ({len(clean)} chars): '{raw_serial}' (expected 11-12 alphanumeric).")
    elif any(c.islower() for c in raw_serial) or '-' in raw_serial or ' ' in raw_serial:
        warnings.append(f"Malformed serial format normalised from '{raw_serial}' to '{clean}'.")
        
    # Amount validation
    amount = row.get('claim_amount_inr', 0)
    try:
        amount_num = float(amount)
        if amount_num <= 0:
            warnings.append(f"Non-positive claim amount: Rs {amount_num}.")
        elif 1900 <= amount_num < 2000:
            warnings.append(f"Threshold gaming alert: Rs {amount_num:,.0f} sits immediately below the Rs 2,000 auto-approval cap.")
    except (ValueError, TypeError):
        warnings.append(f"Invalid claim amount value: '{amount}'.")
        
    # Inspection validation
    inspected = str(row.get('partner_inspected', '')).upper()
    if inspected not in ['Y', 'N', 'YES', 'NO']:
        warnings.append(f"Unrecognised partner inspection flag: '{inspected}'.")
        
    # Customer prior claims
    try:
        prior = int(row.get('customer_prior_claims', 0))
        if prior >= 3:
            warnings.append(f"High-frequency claimant: customer has {prior} previous warranty claims (21% risk cohort).")
    except (ValueError, TypeError):
        warnings.append(f"Invalid customer prior claims: '{row.get('customer_prior_claims')}'.")
        
    return warnings


def clean_and_prepare_dataset(
    df: pd.DataFrame,
    partners_df: pd.DataFrame,
    products_df: pd.DataFrame,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Cleans raw dataset, joins partner and product metadata,
    deduplicates (keeping latest), and constructs baseline flags.
    Preserves original row order when is_train is False.
    """
    data = df.copy()
    data['orig_order'] = np.arange(len(data))
    
    # 1. Parse timestamps (IST)
    data['submitted_dt'] = pd.to_datetime(data['submitted_at'])
    
    # 2. Identify resubmissions and deduplicate
    dup_ids = set(data[data.duplicated(subset=['claim_id'], keep=False)]['claim_id'])
    data['is_resubmission'] = data['claim_id'].apply(lambda cid: 1 if cid in dup_ids else 0)
    
    if is_train:
        # Sort by submitted_dt and keep latest submission for each claim_id in train
        data = data.sort_values('submitted_dt')
        data = data.drop_duplicates(subset=['claim_id'], keep='last').copy()
    else:
        # Preserve original row ordering for test
        data = data.sort_values('orig_order')
        
    # 3. Normalise serial numbers
    data['norm_serial'] = data['product_serial'].apply(clean_serial)
    
    # 4. Binary conversions
    data['photo_attached_bin'] = data['photo_attached'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    data['partner_inspected_bin'] = data['partner_inspected'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)
    data['days_since_purchase'] = pd.to_numeric(data['days_since_purchase'], errors='coerce').fillna(0)
    data['claim_amount_inr'] = pd.to_numeric(data['claim_amount_inr'], errors='coerce').fillna(0)
    data['customer_prior_claims'] = pd.to_numeric(data['customer_prior_claims'], errors='coerce').fillna(0)
    
    # 5. Merge partner metadata
    p_meta = partners_df.copy()
    p_meta['onboarded_dt'] = pd.to_datetime(p_meta['onboarded_date'])
    data = data.merge(p_meta[['partner_id', 'city', 'onboarded_dt', 'partner_type']], on='partner_id', how='left')
    
    # 6. Merge product metadata
    data = data.merge(products_df[['sku', 'family', 'list_price_inr', 'warranty_months']], on='sku', how='left')
    
    # 7. Derived domain features
    data['partner_age_days'] = (data['submitted_dt'] - data['onboarded_dt']).dt.days.clip(lower=0)
    data['is_new_partner'] = (data['partner_age_days'] <= 365).astype(int)
    data['warranty_days'] = data['warranty_months'] * 30.4375
    data['purchase_to_warranty_ratio'] = (data['days_since_purchase'] / data['warranty_days'].replace(0, 365)).clip(0, 3.0)
    data['amount_to_list_ratio'] = (data['claim_amount_inr'] / data['list_price_inr'].replace(0, 1000)).clip(0, 2.0)
    
    # 8. May 2026 policy change and threshold features
    data['post_may'] = (data['submitted_dt'] >= '2026-05-01').astype(int)
    data['is_small_claim'] = (data['claim_amount_inr'] < 2000).astype(int)
    data['is_threshold_gaming'] = ((data['claim_amount_inr'] >= 1900) & (data['claim_amount_inr'] < 2000)).astype(int)
    data['uninspected_small_claim'] = ((data['is_small_claim'] == 1) & (data['partner_inspected_bin'] == 0)).astype(int)
    
    if not is_train:
        data = data.sort_values('orig_order').reset_index(drop=True)
        
    return data
