train.csv             Past warranty claims, with investigation outcome.
test_unlabelled.csv   The most recent claims, same columns without is_fraud.
  claim_id              Claim number
  submitted_at          IST
  partner_id            Key into partners.csv
  sku                   Key into products.csv
  product_serial        Serial as typed by the partner
  days_since_purchase   Days between purchase and claim
  claim_amount_inr      Amount claimed
  photo_attached        Y/N
  partner_inspected     Y/N inspection sign-off
  claim_description     Fault description (free text, partner)
  inspector_note        Inspector's note (free text), blank if not inspected
  customer_prior_claims Customer's earlier warranty claims
  source                crm | legacy_zoho (before 1 Oct 2025)
  is_fraud              1 fraud, 0 not fraud, blank = undecided at export (train only)

partners.csv          partner_id, city, onboarded_date, partner_type
products.csv          sku, family, list_price_inr, warranty_months
sample_submission.csv claim_id, score
ops-policy.pdf        Kestrel operations policy v4.1 â€” costs, claim approval, review capacity, systems.
email-thread.txt      Messages already exchanged about this work.