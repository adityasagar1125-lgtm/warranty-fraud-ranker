/**
 * Kestrel Claim Triage — Interactive Frontend Logic with Conformal Bounds & Counterfactuals
 */

const SCENARIOS = {
  rogue_outlet: {
    claim_id: "WC720101",
    partner_id: "SP3160",
    sku: "KH-MG-02",
    product_serial: "KH-468276030",
    claim_amount_inr: 1950,
    days_since_purchase: 85,
    photo_attached: "N",
    partner_inspected: "N",
    claim_description: "blade jammed",
    inspector_note: "",
    customer_prior_claims: 0
  },
  threshold_gaming: {
    claim_id: "WC720102",
    partner_id: "SP3232",
    sku: "KH-AF-01",
    product_serial: "KH-565562361",
    claim_amount_inr: 1980,
    days_since_purchase: 40,
    photo_attached: "N",
    partner_inspected: "N",
    claim_description: "filter indicator stuck",
    inspector_note: "",
    customer_prior_claims: 1
  },
  repeat_customer: {
    claim_id: "WC720103",
    partner_id: "SP3033",
    sku: "KH-RV-01",
    product_serial: "KH982341092",
    claim_amount_inr: 7500,
    days_since_purchase: 120,
    photo_attached: "Y",
    partner_inspected: "Y",
    claim_description: "motor not running",
    inspector_note: "Motor winding failure confirmed",
    customer_prior_claims: 3
  },
  clean_claim: {
    claim_id: "WC720104",
    partner_id: "SP3265",
    sku: "KH-WP-01",
    product_serial: "KH781290341",
    claim_amount_inr: 4500,
    days_since_purchase: 110,
    photo_attached: "Y",
    partner_inspected: "Y",
    claim_description: "water leaking",
    inspector_note: "Customer has bill, serial verified",
    customer_prior_claims: 0
  },
  serial_reuse: {
    claim_id: "WC720105",
    partner_id: "SP3318",
    sku: "KH-AF-02",
    product_serial: "KH-869773311",
    claim_amount_inr: 1920,
    days_since_purchase: 60,
    photo_attached: "N",
    partner_inspected: "N",
    claim_description: "power button not working",
    inspector_note: "",
    customer_prior_claims: 0
  }
};

// TAB SWITCHING
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  
  const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick')?.includes(tabId));
  if (targetBtn) targetBtn.classList.add('active');
  
  const targetContent = document.getElementById(`tab-${tabId}`);
  if (targetContent) targetContent.classList.add('active');
  
  if (tabId === 'top40') {
    loadTop40Table('2026-07');
  }
}

// INPUT MODE TOGGLING
function setInputMode(mode) {
  const formBtn = document.getElementById('btn-mode-form');
  const jsonBtn = document.getElementById('btn-mode-json');
  const formEl = document.getElementById('claim-form');
  const jsonContainer = document.getElementById('json-mode-container');
  
  if (mode === 'form') {
    formBtn.classList.add('active');
    jsonBtn.classList.remove('active');
    formEl.style.display = 'block';
    jsonContainer.style.display = 'none';
  } else {
    formBtn.classList.remove('active');
    jsonBtn.classList.add('active');
    formEl.style.display = 'none';
    jsonContainer.style.display = 'block';
    syncFormToJson();
  }
}

// SCENARIO LOADER
function loadScenario(key) {
  if (!key || !SCENARIOS[key]) return;
  const data = SCENARIOS[key];
  
  document.getElementById('f-claim-id').value = data.claim_id;
  document.getElementById('f-partner-id').value = data.partner_id;
  document.getElementById('f-sku').value = data.sku;
  document.getElementById('f-serial').value = data.product_serial;
  document.getElementById('f-amount').value = data.claim_amount_inr;
  document.getElementById('f-days').value = data.days_since_purchase;
  document.getElementById('f-prior').value = data.customer_prior_claims;
  document.getElementById('f-photo').value = data.photo_attached;
  document.getElementById('f-inspected').value = data.partner_inspected;
  document.getElementById('f-desc').value = data.claim_description;
  document.getElementById('f-note').value = data.inspector_note;
  
  document.getElementById('json-input').value = JSON.stringify(data, null, 2);
  submitScoreRequest(data);
}

function syncFormToJson() {
  const payload = getFormPayload();
  document.getElementById('json-input').value = JSON.stringify(payload, null, 2);
}

function getFormPayload() {
  return {
    claim_id: document.getElementById('f-claim-id').value.trim(),
    partner_id: document.getElementById('f-partner-id').value.trim(),
    sku: document.getElementById('f-sku').value,
    product_serial: document.getElementById('f-serial').value.trim(),
    claim_amount_inr: parseFloat(document.getElementById('f-amount').value) || 0,
    days_since_purchase: parseFloat(document.getElementById('f-days').value) || 0,
    photo_attached: document.getElementById('f-photo').value,
    partner_inspected: document.getElementById('f-inspected').value,
    claim_description: document.getElementById('f-desc').value.trim(),
    inspector_note: document.getElementById('f-note').value.trim(),
    customer_prior_claims: parseInt(document.getElementById('f-prior').value) || 0,
    source: "crm"
  };
}

function resetForm() {
  document.getElementById('claim-form').reset();
  loadScenario('rogue_outlet');
}

// SCORE SUBMISSIONS
async function handleScoreSubmit(e) {
  e.preventDefault();
  const payload = getFormPayload();
  await submitScoreRequest(payload);
}

async function scoreJsonPayload() {
  try {
    const raw = document.getElementById('json-input').value;
    const payload = JSON.parse(raw);
    await submitScoreRequest(payload);
  } catch (err) {
    alert("Invalid JSON format: " + err.message);
  }
}

async function submitScoreRequest(payload) {
  try {
    const resp = await fetch('/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!resp.ok) {
      const err = await resp.json();
      alert("Error scoring claim: " + (err.detail || resp.statusText));
      return;
    }
    
    const result = await resp.json();
    renderScoreVerdict(result, payload);
  } catch (err) {
    console.error("Scoring error:", err);
    alert("Network error contacting scoring engine.");
  }
}

// RENDER VERDICT
function renderScoreVerdict(res, payload) {
  const isCheck = res.recommendation === 'CHECK';
  
  // Badge & Action
  const badge = document.getElementById('rec-badge');
  const actionText = document.getElementById('res-action');
  
  if (isCheck) {
    badge.className = 'badge badge-check';
    badge.innerText = 'CHECK';
    actionText.className = 'stat-action-text check';
    actionText.innerText = 'DESK CHECK';
  } else {
    badge.className = 'badge badge-pay';
    badge.innerText = 'PAY';
    actionText.className = 'stat-action-text pay';
    actionText.innerText = 'AUTO PAY';
  }
  
  // Numbers & Conformal Bounds
  const pPercent = (res.calibrated_probability * 100).toFixed(1);
  const scoreEl = document.getElementById('res-score');
  scoreEl.innerText = `${pPercent}%`;
  scoreEl.className = `stat-number ${res.calibrated_probability >= 0.15 ? 'danger' : 'success'}`;
  
  const confBox = document.getElementById('res-conformal');
  if (res.conformal_interval) {
    const pLow = (res.conformal_interval[0] * 100).toFixed(1);
    const pHigh = (res.conformal_interval[1] * 100).toFixed(1);
    confBox.innerText = `95% CI: [${pLow}% – ${pHigh}%]`;
  } else {
    confBox.innerText = `Score: ${res.score.toFixed(4)}`;
  }
  
  const savingEl = document.getElementById('res-saving');
  const savingVal = res.expected_saving_inr;
  savingEl.innerText = (savingVal >= 0 ? '+' : '') + `₹${savingVal.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  savingEl.className = `stat-number ${savingVal > 0 ? 'success' : 'danger'}`;
  
  document.getElementById('res-conf').innerText = res.confidence_note || "Standard Confidence";
  
  // Math Breakdown
  const p = res.calibrated_probability;
  const amt = payload.claim_amount_inr || 0;
  const pGain = (p * amt).toFixed(0);
  const fpLoss = ((1 - p) * 380).toFixed(0);
  const net = (p * amt - (1 - p) * 380).toFixed(0);
  
  document.getElementById('math-values').innerHTML = 
    `= (${p.toFixed(3)} × ₹${amt.toLocaleString('en-IN')}) − (${(1 - p).toFixed(3)} × ₹380) = ₹${Number(pGain).toLocaleString('en-IN')} − ₹${Number(fpLoss).toLocaleString('en-IN')} = <strong>${net >= 0 ? '+' : ''}₹${Number(net).toLocaleString('en-IN')}</strong>`;
    
  // Counterfactual What-If Levers
  const cfList = document.getElementById('cf-list');
  cfList.innerHTML = '';
  if (res.counterfactuals && res.counterfactuals.length > 0) {
    res.counterfactuals.forEach(cf => {
      const item = document.createElement('div');
      item.className = 'cf-item';
      item.innerHTML = `
        <div>
          <span class="cf-condition">${cf.condition}</span>
          <div style="font-size: 11px; color: #94A3B8;">Simulated Risk: ${(cf.simulated_prob * 100).toFixed(1)}%</div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="cf-delta">-${cf.prob_delta_pct}% risk</span>
          <span class="cf-badge ${cf.verdict.toLowerCase()}">${cf.verdict}</span>
        </div>
      `;
      cfList.appendChild(item);
    });
    document.getElementById('cf-container').style.display = 'block';
  } else {
    document.getElementById('cf-container').style.display = 'none';
  }

  // Reasons List
  const reasonsList = document.getElementById('reasons-list');
  reasonsList.innerHTML = '';
  if (res.reasons && res.reasons.length > 0) {
    res.reasons.forEach(r => {
      const li = document.createElement('li');
      li.innerText = r;
      reasonsList.appendChild(li);
    });
  } else {
    reasonsList.innerHTML = '<li>Baseline profile: Normal claim parameters.</li>';
  }
  
  // Warnings List
  const warnBox = document.getElementById('warnings-box');
  const warnList = document.getElementById('warnings-list');
  warnList.innerHTML = '';
  if (res.data_warnings && res.data_warnings.length > 0) {
    warnBox.style.display = 'block';
    res.data_warnings.forEach(w => {
      const li = document.createElement('li');
      li.innerText = w;
      warnList.appendChild(li);
    });
  } else {
    warnBox.style.display = 'none';
  }
}

// LOAD TOP 40 TABLE
async function loadTop40Table(month) {
  try {
    const url = `/api/top40?month=${encodeURIComponent(month)}`;
    const resp = await fetch(url);
    if (!resp.ok) return;
    
    const data = await resp.json();
    
    document.getElementById('triage-pool-count').innerText = data.total_in_pool.toLocaleString();
    document.getElementById('triage-total-savings').innerText = `₹${data.total_expected_net_savings_inr.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    document.getElementById('triage-avg-score').innerText = data.average_top_k_score.toFixed(3);
    
    const tbody = document.getElementById('top40-tbody');
    tbody.innerHTML = '';
    
    data.claims.forEach((c, idx) => {
      const tr = document.createElement('tr');
      const isCheck = c.recommendation === 'CHECK';
      const mainReason = (c.reasons && c.reasons.length > 0) ? c.reasons[0] : "Multi-factor risk score";
      
      tr.innerHTML = `
        <td><strong>#${idx + 1}</strong></td>
        <td><code>${c.claim_id}</code></td>
        <td><strong>${c.partner_id}</strong></td>
        <td>${c.city || 'Tier-2'} (${c.partner_type ? c.partner_type.replace(/_/g, ' ') : 'partner'})</td>
        <td>${c.sku}</td>
        <td>₹${c.claim_amount_inr.toLocaleString('en-IN')}</td>
        <td><strong>${(c.score * 100).toFixed(1)}%</strong></td>
        <td style="color: ${c.expected_saving_inr > 0 ? '#10B981' : '#EF4444'}; font-weight: 700;">
          ${c.expected_saving_inr > 0 ? '+' : ''}₹${c.expected_saving_inr.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </td>
        <td><span class="table-badge ${isCheck ? 'check' : 'pay'}">${c.recommendation}</span></td>
        <td style="max-width: 320px; font-size: 12px; color: #CBD5E1;">${mainReason}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading top 40:", err);
  }
}

// INITIALIZATION
window.addEventListener('DOMContentLoaded', () => {
  loadScenario('rogue_outlet');
});
