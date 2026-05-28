# 📋 Things To Note

AI review findings and prioritized UX suggestions. Use this as a strategic roadmap for follow-up work.

---

## 🚨 **Tier 1: Immediate Fixes (Deploy Blockers)**
*Target: 1–2 weeks. Must complete before production release.*

### ⚡ Rainfall Unit Clarification — **CRITICAL**

| Aspect | Details |
|--------|---------|
| **Problem** | `PRECTOTCORR` values (e.g., `1574`, `2048`) appear to be monthly totals, but NASA POWER API returns daily averages (`mm/day`). Unit mismatch causes 30× input scaling errors. |
| **Model Risk** | If trained on daily averages (0–20 mm/day) but UI accepts monthly totals (0–600 mm/month), model sees out-of-distribution input → nonsense predictions. |
| **Solution** | Verify preprocessing pipeline alignment; clearly label all UI and docs. |

**Action Items:**
- [ ] Inspect `data/processed_dataset.csv` PRECTOTCORR values and aggregation method
- [ ] Confirm whether values are **monthly accumulations** or **daily averages**
- [ ] Align labels across: codebase → documentation → UI display
- [ ] **Explicitly state in UI:** *"Monthly precipitation total (mm/month)"* or *"Average daily precipitation (mm/day)"*

---

### 🧮 Validate SHAP/Permutation Attribution — **CRITICAL**

| Concern | Evidence |
|---------|----------|
| **Suspiciously Large Impacts** | Precipitation impact: `-1,577 kg/ha` when final yield ≈ `9,310 kg/ha` = **17% loss from one feature alone** |
| **Root Cause** | May be denormalization error, wrong attribution method, or miscalibrated baseline |

**Validation Checklist:**
- [ ] Confirm whether using **SHAP values** (additive) or **permutation importance** (non-additive)
- [ ] Verify denormalization scaling applied **after** attribution, not before
- [ ] **Sanity test:** Sum of all feature impacts should ≈ `(prediction - baseline)`. If not, attribution is miscalibrated.
- [ ] If errors found, recalibrate and update displayed effect sizes

**Impact:** Scientific credibility depends on correct attribution math.

---

### 💡 Add Confidence Computation Explanation

| Current | Problem | Solution |
|---------|---------|----------|
| Shows `82.7%` confidence | Source unclear to users | Add tooltip explaining method |

**Suggested Explanation:**
> *"Model Ensemble Confidence reflects how well the 5 trained models agree. Calculated as: `1 - (σ_ensemble / μ_prediction)` where σ = model disagreement and μ = mean prediction. Higher % = stronger consensus among ensemble members."*

**Action:**
- [ ] Add expandable tooltip to confidence metric in UI
- [ ] Document in app help/FAQ section

---

## 📈 **Tier 2: Medium-Term Enhancements (1–2 months)**
*High-value UX & trust features. Schedule after Tier 1 completion.*

### 📊 Add Temporal Comparison vs. Historical Baseline

**Feature:** Compare prediction to 30-year regional climatology.

**User-facing output:**
> *"Expected yield: **+12% above 30-year regional average**"* ← Instant context for farmers.

Or: *"This season is **+1.2σ wetter than normal**"* ← Statistical framing.

**Technical Implementation:**
- Compute Z-score: `(user_value - historical_mean) / historical_std`
- Display as percentage deviation or σ units
- Highlight extreme years (top 10%, bottom 10%)

---

### 🌾 Add Agronomic Interpretation Layer

**Goal:** Translate technical model outputs → actionable, plain-language guidance.

**Categories to Add:**

| Category | Example |
|----------|---------|
| **Planting Outlook** | *"Late planting (June) may reduce waterlogging risk under projected rainfall."* |
| **Flood Risk** | *"High precipitation during Months 6–9 risks waterlogging during root bulking."* |
| **Heat Stress Risk** | *"Temperature >35°C in Month 7 may reduce pollination success."* |
| **Disease Pressure** | *"High RH (>80%) + warm nights (>20°C) in Months 5–7 favor cassava bacterial blight."* |
| **Soil-Water Dynamics** | *"Recommend ridge planting to improve drainage under projected rainfall."* |

**Deliverable:** Rule-based heuristic layer (if-then logic mapping climate → agronomic risk).

**Example Rule:**
```
IF (PRECTOTCORR_month6-8 > 300mm) AND (Crop == "Cassava"):
  THEN Risk = "High waterlogging during root bulking"
  RECOMMENDATION = "Consider ridge planting, improved drainage"
```

---

### 🎨 Redesign Feature Importance Chart — Climate Stress Scorecard

**Current Problem:** All-red bar chart (negative impacts) confuses users into thinking "everything is bad."

**Better Design: Climate Stress Scorecard**

```
┌─────────────────────────────────────┐
│ Climate Stress Score: 6.4/10        │
│ (Moderate Risk)                      │
├─────────────────────────────────────┤
│ 🟦 Waterlogging Risk:   HIGH         │
│    (Months 6–8 >300mm/month)        │
│                                      │
│ 🟩 Heat Stress:         LOW          │
│    (T2M_MAX <35°C)                  │
│                                      │
│ 🟨 Disease Pressure:    MODERATE    │
│    (High RH + Warm nights)          │
└─────────────────────────────────────┘
```

**Benefits:**
- Translates ML outputs → management categories extension officers use
- Color coding by stress type (Thermal = 🔴, Hydric = 💧)
- Actionable without requiring climate science expertise

---

## 🔬 **Tier 3: Research Roadmap (Strategic)**
*Long-term value additions. Plan for Q3+ 2026.*

### 🎯 Add Scenario Mode (What-If Analysis)

**Purpose:** Climate adaptation planning; supports "what-if" exploration for farmers.

**Example Scenarios:**
- *"What if rainfall decreases 15%?"*
- *"What if mean temperature rises +2°C?"*
- *"What if I shift planting from April to June?"*
- *"What if soil drainage improves (ridge planting)?"*

**Technical Requirements:**
- Fast re-inference needed (target: <100ms per prediction)
- Simple slider UI for ±20% rainfall, ±2°C temperature
- Display updated yield prediction + risk scorecard

**Strategic Value:** Enables small-holder farmers to stress-test their plans under climate variability → increases adoption.

---

## ✅ **Additional Credibility Enhancements**

### 📚 Add Data Provenance Section

**User Trust:** Farmers want to know where data comes from.

**Include in "About" or "Help" section:**

```
Climate Data Source:     NASA POWER v9 (1999–2023)
Yield Data Source:       FAOSTAT + HarvestStat Africa
Training Years:          1999–2023 (25 years)
Geopolitical Coverage:   Nigeria (6 zones, 36 states)
Crops Included:          Maize, Rice, Cassava, Yam

Preprocessing:
- Climate: Daily → Monthly (aggregation: sum for PRECTOTCORR, mean for others)
- Yield: State-level → Geopolitical Zone level (aggregation: mean)
- Missing data: Filled via median imputation
```

---

### ⚠️ Add Model Limitations & Caveats

**Credibility:** Prevents overconfidence; builds trust via transparency.

**Example disclaimer:**
> ⚠️ **Model Limitations**
> 
> This model predicts yield based on climate patterns. It does NOT account for:
> - Pest outbreaks or epidemics
> - Fertilizer availability or quality
> - Conflict or political instability
> - Catastrophic flooding or drought (tail events)
> - Crop variety changes or improved cultivars
> - Farmer management practices (irrigation, mulching, etc.)
> 
> **Best used for:** Seasonal planning, climate risk assessment, comparative scenario analysis.
> 
> **Not suitable for:** Insurance payouts, precise field-level predictions, or planning without agronomist consultation.

---

### 📊 Add Model Validation Metrics

**Users Will Ask:** *"How accurate is this?"*

**Expose key metrics in expandable "Model Performance" section:**

```
Cross-Validation Performance (5-fold):
├─ Mean Absolute Error (MAE):     ±287 kg/ha
├─ Root Mean Squared Error (RMSE): ±412 kg/ha
├─ R² Score:                        0.73
└─ Out-of-sample region test:       ±298 kg/ha

Interpretation:
- On average, predictions are off by ~287 kg/ha
- Model explains 73% of yield variability
- Holds up well on held-out geographic regions
```

---

## 🎯 **Strategic North Star**

### **The Biggest Value Add: Explainability → Agronomic Action**

| Current State | Next Level | Ultimate Vision |
|---------------|-----------|-----------------|
| "PRECTOTCORR impact: -1,557 kg/ha" | "Excess rainfall likely reduced oxygen in root zone" | **"Risk: severe waterlogging during root bulking. Recommendation: ridge planting, drainage improvement."** |
| Feature importance chart | Climate stress scorecard | Actionable decision support |
| Technical metrics | User trust | Adoption by extension officers & small-holder farmers |

**Key insight:** Real adoption happens when ML outputs → **management decisions farmers can act on.**

---

## 📋 **Master Checklist**

### Tier 1 Blockers
- [ ] Verify PRECTOTCORR units (mm/month vs. mm/day)
- [ ] Fix preprocessing pipeline alignment
- [ ] Update all UI labels and documentation
- [ ] Validate SHAP/permutation attribution math
- [ ] Add confidence computation tooltip

### Tier 2 Medium-term
- [ ] Temporal comparison (vs. 30-year baseline)
- [ ] Agronomic interpretation rules
- [ ] Climate stress scorecard UI redesign
- [ ] Rule-based heuristics for flood/heat/disease risk

### Tier 3 Strategic
- [ ] Scenario mode (what-if sliders)
- [ ] Data provenance documentation
- [ ] Model limitations disclaimer
- [ ] Validation metrics in expandable section

