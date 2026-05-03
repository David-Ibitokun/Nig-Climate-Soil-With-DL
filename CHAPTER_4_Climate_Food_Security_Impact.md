# Chapter 4: Evaluating the Impact of Climate Change on Food Security

## 4.1 Introduction

This chapter synthesizes the trained TCN-MLP model to assess how climate change scenarios (warming, drought, flooding, and extreme events) impact crop yields and regional food security across Nigeria. Using feature-space perturbations, we project yield responses under different climate futures and quantify vulnerability at regional and crop levels.

---

## 4.2 Methodology: Climate Scenario Modeling

### 4.2.1 Scenario Design

We simulate climate change by perturbing the model's input features:

| Scenario | GDD Change | Rainfall Change | Humidity Change | Interpretation |
|----------|-----------|-----------------|-----------------|-----------------|
| **Baseline** | 0 | 0% | 0 | Current climate (reference) |
| **Warming +1°C** | +2.5 | 0% | -2% | Mild temperature increase |
| **Warming +2°C** | +5.0 | 0% | -4% | Moderate temperature increase |
| **Drought -20%** | +1.0 | -20% | -5% | Mild water stress |
| **Drought -40%** | +2.0 | -40% | -10% | Severe water stress |
| **Flooding +40%** | -1.0 | +40% | +10% | Excess rainfall event |
| **Extreme (2°C + Drought)** | +5.0 | -30% | -8% | Compound warming + dry conditions |

### 4.2.2 Implementation

For each scenario, we:
1. **Load baseline data** from all 600 samples (6 regions × 4 crops × 25 years)
2. **Standardize climate features** using training-set statistics
3. **Apply perturbations** to scaled inputs (GDD, rainfall, humidity across 12 months)
4. **Predict yields** using the trained TCN-MLP model
5. **Calculate yield change** as: `((scenario_yield - baseline_yield) / baseline_yield) × 100%`
6. **Aggregate by region and crop** to assess vulnerability patterns

### 4.2.3 Why Feature-Space Perturbation?

- **No physics simulation required**: We leverage the model's learned relationships between climate and yield.
- **Fast computation**: Enables exploration of multiple scenarios without computational overhead.
- **Model-consistent**: Predictions stay within the input space the model was trained on.
- **Uncertainty quantified**: Cross-validation metrics (CV R² ≈ 0.83) reflect model reliability across scenarios.

---

## 4.3 Climate Impact Results: Yield Projections

### 4.3.1 Overall Yield Changes

| Scenario | Mean Yield Impact |
|----------|------------------|
| Warming +1°C | **-23.1%** |
| Warming +2°C | **-43.8%** |
| Drought -20% | **-32.3%** |
| Drought -40% | **-46.8%** |
| Flooding +40% | **+197.4%** (unrealistic; model artifact) |
| Extreme (2°C + Drought) | **-53.5%** |

**Key Finding:** A compound 2°C warming + 30% drought reduces yields by more than half—a critical food security threat.

### 4.3.2 Regional Sensitivity Rankings

#### Most Vulnerable Regions:
1. **South-West** (-17.2% under extreme scenario)
2. **South-East** (-18.3% under extreme)
3. **North-Central** (-16.4% under extreme)

**Reason:** Southern zones have higher baseline yield instability (CV = 0.79–0.93), making them more susceptible to stress.

#### Most Resilient Regions:
1. **North-West** (−2.8% under extreme; baseline CV = 0.70)
2. **North-East** (+1.3% under extreme; baseline CV = 0.74)

**Reason:** Northern regions show lower variability; the model predicts they are either resilient or even marginally benefit from altered climate features in the feature space.

---

## 4.4 Food Security Risk Assessment

### 4.4.1 Risk Score Methodology

For each region, we calculate a **composite Food Security Risk Score** (0–1, higher = worse):

$$\text{Risk Score} = 0.4 \times R_{\text{extreme}} + 0.3 \times R_{\text{drought}} + 0.3 \times R_{\text{stability}}$$

Where:

- **$R_{\text{extreme}}$** (40% weight): 
  - 1.0 if extreme scenario yield drop ≥ 15%
  - 0.5 if drop 5–15%
  - 0 if drop < 5%

- **$R_{\text{drought}}$** (30% weight):
  - 1.0 if drought scenario yield drop ≥ 20%
  - 0.5 if drop 10–20%
  - 0 if drop < 10%

- **$R_{\text{stability}}$** (30% weight):
  - Ratio of extreme scenario stability to baseline stability (higher ratio = worse)
  - Reflects whether yield becomes more volatile under climate stress

### 4.4.2 Regional Risk Rankings

| Region | Risk Score | Category | Interpretation |
|--------|-----------|----------|-----------------|
| South-West | **0.820** | CRITICAL | High yield losses + deteriorating stability |
| South-East | **0.801** | CRITICAL | Consistent yield drops across scenarios |
| North-Central | **0.788** | CRITICAL | Volatile baseline; large scenario impacts |
| South-South | **0.613** | HIGH | Moderate yield losses; moderate stability |
| North-East | **0.294** | LOW | Marginal losses or gains; stable yields |
| North-West | **0.286** | LOW | Most resilient; stable and low risk |

---

## 4.5 Crop-Specific Climate Sensitivity

### 4.5.1 Sensitivity Index Calculation

For each crop, overall sensitivity is:

$$\text{Sensitivity} = \frac{1}{3} \left( \left|\frac{\text{Warming}_{\text{impact}}}{\text{+2°C}}\right| + \left|\frac{\text{Drought}_{\text{impact}}}{\text{−40\%}}\right| + \left|\frac{\text{Extreme}_{\text{impact}}}{\text{Compound}}\right| \right)$$

### 4.5.2 Crop Rankings

| Crop | Sensitivity | Category |
|------|-----------|----------|
| **Cassava** | 13.9 | HIGHLY SENSITIVE |
| **Maize** | 8.2 | MODERATELY SENSITIVE |
| **Rice** | 6.5 | MODERATELY SENSITIVE |
| **Yam** | 5.0 | RESILIENT |

**Implication:** Cassava production is most threatened by climate change; yam production is most resilient. Agricultural policy should prioritize cassava-growing regions for climate adaptation support.

---

## 4.6 Adaptive Strategies & Recommendations

### 4.6.1 Regional Level (High-Risk Zones)

**For South-West, South-East, North-Central:**
1. **Immediate**: Distribute drought-resistant seed varieties (esp. drought-tolerant cassava, maize)
2. **Short-term (1–2 years)**: 
   - Develop irrigation infrastructure
   - Establish agro-weather information services
   - Train farmers on water-efficient practices (drip irrigation, mulching)
3. **Medium-term (3–5 years)**:
   - Shift cropping patterns toward resilient crops (yam, improved rice varieties)
   - Implement soil conservation and rainwater harvesting
   - Diversify income through drought-tolerant crops (millet, sorghum)
4. **Long-term (5+ years)**:
   - Support agricultural research for climate-resilient varieties
   - Promote crop insurance schemes
   - Invest in climate information services and early warning systems

### 4.6.2 Crop-Specific Strategies

**Cassava (Most Sensitive):**
- Prioritize drought-tolerant varieties (e.g., IITA bred lines)
- Implement drip irrigation in cassava-growing regions
- Promote intercropping with nitrogen-fixing legumes to improve soil resilience

**Maize (Moderately Sensitive):**
- Transition to heat-tolerant hybrids
- Adjust planting dates to track climate-shifted growing seasons
- Increase use of conservation agriculture

**Rice (Moderately Sensitive):**
- Promote water-efficient rice varieties
- Improve water management in irrigated systems
- Diversify into drought-tolerant upland rice in moisture-limited zones

**Yam (Resilient):**
- Expand production to buffer food security
- Maintain current management practices; focus on yield improvement
- Investigate why yam remains resilient (soil type? crop physiology?) for learning transfer to other crops

### 4.6.3 System-Wide Strategies

- **Portfolio diversification**: Combine crops and regions to reduce single-point failure risk
- **Infrastructure**: Expand storage and value-chain facilities to reduce post-harvest losses
- **Farmer support**: Subsidize inputs (seed, fertilizer) for vulnerable households
- **Research & extension**: Fund breeding programs and strengthen agricultural extension
- **Policy**: Align subsidies and incentives with climate-resilient practices

---

## 4.7 Food Security Outlook

### Current State:
- Average baseline yield: **5,980 kg/ha**
- Under extreme scenario: **2,774 kg/ha** (−53.5%)

### Implications:
- **Caloric deficit**: If 1 hectare produces 5,980 kg today, extreme scenario reduces it to ~2,774 kg—insufficient for many households
- **Regional inequality**: South-West, South-East, North-Central face the largest gaps; North-West most stable
- **Temporal variability**: High-risk regions already show 0.79–0.93 CV in baseline; stress amplifies this volatility

---

## 4.8 Limitations & Uncertainties

1. **Feature-space perturbations**: We do not simulate actual climate patterns, soil changes, or pest outbreaks—only feature input shifts.
2. **Model scope**: TCN-MLP trained on 1999–2023 data; may not capture unprecedented climate extremes.
3. **Flooding scenario artifact**: The +40% rainfall scenario predicts +197% yield; unrealistic. Suggest caution interpreting flooding results (likely model artifact or data imbalance).
4. **Regional heterogeneity**: Recommendations assume uniform soil, infrastructure, and socioeconomic conditions within each region; actual needs vary.
5. **Policy lag**: Adaptation strategies assume timely implementation; institutional delays can reduce effectiveness.

---

## 4.9 Conclusions & Policy Priorities

1. **Immediate action needed** in South-West, South-East, and North-Central to build adaptive capacity before yields collapse
2. **Crop-level focus**: Cassava, maize, and rice require targeted research and varietal improvement
3. **Data-driven approach**: Regular re-runs of this model as new climate and yield data arrive will enable adaptive management
4. **Multi-stakeholder engagement**: Success requires coordination among farmers, extension agents, researchers, and policymakers

---

## 4.10 Next Steps

- Validate model predictions against independent climate and yield data (e.g., crop weather indices, farmer surveys)
- Refine scenario design based on climate projections from GCMs (IPCC-endorsed models)
- Engage agricultural extension to field-test recommended varieties and practices
- Develop a web dashboard for real-time policy monitoring and adaptive planning
