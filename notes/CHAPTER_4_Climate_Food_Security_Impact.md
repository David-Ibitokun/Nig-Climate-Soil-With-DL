# Chapter 4: Evaluation of Climate Change Impacts on Food Security in Nigeria

## 4.1 Overview

This chapter presents the impact evaluation stage of the study, where the trained TCN-MLP model is used to estimate how projected climate stress conditions may influence crop yield and food security outcomes across Nigeria. The analysis focuses on four major crops (cassava, maize, rice, and yam) across six geopolitical regions over the 1999 to 2023 period.

The objective of this chapter is threefold:
1. quantify yield responses under multiple climate stress scenarios,
2. translate yield responses into a regional food security risk metric, and
3. derive practical adaptation priorities for policy and extension planning.

The chapter is organized into scenario design, implementation workflow, results interpretation, risk evaluation, adaptation strategies, limitations, and policy conclusions.

---

## 4.2 Climate Scenario Modeling Framework

### 4.2.1 Scenario Specification

Climate stress is represented using controlled perturbations on input climate features (GDD, rainfall, and humidity). Table 4.1 summarizes the scenario assumptions.

| Scenario | GDD Change | Rainfall Change | Humidity Change | Interpretation |
|----------|------------|-----------------|-----------------|----------------|
| Baseline | 0.0 | 0% | 0% | Reference climate condition |
| Warming +1C | +2.5 | 0% | -2% | Mild warming signal |
| Warming +2C | +5.0 | 0% | -4% | Moderate warming signal |
| Drought -20% | +1.0 | -20% | -5% | Moderate moisture stress |
| Drought -40% | +2.0 | -40% | -10% | Severe moisture stress |
| Flooding +40% | -1.0 | +40% | +10% | Excess rainfall event |
| Extreme (2C + Drought) | +5.0 | -30% | -8% | Compound heat-drought stress |

### 4.2.2 Computational Procedure

For each scenario, the following pipeline is applied:
1. Load the baseline dataset of 600 observations (6 regions x 4 crops x 25 years).
2. Apply the same feature scaling used during model training.
3. Perturb the scaled climate variables according to the scenario definition.
4. Run forward prediction using the trained TCN-MLP model.
5. Compute percentage yield response using:

$$
\Delta Y(\%) = \frac{Y_{scenario} - Y_{baseline}}{Y_{baseline}} \times 100
$$

6. Aggregate outputs by region and crop to reveal vulnerability structure.

### 4.2.3 Rationale for Feature-Space Perturbation

The adopted approach is suitable for a B.Tech level predictive impact study because it:
1. reuses learned nonlinear climate-yield relationships from the trained model,
2. enables fast, repeatable, and transparent scenario testing,
3. avoids expensive climate-physics coupling outside project scope, and
4. supports uncertainty discussion using available validation statistics.

Model reliability in this chapter is interpreted alongside cross-validation performance (approximately CV R2 = 0.83).

---

## 4.3 Scenario Impact Results

### 4.3.1 Aggregate Yield Response

Table 4.2 reports mean projected yield changes under each scenario.

| Scenario | Mean Yield Impact |
|----------|-------------------|
| Warming +1C | -23.1% |
| Warming +2C | -43.8% |
| Drought -20% | -32.3% |
| Drought -40% | -46.8% |
| Flooding +40% | +197.4% (model artifact risk) |
| Extreme (2C + Drought) | -53.5% |

The dominant result is that compound stress (warming + drought) produces the strongest decline, reducing average yield by more than half relative to baseline.

### 4.3.2 Regional Differentiation

Under the extreme scenario, the most vulnerable regions are:
1. South-East (-18.3%),
2. South-West (-17.2%), and
3. North-Central (-16.4%).

The most resilient regions are:
1. North-West (-2.8%), and
2. North-East (+1.3%).

This regional contrast is consistent with observed baseline variability. Regions with higher baseline yield volatility tend to exhibit larger stress responses.

---

## 4.4 Food Security Risk Quantification

### 4.4.1 Composite Risk Formulation

Regional food security exposure is summarized using a bounded risk index:

$$
Risk = 0.4R_{extreme} + 0.3R_{drought} + 0.3R_{stability}
$$

Component definitions:
1. $R_{extreme}$ captures severe loss under compound stress:
  if loss >= 15%, score = 1.0;
  if loss is 5% to 15%, score = 0.5;
  if loss < 5%, score = 0.
2. $R_{drought}$ captures response to drought:
  if loss >= 20%, score = 1.0;
  if loss is 10% to 20%, score = 0.5;
  if loss < 10%, score = 0.
3. $R_{stability}$ represents volatility amplification under stress using the ratio:

$$
R_{stability} = 0.3 \times \frac{CV_{stress}}{CV_{baseline}}
$$

where higher values indicate reduced production stability.

### 4.4.2 Regional Risk Ranking

| Region | Risk Score | Category | Interpretation |
|--------|------------|----------|----------------|
| South-West | 0.820 | Critical | High losses with worsening stability |
| South-East | 0.801 | Critical | Persistent losses across scenarios |
| North-Central | 0.788 | Critical | High sensitivity with unstable baseline |
| South-South | 0.613 | High | Moderate losses and moderate instability |
| North-East | 0.294 | Low | Relatively stable under tested stresses |
| North-West | 0.286 | Low | Most resilient in this model setting |

The ranking indicates that risk is not uniformly distributed and should be addressed through region-specific adaptation planning.

---

## 4.5 Crop Sensitivity Analysis

### 4.5.1 Sensitivity Metric

Crop sensitivity is computed from normalized responses to warming, drought, and compound stress:

$$
Sensitivity = \frac{1}{3}\left(\left|\frac{Impact_{warming}}{+2C}\right| + \left|\frac{Impact_{drought}}{-40\%}\right| + \left|\frac{Impact_{extreme}}{compound}\right|\right)
$$

### 4.5.2 Crop Ranking

| Crop | Sensitivity Score | Classification |
|------|-------------------|----------------|
| Cassava | 13.9 | Highly sensitive |
| Maize | 8.2 | Moderately sensitive |
| Rice | 6.5 | Moderately sensitive |
| Yam | 5.0 | Relatively resilient |

These results indicate cassava as the most climate-sensitive crop in the current dataset, while yam appears comparatively robust.

---

## 4.6 Adaptation and Policy Implications

### 4.6.1 Region-Level Priorities

For high-risk zones (South-West, South-East, North-Central), priority actions are grouped by implementation horizon.

Immediate actions:
1. distribute drought-tolerant planting materials,
2. strengthen seasonal advisory dissemination for farmers.

Short-term actions (1 to 2 years):
1. expand small-scale irrigation support,
2. train farmers on moisture conservation and climate-smart agronomy,
3. improve local weather-based decision support.

Medium-term actions (3 to 5 years):
1. encourage crop portfolio shifts toward resilient varieties,
2. scale soil and water conservation practices,
3. support livelihood diversification in highly exposed communities.

Long-term actions (beyond 5 years):
1. sustain breeding programs for stress-tolerant varieties,
2. institutionalize agricultural insurance coverage,
3. integrate climate-risk analytics into national food policy planning.

### 4.6.2 Crop-Specific Direction

Cassava:
1. prioritize drought-resilient variety deployment,
2. improve water management around major cassava belts.

Maize:
1. expand heat-tolerant hybrids,
2. optimize planting calendar under shifting temperature profiles.

Rice:
1. strengthen water-efficient production systems,
2. improve suitability matching between rice type and local hydrology.

Yam:
1. use as a strategic resilience crop,
2. investigate transferable resilience characteristics for other crops.

---

## 4.7 Practical Food Security Outlook

Baseline mean yield is approximately 5,980 kg/ha, while the extreme scenario projects about 2,774 kg/ha (a 53.5% reduction). If such a decline occurs in reality, household food availability and market supply would be substantially reduced, especially in already volatile regions.

This supports a key thesis position: climate stress will likely amplify existing regional inequality in agricultural productivity unless targeted adaptation is implemented.

---

## 4.8 Study Limitations

The following limitations should guide interpretation:
1. Scenario simulation is feature-based and not a full climate-physics forecast.
2. Model behavior is constrained by historical data (1999 to 2023).
3. The flooding response (+197.4%) is likely an extrapolation artifact and should be treated cautiously.
4. Socioeconomic, infrastructure, and soil heterogeneity are not fully modeled at sub-regional scale.
5. Policy recommendations assume implementation capacity that may vary across states.

---

## 4.9 Chapter Summary

This chapter demonstrates that the trained TCN-MLP model can be used as a decision-support tool for climate-food security analysis. The core findings are:
1. compound heat-drought stress is the strongest threat to yield,
2. South-West, South-East, and North-Central are priority vulnerability zones,
3. cassava is the most climate-sensitive crop in the study,
4. adaptation planning should combine crop-specific and region-specific strategies.

Overall, the evidence supports proactive climate adaptation as a necessary condition for food security resilience in Nigeria.

---

## 4.10 Recommended Continuation of the Study

Future work should:
1. validate model estimates with independent field observations,
2. align perturbation scenarios with externally sourced climate projection pathways,
3. include additional drivers such as soil fertility dynamics, input use, and pest pressure,
4. deploy the model outputs in a monitoring dashboard for periodic policy review.
