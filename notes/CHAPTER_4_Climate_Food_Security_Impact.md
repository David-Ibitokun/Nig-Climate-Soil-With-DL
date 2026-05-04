# Chapter 4: Evaluation of Climate Change Impacts on Food Security in Nigeria

## 4.1 Overview

This chapter presents the impact evaluation stage of the study. The trained TCN-MLP model is used to estimate how changes in climate conditions may influence crop yield and food security outcomes across Nigeria. The analysis focuses on four major crops, namely cassava, maize, rice, and yam, across the six geopolitical zones over the 1999 to 2023 period.

The chapter has three objectives:
1. quantify yield responses under selected climate stress scenarios,
2. summarize regional food security vulnerability, and
3. derive practical adaptation priorities that can support agricultural planning.

The discussion is organized into scenario design, model-based impact results, regional risk interpretation, crop sensitivity, adaptation priorities, limitations, and conclusion.

---

## 4.2 Climate Scenario Modeling Framework

### 4.2.1 Scenario Specification

Climate stress is represented using controlled perturbations on the input climate features used by the notebook, mainly temperature, rainfall, and soil moisture. Table 4.1 summarizes the scenario assumptions.

| Scenario | Temperature Change | Rainfall Change | Soil Moisture Change | Interpretation |
|----------|-------------------|-----------------|----------------------|----------------|
| Baseline | 0.0 | 0% | 0% | Reference condition |
| Warming +1C | +1.0 | 0% | -5% | Mild warming signal |
| Warming +2C | +2.0 | 0% | -10% | Moderate warming signal |
| Drought -20% | +0.5 | -20% | -15% | Moderate moisture stress |
| Drought -40% | +1.0 | -40% | -30% | Severe moisture stress |
| Flooding +30% | 0.0 | +30% | +25% | Excess rainfall event |
| Extreme (2C + Drought) | +2.0 | -30% | -25% | Compound heat-drought stress |

### 4.2.2 Computational Procedure

For each scenario, the following pipeline is applied:
1. Load the baseline dataset of 600 observations (6 regions x 4 crops x 25 years).
2. Apply the same scaling used during model training.
3. Modify the climate variables according to the scenario definition.
4. Run forward prediction using the trained TCN-MLP model.
5. Compute percentage yield response using:

$$
\Delta Y(\%) = \frac{Y_{scenario} - Y_{baseline}}{Y_{baseline}} \times 100
$$

6. Aggregate outputs by region and crop to reveal the vulnerability structure.

### 4.2.3 Rationale for Feature-Space Perturbation

The adopted approach is suitable for a B.Tech level predictive impact study because it:
1. reuses learned nonlinear climate-yield relationships from the trained model,
2. enables fast, repeatable, and transparent scenario testing, and
3. avoids the need for a full climate-physics coupling, which is outside the project scope.

Model reliability is interpreted alongside the cross-validation performance reported in the notebook.

---

## 4.3 Scenario Impact Results

### 4.3.1 Aggregate Yield Response

Table 4.2 reports mean projected yield changes under each scenario.

| Scenario | Mean Yield Impact |
|----------|-------------------|
| Warming +1C | -3.48% |
| Warming +2C | -4.99% |
| Drought -20% | -2.70% |
| Drought -40% | -5.67% |
| Flooding +30% | +1.37% |
| Extreme (2C + Drought) | -5.26% |

The dominant result is that the compound heat-drought scenario produces the strongest decline in yield. The flooding case shows only a small positive response, so it should be interpreted cautiously rather than treated as a realistic production gain.

### 4.3.2 Regional Differentiation

Under the notebook evaluation, South-West has the highest food security risk score (0.820), followed by South-East (0.801) and North-Central (0.788). North-East (0.294) and North-West (0.286) are the least exposed regions in this model setting.

This regional contrast indicates that vulnerability is not evenly distributed. The higher-risk zones are the best candidates for targeted adaptation support.

---

## 4.4 Food Security Risk Quantification

### 4.4.1 Composite Risk Formulation

Regional food security exposure is summarized using a composite risk score derived from the severe-loss case, the drought response, and the stability of yield under stress. Higher values indicate greater vulnerability.

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

The notebook ranks cassava as the most sensitive crop, with an overall sensitivity score of 3.003. Yam follows at 1.811 and is classified as moderately sensitive. Rice is also moderately sensitive, while maize is the most resilient crop in this evaluation.

These results suggest that crop adaptation should prioritize cassava, especially in regions already identified as highly vulnerable.

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

Baseline mean yield is approximately 5,980 kg/ha, while the notebook reports a potential food security gap of about 12.4% under the extreme scenario. In practical terms, this would still imply a meaningful reduction in available food supply, especially in already volatile regions.

This supports a key thesis position: climate stress will likely amplify existing regional inequality in agricultural productivity unless targeted adaptation is implemented.

---

## 4.8 Study Limitations

The following limitations should guide interpretation:
1. Scenario simulation is feature-based and not a full climate-physics forecast.
2. Model behavior is constrained by historical data from 1999 to 2023.
3. The flooding case gives only a small positive response, which suggests that extrapolation sensitivity should be considered when interpreting non-drought scenarios.
4. Socioeconomic, infrastructure, and soil heterogeneity are not fully modeled at sub-regional scale.
5. Policy recommendations assume implementation capacity that may vary across states.

---

## 4.9 Chapter Summary

This chapter demonstrates that the trained TCN-MLP model can be used as a decision-support tool for climate-food security analysis. The core findings are:
1. compound heat-drought stress is the strongest threat to yield,
2. South-West, South-East, and North-Central are the priority vulnerability zones,
3. cassava is the most climate-sensitive crop in the study, and
4. adaptation planning should combine crop-specific and region-specific strategies.

Overall, the evidence supports proactive climate adaptation as an important condition for improving food security resilience in Nigeria.

---

## 4.10 Recommended Continuation of the Study

Future work should:
1. validate model estimates with independent field observations,
2. align perturbation scenarios with externally sourced climate projection pathways,
3. include additional drivers such as soil fertility, input use, and pest pressure
4. deploy the model outputs in a monitoring dashboard for periodic policy review.
