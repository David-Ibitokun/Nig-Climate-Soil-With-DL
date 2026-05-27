---
title: "Climate Change and Food Security in Nigeria"
subtitle: "Chapter 4: Results"
author: "[Your Name]"
date: "27 May 2026"
---

# Chapter 4: Results

## 4.1 Introduction

This chapter presents the results available from the project’s own app pages and notebooks. The evidence base is limited to app.py, the Streamlit pages in pages/, the preprocessing notebook tcn_mlp_data.ipynb, and the training notebook tcn_mlp_train.ipynb. No external results archive is used here.

## 4.2 Experimental Setup

The data notebook shows that the dataset was assembled from NASA POWER climate records and HarvestStat Africa yield data, filtered to four crops: maize, rice, cassava, and yam. State-level crop records were mapped to Nigeria’s six geopolitical zones and aggregated to yearly crop-region observations. Monthly climate sequences were then constructed from 12 months of the following variables: T2M, T2M_MAX, T2M_MIN, TS, T2MDEW, T2MWET, PRECTOTCORR, RH2M, and QV2M.

The training notebook shows a TCN-MLP ensemble trained with 5-fold stratified cross-validation by crop-region combination. The configuration recorded in the notebook is:

| Setting | Value |
|---|---|
| Seed | 42 |
| Folds | 5 |
| Epochs | 160 |
| Batch size | 32 |
| Loss | Huber, delta 0.35 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 3e-4 |
| Clip norm | 1.0 |
| L2 regularization | 1e-3 |
| Dropout | 0.12 |

The model architecture in the notebook uses a causal TCN branch with residual blocks, region and crop embeddings, and a year branch built from normalized year, sine, and cosine features. The app’s home page reflects that same architecture by describing the system as a TCN–MLP ensemble for climate-sequence crop-yield prediction.

## 4.3 Quantitative Results Available in the App

The app.py home page precomputes an ensemble summary from the loaded evaluation data and displays two key overview metrics when available: average R² across the five folds and average MAE across the five folds. It also reports the number of crops as four and the number of temporal features as nine. These values are surfaced in the interface rather than hard-coded in the source, so the thesis should refer to them as page-reported summary metrics.

The Model Evaluation page in pages/04_Model_Evaluation.py shows the following evaluation outputs when the corresponding data are present:

| Metric family | What the page displays |
|---|---|
| Ensemble performance | Average R², MAE, MAPE, and sMAPE across five folds |
| Fold-level comparison | A grouped bar chart for R², MAPE, sMAPE, and MASE by fold |
| Regional performance | Region-by-region R², MAE, MAPE, sMAPE, and MASE |
| Interpretation | Plain-language explanations of R², MAE, MAPE, sMAPE, MASE, and 5-fold cross-validation |

The notebook itself also prints fold-wise training and evaluation metrics, exports the five fold models, and saves a fold artifact table that records which fold was selected as the best-performing fold. However, because the notebook output is not embedded in the source text, the thesis should use the page outputs or the notebook’s saved figures as the presentation layer for those values.

## 4.4 Historical and Seasonal Patterns

The Historical Trends page renders a 1999–2023 yield trend figure and explains the reading of positive and negative trend slopes. The page text states that the analysis covers maize, rice, cassava, and yam over the 25-year period and that the slope interpretation is positive when productivity is improving and negative when productivity is declining.

The Climate Patterns page provides the seasonal climate context used by the model. It shows monthly temperature, rainfall, and relative humidity patterns derived directly from the processed dataset. The page explains the climate meanings of T2M, T2M_MAX, T2M_MIN, TS, PRECTOTCORR, RH2M, QV2M, T2MDEW, and T2MWET, and notes why these variables matter for yield. It also shows the monsoon and pre-monsoon seasonal interpretation in the interface.

These two pages are the main narrative evidence for the chapter’s climate context. They establish that the model is being interpreted against seasonal rainfall and temperature structure rather than against a purely static crop table.

## 4.5 Qualitative Results from the Notebook Design

The preprocessing notebook documents the full data pipeline: state-to-zone mapping, crop filtering, aggregation to yearly region-level yield, and the construction of monthly climate tensors. This is important for the thesis because it shows that the final dataset preserves both crop identity and regional identity while also encoding 12 months of climate variation.

The training notebook shows that the model uses a temporal branch to capture month-to-month climate dynamics and context branches to let the network distinguish region, crop, and year effects. The architecture therefore supports the thesis claim that yield is shaped by both seasonal climate and agricultural context.

The app pages also make the project interpretable to non-technical readers. The model evaluation page explains each metric in plain language, while the recommendations page turns the model outputs into policy, crop, farm, and institutional recommendations. In this way, the repository already contains the building blocks for the results narrative and the discussion narrative.

## 4.6 Summary of Results

The project provides three kinds of results within the allowed source files:

- A documented preprocessing pipeline that constructs 12-month climate sequences for four crops across six regions.
- A 5-fold TCN-MLP ensemble training setup with explicit regularisation and reproducibility settings.
- Streamlit pages that expose ensemble evaluation, historical trends, seasonal climate patterns, and recommendations in an accessible form.

These source files are sufficient to describe the method and the structure of the results presentation, but the exact numeric evaluation values should be taken from the running app pages rather than claimed directly from the notebook source.

## Appendix A. Screenshot Checklist

1. App overview
   - Capture the main page of app.py showing the model overview metrics and project summary.
   - Use as Figure 4.1.

2. Model evaluation page
   - Capture the output of pages/04_Model_Evaluation.py with the ensemble metrics, fold chart, and regional performance section.
   - Use as Figure 4.2.

3. Historical trends page
   - Capture the output of pages/06_Historical_Trends.py showing the historical yield trend figure.
   - Use as Figure 4.3.

4. Climate patterns page
   - Capture the output of pages/08_Climate_Patterns.py showing the seasonal climate plots.
   - Use as Figure 4.4.

5. Training notebook summary
   - Capture the notebook output from notebooks/tcn_mlp_train.ipynb showing the final ensemble metric printout and model export messages.
   - Use as Figure 4.5.