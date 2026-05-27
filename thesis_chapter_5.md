---
title: "Climate Change and Food Security in Nigeria"
subtitle: "Chapter 5: Discussion & Conclusion"
author: "[Your Name]"
date: "27 May 2026"
---

# Chapter 5: Discussion & Conclusion

## 5.1 Discussion

Within the allowed source files, the most defensible interpretation is that the project links climate sequences to yield outcomes through a TCN-MLP ensemble and then translates those predictions into a human-readable dashboard. The notebooks show why the architecture is appropriate: monthly climate inputs are handled as sequences, while crop, region, and year are modelled as context variables. That design is consistent with a food-security problem in which the same climate conditions can mean different things for different crops and regions.

The Climate Patterns page gives the seasonal context for that interpretation. It shows that rainfall, temperature, and humidity change across the year and that the rainy season is central to the analysis. The data notebook also confirms that the climate inputs were aggregated into monthly windows, which means the model is explicitly seasonal rather than static. That is the right framing for a thesis about climate impact on crop yield.

The Model Evaluation page is important because it exposes the metrics in a decision-oriented format: fold-level R², MAE, MAPE, sMAPE, and MASE, together with a regional summary. The app’s metric layout implies that the correct discussion is not about a single score but about how stable the model is across folds and across regions. That is a better thesis argument than focusing only on one headline number.

The training notebook also supports a cautious interpretation. It uses 5-fold cross-validation, a modest batch size, Huber loss, and regularisation through both L2 and dropout. Those settings are appropriate when the dataset is not large and when the goal is generalisation rather than memorisation. In thesis terms, this means the model is being used as a structured predictor of relative vulnerability, not as a perfect point-estimation system.

The recommendation page is the final bridge from model output to interpretation. It shows that the project is intended to support region-specific adaptation, crop-level planning, farm-level management, and institutional coordination. That matters because the thesis should not stop at prediction; it should explain how the results can guide action.

## 5.2 Conclusion

Based on the source files available in the repository, the thesis can conclude that the project successfully operationalised a climate-yield prediction pipeline for four crops across Nigeria’s six geopolitical zones. The data preparation notebook established the monthly climate sequences, the training notebook defined the ensemble model, and the app presented the outputs through evaluation, trend, pattern, and recommendation pages.

The conclusion should therefore be phrased conservatively: the project demonstrates a workable framework for climate-sensitive yield modelling and gives a clear dashboard for interpreting results, but the final thesis should quote any exact performance figures directly from the running app or its saved notebook outputs.

## 5.3 Future Work

Future work can stay within the same codebase direction while improving the thesis evidence layer.

1. Add explicit notebook output cells or exported summary tables for the evaluation metrics so the thesis can quote exact values directly from the training workflow.
2. Extend the climate-pattern page with annotated seasonal markers so the monsoon and planting windows are even clearer in the final document.
3. Add a dedicated notebook cell that exports a short textual summary of the fold metrics, which would make chapter writing more reproducible.
4. Expand the recommendations page with region-specific examples tied to the page’s own metrics and trend visuals.

## 5.4 Final Remarks

The repository already contains the structure needed for a thesis-ready narrative: data preparation, model training, model evaluation, seasonal interpretation, historical trends, and recommendation logic. The safest thesis wording is to describe that structure directly and only attach exact numbers where they are visible in the app or notebook outputs.