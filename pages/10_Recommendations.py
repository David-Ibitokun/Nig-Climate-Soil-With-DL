import streamlit as st
import pandas as pd

from data_loader import apply_global_style, load_data


def _safe_df(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def render():
    apply_global_style()

    st.title("Conclusion and Recommendations")

    st.markdown(
        """
This chapter presents the concluding part of the study on climate change and food security in Nigeria.
It summarizes the major findings from the TCN-MLP model, explains the implications for agricultural planning, and presents recommendations for policy, practice, and future research.
        """
    )

    data = load_data()
    region_metrics = _safe_df(data.get("region_ensemble_metrics"))
    crop_region = _safe_df(data.get("crop_region_resilience"))
    crop_sensitivity = _safe_df(data.get("crop_sensitivity"))
    ensemble_metrics = _safe_df(data.get("ensemble_metrics"))

    lowest_region = None
    highest_region = None
    lowest_crop_region = None
    if not region_metrics.empty and "Resilience_Index" in region_metrics.columns:
        ordered_regions = region_metrics.sort_values("Resilience_Index", ascending=True)
        lowest_region = ordered_regions.iloc[0]
        highest_region = ordered_regions.iloc[-1]
    else:
        ordered_regions = pd.DataFrame()

    if not crop_region.empty and "Resilience_Index" in crop_region.columns:
        ordered_crop_region = crop_region.sort_values("Resilience_Index", ascending=True)
        lowest_crop_region = ordered_crop_region.iloc[0]
    else:
        ordered_crop_region = pd.DataFrame()

    st.markdown("---")

    st.subheader("Introduction")

    st.markdown(
        """
This chapter closes the study by interpreting the model outputs in a thesis-style format.
The TCN-MLP ensemble was trained on climate-yield data from 1999 to 2023 and used to evaluate how climate stress, crop type, and regional context shape agricultural outcomes.

The results are summarized here so that the technical evaluation is translated into planning language for policy makers, extension officers, and researchers.
        """
    )

    st.markdown("---")

    st.subheader("Summary of the Study")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Years covered", "25", "1999-2023")
    with col2:
        st.metric("Crops studied", "4", "Maize, Rice, Cassava, Yam")
    with col3:
        st.metric("Regions covered", "6", "Nigeria geopolitical zones")
    with col4:
        if not ensemble_metrics.empty and "Ensemble_R2" in ensemble_metrics.columns:
            st.metric("Avg. fold R²", f"{ensemble_metrics['Ensemble_R2'].mean():.3f}")
        else:
            st.metric("Avg. fold R²", "N/A")

    st.markdown(
        """
The study addressed three main questions: how climate stress affects crop yield, whether vulnerability differs by region and crop, and how the model outputs can be converted into a practical food-security interpretation.

The analysis showed that climate variables are strongly associated with yield variation, and that the effects are not uniform across the country.
        """)

    st.markdown("---")

    st.subheader("Major Findings")

    findings = []
    if lowest_region is not None:
        findings.append(
            f"The most vulnerable region in the exported results is {lowest_region['Region']} with a resilience index of {lowest_region['Resilience_Index']:.3f}, which suggests it should receive early adaptation attention."
        )
    if highest_region is not None:
        findings.append(
            f"The strongest region in the exported results is {highest_region['Region']} with a resilience index of {highest_region['Resilience_Index']:.3f}, showing that resilience is unevenly distributed across the country."
        )
    if not crop_sensitivity.empty and "Overall_Sensitivity" in crop_sensitivity.columns:
        crop_row = crop_sensitivity.sort_values("Overall_Sensitivity", ascending=False).iloc[0]
        findings.append(
            f"At the crop level, {crop_row['Crop']} records the highest overall sensitivity in the project outputs, which supports the conclusion that crop response differs materially across staples."
        )
    if lowest_crop_region is not None:
        findings.append(
            f"The lowest crop-region resilience combination in the exported matrix is {lowest_crop_region['CropRegion']}, indicating where the model sees the greatest need for targeted intervention."
        )

    findings.extend([
        "Climate stress has a substantial negative effect on crop yield, especially when warming and drought occur together.",
        "The model indicates that some regions are far more vulnerable than others, so adaptation planning should be spatially differentiated.",
        "The evaluation metrics should be read together: R² shows fit, while MAPE, sMAPE, and MASE show how the prediction errors behave in practice.",
    ])

    for index, item in enumerate(findings, start=1):
        st.markdown(f"{index}. {item}")

    st.markdown("---")

    st.subheader("Conclusions")

    st.markdown(
        """
The analysis supports four main conclusions.

First, the TCN-MLP model demonstrates that climate change is directly linked to crop yield variation in Nigeria.

Second, vulnerability is uneven across regions, meaning that a single national response would be too coarse for effective adaptation.

Third, the four crops do not behave identically under climate stress, so crop planning must be crop-specific as well as region-specific.

Fourth, the project-specific risk and resilience outputs provide a useful decision-support layer for translating model predictions into practical agricultural planning.
        """)

    st.markdown("---")

    st.subheader("Recommendations")

    tab_policy, tab_crop, tab_farm, tab_institution = st.tabs([
        "Policy Recommendations",
        "Crop-Level Recommendations",
        "Farm-Level Recommendations",
        "Institutional Recommendations",
    ])

    with tab_policy:
        st.markdown(
            """
1. National and state agricultural agencies should adopt region-specific climate adaptation strategies rather than a single nationwide plan.
2. Food security planning should prioritize the lowest-resilience regions first, based on the model outputs.
3. Climate-risk analytics should be integrated into extension planning, input subsidy design, and food reserve management.
4. Irrigation development and weather advisory services should be expanded in the high-risk zones identified by the model.
            """
        )

    with tab_crop:
        st.markdown(
            """
1. Cassava breeding and dissemination programs should prioritize drought tolerance and yield stability under heat stress.
2. Maize production should focus on heat-tolerant hybrids, improved planting calendars, and moisture conservation practices.
3. Rice production should emphasize water management, efficient irrigation, and variety-site matching.
4. Yam should be maintained as a resilience-supporting crop while further studies examine why it is relatively more stable in the project outputs.
            """
        )

    with tab_farm:
        st.markdown(
            """
1. Farmers should adopt moisture conservation practices such as mulching, early planting, and soil water retention methods.
2. Crop diversification should be encouraged to reduce dependence on a single highly sensitive crop.
3. Access to seasonal weather forecasts should be improved so farmers can adjust planting decisions in advance.
4. Smallholder farmers should be supported with practical climate adaptation training and access to improved seed varieties.
            """
        )

        if not crop_sensitivity.empty:
            st.markdown("##### Crop sensitivity snapshot")
            st.dataframe(crop_sensitivity.round(4), use_container_width=True, hide_index=True)

    with tab_institution:
        st.markdown(
            """
1. Agricultural extension services should be strengthened to improve the transfer of climate information to farming communities.
2. Research institutions should continue to develop and test climate-resilient crop varieties suited to Nigerian conditions.
3. Insurance and safety-net programs should be expanded to protect farmers in highly vulnerable regions.
4. Government and development partners should support data systems that can monitor climate risk, yield variation, and adaptation progress over time.
            """
        )

    st.markdown("---")

    st.subheader("Limitations of the Study")

    st.markdown(
        """
1. The scenario analysis is based on feature-space perturbation and is not a full climate-physics simulation.
2. The model is trained on historical data from 1999 to 2023, so its reliability is bounded by the patterns present in that period.
3. The results do not fully capture soil variability, pest pressure, market shocks, and farm management differences at a fine spatial scale.
4. Extreme scenario responses should be interpreted cautiously because extrapolation beyond the training distribution can amplify uncertainty.
5. The food security and resilience scores are project-specific decision-support measures rather than universal standards.
        """)

    st.markdown("---")

    st.subheader("Suggestions for Future Research")

    st.markdown(
        """
1. Independent field validation should be used to compare model predictions with observed yield outcomes under climate stress.
2. Climate projections from global or regional climate models should be incorporated to support longer-term forecasting.
3. Additional variables such as soil fertility, pest incidence, input use, and irrigation coverage should be included in future datasets.
4. The model can be extended to additional crops and finer administrative scales for more detailed planning.
5. Future research should evaluate the socioeconomic dimension of food security, including access, affordability, and household coping capacity.
6. A deployment-ready dashboard could be developed to support continual policy monitoring and climate-risk communication.
        """)

    st.markdown("---")

    st.subheader("Final Remark")

    st.markdown(
        """
The study demonstrates that climate change poses a real and measurable challenge to food security in Nigeria.
By combining deep learning with scenario analysis and risk interpretation, the project provides a practical framework for identifying vulnerable crops and regions.
The findings support targeted adaptation planning and emphasize the need for timely intervention to protect agricultural productivity and food-system resilience.
        """)


if __name__ == "__main__":
    render()