import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from data_loader import apply_global_style, load_data
import urllib.parse


ROOT_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURE_SVG = ROOT_DIR / "TCN_MLP_Architecture.svg"


@st.cache_data(show_spinner=False)
def _load_architecture_svg() -> str:
    if ARCHITECTURE_SVG.exists():
        return ARCHITECTURE_SVG.read_text(encoding="utf-8")
    return ""


def render():
    apply_global_style()
    
    st.title("🏗️ Model Architecture")
    
    st.markdown("""
Learn how the TCN-MLP ensemble is wired, what each branch does, and how the full model turns climate, crop, region, and year context into a yield prediction.
    """)
    
    st.markdown("---")

    st.subheader("🧭 Architecture Visualization")

    svg_markup = _load_architecture_svg()
    if svg_markup:
        # Use a smaller iframe height to avoid large vertical gaps on mobile
        components.html(svg_markup, height=520, scrolling=True)
        st.caption("TCN-MLP architecture diagram showing the temporal branch, context branches, fusion layer, and MLP head.")

        # Create a data URL so users can open the SVG in a new tab or download it directly
        try:
            svg_encoded = urllib.parse.quote(svg_markup)
            html_actions = """
            <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
            .actions-container { margin-top:8px; display:flex; gap:8px; align-items:center; font-family: 'Source Sans 3', 'Source Sans', sans-serif; flex-wrap:wrap; justify-content:flex-start; }
            .actions-container button { padding:8px 12px; border-radius:6px; border:0; cursor:pointer; font-weight:600; font-family: inherit; color:white; min-width:120px; flex:0 1 auto; }
            .actions-container button.open { background:#0d6efd; }
            .actions-container button.download { background:#198754; }
            .actions-container span.tip { color:#6c757d; font-size:0.9rem; font-family: inherit; flex:1 1 220px; min-width:160px; }

            /* Responsive: stack buttons on small screens */
            @media (max-width: 600px) {
                .actions-container { flex-direction:column; align-items:stretch; gap:6px; }
                .actions-container button { width:100%; min-width:0; padding:10px; font-size:15px; }
                .actions-container span.tip { font-size:0.9rem; text-align:left; padding-left:4px; }
            }
            </style>
            <div class="actions-container">
                <button class="open" id="openSvgBtn">Open larger view</button>
                <button class="download" id="downloadSvgBtn">Download SVG</button>
                <button class="download" id="downloadPngBtn">Download PNG</button>
            </div>
            <script>
            (function() {
                const svgEncoded = "{SVG_ENCODED}";
                function getSvg() { return decodeURIComponent(svgEncoded); }

                document.getElementById('openSvgBtn').addEventListener('click', function() {
                    try {
                        const svg = getSvg();
                        const blob = new Blob([svg], {type: 'image/svg+xml;charset=utf-8'});
                        const url = URL.createObjectURL(blob);
                        window.open(url, '_blank');
                        // revoke after a short delay
                        setTimeout(() => URL.revokeObjectURL(url), 20000);
                    } catch (e) { console.error(e); alert('Unable to open SVG in new tab.'); }
                });

                document.getElementById('downloadSvgBtn').addEventListener('click', function() {
                    try {
                        const svg = getSvg();
                        const blob = new Blob([svg], {type: 'image/svg+xml;charset=utf-8'});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'TCN_MLP_Architecture.svg';
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        setTimeout(() => URL.revokeObjectURL(url), 20000);
                    } catch (e) { console.error(e); alert('Unable to download SVG.'); }
                });

                // PNG export: render SVG into canvas then save as PNG
                document.getElementById('downloadPngBtn')?.addEventListener('click', function() {
                    try {
                        const svg = getSvg();
                        // determine dimensions from <svg> attributes or viewBox
                        let width = 1200, height = 800;
                        try {
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(svg, 'image/svg+xml');
                            const svgEl = doc.documentElement;
                            const wAttr = svgEl.getAttribute('width');
                            const hAttr = svgEl.getAttribute('height');
                            if (wAttr && hAttr) {
                                width = parseFloat(wAttr);
                                height = parseFloat(hAttr);
                            } else {
                                const vb = svgEl.getAttribute('viewBox');
                                    if (vb) {
                                    const parts = vb.replace(/,/g,' ').trim().split(/\\s+/);
                                    if (parts.length === 4) {
                                        width = parseFloat(parts[2]);
                                        height = parseFloat(parts[3]);
                                    }
                                }
                            }
                        } catch (e) { /* fallback to defaults */ }

                        const svgBlob = new Blob([svg], {type: 'image/svg+xml;charset=utf-8'});
                        const url = URL.createObjectURL(svgBlob);
                        const img = new Image();
                        img.crossOrigin = 'anonymous';
                        img.onload = function() {
                            try {
                                const canvas = document.createElement('canvas');
                                canvas.width = Math.max(1, Math.round(width));
                                canvas.height = Math.max(1, Math.round(height));
                                const ctx = canvas.getContext('2d');
                                // white background
                                ctx.fillStyle = '#ffffff';
                                ctx.fillRect(0, 0, canvas.width, canvas.height);
                                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                                canvas.toBlob(function(blob) {
                                    if (!blob) { alert('PNG export failed'); return; }
                                    const a = document.createElement('a');
                                    const pngUrl = URL.createObjectURL(blob);
                                    a.href = pngUrl;
                                    a.download = 'TCN_MLP_Architecture.png';
                                    document.body.appendChild(a);
                                    a.click();
                                    a.remove();
                                    setTimeout(() => URL.revokeObjectURL(pngUrl), 20000);
                                }, 'image/png');
                            } catch (err) { console.error(err); alert('PNG conversion failed.'); }
                            URL.revokeObjectURL(url);
                        };
                        img.onerror = function(e) { console.error(e); alert('Failed to load SVG for PNG conversion.'); URL.revokeObjectURL(url); };
                        img.src = url;
                    } catch (e) { console.error(e); alert('Unable to export PNG.'); }
                });

            })();
            </script>
            """
            html_actions = html_actions.replace("{SVG_ENCODED}", svg_encoded)
            # Increase actions area height so stacked buttons fit on small screens
            components.html(html_actions, height=180)
        except Exception:
            # Fallback: if data URL creation fails, show simple links/instructions
            st.info("Unable to create download/open links for the SVG. You can still view it in the app.")
    else:
        st.info("Architecture diagram not found at TCN_MLP_Architecture.svg")

    # Explicit mobile zoom note for users
    st.markdown("""
If you want to zoom in on mobile, convert the page to desktop view in your mobile browser.
""")

    st.markdown("""
The model has three working parts before the final prediction layer:

1. The **TCN branch** extracts time-aware climate features from the monthly sequence.
2. The **context branches** turn region, crop, and year into learned feature vectors.
3. The **MLP head** fuses everything and produces the final yield estimate.
    """)
    
    # Architecture Overview
    st.subheader("🔍 Architecture Overview")
    
    st.markdown("""
The architecture is intentionally split into specialized branches so each type of information is handled in the most suitable way.

### **1. Temporal Convolutional Network (TCN)**
- **Purpose**: Learn how climate changes over the 12-month sequence affect yield.
- **How it works**:
  - Starts with a causal Conv1D layer so the model only reads current and past months.
  - Uses batch normalization and dropout to stabilize training and reduce overfitting.
  - Adds two dilated residual blocks to expand the receptive field without losing the original signal.
  - Ends with a 1x1 convolution and dual pooling to compress the sequence into a compact temporal summary.

### **2. Context Branches**
- **Purpose**: Encode the non-sequence information the TCN cannot infer on its own.
- **How it works**:
  - Region and crop IDs are converted into embeddings so the model learns similarity relationships instead of treating IDs as numbers.
  - Year is represented as a small trend vector using the normalized year plus sine and cosine terms.
  - Each branch is flattened or transformed into a vector that can be merged with the TCN output.

### **3. Multi-Layer Perceptron (MLP)**
- **Purpose**: Combine all branches and learn the final nonlinear mapping to yield.
- **How it works**:
  - Concatenates the temporal summary, region embedding, crop embedding, and year features.
  - Uses dense layers with ReLU, batch normalization, and dropout to learn cross-feature interactions.
  - Produces a single linear regression output in transformed yield space.

### **4. Ensemble Approach**
- **Multiple Models**: 5-fold cross-validation creates diverse models.
- **Ensemble Aggregation**: Predictions are averaged across folds for robustness.
- **Confidence Intervals**: Ensemble variance provides prediction uncertainty.
    """)
    
    st.markdown("---")

    st.subheader("📊 Data Flow")

    st.markdown("""
```text
Sequence input (12 monthly climate steps)
        ↓
TCN branch: causal Conv1D → residual dilated blocks → pooling
        ↓
Context branches: region embedding, crop embedding, year trend
        ↓
Feature fusion: concatenate all learned vectors
        ↓
MLP head: dense layers + normalization + dropout
        ↓
Linear output: normalized log-yield
        ↓
Post-processing: invert normalization and exponentiate to kg/ha
```
    """)

    st.markdown("---")

    st.subheader("🧱 Branch-by-Branch Breakdown")

    tab_tcn, tab_context, tab_mlp, tab_output = st.tabs([
        "TCN Branch",
        "Context Branches",
        "MLP Head",
        "Output & Ensemble",
    ])

    with tab_tcn:
        st.markdown("""
#### TCN branch blocks and functions

- **Sequence Input**: receives the 12-step climate sequence in time order.
- **Conv1D Causal**: detects local weather patterns while preventing future leakage.
- **BatchNorm + Dropout**: keeps activations stable and limits overfitting.
- **Residual Block 1**: uses dilation rate 2 to widen the receptive field while keeping the shortcut path intact.
- **Residual Block 2**: uses dilation rate 4 to capture slower seasonal effects and delayed crop response.
- **Conv1D Pointwise**: compresses channel depth without mixing time positions.
- **Global Average Pooling**: summarizes the overall seasonal signal.
- **Global Max Pooling**: preserves the strongest month-level signal.
- **Concatenate Pools**: combines the smooth and peak summaries into one TCN vector.

The TCN is the temporal feature extractor. It is the part that turns raw climate sequences into a compact representation of weather evolution across the growing season.
        """)

    with tab_context:
        st.markdown("""
#### Context branches and functions

- **Region Input → Region Embedding → Flatten**: learns a compact representation for geographic context.
- **Crop Input → Crop Embedding → Flatten**: learns crop-specific response patterns.
- **Year Input**: uses normalized year plus sine and cosine terms to encode long-term drift and cyclical structure.
- **Dense + ReLU on year**: turns the year signal into a richer trend feature.
- **Dropout on year branch**: keeps the model from overusing year as a shortcut.

These branches give the network the static or slowly changing information it needs so the TCN does not have to infer everything from climate alone.
        """)

    with tab_mlp:
        st.markdown("""
#### MLP head blocks and functions

- **Concatenate All Features**: joins the TCN vector, region embedding, crop embedding, and year vector into one fused feature space.
- **Dense + ReLU (40 units)**: learns interactions between climate, crop, region, and time.
- **BatchNorm**: stabilizes the fused representation.
- **Dropout**: reduces overfitting after feature fusion.
- **Dense + ReLU (20 units)**: compresses the representation into a smaller latent form.
- **Dropout**: adds a second regularization step before prediction.
- **Dense Linear Output**: produces the final normalized log-yield value.

The MLP is the decision-making stage. It is where the model learns how to combine all learned features into a single regression prediction.
        """)

    with tab_output:
        st.markdown("""
#### Output path and ensemble behavior

- The raw output is a **normalized log-yield**, not the final kg/ha value.
- During evaluation, the pipeline reverses the crop-level normalization and applies `exp()` to recover yield in kg/ha.
- The 5 folds are kept as separate models, and their predictions are averaged to form the final ensemble estimate.
- The ensemble spread is used as an uncertainty cue, so the app can show a confidence interval around the prediction.

This makes the model more robust than a single fit because each fold sees a slightly different training split and learns a slightly different view of the problem.
        """)

    st.markdown("---")

    st.subheader("🎯 Key Design Choices")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### Why the TCN branch?
- Preserves temporal order in the monthly sequence.
- Uses dilated convolutions to capture both local shocks and longer seasonal dependencies.
- Keeps the number of parameters manageable while still learning deep temporal structure.
- Produces a fixed-size summary that is easy to fuse with context features.
        """)

    with col2:
        st.markdown("""
#### Why the MLP and ensemble?
- The MLP can learn nonlinear interactions after the branches are fused.
- Batch normalization and dropout make the head more stable and less overfit-prone.
- The ensemble reduces variance and improves generalization.
- Fold averaging gives a more reliable prediction than a single model checkpoint.
        """)

    st.markdown("---")

    st.subheader("📈 Model Performance")
    
    data = load_data()
    ensemble_metrics = data.get("ensemble_metrics") if isinstance(data.get("ensemble_metrics"), pd.DataFrame) else pd.DataFrame()

    if not ensemble_metrics.empty:
        avg_r2 = ensemble_metrics["Ensemble_R2"].mean()
        avg_mae = ensemble_metrics["Ensemble_MAE"].mean()
        avg_mape = ensemble_metrics["Ensemble_MAPE"].mean()
        avg_mase = ensemble_metrics["Ensemble_MASE"].mean()
    else:
        avg_r2 = avg_mae = avg_mape = avg_mase = None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Avg R² Score",
            value=f"{avg_r2:.4f}" if avg_r2 is not None else "N/A",
            help="Average coefficient of determination across 5 folds"
        )
    with col2:
        st.metric(
            label="Avg MAE",
            value=f"{avg_mae:.1f} kg/ha" if avg_mae is not None else "N/A",
            help="Average mean absolute error across 5 folds"
        )
    with col3:
        st.metric(
            label="Avg MAPE",
            value=f"{avg_mape:.2f}%" if avg_mape is not None else "N/A",
            help="Average mean absolute percentage error"
        )
    with col4:
        st.metric(
            label="Avg MASE",
            value=f"{avg_mase:.3f}" if avg_mase is not None else "N/A",
            help="Average mean absolute scaled error"
        )
    
    st.markdown("---")
    
    st.subheader("💡 Feature Engineering")
    
    st.markdown("""
The architecture relies on a small set of engineered inputs that make the branches more informative:

1. **Temporal sequence features**: the 12-step climate window processed by the TCN.
2. **Region context**: categorical region IDs learned through embeddings.
3. **Crop context**: categorical crop IDs learned through embeddings.
4. **Year trend features**: normalized year, sine, and cosine values for drift and cycle awareness.
5. **Fusion-ready vectors**: every branch ends in a compact dense representation so the MLP can combine them cleanly.
    """)


if __name__ == "__main__":
    render()
