# TCN-MLP Architecture Diagram

## Mermaid Code

```mermaid
graph TD
    %% Inputs
    SEQ["Sequence Input\n(12, 3) months x features\nT2M_AVG, PRECTOTCORR, GWETROOT"]
    REGION_IN["Region Input\n6 regions"]
    CROP_IN["Crop Input\n4 crops"]
    YEAR_IN["Year Input\nPolynomial + interactions"]
    
    %% TCN Branch
    NOISE["Gaussian Noise\nσ = 0.05"]
    CONV["Causal Conv1D\n28 filters, kernel=3\npadding: causal"]
    BN1["BatchNorm + Dropout\ndropout = 0.25"]
    
    ATTN["Multi-Head Attention\n4 heads, key_dim=8"]
    RES_ADD["Residual Add + BatchNorm\nx = x + attention(x)"]
    GAP["Global Avg Pooling\nOutput: 28-dim"]
    
    %% Categorical Branches
    REGION_EMB["Region Embedding\n6 -> 7 dims"]
    REGION_FLAT["Flatten"]
    
    CROP_EMB["Crop Embedding\n4 -> 4 dims"]
    CROP_FLAT["Flatten"]
    
    %% Trend Branch
    TREND_DENSE["Dense(16)\nReLU + BatchNorm + Dropout"]
    
    %% Concatenation
    CONCAT["Concatenate All\n[28 + 7 + 4 + 16] = 55-dim"]
    
    %% MLP Head
    DENSE1["Dense(20)\nReLU + BatchNorm + Dropout\nL2=1e-3"]
    DENSE2["Dense(14)\nReLU + BatchNorm + Dropout\nL2=1e-3"]
    OUTPUT["Output Neuron\nLinear, bias=7.5\nPredicts log-yield"]
    
    %% Transform
    EXP["Exponential Transform\nexp(output) -> yield (kg/ha)"]
    
    %% Final Output
    FINAL["Final Prediction\nYield (kg/ha)\nTest R2=0.889, MAE=721.99"]
    
    %% Connections
    SEQ --> NOISE
    NOISE --> CONV
    CONV --> BN1
    BN1 --> ATTN
    ATTN --> RES_ADD
    RES_ADD --> GAP
    
    REGION_IN --> REGION_EMB
    REGION_EMB --> REGION_FLAT
    
    CROP_IN --> CROP_EMB
    CROP_EMB --> CROP_FLAT
    
    YEAR_IN --> TREND_DENSE
    
    GAP --> CONCAT
    REGION_FLAT --> CONCAT
    CROP_FLAT --> CONCAT
    TREND_DENSE --> CONCAT
    
    CONCAT --> DENSE1
    DENSE1 --> DENSE2
    DENSE2 --> OUTPUT
    OUTPUT --> EXP
    EXP --> FINAL
    
    %% Styling
    classDef input fill:#1f77b4,stroke:#fff,stroke-width:2px,color:#fff
    classDef tcn fill:#9467bd,stroke:#fff,stroke-width:2px,color:#fff
    classDef categorical fill:#2ca02c,stroke:#fff,stroke-width:2px,color:#fff
    classDef trend fill:#ff7f0e,stroke:#fff,stroke-width:2px,color:#fff
    classDef mlp fill:#c85a54,stroke:#fff,stroke-width:2px,color:#fff
    classDef output fill:#17becf,stroke:#fff,stroke-width:3px,color:#fff
    classDef helper fill:#6c757d,stroke:#fff,stroke-width:1.5px,color:#fff
    
    class SEQ,REGION_IN,CROP_IN,YEAR_IN input
    class NOISE,CONV,BN1,ATTN,RES_ADD,GAP tcn
    class REGION_EMB,REGION_FLAT,CROP_EMB,CROP_FLAT categorical
    class TREND_DENSE trend
    class DENSE1,DENSE2,OUTPUT mlp
    class FINAL output
    class CONCAT helper
    class EXP helper
```

## Architecture Details

| Component | Details |
|-----------|---------|
| **TCN Branch** | Causal Conv1D (28 filters) → BatchNorm/Dropout → Multi-Head Attention (4 heads) → GlobalAvgPooling |
| **Categorical** | Region Embedding (6→7) + Crop Embedding (4→4) + Flattening |
| **Trend Branch** | Dense(16) with ReLU + BatchNorm + Dropout |
| **MLP Head** | Dense(20) → Dense(14) → Output(1, linear) |
| **Output Transform** | exp() to convert log-yield to kg/ha |

## Training Configuration

- **Loss Function**: Huber (δ=0.2) - robust to outliers
- **Optimizer**: AdamW (lr=8e-4, weight_decay=2e-4)
- **Regularization**: L2=1e-3, Dropout=0.25
- **Data Augmentation**: Mixup (α=0.3, 40 samples)
- **Dataset**: 510 train / 90 val / 90 test
- **CV**: 5-fold stratified (0.829 ± 0.113 R²)

## Performance

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| **R² Score** | 0.857 | 0.829 | 0.889 |
| **MAE (kg/ha)** | - | 867.91 | 721.99 |
