# Model Card: Ship Engine Anomaly Detection

## Model Details

### Model Description

This system uses two complementary unsupervised machine learning models for anomaly detection in ship engine sensor data:

1. **One-Class SVM (OCSVM)** - Support Vector Machine trained on normal data to define a decision boundary
2. **Isolation Forest** - Tree-based ensemble that isolates anomalies through random partitioning

Both models use **symbolic regression features** - interpretable mathematical equations automatically discovered from data.

### Model Version

- Version: v1.0
- Training Date: 2025
- Framework: scikit-learn 1.3+

## Intended Use

### Primary Use Cases

- **Predictive Maintenance**: Detect early signs of equipment degradation
- **Real-time Monitoring**: Flag unusual operating conditions during voyages
- **Historical Analysis**: Identify anomalous periods in logged sensor data

### Out-of-Scope Uses

- Safety-critical autonomous decision making (requires human oversight)
- Different equipment types without retraining
- Diagnosis of specific failure modes (detection only, not diagnosis)

## Training Data

### Dataset Characteristics

- **Source**: Ship engine monitoring system
- **Samples**: 19,535 observations
- **Features**: 6 raw sensor readings
- **Labels**: Unsupervised (no ground truth anomaly labels)

### Feature Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Engine RPM | ~700 | ~150 | 400 | 1200 |
| Oil Pressure | ~3.5 | ~1.5 | 0.5 | 8.0 |
| Fuel Pressure | ~8.0 | ~5.0 | 1.0 | 30.0 |
| Coolant Pressure | ~2.5 | ~1.0 | 0.5 | 6.0 |
| Oil Temp | ~80 | ~10 | 50 | 120 |
| Coolant Temp | ~75 | ~8 | 55 | 100 |

### Data Preprocessing

1. Column name normalization
2. Physical range validation
3. Symbolic feature computation (4 derived features)
4. Standard scaling (OCSVM only)

## Model Architecture

### Feature Engineering: Symbolic Regression

Instead of hand-crafted features, we use equations discovered through symbolic regression:

```
sym_oil_temp = log(coolant_temp × √engine_rpm) + 69.69828
sym_coolant_temp = √oil_temp + 69.59798
sym_oil_pressure = exp(exp(√(√(log(engine_rpm) × 0.827) / oil_temp)))
sym_fuel_pressure = 40.246 / (1619.76 - engine_rpm) + 6.426
```

These equations capture physical relationships between variables.

### OCSVM Model

- **Kernel**: Radial Basis Function (RBF)
- **Nu**: 0.02 (expected anomaly proportion)
- **Gamma**: 0.2 (kernel coefficient)
- **Preprocessing**: StandardScaler

### Isolation Forest Model

- **Estimators**: 300 trees
- **Contamination**: 0.02 (expected anomaly proportion)
- **Max Samples**: auto
- **Preprocessing**: None (works on raw symbolic features)

## Performance

### Training Metrics

| Model | Anomaly Rate | Decision Boundary |
|-------|--------------|-------------------|
| OCSVM | 2.00% | Distance from hyperplane |
| Isolation Forest | 2.00% | Isolation depth |

### Inference Performance

- Single prediction: <20ms
- Batch (1000 samples): <500ms
- Memory footprint: ~50MB loaded

### Limitations

1. **No Ground Truth**: Evaluated through unsupervised metrics only
2. **Distribution Shift**: May need retraining if engine operating conditions change significantly
3. **Rare Anomalies**: Very rare failure modes may not be detected
4. **Threshold Sensitivity**: Detection rate depends on configured thresholds

## Evaluation

### Methodology

Without labeled anomalies, we use:

1. **Decision Margin Analysis**: Separation between normal/anomaly scores
2. **Density Contrast**: Local density differences (kNN-based)
3. **Model Agreement**: Jaccard similarity between OCSVM and IF predictions
4. **Parameter Robustness**: Stability across hyperparameter sweeps

### Cross-Model Validation

| Metric | Value |
|--------|-------|
| Jaccard Agreement | 0.44 |
| Spearman Correlation | 0.38 |

Models show moderate agreement, suggesting complementary detection capabilities.

## Ethical Considerations

### Bias Risks

- Training data from a single vessel/engine type may not generalize
- Seasonal/environmental factors not explicitly modeled
- Operator behavior patterns could affect sensor readings

### Mitigation Strategies

- Confidence scores allow human review of uncertain predictions
- Ensemble mode requires model agreement for high-confidence flags
- Clear documentation of model limitations

## Deployment

### Requirements

- Python 3.11+
- scikit-learn 1.3+
- ~50MB memory for models
- CPU inference (no GPU required)

### Monitoring Recommendations

1. Track prediction distribution over time
2. Monitor model confidence trends
3. Log anomaly rates by time period
4. Set up alerts for sustained high anomaly rates

## Maintenance

### Retraining Triggers

- Significant change in operating conditions
- New equipment installation
- Sustained drift in sensor readings
- False positive/negative feedback accumulation

### Update Procedure

1. Collect new training data (normal operating conditions)
2. Run `train_models.py` with new data
3. Validate on held-out data
4. Deploy with A/B testing
5. Monitor production metrics

## Citation

If using this model, please cite:

```
Ship Engine Anomaly Detection System
Feature Engineering Comparison Using Symbolic Regression
2025
```

## Contact

For questions about this model, please contact the development team.
