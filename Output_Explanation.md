# Gap Analysis Output Explanation

## Overview

This document explains the gap analysis results, specifically addressing why gaps can exceed 100% and how to interpret these findings.

---

## Understanding Gaps Greater Than 100%

### Mathematical Foundation

The gap metric is calculated as:

```
Gap (%) = Stated_Reduction (%) - ML_Predicted_Change (%)
```

Where:
- `Stated_Reduction`: Target emission reduction percentage from sustainability commitments
- `ML_Predicted_Change`: Projected emission change from ML model (can be positive or negative)

### Why Gaps Exceed 100%

Gaps exceeding 100% occur when the ML model predicts emission increases while commitments specify emission reductions. This represents scenarios where current trajectory and target objectives move in opposite directions.

**Example:**
```
Baseline Emissions (2018): 100 units
ML Projection (2050): 141.5 units (+41.5% increase)
Commitment Target (2050): 0 units (-100% reduction, net-zero)

Gap = -100% - (+41.5%) = -141.5%
Absolute Gap = 141.5%
```

---

## Linear Regression and Trend Inertia

### Model Behaviour

The linear regression model is trained on historical emission data spanning 1970-2018. During this period, emissions in BASF operating countries demonstrated a consistent upward trend. Given 48 years of increasing emissions and zero instances of sustained reduction in the training data, the model projects continued growth. This is not a model error but expected statistical Behaviour, as the model has no basis to predict trend reversal without data demonstrating such reversals.

**Projection Formula:**
```
Projected_Emissions(t) = Baseline_Emissions + (slope × years_from_baseline)

Where slope > 0 (positive trend from historical data)
```

---

## Interpreting Gap Analysis Results

### Gap Categories

Based on magnitude and direction alignment:

| Gap Range | Classification | Interpretation |
|-----------|----------------|----------------|
| 0-10% | Minor Gap | Current trajectory largely aligned with commitment |
| 10-30% | Moderate Gap | Same direction, insufficient reduction rate |
| 30-70% | Major Gap | Significant misalignment requiring intervention |
| >70% | Critical Gap | Opposite directions or extreme target differential |

### BASF 2050 Net-Zero Commitment

**Data:**
- Baseline (2018): 100 units
- ML Projection (2050): 141.5 units
- Commitment: 0 units (net-zero)
- Gap: 141.5 percentage points

**Analysis:**

1. **Historical Context**: Emissions increased consistently over 1970-2018 period
2. **Model Projection**: Extrapolates growth trend, predicting +41.5% by 2050
3. **Commitment Target**: Complete emission elimination (-100%)
4. **Gap Interpretation**: 141.5 percentage point difference between projected and target states

**Implication:** Achieving net-zero requires:
- Reversing 48-year growth trend
  - Implementing 41.5 percentage points of reduction to reach baseline
- Additional 100 percentage points of reduction to reach zero
- Total transformation: 141.5 percentage points

---

## Model Limitations

### Data Constraints

The ML model operates under the following constraints:

1. **Temporal Scope**: Training data ends in 2018
2. **Information Boundary**: No knowledge of post-2018 developments including:
   - Policy changes 
   - Technology adoption (e.g., renewable energy scale-up)
   - Corporate sustainability initiatives
   - Market transformations
   - Regulatory changes
   - Introduction of Carbon Credits
3. What the Model Cannot Capture
  - Future policy interventions
  - Technological breakthroughs
  - Behavioural changes
  - Economic restructuring
  - Sustainability transformations implemented after 2018

---

## Gap Analysis Interpretation Framework

### The Gap Represents

**Distance Metric**: The gap quantifies the total transformation required to move from the projected trajectory (based on historical patterns) to the target state (based on commitments).

**Components:**
1. **Trend Reversal**: Effort to stop current trajectory
2. **Target Achievement**: Additional effort to reach committed level
3. **Total Distance**: Sum of both components

### The Gap Does NOT Represent

- **Feasibility Assessment**: Large gaps do not imply impossibility
- **Model Prediction Error**: Gaps reflect target ambition vs. historical trends
- **Timeline Likelihood**: Does not assess probability of achievement
- **Resource Requirements**: Does not quantify cost or resources needed

---

## Conclusion

Gaps exceeding 100% are mathematically valid outcomes when:
1. Historical data shows consistent emission growth
2. Linear regression extrapolates this growth forward
3. Commitments specify aggressive reductions or elimination

These gaps provide a quantitative measure of the transformation challenge, indicating the magnitude of change required to align historical trajectories with future sustainability targets. The analysis does not assess feasibility but rather quantifies the distance between projected and committed states, serving as a planning and prioritization tool for sustainability initiatives.

---

## References

- Dataset: EDGAR CO2 Emissions (1970-2018)
- Methodology: Linear Regression with temporal features
- Baseline: 2018 emission levels
- Scope: BASF operating countries (aggregated emissions across all sectors)
