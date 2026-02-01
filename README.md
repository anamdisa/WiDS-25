WiDS Final Project: Data-Driven Carbon Footprint Analytics 
Complete pipeline for analyzing emissions data using Machine Learning and comparing with sustainability commitments using Natural Language Processing.
Check for final codes, data, etc in Week-4 Folder.

**Project Overview**

This project consists of three integrated tasks:

1. **Task 1**: ML-based emission prediction using EDGAR CO2 emissions dataset
2. **Task 2**: NLP-based extraction of sustainability commitments from company reports
3. **Task 3**: Gap analysis comparing ML predictions with stated commitments

**Directory Structure**

```
data/
├── data_edgar.csv                     Input: EDGAR emissions dataset
├── reports/                           Input: PDF of sustainability reports
│   ├── BASF_report.pdf
│   └── ...
├── task1_outputs/                     Task 1 ML outputs
│   ├── task1_model_metrics.csv
│   ├── task1_yearly_aggregated.csv
│   ├── task1_trend_summary.csv
│   ├── task1_predictions_detailed.csv
│   └── *.png (visualizations)
├── nlp_output/                        Task 2 NLP outputs
│   ├── all_commitments_combined.csv
│   ├── *_commitments.csv
│   ├── *_wordcloud.png
│   └── *.txt (extracted text)
└── task3_outputs/                     Task 3 Gap Analysis outputs
    ├── task3_gap_analysis.csv
    ├── task3_future_projections.csv
    ├── task3_analysis_summary.csv
    ├── task3_gap_analysis_report.txt
    └── *.png (visualizations)
```

**Installation**

 Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install xgboost shap                                       #recommended for Task 1
pip install spacy pdfplumber wordcloud                         #Required for Task 2
python -m spacy download en_core_web_sm   Download spaCy model
```

**Usage Guide**

Task 1: ML Emission Prediction

**Purpose**: Train ML models to predict future emissions based on historical data

**Input**: 
- `data/data_edgar.csv` - EDGAR CO2 emissions dataset

**Run**: Paste in Terminal
```
python task1.py
```

**What it does**:
1. Loads and filters data for BASF operating countries
2. Engineers time-series features (lags, rolling averages, trends)
3. Trains 3 models: Ridge Regression, Random Forest, XGBoost
4. Evaluates with RMSE, MAE, R² metrics
5. Extracts feature importance
6. Generates predictions and trend analysis
7. Saves outputs for Task 3 integration

**Key Outputs**:
- `task1_yearly_aggregated.csv` - Yearly emission totals and predictions
- `task1_trend_summary.csv` - Trend statistics (slope, R², avg change)
- `task1_model_evaluation.png` - Comprehensive model comparison
- `task1_emission_trends.png` - Historical and predicted trends

---

 Task 2: NLP Commitment Extraction

**Purpose**: Extract sustainability commitments from company ESG reports

**Input**: 
- `data/reports/*.pdf` - PDF sustainability reports

**Setup**:
```bash
 Create reports directory and add PDFs
mkdir -p data/reports
 Place your PDF files in data/reports/
```

**Run**:
```bash
python Task2.py
```

**What it does**:
1. Extracts text from PDF reports using pdfplumber
2. Cleans and tokenizes text using spaCy
3. Identifies emission-related sentences
4. Extracts structured commitment statements
5. Generates word clouds and keyword frequency plots
6. Saves commitment tables with target years and reduction percentages

**Key Outputs**:
- `all_commitments_combined.csv` - All extracted commitments
- `*_commitments.csv` - Individual company commitments
- `*_wordcloud.png` - Emission keyword visualizations
- `*_emission_sentences.txt` - Relevant text excerpts

**Commitment Format**:
```
| Company | Commitment | Target Year | Reduction % | Metric |
|---------|-----------|-------------|-------------|---------|
| BASF    | Reduce CO2 by 25% by 2030 | 2030 | 25% | CO2 |
```

---

 Task 3: Gap Analysis

**Purpose**: Compare ML-predicted emission trends with stated sustainability commitments

**Prerequisites**:
- Task 1 must be completed (outputs in `data/task1_outputs/`)
- Task 2 must be completed (outputs in `data/nlp_output/`)

**Run**:
```bash
python task3_gap_analysis.py
```

**What it does**:
1. Loads ML predictions and trend data from Task 1
2. Loads NLP-extracted commitments from Task 2
3. Projects future emissions using ML trend slope
4. Compares ML-predicted reductions with stated commitments
5. Calculates gaps for each commitment
6. Classifies alignment status (Aligned/Caution/Gap)
7. Generates comprehensive visualizations and reports

**Gap Analysis Logic**:
```python
 For each commitment:
Stated_Reduction = 25%   From NLP extraction
ML_Predicted_Reduction = 15%   From ML projection
Gap = Stated - Predicted = 10%   Shortfall

 Status Classification:
Gap ≤ 5%   →  ALIGNED
Gap 5-15%  →  CAUTION  
Gap > 15%  →  GAP (significant shortfall)
```

**Key Outputs**:
- `task3_gap_analysis.csv` - Detailed gap analysis for each commitment
- `task3_future_projections.csv` - ML projections to 2025-2050
- `task3_analysis_summary.csv` - Aggregate statistics
- `task3_gap_analysis_report.txt` - Human-readable report
- `task3_comprehensive_analysis.png` - 4-panel dashboard
- `task3_emission_trajectory.png` - Timeline showing gaps

---

Understanding the Results

 Model Performance (Task 1)

Look for:
- **R² close to 1.0** = Excellent model fit
- **Low RMSE/MAE** = Accurate predictions
- **Feature importance** = What drives emissions

 Commitment Extraction (Task 2)

Verify:
- Target years are correctly extracted (e.g., 2030, 2050)
- Reduction percentages are accurate
- Commitments align with report content

 Gap Analysis (Task 3)

Interpret:
- **Positive Gap** = ML predicts less reduction than stated (shortfall)
- **Negative Gap** = ML predicts more reduction than stated (surplus)
- **Large Gaps (>15%)** = Commitments may be unrealistic given current trends

**Example**:
```
Company: BASF
Target: 2030 (12 years from baseline)
Stated: 25% reduction
ML Predicted: 12% reduction  
Gap: 13% shortfall → ⚠ CAUTION
```

This means based on current trends, BASF is projected to achieve only 12% reduction, falling short of their 25% target by 13 percentage points.

---

Customization

 Modify Target Years
Edit `task3_gap_analysis.py`:
```python
target_years = [2025, 2030, 2035, 2040, 2045, 2050]   Add/remove years
```

 Change Countries (Task 1)
Edit `task1.py`:
```python
BASF_COUNTRIES = [
    'Germany',
    'United States',
     Add more countries
]
```

 Adjust Gap Thresholds (Task 3)
Edit `task3_gap_analysis.py`:
```python
if gap <= 5:   Change threshold
    status = "✓ ALIGNED"
elif gap <= 15:   Change threshold
    status = "⚠ CAUTION"
```

 Add More Keywords (Task 2)
Edit `Task2.py`:
```python
EMISSION_KEYWORDS = [
    'co2', 'carbon', 'emission',
     Add more keywords
    'renewable', 'clean energy'
]
```

---

Troubleshooting

 Task 1 Issues

**Problem**: "File not found: data/data_edgar.csv"
```bash
 Solution: Ensure dataset is in correct location
cp data.csv data/data_edgar.csv
```

**Problem**: "XGBoost not available"
```bash
 Solution: Install XGBoost (optional)
pip install xgboost
```

 Task 2 Issues

**Problem**: "No PDF files found"
```bash
 Solution: Create directory and add PDFs
mkdir -p data/reports
 Then add your PDF files
```

**Problem**: "spaCy model not found"
```bash
 Solution: Download spaCy language model
python -m spacy download en_core_web_sm
```

 Task 3 Issues

**Problem**: "Could not load Task 1 outputs"
```bash
 Solution: Run Task 1 first
python task1.py
```

**Problem**: "No commitments found"
```bash
 Solution: Run Task 2 first or check PDF content
python Task2.py
```

---

Sample Output Interpretation

 Best Case Scenario
```
Gap Analysis:
  BASF (2030): Gap = 2% → ✓ ALIGNED
  Status: Current trends support stated commitments
```

 Moderate Gap
```
Gap Analysis:
  BASF (2030): Gap = 12% → ⚠ CAUTION
  Status: Additional measures needed to meet target
```

 Significant Gap
```
Gap Analysis:
  BASF (2030): Gap = 22% → ❌ GAP
  Status: Substantial changes required to achieve commitment
```

---

Notes

1. **Baseline Year**: Task 3 uses the latest year in the dataset as baseline (typically 2018)
2. **Linear Projection**: Uses simple linear trend - real-world may be non-linear
3. **Data Quality**: Results depend on quality of both emissions data and PDF reports
4. **Commitment Parsing**: Regex-based, may miss complex phrasing
5. **Scope**: Analysis focuses on total emissions across all sectors

---

 Expected Outcomes

After running all three tasks, you should have:

1. ✅ Trained ML models with performance metrics
2. ✅ Extracted sustainability commitments from reports
3. ✅ Gap analysis showing alignment between predictions and commitments
4. ✅ Comprehensive visualizations and reports
5. ✅ Data-driven insights for sustainability planning

---

References

- EDGAR CO2 Emissions: https://github.com/openclimatedata/edgar-co2-emissions
- BASF Sustainability: https://www.basf.com/global/en.html

---

Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure input files are in correct locations
4. Check console output for specific error messages

---

**Last Updated**: February 2026

