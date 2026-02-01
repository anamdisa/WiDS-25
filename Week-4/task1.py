"""
WiDS Week 4 - Task 1: Predict Future Emissions Using Machine Learning
Dataset: EDGAR CO2 Emissions Dataset
Focus: BASF Operating Countries (Germany, USA, China, Belgium, Japan, Netherlands, UK, India, Brazil, France)
Models: Ridge Regression, Random Forest, and XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# For XGBoost (install if needed: pip install xgboost)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# For SHAP values (install if needed: pip install shap)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Will use feature importance from Random Forest.")

print("="*80)
print("TASK 1: BASF EMISSION PREDICTION MODEL")
print("="*80)
OUTPUT_DIR = Path("data/task1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n[1] Loading EDGAR Dataset...")
df = pd.read_csv('data/data_edgar.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(10))

print(f"\nYear Range: {df['Year'].min()} to {df['Year'].max()}")
print(f"Unique Countries: {df['Name'].nunique()}")
print(f"Unique Sectors: {df['Sector'].nunique()}")
print(f"Sectors: {df['Sector'].unique()}")

# ============================================================================
# 2. FILTER FOR BASF OPERATING COUNTRIES
# ============================================================================
print("\n[2] Filtering for BASF Operating Countries...")

# BASF major operating countries
BASF_COUNTRIES = [
    'Germany',          # Headquarters
    'United States',    # Major operations
    'China',           # Major operations
    'Belgium',         # European operations
    'Japan',           # Asian operations
    'Netherlands',     # European operations
    'United Kingdom',  # European operations
    'India',           # Asian operations
    'Brazil',          # South American operations
    'France and Monaco'  # European operations (Monaco emissions negligible)
]

# Filter dataset
df_basf = df[df['Name'].isin(BASF_COUNTRIES)].copy()

print(f"\nFiltered Dataset Shape: {df_basf.shape}")
print(f"Countries included: {sorted(df_basf['Name'].unique())}")
print(f"Missing countries: {set(BASF_COUNTRIES) - set(df_basf['Name'].unique())}")

# Verify all sectors are included
print(f"\nSectors included (all): {sorted(df_basf['Sector'].unique())}")

# Check data distribution
print(f"\nData distribution by country:")
print(df_basf.groupby('Name').size())

print(f"\nData distribution by sector:")
print(df_basf.groupby('Sector').size())

# ============================================================================
# 3. DATA CLEANING AND PREPROCESSING
# ============================================================================
print("\n[3] Data Cleaning and Preprocessing...")

# Check for missing values
print(f"\nMissing values:")
print(df_basf.isnull().sum())

# Remove any missing values in Emissions
df_clean = df_basf.dropna(subset=['Emissions'])
print(f"Rows after removing missing emissions: {len(df_clean)}")

# Sort by country, sector, and year for time-series features
df_clean = df_clean.sort_values(['Name', 'Sector', 'Year']).reset_index(drop=True)

print(f"\nCleaned dataset shape: {df_clean.shape}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n[4] Feature Engineering...")

# Encode categorical variables
le_country = LabelEncoder()
le_sector = LabelEncoder()

df_clean['Country_Encoded'] = le_country.fit_transform(df_clean['Name'])
df_clean['Sector_Encoded'] = le_sector.fit_transform(df_clean['Sector'])

# Create time-based features
df_clean['Years_Since_1970'] = df_clean['Year'] - 1970
df_clean['Decade'] = (df_clean['Year'] // 10) * 10

# Create historical features (lagged emissions)
df_clean['Emissions_Lag_1'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].shift(1)
df_clean['Emissions_Lag_2'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].shift(2)
df_clean['Emissions_Lag_3'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].shift(3)

# Create rolling mean features
df_clean['Emissions_Rolling_Mean_3'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
df_clean['Emissions_Rolling_Mean_5'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# Create trend features (year-over-year change)
df_clean['Emissions_Change'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].diff()
df_clean['Emissions_Pct_Change'] = df_clean.groupby(['Name', 'Sector'])['Emissions'].pct_change()

# Fill NaN values created by lagging and differencing with 0
df_clean = df_clean.fillna(0)

print(f"Features created successfully")
print(f"Dataset shape after feature engineering: {df_clean.shape}")

# ============================================================================
# 5. PREPARE TRAIN/TEST SPLIT (TIME-BASED)
# ============================================================================
print("\n[5] Preparing Train/Test Split...")

# Define features and target
feature_cols = [
    'Country_Encoded', 
    'Sector_Encoded', 
    'Year',
    'Years_Since_1970',
    'Decade',
    'Emissions_Lag_1',
    'Emissions_Lag_2',
    'Emissions_Lag_3',
    'Emissions_Rolling_Mean_3',
    'Emissions_Rolling_Mean_5',
    'Emissions_Change',
    'Emissions_Pct_Change'
]

X = df_clean[feature_cols]
y = df_clean['Emissions']

# Time-based split: 80% train, 20% test
split_year = int(df_clean['Year'].min() + 0.8 * (df_clean['Year'].max() - df_clean['Year'].min()))
print(f"\nUsing time-based split at year {split_year}")
print(f"Training years: {df_clean['Year'].min()} - {split_year}")
print(f"Testing years: {split_year + 1} - {df_clean['Year'].max()}")

train_mask = df_clean['Year'] <= split_year
test_mask = df_clean['Year'] > split_year

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Split ratio: {len(X_train)/(len(X_train)+len(X_test)):.2%} train / {len(X_test)/(len(X_train)+len(X_test)):.2%} test")

# ============================================================================
# 6. TRAIN MACHINE LEARNING MODELS
# ============================================================================
print("\n[6] Training Machine Learning Models...")

models = {}
predictions = {}

# Model 1: Ridge Regression
print("\n  [Model 1/3] Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
models['Ridge Regression'] = ridge_model
predictions['Ridge Regression'] = ridge_model.predict(X_test)
print("  ✓ Ridge Regression trained")

# Model 2: Random Forest
print("  [Model 2/3] Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_model.predict(X_test)
print("  ✓ Random Forest trained")

# Model 3: XGBoost (if available)
if XGBOOST_AVAILABLE:
    print("  [Model 3/3] Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_model.predict(X_test)
    print("  ✓ XGBoost trained")
else:
    print("  [Model 3/3] XGBoost not available - skipping")

print(f"\n  Total models trained: {len(models)}")

# ============================================================================
# 7. EVALUATE MODELS (RMSE, MAE, R²)
# ============================================================================
print("\n[7] Evaluating Models...")

results = []
for model_name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
    
    print(f"\n  {model_name}:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    R²:   {r2:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))

# Identify best model
best_model_idx = results_df['R²'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name} (R² = {results_df.loc[best_model_idx, 'R²']:.4f})")

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[8] Extracting Feature Importance...")

# Get feature importance from Random Forest
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("="*80)
print(rf_importance.to_string(index=False))

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n[9] Creating Visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Comprehensive Model Evaluation
fig = plt.figure(figsize=(18, 12))

# Plot 1: RMSE Comparison
ax1 = plt.subplot(2, 3, 1)
metrics_df = results_df.set_index('Model')
metrics_df[['RMSE']].plot(kind='bar', ax=ax1, color='coral', alpha=0.8)
ax1.set_title('Model Performance: RMSE', fontsize=14, fontweight='bold')
ax1.set_ylabel('RMSE (Lower is Better)', fontsize=11)
ax1.set_xlabel('Model', fontsize=11)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: MAE Comparison
ax2 = plt.subplot(2, 3, 2)
metrics_df[['MAE']].plot(kind='bar', ax=ax2, color='skyblue', alpha=0.8)
ax2.set_title('Model Performance: MAE', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE (Lower is Better)', fontsize=11)
ax2.set_xlabel('Model', fontsize=11)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: R² Score Comparison
ax3 = plt.subplot(2, 3, 3)
metrics_df[['R²']].plot(kind='bar', ax=ax3, color='lightgreen', alpha=0.8)
ax3.set_title('Model Performance: R² Score', fontsize=14, fontweight='bold')
ax3.set_ylabel('R² (Higher is Better)', fontsize=11)
ax3.set_xlabel('Model', fontsize=11)
ax3.set_ylim([0, 1])
ax3.tick_params(axis='x', rotation=45)
ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='0.9 threshold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Feature Importance (Top 10)
ax4 = plt.subplot(2, 3, 4)
top_features = rf_importance.head(10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
ax4.barh(range(len(top_features)), top_features['Importance'], color=colors)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'])
ax4.set_xlabel('Importance Score', fontsize=11)
ax4.set_title('Top 10 Features (Random Forest)', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# Plot 5-6: Actual vs Predicted (best 2 models)
best_models = results_df.nlargest(2, 'R²')['Model'].tolist()

for idx, model_name in enumerate(best_models, start=5):
    ax = plt.subplot(2, 3, idx)
    y_pred = predictions[model_name]
    
    # Sample for visualization (max 2000 points)
    if len(y_test) > 2000:
        sample_idx = np.random.choice(len(y_test), 2000, replace=False)
        y_test_sample = y_test.iloc[sample_idx]
        y_pred_sample = y_pred[sample_idx]
    else:
        y_test_sample = y_test
        y_pred_sample = y_pred
    
    ax.scatter(y_test_sample, y_pred_sample, alpha=0.4, s=15, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(y_test_sample.min(), y_pred_sample.min())
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.8)
    
    ax.set_xlabel('Actual Emissions', fontsize=11)
    ax.set_ylabel('Predicted Emissions', fontsize=11)
    ax.set_title(f'Actual vs Predicted: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add R² score to plot
    r2 = r2_score(y_test, predictions[model_name])
    ax.text(0.95, 0.05, f'R² = {r2:.4f}', 
            transform=ax.transAxes, 
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

plt.suptitle('BASF Emission Prediction - Model Evaluation Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('data/task1_outputs/task1_model_evaluation.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: data/task1_outputs/task1_model_evaluation.png")
plt.close()

# Figure 2: Aggregated Emission Trends
print("  Creating aggregated emission trends visualization...")

# Aggregate emissions by year (sum across all countries and sectors)
yearly_emissions = df_clean.groupby('Year')['Emissions'].sum().reset_index()

# Get predictions for full dataset (for trend visualization)
X_full = df_clean[feature_cols]
best_model = models[best_model_name]
y_pred_full = best_model.predict(X_full)

df_clean['Predicted_Emissions'] = y_pred_full

# Aggregate predictions by year
yearly_predictions = df_clean.groupby('Year')['Predicted_Emissions'].sum().reset_index()

# Create plot
fig2, ax = plt.subplots(figsize=(14, 7))

ax.plot(yearly_emissions['Year'], yearly_emissions['Emissions'], 
        'o-', label='Actual Emissions', linewidth=2.5, markersize=6, color='steelblue')
ax.plot(yearly_predictions['Year'], yearly_predictions['Predicted_Emissions'], 
        's-', label=f'Predicted Emissions ({best_model_name})', 
        linewidth=2.5, markersize=6, color='coral', alpha=0.8)

# Add vertical line for train/test split
ax.axvline(x=split_year, color='red', linestyle='--', linewidth=2, 
           label=f'Train/Test Split ({split_year})', alpha=0.7)

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Emissions (All Sectors)', fontsize=13, fontweight='bold')
ax.set_title('BASF Operating Countries: Total Emission Trends (1970-2018)', 
             fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/task1_outputs/task1_emission_trends.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: data/task1_outputs/task1_emission_trends.png")
plt.close()

# ============================================================================
# 10. SAVE RESULTS FOR TASK 3 GAP ANALYSIS
# ============================================================================
print("\n[10] Saving Results for Task 3 Integration...")

# Save 1: Model metrics comparison table
results_df.to_csv('data/task1_outputs/task1_model_metrics.csv', index=False)
print("  ✓ Saved: data/task1_outputs/task1_model_metrics.csv")

# Save 2: Feature importance
rf_importance.to_csv('data/task1_outputs/task1_feature_importance.csv', index=False)
print("  ✓ Saved: data/task1_outputs/task1_feature_importance.csv")

# Save 3: Full predictions with metadata (for Task 3)
predictions_export = df_clean[['Code', 'Name', 'Sector', 'Year', 'Emissions']].copy()
predictions_export['Predicted_Emissions'] = y_pred_full

# Add predictions from all models
for model_name in models.keys():
    predictions_export[f'Predicted_{model_name.replace(" ", "_")}'] = models[model_name].predict(X_full)

predictions_export.to_csv('data/task1_outputs/task1_predictions_detailed.csv', index=False)
print("  ✓ Saved: data/task1_outputs/task1_predictions_detailed.csv")

# Save 4: Yearly aggregated emissions (for Task 3 trend analysis)
yearly_summary = pd.DataFrame({
    'Year': yearly_emissions['Year'],
    'Actual_Total_Emissions': yearly_emissions['Emissions'],
    'Predicted_Total_Emissions': yearly_predictions['Predicted_Emissions']
})

# Calculate year-over-year change
yearly_summary['Actual_YoY_Change'] = yearly_summary['Actual_Total_Emissions'].pct_change() * 100
yearly_summary['Predicted_YoY_Change'] = yearly_summary['Predicted_Total_Emissions'].pct_change() * 100

# Calculate trend (linear fit)
from scipy import stats
years_numeric = yearly_summary['Year'].values
actual_slope, actual_intercept, _, _, _ = stats.linregress(years_numeric, 
                                                            yearly_summary['Actual_Total_Emissions'])
pred_slope, pred_intercept, _, _, _ = stats.linregress(years_numeric, 
                                                        yearly_summary['Predicted_Total_Emissions'])

yearly_summary['Actual_Trend'] = actual_slope * years_numeric + actual_intercept
yearly_summary['Predicted_Trend'] = pred_slope * years_numeric + pred_intercept

yearly_summary.to_csv('data/task1_outputs/task1_yearly_aggregated.csv', index=False)
print("  ✓ Saved: data/task1_outputs/task1_yearly_aggregated.csv")

# Save 5: Trend summary for Task 3
trend_summary = {
    'Actual_Slope_per_Year': actual_slope,
    'Predicted_Slope_per_Year': pred_slope,
    'Actual_Avg_Annual_Change_%': yearly_summary['Actual_YoY_Change'].mean(),
    'Predicted_Avg_Annual_Change_%': yearly_summary['Predicted_YoY_Change'].mean(),
    'Data_Start_Year': int(yearly_summary['Year'].min()),
    'Data_End_Year': int(yearly_summary['Year'].max()),
    'Train_Test_Split_Year': split_year,
    'Best_Model': best_model_name,
    'Best_Model_R2': float(results_df.loc[best_model_idx, 'R²']),
    'Best_Model_RMSE': float(results_df.loc[best_model_idx, 'RMSE']),
    'Best_Model_MAE': float(results_df.loc[best_model_idx, 'MAE'])
}

trend_summary_df = pd.DataFrame([trend_summary])
trend_summary_df.to_csv('data/task1_outputs/task1_trend_summary.csv', index=False)
print("  ✓ Saved: data/task1_outputs/task1_trend_summary.csv")

# ============================================================================
# 11. OPTIONAL: SHAP VALUES
# ============================================================================
if SHAP_AVAILABLE:
    print("\n[11] Computing SHAP Values...")
    try:
        # Use sample for computational efficiency
        sample_size = min(500, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)
        
        if best_model_name == 'Random Forest':
            explainer = shap.TreeExplainer(models['Random Forest'])
        elif best_model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            explainer = shap.TreeExplainer(models['XGBoost'])
        else:
            explainer = shap.TreeExplainer(rf_model)  # Fallback to RF
        
        shap_values = explainer.shap_values(X_sample)
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig('data/task1_outputs/task1_shap_summary.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: data/task1_outputs/task1_shap_summary.png")
        plt.close()
        
    except Exception as e:
        print(f"  ⚠ SHAP calculation failed: {e}")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("="*80)

print("\n📊 OUTPUTS GENERATED:")
print("  1. task1_model_metrics.csv          - Model performance comparison (RMSE, MAE, R²)")
print("  2. task1_feature_importance.csv     - Feature importance rankings")
print("  3. task1_predictions_detailed.csv   - Full predictions with metadata")
print("  4. task1_yearly_aggregated.csv      - Yearly emission trends (for Task 3)")
print("  5. task1_trend_summary.csv          - Trend statistics (for Task 3)")
print("  6. task1_model_evaluation.png       - Model evaluation dashboard")
print("  7. task1_emission_trends.png        - Emission trend visualization")
if SHAP_AVAILABLE:
    print("  8. task1_shap_summary.png           - SHAP feature importance")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"\n🎯 Best Model: {best_model_name}")
print(f"   - RMSE: {results_df.loc[best_model_idx, 'RMSE']:.4f}")
print(f"   - MAE:  {results_df.loc[best_model_idx, 'MAE']:.4f}")
print(f"   - R²:   {results_df.loc[best_model_idx, 'R²']:.4f}")

print(f"\n📈 Emission Trends:")
print(f"   - Actual average annual change: {yearly_summary['Actual_YoY_Change'].mean():.2f}%")
print(f"   - Predicted average annual change: {yearly_summary['Predicted_YoY_Change'].mean():.2f}%")
print(f"   - Data coverage: {yearly_summary['Year'].min()} - {yearly_summary['Year'].max()}")

print("\n🔝 Top 5 Most Important Features:")
for idx, row in rf_importance.head().iterrows():
    print(f"   {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "="*80)
print("READY FOR TASK 3: Gap Analysis Integration")
print("="*80)
print("\nTask 3 can use:")
print("  • task1_yearly_aggregated.csv - for emission trend slopes")
print("  • task1_trend_summary.csv - for annual % change rates")
print("  • task1_predictions_detailed.csv - for detailed predictions")

print("\n" + "="*80)