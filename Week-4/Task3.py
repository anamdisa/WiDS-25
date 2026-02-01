"""
WiDS Week 4 - Task 3: Gap Analysis Between ML Predictions and Sustainability Commitments
Compares ML-predicted emission trends with NLP-extracted commitments from sustainability reports
Identifies alignment gaps and generates comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TASK 3: EMISSION TRENDS vs SUSTAINABILITY COMMITMENTS GAP ANALYSIS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
TASK1_DIR = Path("data/task1_outputs")
TASK2_DIR = Path("data/nlp_output")
OUTPUT_DIR = Path("data/task3_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD ML PREDICTIONS FROM TASK 1
# ============================================================================
print("\n[1] Loading ML Predictions from Task 1...")

try:
    # Load yearly aggregated emissions and predictions
    yearly_data = pd.read_csv(TASK1_DIR / 'task1_yearly_aggregated.csv')
    print(f"   Loaded yearly aggregated data: {len(yearly_data)} years")
    
    # Load trend summary
    trend_summary = pd.read_csv(TASK1_DIR / 'task1_trend_summary.csv')
    print(f"   Loaded trend summary")
    
    # Load detailed predictions
    predictions_detailed = pd.read_csv(TASK1_DIR / 'task1_predictions_detailed.csv')
    print(f"   Loaded detailed predictions: {len(predictions_detailed)} records")
    
    print(f"\n  ML Data Coverage:")
    print(f"    - Year Range: {yearly_data['Year'].min()} - {yearly_data['Year'].max()}")
    print(f"    - Best Model: {trend_summary['Best_Model'].values[0]}")
    print(f"    - Model R²: {trend_summary['Best_Model_R2'].values[0]:.4f}")
    
except Exception as e:
    print(f"\n   ERROR: Could not load Task 1 outputs!")
    print(f"     {e}")
    print(f"     Please ensure Task 1 has been run and outputs exist in: {TASK1_DIR}")
    exit(1)

# ============================================================================
# 2. LOAD NLP COMMITMENTS FROM TASK 2
# ============================================================================
print("\n[2] Loading Sustainability Commitments from Task 2...")

try:
    # Try to load combined commitments
    commitments_file = TASK2_DIR / 'all_commitments_combined.csv'
    
    if commitments_file.exists():
        commitments = pd.read_csv(commitments_file)
        print(f"   Loaded combined commitments: {len(commitments)} commitments")
    else:
        # Look for individual commitment files
        commitment_files = list(TASK2_DIR.glob('*_commitments.csv'))
        
        if not commitment_files:
            print(f"\n   WARNING: No commitment files found in {TASK2_DIR}")
            print(f"     Please run Task 2 first to extract commitments from sustainability reports")
            print(f"     Creating sample commitments for demonstration...")
            
            # Create sample commitments for demonstration
            commitments = pd.DataFrame({
                'Company': ['BASF', 'BASF', 'BASF'],
                'Commitment': [
                    'Reduce CO2 emissions by 25% by 2030',
                    'Achieve net-zero emissions by 2050',
                    'Reduce Scope 1 and 2 emissions by 60% by 2030'
                ],
                'Target Year': ['2030', '2050', '2030'],
                'Reduction %': ['25%', '100%', '60%'],
                'Metric': ['CO2 emissions', 'net-zero', 'Scope 1 and 2 emissions']
            })
        else:
            # Combine individual files
            commitment_dfs = []
            for file in commitment_files:
                df = pd.read_csv(file)
                commitment_dfs.append(df)
            commitments = pd.concat(commitment_dfs, ignore_index=True)
            print(f"   Loaded commitments from {len(commitment_files)} file(s): {len(commitments)} total")
    
    print(f"\n  Commitment Summary:")
    print(f"    - Total commitments: {len(commitments)}")
    print(f"    - Companies: {commitments['Company'].nunique()}")
    
except Exception as e:
    print(f"\n   ERROR: Could not load Task 2 outputs!")
    print(f"     {e}")
    print(f"     Creating sample commitments for demonstration...")
    
    commitments = pd.DataFrame({
        'Company': ['BASF', 'BASF', 'BASF'],
        'Commitment': [
            'Reduce CO2 emissions by 25% by 2030',
            'Achieve net-zero emissions by 2050',
            'Reduce Scope 1 and 2 emissions by 60% by 2030'
        ],
        'Target Year': ['2030', '2050', '2030'],
        'Reduction %': ['25%', '100%', '60%'],
        'Metric': ['CO2 emissions', 'net-zero', 'Scope 1 and 2 emissions']
    })

# ============================================================================
# 3. PROCESS AND STANDARDIZE COMMITMENTS
# ============================================================================
print("\n[3] Processing Commitments...")

# Parse target years (handle string/int issues)
commitments['Target_Year_Int'] = commitments['Target Year'].astype(str).str.extract(r'(\d{4})')[0].astype(float)

# Parse reduction percentages
def parse_percentage(pct_str):
    """Extract numeric percentage from string"""
    if pd.isna(pct_str) or pct_str == 'Not specified':
        return None
    
    # Try to extract number
    match = pd.Series([str(pct_str)]).str.extract(r'(\d+\.?\d*)')[0].values[0]
    if match:
        return float(match)
    return None

commitments['Reduction_Pct_Numeric'] = commitments['Reduction %'].apply(parse_percentage)

# Filter valid commitments with both year and percentage
valid_commitments = commitments[
    (commitments['Target_Year_Int'].notna()) & 
    (commitments['Reduction_Pct_Numeric'].notna())
].copy()

print(f"   Valid commitments with targets: {len(valid_commitments)}")

if len(valid_commitments) > 0:
    print(f"\n  Commitment Details:")
    for idx, row in valid_commitments.iterrows():
        print(f"    • {row['Company']}: {row['Reduction_Pct_Numeric']:.0f}% reduction by {int(row['Target_Year_Int'])}")

# ============================================================================
# 4. CALCULATE ML-PREDICTED EMISSION TRENDS
# ============================================================================
print("\n[4] Analyzing ML-Predicted Emission Trends...")

# Get baseline year (latest year in historical data)
baseline_year = yearly_data['Year'].max()
baseline_emissions = yearly_data[yearly_data['Year'] == baseline_year]['Predicted_Total_Emissions'].values[0]

print(f"\n  Baseline Year: {baseline_year}")
print(f"  Baseline Emissions: {baseline_emissions:.2f}")

# Calculate emission trend slope (using linear regression on predicted emissions)
years = yearly_data['Year'].values
predicted_emissions = yearly_data['Predicted_Total_Emissions'].values

# Fit linear trend
slope, intercept, r_value, p_value, std_err = stats.linregress(years, predicted_emissions)

print(f"\n  ML Trend Analysis:")
print(f"    - Slope (emissions/year): {slope:.4f}")
print(f"    - R² of trend fit: {r_value**2:.4f}")
print(f"    - P-value: {p_value:.6f}")

# Calculate average year-over-year percentage change
avg_yoy_change = yearly_data['Predicted_YoY_Change'].mean()
print(f"    - Average YoY change: {avg_yoy_change:.2f}%")

# Calculate percentage change per year (relative to baseline)
pct_change_per_year = (slope / baseline_emissions) * 100
print(f"    - Annual % change (relative to baseline): {pct_change_per_year:.2f}%")

# ============================================================================
# 5. PROJECT FUTURE EMISSIONS TO TARGET YEARS
# ============================================================================
print("\n[5] Projecting Emissions to Target Years...")

def project_emissions(baseline_year, baseline_emissions, slope, target_year):
    """Project emissions to target year using linear trend"""
    years_ahead = target_year - baseline_year
    projected_emissions = baseline_emissions + (slope * years_ahead)
    
    # Calculate percentage change from baseline
    pct_change = ((projected_emissions - baseline_emissions) / baseline_emissions) * 100
    
    return projected_emissions, pct_change

# Create projection summary
projections = []

# Project to common target years
target_years = [2025, 2030, 2035, 2040, 2045, 2050]

for target_year in target_years:
    if target_year > baseline_year:
        projected_emissions, pct_change = project_emissions(
            baseline_year, baseline_emissions, slope, target_year
        )
        
        projections.append({
            'Target_Year': target_year,
            'Projected_Emissions': projected_emissions,
            'Change_from_Baseline_%': pct_change,
            'Years_Ahead': target_year - baseline_year
        })

projections_df = pd.DataFrame(projections)
print(f"\n  Future Emission Projections:")
print(projections_df.to_string(index=False))

# ============================================================================
# 6. COMPARE ML PREDICTIONS WITH COMMITMENTS (GAP ANALYSIS)
# ============================================================================
print("\n[6] Gap Analysis: ML Predictions vs Stated Commitments...")

gap_analysis = []

for idx, commitment in valid_commitments.iterrows():
    target_year = int(commitment['Target_Year_Int'])
    stated_reduction = commitment['Reduction_Pct_Numeric']
    company = commitment['Company']
    commitment_text = commitment['Commitment'][:100]
    
    # Skip if target year is before baseline
    if target_year <= baseline_year:
        continue
    
    # Project emissions to target year
    projected_emissions, ml_predicted_change = project_emissions(
        baseline_year, baseline_emissions, slope, target_year
    )
    
    # Calculate ML-predicted reduction (negative change = reduction)
    ml_predicted_reduction = -ml_predicted_change
    
    # Calculate gap
    gap = stated_reduction - ml_predicted_reduction
    
    # Determine alignment status
    if gap <= 5:  # Within 5% tolerance
        status = " ALIGNED"
        status_color = 'green'
    elif gap <= 15:
        status = " CAUTION"
        status_color = 'orange'
    else:
        status = " GAP"
        status_color = 'red'
    
    gap_analysis.append({
        'Company': company,
        'Commitment': commitment_text,
        'Target_Year': target_year,
        'Years_Until_Target': target_year - baseline_year,
        'Stated_Reduction_%': stated_reduction,
        'ML_Predicted_Reduction_%': ml_predicted_reduction,
        'Gap_%': gap,
        'Status': status,
        'Projected_Emissions': projected_emissions,
        'Baseline_Emissions': baseline_emissions
    })
    
    print(f"\n  📊 {company} - Target {target_year}:")
    print(f"     Stated Commitment: {stated_reduction:.1f}% reduction")
    print(f"     ML Predicted:      {ml_predicted_reduction:.1f}% reduction")
    print(f"     Gap:               {gap:.1f}% {'shortfall' if gap > 0 else 'surplus'}")
    print(f"     Status:            {status}")

gap_df = pd.DataFrame(gap_analysis)

if len(gap_df) > 0:
    print(f"\n  Gap Analysis Summary:")
    print(gap_df[['Company', 'Target_Year', 'Stated_Reduction_%', 
                   'ML_Predicted_Reduction_%', 'Gap_%', 'Status']].to_string(index=False))
    
    # Save gap analysis
    gap_df.to_csv(OUTPUT_DIR / 'task3_gap_analysis.csv', index=False)
    print(f"\n   Saved: {OUTPUT_DIR / 'task3_gap_analysis.csv'}")

# ============================================================================
# 7. CREATE COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("\n[7] Creating Visualizations...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure 1: Emission Trends with Commitments
fig1 = plt.figure(figsize=(16, 10))

# Plot 1: Historical and Projected Emissions with Commitment Targets
ax1 = plt.subplot(2, 2, 1)

# Plot historical actual and predicted
ax1.plot(yearly_data['Year'], yearly_data['Actual_Total_Emissions'], 
         'o-', label='Actual (Historical)', linewidth=2.5, markersize=5, color='steelblue')
ax1.plot(yearly_data['Year'], yearly_data['Predicted_Total_Emissions'], 
         's-', label='ML Predicted (Historical)', linewidth=2.5, markersize=5, 
         color='coral', alpha=0.8)

# Extend trend line to future
future_years = np.arange(baseline_year + 1, 2051)
future_emissions = baseline_emissions + slope * (future_years - baseline_year)
ax1.plot(future_years, future_emissions, '--', 
         label='ML Trend Projection', linewidth=2, color='coral', alpha=0.6)

# Plot commitment target points
if len(valid_commitments) > 0:
    for idx, commitment in valid_commitments.iterrows():
        target_year = int(commitment['Target_Year_Int'])
        reduction_pct = commitment['Reduction_Pct_Numeric']
        
        if target_year > baseline_year:
            # Calculate target emissions based on stated reduction
            target_emissions = baseline_emissions * (1 - reduction_pct / 100)
            
            ax1.scatter(target_year, target_emissions, s=200, marker='*', 
                       c='green', edgecolors='black', linewidths=2, 
                       zorder=5, label=f'Commitment Target ({int(reduction_pct)}% reduction)')

ax1.axvline(x=baseline_year, color='red', linestyle=':', linewidth=2, 
           label=f'Baseline ({baseline_year})', alpha=0.7)

ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total Emissions', fontsize=11, fontweight='bold')
ax1.set_title('Emission Trends: Historical, Predicted & Commitments', 
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Gap Analysis Bar Chart
if len(gap_df) > 0:
    ax2 = plt.subplot(2, 2, 2)
    
    x_pos = np.arange(len(gap_df))
    colors = ['green' if g <= 5 else 'orange' if g <= 15 else 'red' 
              for g in gap_df['Gap_%']]
    
    bars = ax2.barh(x_pos, gap_df['Gap_%'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([f"{row['Company']}\n({int(row['Target_Year'])})" 
                          for _, row in gap_df.iterrows()], fontsize=9)
    ax2.set_xlabel('Gap (Stated - Predicted) %', fontsize=11, fontweight='bold')
    ax2.set_title('Commitment Gap Analysis', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=15, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add gap values on bars
    for i, (idx, row) in enumerate(gap_df.iterrows()):
        gap_val = row['Gap_%']
        ax2.text(gap_val + 1, i, f"{gap_val:.1f}%", 
                va='center', fontsize=9, fontweight='bold')

# Plot 3: Reduction Comparison
if len(gap_df) > 0:
    ax3 = plt.subplot(2, 2, 3)
    
    x = np.arange(len(gap_df))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, gap_df['Stated_Reduction_%'], width, 
                    label='Stated Commitment', color='green', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, gap_df['ML_Predicted_Reduction_%'], width, 
                    label='ML Predicted', color='coral', alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('Commitment', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Reduction %', fontsize=11, fontweight='bold')
    ax3.set_title('Stated vs ML-Predicted Reduction Comparison', 
                 fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{row['Company']}\n{int(row['Target_Year'])}" 
                         for _, row in gap_df.iterrows()], fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Plot 4: Trend Projection Table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Create summary statistics table
summary_data = [
    ['Metric', 'Value'],
    ['─' * 30, '─' * 20],
    ['Baseline Year', f'{baseline_year}'],
    ['Baseline Emissions', f'{baseline_emissions:.2f}'],
    ['ML Trend Slope (per year)', f'{slope:.4f}'],
    ['Avg Annual Change', f'{pct_change_per_year:.2f}%'],
    ['─' * 30, '─' * 20],
]

if len(gap_df) > 0:
    summary_data.append(['Avg Stated Reduction', f"{gap_df['Stated_Reduction_%'].mean():.1f}%"])
    summary_data.append(['Avg ML Predicted Reduction', f"{gap_df['ML_Predicted_Reduction_%'].mean():.1f}%"])
    summary_data.append(['Avg Gap', f"{gap_df['Gap_%'].mean():.1f}%"])
    summary_data.append(['─' * 30, '─' * 20])
    summary_data.append(['Aligned Commitments', f"{len(gap_df[gap_df['Gap_%'] <= 5])}"])
    summary_data.append(['Gap Commitments', f"{len(gap_df[gap_df['Gap_%'] > 15])}"])

table = ax4.table(cellText=summary_data, loc='center', cellLoc='left',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'task3_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'task3_comprehensive_analysis.png'}")
plt.close()

# Figure 2: Detailed Projection Timeline
fig2, ax = plt.subplots(figsize=(14, 8))

# Plot historical trend
ax.plot(yearly_data['Year'], yearly_data['Predicted_Total_Emissions'], 
        'o-', label='Historical ML Predictions', linewidth=2.5, 
        markersize=6, color='steelblue')

# Plot future projection
future_years_extended = np.arange(baseline_year, 2051)
future_emissions_extended = baseline_emissions + slope * (future_years_extended - baseline_year)
ax.plot(future_years_extended, future_emissions_extended, '--', 
        label='ML Trend Projection', linewidth=2.5, color='coral', alpha=0.7)

# Plot commitment targets
if len(valid_commitments) > 0:
    for idx, commitment in valid_commitments.iterrows():
        target_year = int(commitment['Target_Year_Int'])
        reduction_pct = commitment['Reduction_Pct_Numeric']
        
        if target_year > baseline_year and target_year <= 2050:
            # Calculate target emissions
            target_emissions = baseline_emissions * (1 - reduction_pct / 100)
            
            # Plot target point
            ax.scatter(target_year, target_emissions, s=300, marker='*', 
                      c='green', edgecolors='black', linewidths=2, zorder=5)
            
            # Get ML projection for same year
            projected_emissions, _ = project_emissions(
                baseline_year, baseline_emissions, slope, target_year
            )
            
            # Draw gap line
            ax.plot([target_year, target_year], 
                   [target_emissions, projected_emissions],
                   'r--', linewidth=2, alpha=0.5)
            
            # Add labels
            gap_emissions = projected_emissions - target_emissions
            gap_pct = (gap_emissions / baseline_emissions) * 100
            
            ax.text(target_year, (target_emissions + projected_emissions) / 2,
                   f'Gap: {gap_pct:.1f}%',
                   ha='right', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.axvline(x=baseline_year, color='red', linestyle=':', linewidth=2, 
          label=f'Baseline ({baseline_year})', alpha=0.7)

ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Emissions', fontsize=13, fontweight='bold')
ax.set_title('Emission Trajectory: ML Predictions vs Sustainability Commitments', 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(yearly_data['Year'].min() - 2, 2052)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'task3_emission_trajectory.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'task3_emission_trajectory.png'}")
plt.close()

# ============================================================================
# 8. SAVE COMPREHENSIVE RESULTS
# ============================================================================
print("\n[8] Saving Comprehensive Results...")

# Save projection summary
projections_df.to_csv(OUTPUT_DIR / 'task3_future_projections.csv', index=False)
print(f"   Saved: {OUTPUT_DIR / 'task3_future_projections.csv'}")

# Create comprehensive report
report_data = {
    'Analysis_Type': 'Gap Analysis Summary',
    'Baseline_Year': baseline_year,
    'Baseline_Emissions': baseline_emissions,
    'ML_Trend_Slope': slope,
    'ML_Annual_Change_%': pct_change_per_year,
    'Avg_YoY_Change_%': avg_yoy_change,
    'Number_of_Commitments': len(valid_commitments),
    'Number_with_Targets': len(gap_df)
}

if len(gap_df) > 0:
    report_data.update({
        'Avg_Stated_Reduction_%': gap_df['Stated_Reduction_%'].mean(),
        'Avg_ML_Predicted_Reduction_%': gap_df['ML_Predicted_Reduction_%'].mean(),
        'Avg_Gap_%': gap_df['Gap_%'].mean(),
        'Max_Gap_%': gap_df['Gap_%'].max(),
        'Min_Gap_%': gap_df['Gap_%'].min(),
        'Aligned_Commitments': len(gap_df[gap_df['Gap_%'] <= 5]),
        'Caution_Commitments': len(gap_df[(gap_df['Gap_%'] > 5) & (gap_df['Gap_%'] <= 15)]),
        'Gap_Commitments': len(gap_df[gap_df['Gap_%'] > 15])
    })

report_df = pd.DataFrame([report_data])
report_df.to_csv(OUTPUT_DIR / 'task3_analysis_summary.csv', index=False)
print(f"   Saved: {OUTPUT_DIR / 'task3_analysis_summary.csv'}")

# ============================================================================
# 9. GENERATE TEXT REPORT
# ============================================================================
print("\n[9] Generating Text Report...")

report_lines = []
report_lines.append("="*80)
report_lines.append("TASK 3: GAP ANALYSIS REPORT")
report_lines.append("Emission Trends vs Sustainability Commitments")
report_lines.append("="*80)

report_lines.append(f"\n1. BASELINE & ML TREND ANALYSIS")
report_lines.append(f"   • Baseline Year: {baseline_year}")
report_lines.append(f"   • Baseline Emissions: {baseline_emissions:.2f}")
report_lines.append(f"   • ML Trend Slope: {slope:.4f} emissions/year")
report_lines.append(f"   • Annual % Change: {pct_change_per_year:.2f}%")
report_lines.append(f"   • Trend Direction: {'INCREASING' if slope > 0 else 'DECREASING'}")

report_lines.append(f"\n2. FUTURE PROJECTIONS (ML-Based)")
for _, proj in projections_df.iterrows():
    report_lines.append(f"   • {int(proj['Target_Year'])}: {proj['Projected_Emissions']:.2f} "
                       f"({proj['Change_from_Baseline_%']:+.1f}% from baseline)")

if len(gap_df) > 0:
    report_lines.append(f"\n3. SUSTAINABILITY COMMITMENTS")
    for idx, row in gap_df.iterrows():
        report_lines.append(f"   • {row['Company']} ({int(row['Target_Year'])}): "
                          f"{row['Stated_Reduction_%']:.0f}% reduction target")

    report_lines.append(f"\n4. GAP ANALYSIS RESULTS")
    for idx, row in gap_df.iterrows():
        status_symbol = "" if row['Gap_%'] <= 5 else "" if row['Gap_%'] <= 15 else ""
        report_lines.append(f"   {status_symbol} {row['Company']} ({int(row['Target_Year'])})")
        report_lines.append(f"      - Stated: {row['Stated_Reduction_%']:.1f}% reduction")
        report_lines.append(f"      - ML Predicted: {row['ML_Predicted_Reduction_%']:.1f}% reduction")
        report_lines.append(f"      - Gap: {row['Gap_%']:.1f}% {row['Status']}")

    report_lines.append(f"\n5. SUMMARY STATISTICS")
    report_lines.append(f"   • Average Stated Reduction: {gap_df['Stated_Reduction_%'].mean():.1f}%")
    report_lines.append(f"   • Average ML Predicted Reduction: {gap_df['ML_Predicted_Reduction_%'].mean():.1f}%")
    report_lines.append(f"   • Average Gap: {gap_df['Gap_%'].mean():.1f}%")
    report_lines.append(f"   • Aligned Commitments (≤5% gap): {len(gap_df[gap_df['Gap_%'] <= 5])}")
    report_lines.append(f"   • Gap Commitments (>15% gap): {len(gap_df[gap_df['Gap_%'] > 15])}")

    report_lines.append(f"\n6. KEY FINDINGS")
    max_gap_idx = gap_df['Gap_%'].idxmax()
    max_gap_row = gap_df.loc[max_gap_idx]
    
    if gap_df['Gap_%'].mean() > 15:
        report_lines.append(f"    SIGNIFICANT GAP DETECTED")
        report_lines.append(f"   The average gap of {gap_df['Gap_%'].mean():.1f}% indicates that")
        report_lines.append(f"   current emission trends are not aligned with stated commitments.")
    elif gap_df['Gap_%'].mean() > 5:
        report_lines.append(f"    MODERATE GAP")
        report_lines.append(f"   The average gap of {gap_df['Gap_%'].mean():.1f}% suggests some")
        report_lines.append(f"   misalignment between trends and commitments.")
    else:
        report_lines.append(f"    ALIGNED")
        report_lines.append(f"   Current trends appear aligned with stated commitments.")
    
    report_lines.append(f"\n   Largest Gap:")
    report_lines.append(f"   • {max_gap_row['Company']} ({int(max_gap_row['Target_Year'])})")
    report_lines.append(f"   • Stated: {max_gap_row['Stated_Reduction_%']:.1f}% reduction")
    report_lines.append(f"   • Predicted: {max_gap_row['ML_Predicted_Reduction_%']:.1f}% reduction")
    report_lines.append(f"   • Gap: {max_gap_row['Gap_%']:.1f}%")

else:
    report_lines.append(f"\n3. NO VALID COMMITMENTS FOUND")
    report_lines.append(f"   Unable to perform gap analysis without sustainability commitments.")

report_lines.append("\n" + "="*80)
report_lines.append("END OF REPORT")
report_lines.append("="*80)

# Save text report
report_text = "\n".join(report_lines)
with open(OUTPUT_DIR / 'task3_gap_analysis_report.txt', 'w') as f:
    f.write(report_text)

print(f"   Saved: {OUTPUT_DIR / 'task3_gap_analysis_report.txt'}")

# Print report to console
print("\n" + report_text)

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TASK 3 COMPLETED SUCCESSFULLY!")
print("="*80)

print("\n OUTPUTS GENERATED:")
print(f"  1. task3_gap_analysis.csv              - Detailed gap analysis table")
print(f"  2. task3_future_projections.csv        - ML-based emission projections")
print(f"  3. task3_analysis_summary.csv          - Summary statistics")
print(f"  4. task3_gap_analysis_report.txt       - Comprehensive text report")
print(f"  5. task3_comprehensive_analysis.png    - Multi-panel visualization")
print(f"  6. task3_emission_trajectory.png       - Timeline with gaps")

print(f"\n All outputs saved to: {OUTPUT_DIR.absolute()}")

print("\n" + "="*80)
print("WIDS WEEK 4 PROJECT COMPLETE!")
print("="*80)
print("\nYou have successfully completed:")
print("   Task 1: ML-based emission prediction")
print("   Task 2: NLP-based commitment extraction")
print("   Task 3: Gap analysis and integration")

print("\n" + "="*80)