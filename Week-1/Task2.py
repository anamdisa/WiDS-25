import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the dataset
df = pd.read_csv('Task2_Dataset.csv')

# column names 
columns = ['X1_Relative_Compactness', 'X2_Surface_Area', 'X3_Wall_Area', 'X4_Roof_Area', 'X5_Overall_Height', 'X6_Orientation', 'X7_Glazing_Area', 'X8_Glazing_Area_Distribution', 'Y1_Heating_Load', 'Y2_Cooling_Load']

# Assign column names if they don't exist
if df.shape[1] == 10:
    df.columns = columns

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

print("\n" + "="*80)
print("DATASET INFORMATION")
print("="*80)
print(df.info())

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(df.dtypes)

# Create a time index for plotting over time
df['Sample_Index'] = range(len(df))

# PLOTS OVER TIME (using sample index as proxy)
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Process Variables Over Sample Index (Time Proxy)', fontsize=16, fontweight='bold')

# Plot 1: Temperature-related variables (Heating and Cooling Loads)
axes[0].plot(df['Sample_Index'], df['Y1_Heating_Load'], label='Y1 Heating Load', alpha=0.7, linewidth=1)
axes[0].plot(df['Sample_Index'], df['Y2_Cooling_Load'], label='Y2 Cooling Load', alpha=0.7, linewidth=1)
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Load')
axes[0].set_title('Temperature-Related: Heating & Cooling Loads Over Samples')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Pressure-related variables (Height and Areas)
axes[1].plot(df['Sample_Index'], df['X5_Overall_Height'], label='X5 Overall Height', alpha=0.7, linewidth=1)
axes[1].plot(df['Sample_Index'], df['X2_Surface_Area'], label='X2 Surface Area', alpha=0.7, linewidth=1)
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Pressure-Related: Height & Surface Area Over Samples')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Flow-related variables (Glazing Area)
axes[2].plot(df['Sample_Index'], df['X7_Glazing_Area'], label='X7 Glazing Area', alpha=0.7, linewidth=1)
axes[2].plot(df['Sample_Index'], df['X8_Glazing_Area_Distribution'], label='X8 Glazing Distribution', alpha=0.7, linewidth=1)
axes[2].set_xlabel('Sample Index')
axes[2].set_ylabel('Glazing Metrics')
axes[2].set_title('Flow-Related: Glazing Area & Distribution Over Samples')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# CORRELATION MATRIX
print("\n" + "="*80)
print("CORRELATION MATRIX")
print("="*80)

# Compute correlation matrix
correlation_matrix = df[columns].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm', 
            center=0,
            square=True, 
            linewidths=1, 
            cbar_kws={"shrink": 0.8},
            ax=ax,
            vmin=-1, 
            vmax=1)
ax.set_title('Correlation Matrix Heatmap - All Variables', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# CORRELATION WITH TARGET VARIABLES
print("\n" + "="*80)
print("CORRELATION WITH TARGET VARIABLES (Y1 & Y2)")
print("="*80)

# Correlation with Y1 (Heating Load)
y1_corr = df[columns].corr()['Y1_Heating_Load'].sort_values(ascending=False)
print("\nCorrelation with Y1 (Heating Load):")
print(y1_corr)

# Correlation with Y2 (Cooling Load)
y2_corr = df[columns].corr()['Y2_Cooling_Load'].sort_values(ascending=False)
print("\nCorrelation with Y2 (Cooling Load):")
print(y2_corr)

# Visualize feature correlations with outputs
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Y1 correlations
y1_corr_features = y1_corr.drop(['Y1_Heating_Load', 'Y2_Cooling_Load'])
axes[0].barh(range(len(y1_corr_features)), y1_corr_features.values, color='steelblue')
axes[0].set_yticks(range(len(y1_corr_features)))
axes[0].set_yticklabels(y1_corr_features.index)
axes[0].set_xlabel('Correlation Coefficient')
axes[0].set_title('Feature Correlations with Y1 (Heating Load)', fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
axes[0].grid(True, alpha=0.3)

# Y2 correlations
y2_corr_features = y2_corr.drop(['Y1_Heating_Load', 'Y2_Cooling_Load'])
axes[1].barh(range(len(y2_corr_features)), y2_corr_features.values, color='coral')
axes[1].set_yticks(range(len(y2_corr_features)))
axes[1].set_yticklabels(y2_corr_features.index)
axes[1].set_xlabel('Correlation Coefficient')
axes[1].set_title('Feature Correlations with Y2 (Cooling Load)', fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# DISTRIBUTION PLOTS
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('Distribution of All Variables', fontsize=16, fontweight='bold')

for idx, col in enumerate(columns):
    row = idx // 4
    col_idx = idx % 4
    
    axes[row, col_idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[row, col_idx].set_title(col, fontsize=10)
    axes[row, col_idx].set_xlabel('Value')
    axes[row, col_idx].set_ylabel('Frequency')
    axes[row, col_idx].grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(columns), 12):
    row = idx // 4
    col_idx = idx % 4
    axes[row, col_idx].axis('off')

plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# SUMMARY STATISTICS
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"\nHighest correlation with Y1 (Heating Load): {y1_corr_features.index[0]} ({y1_corr_features.values[0]:.3f})")
print(f"Highest correlation with Y2 (Cooling Load): {y2_corr_features.index[0]} ({y2_corr_features.values[0]:.3f})")
print(f"\nY1 and Y2 correlation: {correlation_matrix.loc['Y1_Heating_Load', 'Y2_Cooling_Load']:.3f}")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. time_series_plots.png")
print("  2. correlation_heatmap.png")
print("  3. target_correlations.png")
print("  4. distributions.png")