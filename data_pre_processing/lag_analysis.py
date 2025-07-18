# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 01:56:28 2025

@author: gmfet
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 00:39:04 2025
@author: gmfet
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Data ===
data_path = r"C:\Users\gmfet\vgd_italy\output\cross_correlations20250708_152435.csv"
df = pd.read_csv(data_path)

# === Extract Relevant Lag Columns ===
cols = [col for col in df.columns if col.startswith('lag_')]
output_df = df[cols]

# cols = [col for col in df.columns if col.startswith('lag_') and 'vgd' in col]

# === Pairwise Scatter Matrix ===
g = sns.PairGrid(output_df, corner=False, diag_sharey=False)
g.map_lower(sns.scatterplot, s=20, alpha=0.6)
g.map_diag(sns.histplot, kde=True)

plt.tight_layout()
plt.savefig(r"C:\Users\gmfet\vgd_italy\output\scatter_matrix_lag2.png")

# === Compute Mean Lags ===
overall_lags = output_df.mean().sort_values()
print("Average Lags (sorted):\n", overall_lags)

# === Construct Lag Matrix ===
lag_matrix = pd.DataFrame(index=[], columns=[])
for col in cols:
    parts = col.replace("lag_", "").split("_to_")
    src, tgt = parts[0], parts[1]
    lag_matrix.loc[src, tgt] = overall_lags[col]

lag_matrix = lag_matrix.astype(float)
print("\nLag Matrix:\n", lag_matrix)

# === Heatmap of Lag Matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(lag_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Average Lag Between Variables")
plt.tight_layout()
plt.savefig(r"C:\Users\gmfet\vgd_italy\output\lag_matrix_heatmap2.png")
plt.show()

# === Horizontal Barplot from Computed Lags ===
# Build DataFrame for plotting
plot_df = overall_lags.reset_index()
plot_df.columns = ['Lag Name', 'Lag']
plot_df['Variable Pair'] = plot_df['Lag Name'].str.replace('lag_', '', regex=False)
plot_df['Variable Pair'] = plot_df['Variable Pair'].str.replace('_to_', ' → ', regex=False)

# Sort by lag value for visualization
plot_df = plot_df.sort_values(by='Lag')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Lag', y='Variable Pair', data=plot_df, palette='coolwarm', orient='h')
plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.title('Lag Values Between VGD and Climatic/Seismic Variables (Computed)', fontsize=14)
plt.xlabel('Lag (time units)')
plt.ylabel('Variable Relationship')

# Annotate bars
for index, row in plot_df.iterrows():
    plt.text(row['Lag'] + (0.05 if row['Lag'] >= 0 else -0.35), index, f"{row['Lag']:.2f}", 
             color='black', va='center', fontsize=9)

plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.savefig(r"C:\Users\gmfet\vgd_italy\output\computed_lag_barplot2.png")
plt.show()

