# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 02:21:58 2025

@author: gmfet
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load Data ===
data_path = r"C:\Users\gmfet\vgd_italy\output\cross_correlations20250708_152435.csv"
df = pd.read_csv(data_path)

granger_cols = [col for col in df.columns if 'granger' in col]
granger_df = df[granger_cols]


average_pvalues = granger_df.mean().sort_values()
print("Average Granger p-values (sorted):\n", average_pvalues)


significant = average_pvalues
# [average_pvalues < 0.05]
print("\nSignificant Granger Causalities (p < 0.05):\n", significant)


results_df = significant.reset_index()
results_df.columns = ['Causal Pair', 'Avg p-value']
results_df['Causal Relationship'] = results_df['Causal Pair'].str.replace('granger_', '', regex=False)
results_df['Causal Relationship'] = results_df['Causal Relationship'].str.replace('_to_', ' → ', regex=False)



plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Avg p-value', y='Causal Relationship', palette='crest', orient='h')
plt.axvline(0.05, color='red', linestyle='--', label='p = 0.05')
plt.title('Significant Granger Causal Relationships (p < 0.05)')
plt.xlabel('Average p-value')
plt.legend()
plt.tight_layout()
plt.show()
