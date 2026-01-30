# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 06:14:40 2026

@author: gmfet
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Load your data
data1 = np.load(r"C:\Users\gmfet\vgd_italy\regression\original_data\training\targets.npy")
data2 = np.load(r"C:\Users\gmfet\vgd_italy\regression\original_data\test\targets.npy")
data3 = np.load(r"C:\Users\gmfet\vgd_italy\regression\original_data\validation\targets.npy")

data = np.concatenate((data1, data2, data3), axis=0)



# 1. Calculate ACF for each row
# nlags=301 because we have 302 time steps (max lag is N-1)
all_acf = np.array([acf(row, nlags=301, fft=True) for row in data])

# 2. Calculate the mean ACF across all samples
mean_acf = np.mean(all_acf, axis=0)
lags = np.arange(len(mean_acf))

# 3. Plotting
plt.figure(figsize=(10, 6))
plt.plot(lags, mean_acf, color='tab:blue', lw=2, label='Average ACF')
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.title('Average Autocorrelation')
plt.xlabel('Lag (Time Steps)')
plt.ylabel('Average Correlation Coefficient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Programmatic suggestion for seq_length
# 3. Calculate the 95% Confidence Interval
# N is the length of the time series
N = data.shape[1] 
ci = 1.96 / np.sqrt(N)

# 4. Find the first lag where ACF drops below the confidence interval
# This is our "Optimal Sequence Length"
crossing_points = np.where(mean_acf < ci)[0]
opt_len = crossing_points[0] if len(crossing_points) > 0 else len(lags)

# 5. Plotting
plt.figure(figsize=(10, 6))
plt.plot(lags, mean_acf, color='tab:blue', lw=2, label='Average ACF')

# Add Confidence Interval shaded region
plt.axhspan(-ci, ci, color='gray', alpha=0.2, label='95% Confidence Interval')
plt.axhline(0, color='black', linestyle='-', alpha=0.3)

# Highlight the Optimal Sequence Length
plt.axvline(opt_len, color='red', linestyle='--', alpha=0.8, 
            label=f'Optimal Seq Len: {opt_len}')

plt.title('Information Decay & Optimal Sequence Length')
plt.xlabel('Lag (Time Steps)')
plt.ylabel('Average Correlation Coefficient')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()