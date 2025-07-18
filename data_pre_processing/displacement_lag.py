import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf



data_path = r"C:\Users\gmfet\Desktop\emilia\data\vgd_targets\mp_20.npy"



data_path = np.load(data_path)



max_lag = 64  # how many lags to show

fig, axs = plt.subplots(8, 8, figsize=(40, 40))  # adjust to match max_lag
axs = axs.flatten()

for k in range(1, max_lag + 1):
    axs[k - 1].scatter(data_path[:-k], data_path[k:], s=10)
    axs[k - 1].set_title(f"Lag {k}")
    axs[k - 1].set_xlabel("VGD")
    axs[k - 1].set_ylabel(f"lag(vgd, {k})")

plt.tight_layout()
plt.show()



plot_acf(data_path, lags=300)
plt.title("Autocorrelation of VGD")
plt.show()
