import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 3 Scatterplots

df = pd.read_csv('kaggle_train_Stokes.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(df["h"], df["z*"], alpha=0.5)
axes[0].set_xlabel("h (Depth of Water)")
axes[0].set_ylabel("z* (Height of Max Flow)")
axes[0].set_title("h vs. z*")

axes[1].scatter(df["omega"], df["z*"], alpha=0.5)
axes[1].set_xlabel("ω (Flow Frequency)")
axes[1].set_ylabel("z* (Height of Max Flow)")
axes[1].set_title("ω vs. z*")
axes[1].set_xscale("log") 

axes[2].scatter(df["nu"], df["z*"], alpha=0.5)
axes[2].set_xlabel("ν (Fluid Viscosity)")
axes[2].set_ylabel("z* (Height of Max Flow)")
axes[2].set_title("ν vs. z*")
axes[2].set_xscale("log") 

plt.tight_layout()
plt.show()

# Color Gradient Plot

df_dimensionless = pd.read_csv('kaggle_train_Stokes.csv')

df_dimensionless["x1"] = df_dimensionless["nu"] / (df_dimensionless["h"]**2 * df_dimensionless["omega"]) 
df_dimensionless["x2"] = (df_dimensionless["omega"] * df_dimensionless["h"]**2) / df_dimensionless["nu"] 
df_dimensionless["z_bar"] = df_dimensionless["z*"] / df_dimensionless["h"] 

plt.figure(figsize=(7, 6))
sc = plt.scatter(df_dimensionless["x1"], df_dimensionless["x2"], c=df_dimensionless["z_bar"], cmap="viridis", alpha=0.7)
plt.colorbar(sc, label=r"$z_{\mathrm{bar}} = \frac{z^*}{h}$")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$\nu h^{-2} \omega^{-1}$")
plt.ylabel(r"$\omega h^2 \nu^{-1}$")
plt.title("Dimensionless Visualization of the Dataset")

plt.show()
