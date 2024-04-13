import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

energy_path = "./data/france_production.csv"

energy_df = pd.read_csv(energy_path)

solar = energy_df[energy_df["Date and Hour"].str.contains("2021") & energy_df["Source"].str.contains("Solar")]
wind = energy_df[energy_df["Date and Hour"].str.contains("2021") & energy_df["Source"].str.contains("Wind")]
fig, ax = plt.subplots(figsize=(16, 8))
ax.scatter(solar["Date and Hour"], solar["Production"], c="yellow", label="Solar", marker="o", linewidths=0.5)
ax.scatter(wind["Date and Hour"], wind["Production"], c="blue", label="Wind", marker="x", linewidths=0.5)
plt.legend()
plt.savefig("energy_production.png")
