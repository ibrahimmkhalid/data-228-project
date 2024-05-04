# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as animation
import pandas as pd
import datetime
import seaborn as sns
import os

# %%
energy_path = "./data/france_production.csv"
weather_path = "./data/france_weather.csv"
output_path = "./output"

# %%
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %%
try:
    weather_df = pd.read_csv(weather_path)
    energy_df = pd.read_csv(energy_path)
except FileNotFoundError:
    print("./data/france_production.csv or ./data/france_weather.csv not found")
    exit(1)

# %%
energy_df.columns

# %%
energy_df.head()

# %%
energy_df["Date and Hour"] = pd.to_datetime(energy_df["Date and Hour"], utc=True)
weather_df["dt_iso"] = weather_df["dt_iso"].map(lambda x: x.replace(" UTC", ""))
weather_df["dt_iso"] = pd.to_datetime(weather_df["dt_iso"])

# %%
energy_df["StartHour"] = energy_df["StartHour"].astype(str)
energy_df["EndHour"] = energy_df["EndHour"].astype(str)
energy_df["StartHour"] = energy_df["StartHour"].map(lambda x: x.replace(":", ""))
energy_df["EndHour"] = energy_df["EndHour"].map(lambda x: x.replace(":", ""))
energy_df["StartHour"] = energy_df["StartHour"].astype(int)
energy_df["EndHour"] = energy_df["EndHour"].astype(int)
diff = energy_df["EndHour"] - energy_df["StartHour"]
print(diff.value_counts())
print(energy_df.shape)

# %% [markdown]
# all the differences are 1, so we can drop Date, StartHour, EndHour, dayOfYear, dayName, monthName and use only "Date and Hour" column

# %%
energy_df.drop(
    columns=["Date", "StartHour", "EndHour", "dayOfYear", "dayName", "monthName"],
    inplace=True,
)
energy_df.rename(
    columns={
        "Date and Hour": "dt_iso",
        "Production": "production",
        "Source": "source",
    },
    inplace=True,
)
energy_df.head()

# %%
energy_df.isna().sum()
display(energy_df[energy_df["production"].isna()])

# %% [markdown]
# only 2 rows, we can drop them

# %%
energy_df.dropna(inplace=True)

# %%
weather_df.columns

# %%
weather_df.head()

# %%
print(weather_df.shape)

# %%
print(weather_df.isna().sum())

# %% [markdown]
# we can drop all columns that are empty and also the ones that are not useful for analysis

# %%
weather_df.drop(
    columns=[
        "dt",
        "timezone",
        "city_name",
        "lat",
        "lon",
        "visibility",
        "sea_level",
        "grnd_level",
        "wind_gust",
        "rain_3h",
        "snow_3h",
        "weather_id",
        "weather_description",
        "weather_main",
        "weather_icon",
    ],
    inplace=True,
)

# %%
print(weather_df.isna().sum())

# %%
display(weather_df[[not x for x in weather_df["rain_1h"].isna()]].head())

# %%
display(weather_df[[not x for x in weather_df["snow_1h"].isna()]].head())

# %%
weather_df["rain_1h"].fillna(0, inplace=True)
weather_df["snow_1h"].fillna(0, inplace=True)

# %%
print(weather_df.isna().sum())

# %%
print(weather_df.shape)
print(energy_df.shape)

# %% [markdown]
# peculiarity of the datasets, the energy dataset has more than double the rows of the weather dataset.

# %%
energy_df.sort_values(by="dt_iso", inplace=True)
display(energy_df)

# %% [markdown]
# we have been bamboozled. The energy dataset was not originally sorted, so the
# purchase of the weather dataset does not cover all the covered period of the
# energy dataset. We will have to drop the rows that are not covered by the
# weather dataset.

# %%
energy_df = energy_df[energy_df["dt_iso"].isin(weather_df["dt_iso"])]

# %%
print(energy_df.shape)
print(weather_df.shape)

# %%
energy_df["dt_iso"].value_counts().sort_values(ascending=False)

# %%
display(energy_df[energy_df["dt_iso"] == "2021-10-31 01:00:00+00:00"])
display(energy_df[energy_df["dt_iso"] == "2020-10-25 01:00:00+00:00"])

# %%
# set the production for the above rows to 0
energy_df.loc[energy_df["dt_iso"] == "2021-10-31 01:00:00+00:00", "production"] = (4379.0 + 4698.0) / 2
energy_df.loc[energy_df["dt_iso"] == "2020-10-25 01:00:00+00:00", "production"] = (10982.0 + 10682.0) / 2
energy_df.drop(inplace=True, index=30433)
energy_df.drop(inplace=True, index=7211)

# %%
energy_df["dt_iso"].value_counts().sort_values(ascending=False)

# %%
print(energy_df.shape)
print(weather_df.shape)

# %%
wind = energy_df[energy_df["source"] == "Wind"].drop(columns=["source"]).reset_index(drop=True)
solar = energy_df[energy_df["source"] == "Solar"].drop(columns=["source"]).reset_index(drop=True)
energy_df = wind.merge(solar, on="dt_iso", suffixes=("_wind", "_solar"))

# %%
weather_df = weather_df[weather_df["dt_iso"].isin(energy_df["dt_iso"])].reset_index(drop=True)

# %%
df = weather_df.merge(energy_df, on="dt_iso")

# %%
df.columns

# %%
df.head()

# %%
df["dt_iso"] = pd.to_datetime(df["dt_iso"])
print(df["dt_iso"].min())
print(df["dt_iso"].max())

# %%
plt.rcParams.update({'font.size': 14})  # Set font size to 14
number_of_days = 7
start_date = "2022-01-13"
end_date = datetime.datetime.strptime(start_date, "%Y-%m-%d") + datetime.timedelta(days=number_of_days)
end_date = end_date.strftime("%Y-%m-%d")
df_plot = df[df["dt_iso"] >= start_date]
df_plot = df_plot[df_plot["dt_iso"] <= end_date]
plt.figure(figsize=(12, 5))
plt.plot(df_plot["dt_iso"], df_plot["production_wind"], label="Wind")
plt.plot(df_plot["dt_iso"], df_plot["production_solar"], label="Solar")
plt.xlabel("Date")
plt.ylabel("Production (MW)")
plt.legend()
plt.savefig(f"{output_path}/energy_production_sample_week.png")
plt.show()
plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

# %%
plt.rcParams.update({'font.size': 14})  # Set font size to 14
fig, ax = plt.subplots(figsize=(12, 5))
line1, = ax.plot([], [], label="Wind")
line2, = ax.plot([], [], label="Solar")
ax.legend()
ax.set_ylim(0, max(df["production_wind"].max(), df["production_solar"].max()))
ax.set_xlabel("Date")
ax.set_ylabel("Production (MW)")
def animate(i):
    start_date = df["dt_iso"].min() + datetime.timedelta(days=i)
    end_date = start_date + datetime.timedelta(days=number_of_days)
    df_plot = df[(df["dt_iso"] >= start_date) & (df["dt_iso"] <= end_date)]
    line1.set_data(df_plot["dt_iso"], df_plot["production_wind"])
    line2.set_data(df_plot["dt_iso"], df_plot["production_solar"])
    ax.set_xlim(start_date, end_date)
    return line1, line2

ani = animation.FuncAnimation(fig,
                              animate,
                              frames=int((df["dt_iso"].max() - df["dt_iso"].min()).days - number_of_days + 1),
                              interval=75,
                              blit=True)
ani.save(f"{output_path}/energy_production.mp4")
plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True)
plt.savefig(f"{output_path}/corr_all.png")
plt.show()

# %% [markdown]
# - The temp column perfectly correlates with temp_min and temp_max, therefore we can remove temp_min and temp_max without losing much information
# - The feels like column also correlates very strongly with temp, so it can be removed as well
# - The rain_1h and snow_1h columns have very small correlations (< 0.1) with either solar or wind production, so they can also be safely discarded
# - pressure has a very small correlation with solar production, but has a not insignificant negative correlation with wind production, so we will keep it
# - humidity has a very small correlation with wind production, but has a strong negative correlation with solar production, so it will be kept

# %%
df.drop(columns=["feels_like","temp_min", "temp_max", "rain_1h", "snow_1h"], inplace=True)
df.to_csv("./data/france_weather_energy_with_date.csv", index=False)

# %%
df.describe(include="all")

# %%
df = df.drop(columns=["dt_iso"]).reset_index(drop=True)

# %%
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.savefig(f"{output_path}/corr_select.png")
plt.show()

# %%
plt.bar(corr["production_solar"].index[:-2], corr["production_solar"].values[:-2])
plt.ylim(-0.6, 0.8)
plt.xticks(rotation=45)
plt.xlabel("feature")
plt.ylabel("correlation")
plt.savefig(f"{output_path}/corr_solar.png")
plt.show()

# %%
plt.bar(corr["production_wind"].index[:-2], corr["production_wind"].values[:-2])
plt.ylim(-0.6, 0.8)
plt.xticks(rotation=45)
plt.xlabel("feature")
plt.ylabel("correlation")
plt.savefig(f"{output_path}/corr_wind.png")
plt.show()

# %%
df.to_csv("./data/france_weather_energy.csv", index=False)
