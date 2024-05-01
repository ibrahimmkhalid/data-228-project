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
try:
    from google.colab import drive
    drive.mount("/content/drive")
    is_colab = True
except:
    is_colab = False

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

random_state = 228

# %%
if is_colab:
    prepend_path = "/content/drive/MyDrive/Syncable/sjsu/data-228/DATA 228 Project Files"
else:
    prepend_path = "."
data_path = f"{prepend_path}/data/france_weather_energy.csv"
data_path_dates = f"{prepend_path}/data/france_weather_energy_with_date.csv"

# %%
df = pd.read_csv(data_path)
df.head()

# %%
df.shape

# %%
y_cols = ["production_wind", "production_solar"]
X_ = df.drop(columns=y_cols)
y = df[y_cols]
display(X_.head())
display(y.head())

# %%
corr = df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True)
plt.show()

# %%
significant_cols = corr[y_cols].abs().gt(0.1)
significant_cols.drop(index=y_cols, inplace=True)
display(significant_cols)

# %%
wind_significant_cols = list(
    k for k, v in significant_cols["production_wind"].items() if v == True
)
solar_significant_cols = list(
    k for k, v in significant_cols["production_solar"].items() if v == True
)

print("Wind significant columns:", wind_significant_cols)
print("Solar significant columns:", solar_significant_cols)

# %%
standard_scalar = StandardScaler()
standard_scalar.fit(X_)
X = standard_scalar.transform(X_)
X = pd.DataFrame(X, columns=X_.columns)
X.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# %%
y_wind_train = y_train["production_wind"]
y_solar_train = y_train["production_solar"]
y_wind_test = y_test["production_wind"]
y_solar_test = y_test["production_solar"]

# %%
X_wind_train = X_train[wind_significant_cols]
X_solar_train = X_train[solar_significant_cols]
X_wind_test = X_test[wind_significant_cols]
X_solar_test = X_test[solar_significant_cols]

# %%
grid_params = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
}

# %%
grid_search_wind = GridSearchCV(SVR(), grid_params, refit=True, n_jobs=-1, cv=3)
grid_search_solar = GridSearchCV(SVR(), grid_params, refit=True, n_jobs=-1, cv=3)

# %%
grid_search_wind.fit(X_wind_train, y_wind_train)

# %%
grid_search_solar.fit(X_solar_train, y_solar_train)

# %%
print("Best wind params:", grid_search_wind.best_params_)
print("Best solar params:", grid_search_solar.best_params_)

# %%
y_wind_pred = grid_search_wind.predict(X_wind_test)
y_solar_pred = grid_search_solar.predict(X_solar_test)

# %%
print("Wind regression report:")
rmse_wind = mean_squared_error(y_wind_test, y_wind_pred, squared=False)
r2_wind = r2_score(y_wind_test, y_wind_pred)
print("RMSE: ", rmse_wind)
print("R2: ", r2_wind)

# %%
print("Solar regression report:")
rmse_solar = mean_squared_error(y_solar_test, y_solar_pred, squared=False)
r2_solar = r2_score(y_solar_test, y_solar_pred)
print("RMSE: ", rmse_solar)
print("R2: ", r2_solar)

# %%
figsize = (8, 6)

# %%
range_n = 24 * 2
figsize = (8, 6)
plt.figure(figsize=figsize)
plt.plot(
    y_wind_test.to_list()[:range_n],
    label="True wind production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
plt.plot(
    y_wind_pred[:range_n],
    label="Predicted wind production",
    color="orange",
    linestyle="dashed",
)
plt.legend()
plt.show()

# %%
plt.figure(figsize=figsize)
plt.plot(
    y_wind_test.to_list(),
    label="True wind production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
plt.plot(
    y_wind_pred,
    label="Predicted wind production",
    color="orange",
    linestyle="solid"
)
plt.legend()
plt.show()

# %%
plt.figure(figsize=figsize)
plt.plot(
    y_solar_test.to_list()[:range_n],
    label="True solar production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
plt.plot(
    y_solar_pred[:range_n],
    label="Predicted solar production",
    color="orange",
    linestyle="dashed",
)
plt.legend()
plt.show()

# %%
plt.figure(figsize=figsize)
plt.plot(
    y_solar_test.to_list(),
    label="True solar production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
plt.plot(
    y_solar_pred,
    label="Predicted solar production",
    color="orange",
    linestyle="solid",
)
plt.legend()
plt.show()

# %%
print("Total solar production:", np.sum(y_solar_pred))
print("Total wind production :", np.sum(y_wind_pred))

# %%
grid_params = {
    "estimator__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "estimator__C": [0.1, 1, 10, 100, 1000],
}

# %%
# selected params from above tests that were common in both wind and solar prediction
grid = GridSearchCV(
    MultiOutputRegressor(SVR(kernel="rbf")),
    grid_params,
    refit=True,
    n_jobs=-1,
    cv=3,
)

# %%
grid.fit(X_train, y_train)

# %%
y_multi_pred = grid.predict(X_test)

# %%
print("Regression report:")
rmse_multi = mean_squared_error(y_test, y_multi_pred, squared=False)
r2_multi = r2_score(y_test, y_multi_pred)
print("RMSE: ", rmse_multi)
print("R2: ", r2_multi)

# %%
print("Best combinded params:", grid.best_params_)

# %%
df = pd.read_csv(data_path_dates)
X_ = df.drop(columns=y_cols)
y = df[y_cols]
dates = X_["dt_iso"]
X_ = X_.drop(columns=["dt_iso"])
X = standard_scalar.transform(X_)
X = pd.DataFrame(X, columns=X_.columns)

# %%
start_n = 100
range_n = 24 * 2 + start_n
y_pred = grid.predict(X[start_n:range_n])
y_true = y[start_n:range_n]

# %%
dates = pd.to_datetime(dates)
selected_dates = dates[start_n:range_n:8]


# %%
def format_date_time(dt):
    return dt.strftime("%Y-%m-%d\n%H:%M:%S")


# %%
formatted_dates = [format_date_time(dt) for dt in selected_dates]

# %%
fig, ax = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)
ax[0].plot(
    dates[start_n:range_n],
    y_true["production_wind"].to_list(),
    label="True wind production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
ax[0].plot(
    dates[start_n:range_n],
    y_pred[:, 0],
    label="Predicted wind production",
    color="orange",
    linestyle="dashed",
)
ax[1].plot(
    dates[start_n:range_n],
    y_true["production_solar"].to_list(),
    label="True solar production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
ax[1].plot(
    dates[start_n:range_n],
    y_pred[:, 1],
    label="Predicted solar production",
    color="orange",
    linestyle="dashed",
)
ax[0].legend()
ax[1].legend()
ax[0].set_ylabel("Wind production (MW)")
ax[1].set_ylabel("Solar production (MW)")
plt.xlabel("Instance in time")
plt.xticks(selected_dates, formatted_dates)
plt.show()

# %%
print("Total solar production:", np.sum(y_pred[:, 1]))
print("Total wind production :", np.sum(y_pred[:, 0]))

# %%
print("Comparing rmse values with original data standard deviation:")
print("Original wind production std:", df["production_wind"].std())
print("Original solar production std:", df["production_solar"].std())
print()
print("Predicted wind production rmse:", rmse_wind)
print("Predicted solar production rmse:", rmse_solar)
print("Multi-output regression rmse:", rmse_multi)

# %%
y_pred = grid.predict(X)
y_true = y
dates = pd.to_datetime(dates)
selected_dates = dates[::5000]
formatted_dates = [format_date_time(dt) for dt in selected_dates]

# %%
fig, ax = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)
ax[0].plot(
    dates,
    y_true["production_wind"].to_list(),
    label="True wind production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
ax[0].plot(
    dates,
    y_pred[:, 0],
    label="Predicted wind production",
    color="orange",
    linestyle="solid",
)
ax[1].plot(
    dates,
    y_true["production_solar"].to_list(),
    label="True solar production",
    color="blue",
    linestyle="solid",
    alpha=0.5,
)
ax[1].plot(
    dates,
    y_pred[:, 1],
    label="Predicted solar production",
    color="orange",
    linestyle="solid",
)
ax[0].legend()
ax[1].legend()
ax[0].set_ylabel("Wind production (MW)")
ax[1].set_ylabel("Solar production (MW)")
plt.xlabel("Instance in time")
plt.xticks(selected_dates, formatted_dates)
plt.show()
