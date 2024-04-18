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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
random_state = 228


# %%
df = pd.read_csv("./data/france_weather_energy.csv")
df.head()

# %%
df.shape

# %%
df = df.sample(4000, random_state=random_state)
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
wind_significant_cols = list(k for k, v in significant_cols["production_wind"].items() if v==True)
solar_significant_cols = list(k for k, v in significant_cols["production_solar"].items() if v==True)

print("Wind significant columns:", wind_significant_cols)
print("Solar significant columns:", solar_significant_cols)

# %%
standard_scalar = StandardScaler()
standard_scalar.fit(X_)
X = standard_scalar.transform(X_)
X = pd.DataFrame(X, columns=X_.columns)
X.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

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
    "kernel": ["rbf", "linear", "poly", "sigmoid"]
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
print(rmse_wind)

# %%
print("Solar regression report:")
rmse_solar = mean_squared_error(y_solar_test, y_solar_pred, squared=False)
print(rmse_solar)

# %%
range_n = 24*3
plt.figure(figsize=(14, 10))
plt.plot(y_wind_test.to_list()[:range_n], label="True wind production", color="blue", linestyle="solid", alpha=0.5)
plt.plot(y_wind_pred[:range_n], label="Predicted wind production", color="blue", linestyle="dashed")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(14, 10))
plt.plot(y_solar_test.to_list()[:range_n], label="True solar production", color="orange", linestyle="solid", alpha=0.5)
plt.plot(y_solar_pred[:range_n], label="Predicted solar production", color="orange", linestyle="dashed")
plt.legend()
plt.show()

# %%
print("Total solar production:", np.sum(y_solar_pred))
print("Total wind production :", np.sum(y_wind_pred))


# %% 
grid_params = {
    "estimator__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
}

# %%
# selected params from above tests that were common in both wind and solar prediction
grid = GridSearchCV(MultiOutputRegressor(SVR(kernel="rbf", C=1000)), grid_params, refit=True, n_jobs=-1, cv=3)

# %%
grid.fit(X_train, y_train)

# %%
y_multi_pred = grid.predict(X_test)

# %%
print("Regression report:")
rmse_multi = mean_squared_error(y_test, y_multi_pred, squared=False)
print(rmse_multi)


# %%
range_n = 24*2
y_pred = grid.predict(X[:range_n])
y_true = y[:range_n]

# %%
plt.figure(figsize=(14, 10))
plt.plot(y_true["production_wind"].to_list(), label="True wind production", color="blue", linestyle="solid", alpha=0.5)
plt.plot(y_pred[:, 0], label="Predicted wind production", color="blue", linestyle="dashed")
plt.plot(y_true["production_solar"].to_list(), label="True solar production", color="orange", linestyle="solid", alpha=0.5)
plt.plot(y_pred[:, 1], label="Predicted solar production", color="orange", linestyle="dashed")
plt.legend()
plt.ylabel("Production (MW)")
plt.xlabel("Instance in time")
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
