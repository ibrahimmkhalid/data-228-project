import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as animation
import pandas as pd
import datetime

df = pd.read_csv("./data/france_weather_energy_with_date.csv")

df["dt_iso"] = pd.to_datetime(df["dt_iso"])

number_of_days = 7
start_date = "2022-01-13"
end_date = datetime.datetime.strptime(start_date, "%Y-%m-%d") + datetime.timedelta(days=number_of_days)
end_date = end_date.strftime("%Y-%m-%d")
df_plot = df[df["dt_iso"] >= start_date]
df_plot = df_plot[df_plot["dt_iso"] <= end_date]
plt.figure(figsize=(10, 6))
plt.plot(df_plot["dt_iso"], df_plot["production_wind"], label="Wind")
plt.plot(df_plot["dt_iso"], df_plot["production_solar"], label="Solar")
plt.xlabel("Date")
plt.ylabel("Production (MW)")
plt.legend()
plt.savefig("energy_production_2.png")


fig, ax = plt.subplots(figsize=(14, 10))
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
ani.save("energy_production.mp4")
