# @name The Keeling curve
import matplotlib.pyplot as plt

plt.style.use("strata")

fig, (curve, year) = plt.subplots(
    1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [4, 1]}
)

curve.plot(monthly["date"], monthly["co2_ppm"], linewidth=1, label="monthly mean")
curve.plot(monthly["date"], monthly["trend_ppm"], linewidth=2.5, label="yearly cycle removed")
latest = monthly.iloc[-1]
curve.annotate(
    f"{latest['co2_ppm']:.1f} ppm\n{latest['date']:%B %Y}",
    (latest["date"], latest["co2_ppm"]),
    xytext=(-15, -100),
    textcoords="offset points",
    ha="right",
)
curve.set_title("CO₂ at Mauna Loa, parts per million")
curve.legend(loc="upper left")

year.bar(cycle["month"], cycle["cycle_ppm"], color="C2")
year.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
year.set_title("The average year")

fig.tight_layout()
fig
