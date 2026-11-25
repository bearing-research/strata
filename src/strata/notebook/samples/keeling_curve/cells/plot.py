# @name The Keeling curve
import matplotlib.pyplot as plt

plt.style.use("strata")


def draw(record, average_year):
    fig, (curve, year) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [4, 1]}
    )

    curve.plot(record["date"], record["co2_ppm"], linewidth=1, label="monthly mean")
    curve.plot(record["date"], record["trend_ppm"], linewidth=2.5, label="yearly cycle removed")
    latest = record.iloc[-1]
    curve.annotate(
        f"{latest['co2_ppm']:.1f} ppm\n{latest['date']:%B %Y}",
        (latest["date"], latest["co2_ppm"]),
        xytext=(-15, -100),
        textcoords="offset points",
        ha="right",
        arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.8},
    )
    curve.set_title("CO₂ at Mauna Loa, parts per million")
    curve.legend(loc="upper left")

    year.bar(average_year["month"], average_year["cycle_ppm"], color="C2")
    year.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
    year.set_title("The average year")

    fig.tight_layout()
    return fig


draw(monthly, cycle)
