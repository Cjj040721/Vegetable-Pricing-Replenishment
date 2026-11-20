from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def save_report_figure(plan: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    profit = plan.groupby("sku")["expected_profit"].sum().sort_values(ascending=False)
    profit.plot(kind="bar", ax=axes[0], color="#2AA887")
    axes[0].set_title("Expected profit by selected SKU")
    axes[0].set_ylabel("Synthetic currency units")
    axes[0].tick_params(axis="x", rotation=45)
    daily = plan.groupby("forecast_date").agg(
        replenishment=("replenishment_kg", "sum"),
        expected_profit=("expected_profit", "sum"),
    )
    daily["replenishment"].plot(ax=axes[1], marker="o", color="#6E56CF")
    axes[1].set_title("Seven-day replenishment plan")
    axes[1].set_ylabel("Kilograms")
    axes[1].ticklabel_format(style="plain", axis="y", useOffset=False)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].tick_params(axis="x", rotation=30)
    figure.suptitle("Demand forecasting + mixed-integer optimisation (synthetic demo)")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
