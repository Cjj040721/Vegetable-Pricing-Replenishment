import argparse
import json
from pathlib import Path

import pandas as pd

from .forecast import train_evaluate_and_forecast
from .optimise import optimise_plan
from .reporting import save_report_figure

REQUIRED_COLUMNS = {"date", "sku", "category", "units_sold", "unit_price", "unit_cost"}


def run(data_path: Path, output_dir: Path, validation_days: int, min_skus: int, max_skus: int, elasticity: float) -> dict:
    frame = pd.read_csv(data_path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame["units_sold"].lt(0).any() or frame["unit_price"].lt(0).any() or frame["unit_cost"].lt(0).any():
        raise ValueError("Sales and monetary values must be non-negative")
    metrics, forecast = train_evaluate_and_forecast(frame, validation_days=validation_days)
    plan, optimisation = optimise_plan(
        forecast,
        min_skus_per_day=min_skus,
        max_skus_per_day=max_skus,
        elasticity=elasticity,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "demand_forecast.csv", index=False)
    plan.to_csv(output_dir / "replenishment_plan.csv", index=False)
    save_report_figure(plan, output_dir / "demo_report.png")
    summary = {"validation": metrics, "optimisation": optimisation}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast demand and optimise a seven-day pricing/replenishment plan")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument("--min-skus", type=int, default=6)
    parser.add_argument("--max-skus", type=int, default=10)
    parser.add_argument("--elasticity", type=float, default=2.2)
    args = parser.parse_args()
    summary = run(args.data, args.output, args.validation_days, args.min_skus, args.max_skus, args.elasticity)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
