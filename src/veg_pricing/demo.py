import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_sales(days: int = 240, seed: int = 42) -> pd.DataFrame:
    if days < 90:
        raise ValueError("days must be at least 90")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    categories = ["Leafy", "Pepper", "Mushroom", "Root", "Cauliflower", "Eggplant"]
    rows = []
    for sku_index in range(12):
        sku = f"SKU-{sku_index + 1:02d}"
        category = categories[sku_index % len(categories)]
        base_demand = rng.uniform(5.0, 24.0)
        base_price = rng.uniform(6.0, 18.0)
        unit_cost = base_price * rng.uniform(0.48, 0.68)
        elasticity = rng.uniform(1.1, 1.8)
        phase = rng.uniform(0, 2 * np.pi)
        for day_index, date in enumerate(dates):
            promotion = 0.95 if day_index % 21 < 3 else 1.0
            unit_price = base_price * promotion
            weekly = 1 + 0.18 * np.sin(2 * np.pi * date.dayofweek / 7 + phase)
            seasonal = 1 + 0.22 * np.sin(2 * np.pi * day_index / 365 + phase)
            price_response = (base_price / unit_price) ** elasticity
            noise = rng.normal(0, 0.09)
            expected = base_demand * weekly * seasonal * price_response * np.exp(noise)
            units_sold = round(max(0, expected))
            rows.append({
                "date": date,
                "sku": sku,
                "category": category,
                "units_sold": units_sold,
                "unit_price": round(unit_price, 4),
                "unit_cost": round(unit_cost, 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic sales data")
    parser.add_argument("--output", type=Path, default=Path("data/demo_sales.csv"))
    parser.add_argument("--days", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = generate_demo_sales(args.days, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} synthetic rows for {frame['sku'].nunique()} SKUs to {args.output}")


if __name__ == "__main__":
    main()
