import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def optimise_plan(
    forecast: pd.DataFrame,
    min_skus_per_day: int = 6,
    max_skus_per_day: int = 10,
    min_display_kg: float = 2.5,
    elasticity: float = 2.2,
    price_factors: tuple[float, ...] = (0.95, 1.0, 1.05),
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    scenarios = forecast.loc[forecast.index.repeat(len(price_factors))].copy().reset_index(drop=True)
    scenarios["price_factor"] = np.tile(price_factors, len(forecast))
    scenarios["price"] = scenarios["unit_price"] * scenarios["price_factor"]
    scenarios["adjusted_demand"] = scenarios["predicted_demand"] * scenarios["price_factor"] ** (-elasticity)
    scenarios["replenishment_kg"] = scenarios["adjusted_demand"].clip(lower=min_display_kg)
    scenarios["waste_kg"] = scenarios["replenishment_kg"] - scenarios["adjusted_demand"]
    scenarios["expected_profit"] = (
        scenarios["adjusted_demand"] * (scenarios["price"] - scenarios["unit_cost"])
        - scenarios["waste_kg"] * scenarios["unit_cost"]
    )

    groups = list(scenarios.groupby(["forecast_date", "sku"], sort=False).indices.values())
    dates = list(scenarios.groupby("forecast_date", sort=False).indices.values())
    constraint_count = len(groups) + len(dates)
    matrix = lil_matrix((constraint_count, len(scenarios)), dtype=float)
    lower = np.zeros(constraint_count)
    upper = np.ones(constraint_count)
    for row, indices in enumerate(groups):
        matrix[row, indices] = 1
    date_offset = len(groups)
    for offset, indices in enumerate(dates):
        row = date_offset + offset
        matrix[row, indices] = 1
        lower[row] = min_skus_per_day
        upper[row] = max_skus_per_day

    result = milp(
        c=-scenarios["expected_profit"].to_numpy(),
        integrality=np.ones(len(scenarios)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 30},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    selected = scenarios[result.x > 0.5].copy()
    selected = selected.sort_values(["forecast_date", "sku"])
    summary = {
        "solver_status": result.message,
        "selected_rows": int(len(selected)),
        "expected_replenishment_kg": round(float(selected["replenishment_kg"].sum()), 4),
        "expected_profit": round(float(selected["expected_profit"].sum()), 4),
    }
    columns = [
        "forecast_date", "sku", "category", "price_factor", "price", "unit_cost",
        "predicted_demand", "adjusted_demand", "replenishment_kg", "waste_kg", "expected_profit",
    ]
    return selected[columns], summary
