from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
output = root / "outputs"

training = pd.read_csv(output / "training_data.csv")
future = pd.read_csv(output / "scenario_inputs.csv")

variables = [
    "temperature", "ph_total", "chl_a", "practical_salinity",
    "doc", "do", "stratification", "aou",
]
experiments = ["ssp245", "ssp585"]
regions = ["ECS", "YS", "BS"]
required = ["experiment", "year", "region", "cell_id", "area_m2", "thg_reduction"]

if not set(variables).issubset(training.columns):
    raise ValueError("Training data are missing required columns.")
if not set(required + variables).issubset(future.columns):
    raise ValueError("Scenario inputs are missing required columns.")

future = future.loc[(future["year"] == 2060) & future["experiment"].isin(experiments)]
if future[required + variables].isna().any().any():
    raise ValueError("Scenario inputs contain missing values.")
future = future.drop_duplicates(["experiment", "region", "cell_id"])
if not np.isfinite(future[variables + ["area_m2"]].to_numpy()).all():
    raise ValueError("Scenario inputs contain non-finite values.")
if (future["area_m2"] <= 0).any():
    raise ValueError("Grid-cell areas must be positive.")

rows = []
for variable in variables:
    training_min = training[variable].min()
    training_max = training[variable].max()
    if not np.isfinite([training_min, training_max]).all():
        raise ValueError(f"No valid training range for {variable}.")
    for experiment in experiments:
        for region in regions:
            group = future.loc[(future["experiment"] == experiment) & (future["region"] == region)]
            if group.empty:
                raise ValueError(f"No 2060 inputs for {experiment}, {region}.")
            outside = (group[variable] < training_min) | (group[variable] > training_max)
            outside_percent = 100 * group.loc[outside, "area_m2"].sum() / group["area_m2"].sum()
            rows.append({
                "variable": variable, "experiment": experiment, "region": region,
                "training_min": training_min, "training_max": training_max,
                "future_min": group[variable].min(), "future_max": group[variable].max(),
                "area_outside_percent": outside_percent,
            })

pd.DataFrame(rows).to_csv(output / "input_ranges_2060.csv", index=False)
