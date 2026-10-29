from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import PyCO2SYS


root = Path(__file__).resolve().parent
input_file = root / "data" / "observations.csv"
output_dir = root / "outputs"

predictors = [
    "temperature", "water_depth", "do", "aou", "doc", "spm",
    "stratification", "nitrate", "log10_thg", "chl_a", "ph_total",
    "practical_salinity",
]

data = pd.read_csv(input_file)
data["doc"] = data["doc_mg_c_l"] * 1000 / 12.011

pressure = gsw.p_from_z(-data["water_depth"], data["latitude"])
absolute_salinity = gsw.SA_from_SP(
    data["practical_salinity"], pressure, data["longitude"], data["latitude"]
)
conservative_temperature = gsw.CT_from_t(
    absolute_salinity, data["temperature"], pressure
)
density = gsw.rho(absolute_salinity, conservative_temperature, pressure)
oxygen_saturation = gsw.O2sol(
    absolute_salinity, conservative_temperature, pressure,
    data["longitude"], data["latitude"],
)
measured_oxygen = data["do"] * 1000000 / (31.998 * density)
data["aou"] = oxygen_saturation - measured_oxygen
data["sigma0"] = gsw.sigma0(absolute_salinity, conservative_temperature)

data["stratification"] = np.nan
for profile_id, profile in data.groupby("profile_id"):
    valid = profile.dropna(subset=["water_depth", "sigma0"])
    valid = valid.sort_values("water_depth")
    if len(valid) >= 2 and valid["water_depth"].nunique() >= 2:
        difference = valid["sigma0"].iloc[-1] - valid["sigma0"].iloc[0]
        data.loc[profile.index, "stratification"] = difference

if "ph_total" not in data.columns:
    data["ph_total"] = np.nan

if "ph_original" in data.columns:
    available_scales = data["ph_input_scale"].dropna().unique()
    for scale in available_scales:
        rows = data["ph_total"].isna() & data["ph_original"].notna()
        rows &= data["ph_input_scale"].eq(scale)
        rows &= data[["temperature", "practical_salinity", "water_depth"]].notna().all(axis=1)
        if rows.any():
            converted = PyCO2SYS.sys(
                par1=data.loc[rows, "ph_original"].to_numpy(),
                par1_type=3,
                salinity=data.loc[rows, "practical_salinity"].to_numpy(),
                temperature=data.loc[rows, "temperature"].to_numpy(),
                pressure=np.asarray(pressure)[rows.to_numpy()],
                opt_pH_scale=int(scale),
            )
            data.loc[rows, "ph_total"] = converted["pH_total"]

if not np.isfinite(data["mehg"]).all() or data["mehg"].le(0).any():
    raise ValueError("MeHg must contain finite positive concentrations.")

data["log10_mehg"] = np.log10(data["mehg"])
data["log10_thg"] = np.log10(data["thg"].where(data["thg"] > 0))
data[predictors] = data[predictors].replace([np.inf, -np.inf], np.nan)

output_dir.mkdir(parents=True, exist_ok=True)
columns = ["station_id", "profile_id", "log10_mehg", *predictors]
data[columns].to_csv(output_dir / "training_data.csv", index=False)
print("Training data saved:", len(data), "rows")
