from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr


root = Path(__file__).resolve().parent
data_dir = root / "data"
output_dir = root / "outputs"
experiments = ["ssp245", "ssp585"]
years = list(range(2017, 2061))
reductions = [0.0, 0.1, 0.3, 0.5, 0.8]
variables = ["thetao", "so", "o2", "ph", "dissoc", "chl"]
environment = ["temperature", "practical_salinity", "do", "ph_total", "doc", "chl_a"]

grid = pd.read_csv(data_dir / "ocean_grid.csv")
stations = pd.read_csv(data_dir / "station_endpoints.csv")
background = pd.read_csv(data_dir / "background_2017.csv")
annual_tables = []
station_tables = []

for experiment in experiments:
    fields = []
    for variable in variables:
        source = xr.open_dataset(data_dir / "cmip6" / experiment / f"{variable}.nc")
        source = source.sortby("lat").sortby("lon")
        source = source.sel(time=slice("2017-01-01", "2060-12-31"))
        source = source.sel(lon=slice(116.5, 130.5), lat=slice(22.5, 41.5))
        if "lev" in source.dims:
            source = source.sel(lev=slice(0, 250))
        if variable == "thetao":
            depth_bounds = source["lev_bnds"].load()
        fields.append(source[[variable]])

    monthly = xr.merge(fields, join="exact")
    if monthly.sizes["time"] != len(years) * 12:
        raise ValueError("Expected 12 months for every year from 2017 to 2060.")

    pressure = xr.apply_ufunc(gsw.p_from_z, -monthly["lev"], monthly["lat"])
    absolute_salinity = xr.apply_ufunc(
        gsw.SA_from_SP, monthly["so"], pressure, monthly["lon"], monthly["lat"]
    )
    conservative_temperature = xr.apply_ufunc(
        gsw.CT_from_pt, absolute_salinity, monthly["thetao"]
    )
    temperature = xr.apply_ufunc(
        gsw.t_from_CT, absolute_salinity, conservative_temperature, pressure
    )

    converted = xr.Dataset({
        "temperature": temperature,
        "practical_salinity": monthly["so"],
        "do": monthly["o2"] * 31.998,
        "ph_total": monthly["ph"],
        "doc": monthly["dissoc"] * 1000,
        "chl_a": monthly["chl"] * 1000000,
    })

    lower = depth_bounds.isel(bnds=0).clip(min=0, max=10)
    upper = depth_bounds.isel(bnds=1).clip(min=0, max=10)
    overlap = (upper - lower).clip(min=0)
    levels = overlap["lev"].where(overlap > 0, drop=True)
    surface = converted.sel(lev=levels).weighted(overlap.sel(lev=levels)).mean("lev", skipna=False)
    month_days = surface["time"].dt.days_in_month
    annual = (surface * month_days).groupby("time.year").sum("time", skipna=False)
    annual = annual / month_days.groupby("time.year").sum("time")
    annual = annual.sel(year=years)

    table = annual.to_dataframe().reset_index()
    table = table.rename(columns={"lon": "longitude", "lat": "latitude"})
    table = table.merge(grid, on=["longitude", "latitude"], how="inner", validate="many_to_one")
    table["experiment"] = experiment
    annual_tables.append(table)

    for station in stations.itertuples(index=False):
        profile = monthly[["thetao", "so"]].sel(
            lon=station.longitude, lat=station.latitude, method="nearest"
        )
        shallow_depth = max(station.shallow_depth, float(profile["lev"].min()))
        deep_depth = station.deep_depth
        if station.shallow_depth < 2.5:
            shallow_depth = float(profile["lev"].min())
        if deep_depth > float(profile["lev"].max()) or deep_depth <= shallow_depth:
            continue

        endpoints = profile.interp(lev=[shallow_depth, deep_depth])
        endpoint_pressure = xr.apply_ufunc(gsw.p_from_z, -endpoints["lev"], station.latitude)
        endpoint_salinity = xr.apply_ufunc(
            gsw.SA_from_SP, endpoints["so"], endpoint_pressure, station.longitude, station.latitude
        )
        endpoint_temperature = xr.apply_ufunc(
            gsw.CT_from_pt, endpoint_salinity, endpoints["thetao"]
        )
        sigma0 = xr.apply_ufunc(gsw.sigma0, endpoint_salinity, endpoint_temperature)
        difference = sigma0.isel(lev=1) - sigma0.isel(lev=0)
        if not np.isfinite(difference.to_numpy()).all():
            continue
        days = difference["time"].dt.days_in_month
        station_annual = (difference * days).groupby("time.year").sum("time", skipna=False)
        station_annual = station_annual / days.groupby("time.year").sum("time")
        station_table = station_annual.to_dataframe(name="stratification").reset_index()
        station_table["station_id"] = station.station_id
        station_table["region"] = station.region
        station_table["experiment"] = experiment
        station_tables.append(station_table[["experiment", "year", "station_id", "region", "stratification"]])

annual_inputs = pd.concat(annual_tables, ignore_index=True)
valid_rows = np.isfinite(annual_inputs[environment]).all(axis=1)
annual_inputs["valid"] = valid_rows
cell_validity = annual_inputs.groupby("cell_id")["valid"].agg(["all", "count"])
valid_cells = cell_validity.index[cell_validity["all"] & cell_validity["count"].eq(len(years) * len(experiments))]
annual_inputs = annual_inputs[annual_inputs["cell_id"].isin(valid_cells)].drop(columns="valid")

station_inputs = pd.concat(station_tables, ignore_index=True)
station_counts = station_inputs.groupby("station_id").size()
fixed_stations = station_counts.index[station_counts.eq(len(years) * len(experiments))]
station_inputs = station_inputs[station_inputs["station_id"].isin(fixed_stations)]
stratification = station_inputs.groupby(["experiment", "year", "region"], as_index=False)["stratification"].mean()
annual_inputs = annual_inputs.merge(stratification, on=["experiment", "year", "region"], how="left")
annual_inputs = annual_inputs.merge(background, on="cell_id", how="left", validate="many_to_one")

if annual_inputs.empty or annual_inputs[["stratification", "thg", "spm", "nitrate"]].isna().any().any():
    raise ValueError("Annual inputs require fixed valid cells, stations, and 2017 background values.")
if annual_inputs["thg"].le(0).any():
    raise ValueError("2017 THg concentrations must be positive.")

annual_inputs["water_depth"] = 5.0
pressure = gsw.p_from_z(-annual_inputs["water_depth"], annual_inputs["latitude"])
absolute_salinity = gsw.SA_from_SP(
    annual_inputs["practical_salinity"], pressure,
    annual_inputs["longitude"], annual_inputs["latitude"],
)
conservative_temperature = gsw.CT_from_t(absolute_salinity, annual_inputs["temperature"], pressure)
density = gsw.rho(absolute_salinity, conservative_temperature, pressure)
saturation = gsw.O2sol(
    absolute_salinity, conservative_temperature, pressure,
    annual_inputs["longitude"], annual_inputs["latitude"],
)
annual_inputs["aou"] = saturation - annual_inputs["do"] * 1000000 / (31.998 * density)

scenario_tables = []
for reduction in reductions:
    scenario = annual_inputs.copy()
    progress = (scenario["year"] - 2017) / (2060 - 2017)
    scenario["thg"] = scenario["thg"] * (1 - reduction * progress)
    scenario["log10_thg"] = np.log10(scenario["thg"])
    scenario["thg_reduction"] = reduction
    scenario_tables.append(scenario)

scenario_inputs = pd.concat(scenario_tables, ignore_index=True)
output_dir.mkdir(parents=True, exist_ok=True)
scenario_inputs.to_csv(output_dir / "scenario_inputs.csv", index=False)
print("Scenario inputs saved:", len(scenario_inputs), "rows")
