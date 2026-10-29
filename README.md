# Regional MeHg analysis

Four ordinary Python scripts follow the Supporting Information. Each script is organized from input to output. Paths and study settings are written near the beginning of each file.

| File | Content |
| --- | --- |
| `01_observations.py` | Observation units, pH, AOU, stratification, and model inputs |
| `02_cmip6_scenarios.py` | Monthly CMIP6 processing, annual inputs, and THg pathways |
| `03_model_training.py` | Six algorithms, tuning, continuous ensemble weights, comparison, and scenario prediction |
| `04_input_ranges.py` | Training ranges, 2060 ranges, and area outside the training range |

The scripts form a working outline for connecting local data.  Input names below define the expected file layout and can be changed at the beginning of each script.

## Input layout

```text
data/
  observations.csv
  ocean_grid.csv
  station_endpoints.csv
  background_2017.csv
  cmip6/
    ssp245/
      thetao.nc
      so.nc
      o2.nc
      ph.nc
      dissoc.nc
      chl.nc
    ssp585/
      thetao.nc
      so.nc
      o2.nc
      ph.nc
      dissoc.nc
      chl.nc
```

`observations.csv` contains the retained observation records. Columns are `station_id`, `profile_id`, `longitude`, `latitude`, `water_depth`, `temperature`, `practical_salinity`, `do`, `doc_mg_c_l`, `spm`, `nitrate`, `thg`, `chl_a`, `mehg`, and either `ph_total` or `ph_original` with `ph_input_scale`. Original pH scale codes follow the [PyCO2SYS documentation](https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/). Concentrations of MeHg and THg are in ng/L. Other units follow Table S1. No additional sample-screening rules are introduced.

`ocean_grid.csv` contains one row per ocean cell, with `cell_id`, `region`, `longitude`, `latitude`, and `area_m2`. Region codes are ECS, YS, and BS. Coordinates match the climate grid. `background_2017.csv` contains `cell_id`, `thg`, `spm`, and `nitrate`.

`station_endpoints.csv` contains `station_id`, `region`, `longitude`, `latitude`, `shallow_depth`, and `deep_depth`.

CMIP6 files use coordinate names `time`, `lev`, `lat`, and `lon`. Depth bounds use `lev_bnds` with a `bnds` dimension. The scripts use these standard names as the data-connection framework.

## Running order

```bash
python -m pip install -r requirements.txt
python 01_observations.py
python 02_cmip6_scenarios.py
python 03_model_training.py
python 04_input_ranges.py
```

The training script compares tuned single models and continuously optimized ensembles. It does not prescribe a winning model or return fixed weights. `search_spaces.json` records Table S3. Outputs are written under `outputs/`.

## Main outputs

| File | Content |
| --- | --- |
| `training_data.csv` | Target and 12 predictors |
| `scenario_inputs.csv` | Annual climate inputs and five THg pathways |
| `model_performance.csv` | Pooled outer-fold metrics |
| `model_comparison.csv` | Inner-fold model comparison and selected parameters |
| `pooled_oof_predictions.csv` | Held-out predictions |
| `selected_model.json` | Final selection, parameters, and weights |
| `model.joblib` | Fitted model members |
| `regional_scenario_predictions.csv` | Area-weighted regional concentrations and responses |
| `input_ranges_2060.csv` | Environmental range comparison |
