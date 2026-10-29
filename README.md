# MeHg-ML

Code and data for regional seawater methylmercury (MeHg) analysis in the East China Sea, Yellow Sea, and Bohai Sea.

## Code

| File | Description |
| --- | --- |
| [01_observations.py](01_observations.py) | Processes observation units, pH, AOU, stratification, and log-transformed model inputs. |
| [02_cmip6_scenarios.py](02_cmip6_scenarios.py) | Converts monthly GFDL-ESM4 data into annual environmental inputs and constructs five THg decline pathways. |
| [03_model_training.py](03_model_training.py) | Compares six machine learning algorithms and weighted ensembles, tunes model parameters and ensemble weights, and predicts regional MeHg responses. |
| [04_input_ranges.py](04_input_ranges.py) | Compares 2060 environmental inputs with observed ranges and calculates the grid-cell area outside those ranges. |
| [search_spaces.json](search_spaces.json) | Hyperparameter search ranges for ExtraTrees, random forest, LightGBM, XGBoost, CatBoost, and histogram-based gradient boosting. |
| [requirements.txt](requirements.txt) | Python packages used by the scripts. |

## Data

### Observations

[date.xlsx](date.xlsx) contains 1,004 seawater observation records in the `Data` sheet, with 16 fields. These include water depth, temperature, salinity, dissolved oxygen, pH, chlorophyll a, DOC, nitrate, SPM, THg, and MeHg. Variable names and units are given in the column headings.

### Annual scenario results

[annual_regional_mehg_scenarios_2017_2060.csv](annual_regional_mehg_scenarios_2017_2060.csv) contains 1,320 annual regional results for 2017–2060. It covers the East China Sea, Yellow Sea, and Bohai Sea under SSP2-4.5 and SSP5-8.5, with THg reductions of 0%, 10%, 30%, 50%, and 80% by 2060.

| Column | Description |
| --- | --- |
| `region` | Sea region. |
| `climate_scenario` | Climate scenario: SSP2-4.5 or SSP5-8.5. |
| `thg_reduction_2060_pct` | Prescribed THg reduction by 2060 relative to 2017 (%). |
| `year` | Calendar year. |
| `thg_fraction_remaining` | THg fraction remaining relative to 2017. |
| `predicted_mehg_ng_L` | Predicted regional MeHg concentration (ng/L). |
| `reference_mehg_ng_L` | Reference MeHg concentration used to calculate the response (ng/L). |
| `mehg_response_pct` | MeHg change relative to the reference (%). |
