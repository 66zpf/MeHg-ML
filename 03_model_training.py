import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from xgboost import XGBRegressor


root = Path(__file__).resolve().parent
output = root / "outputs"
seed = 42
trials_per_algorithm = 64
startup_trials = 12
predictors = [
    "temperature", "water_depth", "do", "aou", "doc", "spm",
    "stratification", "nitrate", "log10_thg", "chl_a", "ph_total",
    "practical_salinity",
]
search_spaces = json.loads((root / "search_spaces.json").read_text(encoding="utf-8"))
algorithms = list(search_spaces)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_model(algorithm, parameters):
    settings = dict(search_spaces[algorithm]["fixed"])
    settings.update(parameters)
    if algorithm == "extratrees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **settings)
    if algorithm == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **settings)
    if algorithm == "lightgbm":
        return LGBMRegressor(
            random_state=seed, n_jobs=-1, verbosity=-1,
            subsample_freq=1, **settings,
        )
    if algorithm == "xgboost":
        return XGBRegressor(
            random_state=seed, n_jobs=-1, tree_method="hist",
            objective="reg:squarederror", **settings,
        )
    if algorithm == "catboost":
        return CatBoostRegressor(
            random_seed=seed, verbose=False, allow_writing_files=False,
            bootstrap_type="Bayesian", **settings,
        )
    if algorithm == "hgb":
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **settings)
    raise ValueError(algorithm)


def select_models(features, target, stations, grouped=False):
    if grouped:
        inner_folds = list(GroupKFold(n_splits=3).split(features, target, stations))
    else:
        inner_folds = list(KFold(n_splits=3, shuffle=True, random_state=seed).split(features))
    tested = {}
    candidates = []

    for algorithm in algorithms:
        tested[algorithm] = []

        def objective(trial):
            parameters = {}
            for name, space in search_spaces[algorithm]["parameters"].items():
                if space["type"] == "categorical":
                    parameters[name] = trial.suggest_categorical(name, space["choices"])
                elif space["type"] == "int":
                    parameters[name] = trial.suggest_int(name, space["low"], space["high"], step=space.get("step", 1))
                else:
                    parameters[name] = trial.suggest_float(name, space["low"], space["high"], log=space.get("log", False))
            model = make_model(algorithm, parameters)
            prediction = cross_val_predict(model, features, target, cv=inner_folds, n_jobs=1)
            score = float(r2_score(target, prediction))
            tested[algorithm].append({"parameters": parameters, "prediction": prediction, "r2": score})
            return score

        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=trials_per_algorithm)
        best = max(tested[algorithm], key=lambda item: item["r2"])
        candidates.append({
            "name": algorithm,
            "algorithms": [algorithm],
            "parameters": [best["parameters"]],
            "weights": [1.0],
            "r2": best["r2"],
            "rmse": float(root_mean_squared_error(target, best["prediction"])),
            "mae": float(mean_absolute_error(target, best["prediction"])),
        })

    candidates.sort(key=lambda item: item["r2"], reverse=True)
    ranked_algorithms = [item["algorithms"][0] for item in candidates]
    ensemble_pairs = list(combinations(ranked_algorithms[:3], 2))
    ensemble_pairs += [pair for pair in combinations(ranked_algorithms, 2) if pair not in ensemble_pairs]

    for first, second in ensemble_pairs:
        best_error = np.inf
        best_pair = None
        for first_trial in tested[first]:
            for second_trial in tested[second]:
                first_prediction = first_trial["prediction"]
                second_prediction = second_trial["prediction"]
                difference = first_prediction - second_prediction
                denominator = np.dot(difference, difference)
                if denominator > 0:
                    numerator = np.dot(target - second_prediction, difference)
                    weight = float(np.clip(numerator / denominator, 0, 1))
                else:
                    weight = 0.5
                prediction = weight * first_prediction + (1 - weight) * second_prediction
                error = float(np.dot(target - prediction, target - prediction))
                if error < best_error:
                    best_error = error
                    best_pair = {
                        "name": " + ".join(sorted([first, second])),
                        "algorithms": [first, second],
                        "parameters": [first_trial["parameters"], second_trial["parameters"]],
                        "weights": [weight, 1 - weight],
                        "r2": float(r2_score(target, prediction)),
                        "rmse": float(root_mean_squared_error(target, prediction)),
                        "mae": float(mean_absolute_error(target, prediction)),
                    }
        candidates.append(best_pair)

    candidates.sort(key=lambda item: item["r2"], reverse=True)
    return candidates[0], candidates


data = pd.read_csv(output / "training_data.csv")
features = data[predictors]
target = data["log10_mehg"].to_numpy()
stations = data["station_id"].to_numpy()
prediction_tables = []
comparison_rows = []

for scheme in ["random", "station_grouped"]:
    if scheme == "random":
        outer_folds = KFold(n_splits=5, shuffle=True, random_state=seed).split(features)
    else:
        outer_folds = GroupKFold(n_splits=5).split(features, target, stations)

    for fold, (train_index, test_index) in enumerate(outer_folds, start=1):
        print(scheme, fold)
        train_features = features.iloc[train_index]
        train_target = target[train_index]
        selected, candidates = select_models(
            train_features, train_target, stations[train_index], grouped=scheme == "station_grouped"
        )
        for candidate in candidates:
            comparison_rows.append({"scheme": scheme, "fold": fold, **candidate})

        test_candidates = [candidate for candidate in candidates if len(candidate["algorithms"]) == 1]
        test_candidates.append({**selected, "name": "selected_model"})
        fitted = {}
        for candidate in test_candidates:
            prediction = np.zeros(len(test_index))
            for algorithm, parameters, weight in zip(
                candidate["algorithms"], candidate["parameters"], candidate["weights"]
            ):
                key = (algorithm, json.dumps(parameters, sort_keys=True))
                if key not in fitted:
                    fitted[key] = make_model(algorithm, parameters)
                    fitted[key].fit(train_features, train_target)
                prediction += weight * fitted[key].predict(features.iloc[test_index])
            prediction_tables.append(pd.DataFrame({
                "row_id": test_index,
                "station_id": stations[test_index],
                "scheme": scheme,
                "fold": fold,
                "model": candidate["name"],
                "algorithms": " + ".join(candidate["algorithms"]),
                "observed_log10_mehg": target[test_index],
                "predicted_log10_mehg": prediction,
            }))

predictions = pd.concat(prediction_tables, ignore_index=True)
predictions.to_csv(output / "pooled_oof_predictions.csv", index=False)
performance_rows = []
for (scheme, model), group in predictions.groupby(["scheme", "model"], sort=False):
    observed = group["observed_log10_mehg"]
    predicted = group["predicted_log10_mehg"]
    performance_rows.append({
        "scheme": scheme,
        "model": model,
        "r2": r2_score(observed, predicted),
        "rmse": root_mean_squared_error(observed, predicted),
        "mae": mean_absolute_error(observed, predicted),
    })
pd.DataFrame(performance_rows).to_csv(output / "model_performance.csv", index=False)

selected, candidates = select_models(features, target, stations)
for candidate in candidates:
    comparison_rows.append({"scheme": "full_data_inner", "fold": 0, **candidate})
comparison = pd.DataFrame(comparison_rows)
for column in ["algorithms", "parameters", "weights"]:
    comparison[column] = comparison[column].map(json.dumps)
comparison.to_csv(output / "model_comparison.csv", index=False)

members = []
for algorithm, parameters, weight in zip(
    selected["algorithms"], selected["parameters"], selected["weights"]
):
    model = make_model(algorithm, parameters)
    model.fit(features, target)
    members.append({"algorithm": algorithm, "weight": weight, "model": model})
joblib.dump({"predictors": predictors, "prediction_scale": "log10_mehg", "members": members}, output / "model.joblib")
(output / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
print(selected["name"], selected["weights"])

future = pd.read_csv(output / "scenario_inputs.csv")
future_features = future[predictors]
future_log10_mehg = np.zeros(len(future))
for member in members:
    future_log10_mehg += member["weight"] * member["model"].predict(future_features)
future["predicted_mehg"] = 10 ** future_log10_mehg
if not np.isfinite(future["predicted_mehg"]).all():
    raise ValueError("Scenario predictions must be finite")
if not np.isfinite(future["area_m2"]).all() or (future["area_m2"] <= 0).any():
    raise ValueError("Grid-cell areas must be finite and positive")

future["weighted_mehg"] = future["predicted_mehg"] * future["area_m2"]
group_columns = ["experiment", "year", "region", "thg_reduction"]
regional = future.groupby(group_columns, as_index=False).agg(
    weighted_mehg=("weighted_mehg", "sum"),
    area_m2=("area_m2", "sum"),
)
regional["predicted_mehg"] = regional["weighted_mehg"] / regional["area_m2"]
regional = regional.drop(columns="weighted_mehg")
baseline = regional.loc[
    (regional["experiment"] == "ssp245")
    & (regional["year"] == 2017)
    & (regional["thg_reduction"] == 0),
    ["region", "predicted_mehg"],
].rename(columns={"predicted_mehg": "baseline_mehg"})
regional = regional.merge(baseline, on="region", how="left", validate="many_to_one")
if regional["baseline_mehg"].isna().any() or (regional["baseline_mehg"] <= 0).any():
    raise ValueError("Each sea requires a positive SSP2-4.5 baseline for 2017")
regional["response_percent"] = 100 * (
    regional["predicted_mehg"] / regional["baseline_mehg"] - 1
)
regional.to_csv(output / "regional_scenario_predictions.csv", index=False)
