import logging
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "nsites", "volume_per_atom", "density", "nelements",
    "mean_atomic_mass",       "std_atomic_mass",       "range_atomic_mass",
    "mean_electronegativity", "std_electronegativity", "range_electronegativity",
    "mean_atomic_radius",     "std_atomic_radius",     "range_atomic_radius",
    "mean_atomic_number",     "std_atomic_number",     "range_atomic_number",
    "mean_row",               "std_row",               "range_row",
    "mean_group",             "std_group",             "range_group",
]

def build_pipelines():
    return {
        "rf_bandgap": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42
            )),
        ]),
        "rf_formation": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42
            )),
        ]),
        "gb_bandgap": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42
            )),
        ]),
        "gb_formation": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42
            )),
        ]),
    }

def train_and_evaluate(
    X_train,
    X_test,
    y_bg_train,
    y_bg_test,
    y_fe_train,
    y_fe_test,
    cv_folds = 5,
    ):

    pipelines = build_pipelines()

    task_map = {
        "rf_bandgap":   (y_bg_train, y_bg_test,  "band_gap",         "Random Forest"),
        "rf_formation":  (y_fe_train, y_fe_test,  "formation_energy", "Random Forest"),
        "gb_bandgap":   (y_bg_train, y_bg_test,  "band_gap",         "Gradient Boosting"),
        "gb_formation":  (y_fe_train, y_fe_test,  "formation_energy", "Gradient Boosting"),
    }

    records = []
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, (y_tr, y_te, target, model_label) in task_map.items():
        logger.info("Training %s …", name)
        pipe = pipelines[name]

        cv_scores = cross_val_score(
            pipe, X_train, y_tr, cv=cv,
            scoring="neg_mean_absolute_error", n_jobs=-1
        )
        cv_mae = -cv_scores.mean()

        pipe.fit(X_train, y_tr)
        y_pred = pipe.predict(X_test)

        test_mae  = mean_absolute_error(y_te, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        test_r2   = r2_score(y_te, y_pred)

        logger.info(
            "  %s: CV MAE=%.4f | Test MAE=%.4f | R²=%.4f",
            name, cv_mae, test_mae, test_r2,
        )
        records.append({
            "model":     model_label,
            "target":    target,
            "cv_mae":    round(cv_mae, 4),
            "test_mae":  round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "test_r2":   round(test_r2, 4),
        })

    return pd.DataFrame(records), pipelines

def get_feature_importance(pipelines, top_n = 10):
    out = {}
    for key in ("rf_bandgap", "rf_formation"):
        if key not in pipelines:
            continue
        importances = pipelines[key].named_steps["model"].feature_importances_
        df = (
            pd.DataFrame({"feature": FEATURE_NAMES[: len(importances)], "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        out[key] = df
    return out

def save_models(pipelines, save_dir = "models/baselines"):
    os.makedirs(save_dir, exist_ok=True)
    for name, pipe in pipelines.items():
        path = os.path.join(save_dir, f"{name}.joblib")
        joblib.dump(pipe, path)
        logger.info("Saved %s as %s", name, path)

def load_models(save_dir = "models/baselines"):
    pipes = {}
    for fname in os.listdir(save_dir):
        if fname.endswith(".joblib"):
            name = fname[:-7]
            pipes[name] = joblib.load(os.path.join(save_dir, fname))
    return pipes
