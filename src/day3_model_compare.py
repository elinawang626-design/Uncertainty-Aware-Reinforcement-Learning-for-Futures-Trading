"""Fit the original Random Forest and Ridge baselines on the training partition."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .day2_conformal import ALPHAS, add_interval, conformal_q, volatility_scale
from .prep_day1 import FEATURES


def fit_forecasters(train, calibration):
    models = {
        "rf": RandomForestRegressor(n_estimators=600, min_samples_leaf=5, random_state=42, n_jobs=1),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train["market_return"])
        residuals = np.abs(calibration["market_return"] - model.predict(calibration[FEATURES]))
        scale = volatility_scale(calibration)
        quantiles = {
            round(100 * (1 - alpha)): (conformal_q(residuals / scale, alpha), conformal_q(residuals, alpha))
            for alpha in ALPHAS
        }
        fitted[name] = model, quantiles
    return fitted


def forecast(frame, forecaster):
    model, quantiles = forecaster
    out = frame[["entry_date", "exit_date", "market_return"]].copy()
    out["prediction"] = model.predict(frame[FEATURES])
    scale = volatility_scale(frame)
    for level, (scaled_q, fixed_q) in quantiles.items():
        add_interval(out, scaled_q * scale, level)
        add_interval(out, fixed_q, level, "fixed")
    return out


def forecast_metrics(name, prediction):
    error = prediction["market_return"] - prediction["prediction"]
    rows = []
    for method in ("scaled", "fixed"):
        for level in (90, 95):
            width = prediction[f"{method}_{level}_width"]
            rows.append({
                "model": name, "method": method, "nominal_coverage": level / 100,
                "coverage": prediction[f"{method}_{level}_covered"].mean(),
                "mean_width": width.mean(), "min_width": width.min(), "max_width": width.max(),
                "rmse": np.sqrt(np.mean(error**2)),
                "zero_forecast_rmse": np.sqrt(np.mean(prediction["market_return"] ** 2)),
            })
    return rows
