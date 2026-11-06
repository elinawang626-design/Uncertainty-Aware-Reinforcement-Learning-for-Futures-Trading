"""Matched controls and an error-scale diagnostic on the previously seen windows."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from .day2_conformal import add_interval, conformal_q, volatility_scale
from .day3_uncertainty_analysis import position_sizes, width_edges
from .day4_rule_based_backtest import SESSIONS_PER_YEAR, return_risk
from .prep_day1 import FEATURES

LONG_SIZED = "Long-Only + Same Sizing"
MYOPIC = "RL (gamma=0)"
MYOPIC_SIZED = f"{MYOPIC} + sizing"
LEARNED_SIZED = "RL + learned error sizing"
SCALE_FOLDS = (2003, 2007, 2011, 2015, 2019)
SIZE_QUANTILES = (0.30, 0.70)
BOOTSTRAP_DRAWS = 2_000
BLOCK_SESSIONS = 10
BOOTSTRAP_SEED = 42


def fit_error_scale(train, point_model):
    """Pool predictions from expanding past-only fits inside the training partition."""
    folds = []
    for start_year, end_year in zip(SCALE_FOLDS, SCALE_FOLDS[1:]):
        start, end = pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year, 1, 1)
        history = train.loc[train["exit_date"] < start]
        held_out = train.loc[(train.index >= start) & (train["exit_date"] < end)]
        point = clone(point_model).fit(history[FEATURES], history["market_return"])
        errors = (held_out["market_return"] - point.predict(held_out[FEATURES])).abs()
        folds.append(held_out[FEATURES].assign(abs_error=errors))
    oof = pd.concat(folds)
    scale = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=7, min_samples_leaf=50, early_stopping=False, random_state=42,
    ).fit(oof[FEATURES], oof["abs_error"])
    return scale, [len(fold) for fold in folds]


def predict_scale(frame, model):
    return np.maximum(model.predict(frame[FEATURES]), 0.0001)


def learned_interval(frame, prediction, scale_model, quantile):
    result = prediction[["prediction", "market_return"]].copy()
    add_interval(result, quantile * predict_scale(frame, scale_model), 90)
    return result


def calibrate_error_scale(frame, rf_prediction, scale_model):
    residuals = (rf_prediction["market_return"] - rf_prediction["prediction"]).abs()
    q = conformal_q(residuals / predict_scale(frame, scale_model), 0.10)
    learned = learned_interval(frame, rf_prediction, scale_model, q)
    return q, width_edges(learned["scaled_90_width"], SIZE_QUANTILES)


def sizing_diagnostic(frame, rf_prediction, volatility_sizes, scale_model, q, learned_edges):
    """The two overlays share the forecast center, direction, and sizing tiers."""
    learned = learned_interval(frame, rf_prediction, scale_model, q)
    learned_sizes = position_sizes(learned["scaled_90_width"], learned_edges)
    rank = learned["scaled_90_width"].rank()
    vol_rank = pd.Series(volatility_scale(frame), index=frame.index).rank()
    abs_error = (rf_prediction["market_return"] - rf_prediction["prediction"]).abs()
    return learned, learned_sizes, {
        "scale_rank_correlation": rank.corr(vol_rank),
        "sessions_with_different_sizes": int(np.count_nonzero(learned_sizes != volatility_sizes)),
        "volatility_abs_error_rank_correlation": vol_rank.corr(abs_error.rank()),
        "learned_abs_error_rank_correlation": rank.corr(abs_error.rank()),
    }


def direction_audit(ledger, sizes):
    """Compare every realized exposure, net return and turnover on matched dates."""
    plain = ledger["RL"]
    sized = ledger["RL + sizing"]
    control = ledger[LONG_SIZED]
    daily = plain[["entry_date", "exit_date", "contract", "roll"]].assign(
        raw_rl_position=plain["position"], size_factor=sizes,
        raw_rl_short=plain["position"].lt(0),
        short_suppressed=plain["position"].lt(0) & sized["position"].eq(0),
    )
    for column in ("position", "net_return", "turnover"):
        daily[f"sized_rl_{column}"] = sized[column]
        daily[f"long_only_{column}"] = control[column]
        daily[f"delta_{column}"] = sized[column] - control[column]
    return daily, {
        "sessions": len(daily),
        "raw_short_sessions": int(daily["raw_rl_short"].sum()),
        "raw_short_entries": int((daily["raw_rl_short"] & ~daily["raw_rl_short"].shift(fill_value=False)).sum()),
        "suppressed_short_sessions": int(daily["short_suppressed"].sum()),
        "sized_short_sessions": int(sized["position"].lt(0).sum()),
        **{f"different_{column}_sessions": int(daily[f"delta_{column}"].ne(0).sum())
           for column in ("position", "net_return", "turnover")},
    }


def paired_intervals(ledger, contrasts):
    """Circular block bootstrap of paired, already realized daily net returns."""
    n = len(next(iter(ledger.values())))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = rng.integers(n, size=(BOOTSTRAP_DRAWS, (n + BLOCK_SESSIONS - 1) // BLOCK_SESSIONS))
    indices = ((starts[..., None] + np.arange(BLOCK_SESSIONS)) % n).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]

    def metrics(returns):
        _, sharpe = return_risk(returns)
        annualized = np.expm1(np.log1p(returns).mean(axis=-1) * SESSIONS_PER_YEAR)
        return {"annualized_return": annualized, "sharpe": sharpe}

    rows = []
    for comparison, reference, candidate in contrasts:
        before = ledger[reference]["net_return"].to_numpy()
        after = ledger[candidate]["net_return"].to_numpy()
        point_before, point_after = metrics(before), metrics(after)
        sample_before, sample_after = metrics(before[indices]), metrics(after[indices])
        for metric in point_before:
            delta = sample_after[metric] - sample_before[metric]
            finite = delta[np.isfinite(delta)]
            lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            rows.append({
                "comparison": comparison, "reference": reference, "candidate": candidate, "metric": metric,
                "estimate": point_after[metric] - point_before[metric], "ci_2.5": lo, "ci_97.5": hi,
                "valid_draws": len(finite), "block_sessions": BLOCK_SESSIONS,
            })
    return pd.DataFrame(rows)
