"""Calibration-only thresholds and interpretable uncertainty buckets."""

import numpy as np
import pandas as pd


def width_edges(widths, quantiles):
    widths = np.asarray(widths, dtype=float)
    if not len(widths) or not np.isfinite(widths).all() or (widths < 0).any():
        raise ValueError("Expected nonempty finite nonnegative calibration widths")
    if np.allclose(widths, widths[0], rtol=1e-12, atol=1e-14):
        return np.array([])
    edges = np.unique(np.quantile(widths, quantiles))
    return edges[edges < widths.max()]


def position_sizes(widths, edges):
    # If both percentiles coincide, use a single cut and keep both regimes.
    factors = {0: (0.5,), 1: (1.0, 0.0), 2: (1.0, 0.5, 0.0)}
    return np.asarray(factors[len(edges)])[np.searchsorted(edges, widths, side="left")]


def bucket_stats(prediction, edges):
    labels = {
        0: ("all",), 1: ("low", "high"), 2: ("low", "middle", "high"),
        3: ("low", "mid_low", "mid_high", "high"),
    }[len(edges)]
    bins = np.searchsorted(edges, prediction["scaled_90_width"], side="left")
    stats = prediction.assign(
        bucket=pd.Categorical.from_codes(bins, categories=labels, ordered=True),
        abs_return=prediction["market_return"].abs(),
        abs_error=(prediction["market_return"] - prediction["prediction"]).abs(),
    ).groupby("bucket", observed=False).agg(
        sessions=("market_return", "size"),
        mean_abs_return=("abs_return", "mean"),
        mean_abs_error=("abs_error", "mean"),
        mean_width=("scaled_90_width", "mean"),
        coverage=("scaled_90_covered", "mean"),
    )
    return stats
