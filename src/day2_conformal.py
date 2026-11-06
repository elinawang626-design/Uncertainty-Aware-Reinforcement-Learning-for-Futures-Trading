"""Split conformal intervals with a trailing-volatility scale."""

import numpy as np

ALPHAS = (0.10, 0.05)


def conformal_q(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Expected finite nonnegative scores and 0 < alpha < 1")
    k = int(np.ceil((1 - alpha) * (len(scores) + 1)))
    if k > len(scores):
        raise ValueError("Too few calibration scores for the requested interval")
    return float(np.partition(scores, k - 1)[k - 1])


def volatility_scale(frame):
    return frame["vol_20"].clip(lower=0.0001).to_numpy()


def add_interval(frame, half_width, level, method="scaled"):
    prefix = f"{method}_{level}"
    frame[f"{prefix}_lo"] = frame["prediction"] - half_width
    frame[f"{prefix}_hi"] = frame["prediction"] + half_width
    frame[f"{prefix}_width"] = 2 * half_width
    frame[f"{prefix}_covered"] = (
        frame["market_return"].ge(frame[f"{prefix}_lo"])
        & frame["market_return"].le(frame[f"{prefix}_hi"])
    )
