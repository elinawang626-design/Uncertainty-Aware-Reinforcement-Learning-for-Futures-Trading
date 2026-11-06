"""Build trailing features and next-open execution intervals."""

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 60)
FEATURES = ["ret", "log_ret", "vol_20", *(f"close_over_ma_{w}" for w in WINDOWS)]
ES_START = "2024-12-16"
ES_LAST_OPEN = "2026-03-13"
# CME customary equity-index roll sessions, effective at the provider's daily open.
ES_ROLLS = {
    "2025-03-17": "ESM25",
    "2025-06-16": "ESU25",
    "2025-09-15": "ESZ25",
    "2025-12-15": "ESH26",
}


def dated_bars(frame):
    """Validate without forward filling, reordering or silently dropping bars."""
    bars = frame.set_index(pd.to_datetime(frame["date"], errors="raise")).drop(columns="date")
    if bars.empty or bars.index.hasnans or not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("Expected nonempty bars with unique and increasing session dates")
    prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Expected finite and positive OHLC")
    open_, high, low, close = prices.T
    if ((high < np.maximum(open_, close)) | (low > np.minimum(open_, close))).any():
        raise ValueError("Inconsistent OHLC bounds")
    return bars


def add_features(close, intervals):
    """Join each close at t to its execution interval from open t+1 to open t+2."""
    features = pd.DataFrame({"ret": close.pct_change(), "log_ret": np.log(close).diff()})
    features["vol_20"] = features["log_ret"].rolling(20).std()
    for window in WINDOWS:
        features[f"close_over_ma_{window}"] = close / close.rolling(window).mean() - 1
    return features.join(intervals, how="inner").dropna(subset=FEATURES)


def load_spy(path):
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.lower().str.replace(" ", "_")
    bars = dated_bars(raw)
    factor = bars["adj_close"] / bars["close"]
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ValueError("Invalid SPY adjustment factors")
    # Use relative adjusted prices for features and holding returns. Later common
    # back-adjustment factors cancel in returns and price-to-average ratios.
    opens = (bars["open"] * factor).to_numpy()
    dates = bars.index
    intervals = pd.DataFrame({
        "entry_date": dates[1:-1], "exit_date": dates[2:],
        "contract": "SPY", "roll": False, "entry_open": opens[1:-1],
        "exit_open": opens[2:], "market_return": opens[2:] / opens[1:-1] - 1,
    }, index=dates[:-2])
    return add_features(bars["adj_close"], intervals)


def build_es(contracts, start=ES_START, last_open=ES_LAST_OPEN, rolls=ES_ROLLS, first="ESH25"):
    """Build a forward chain and same-contract P&L, including around roll dates."""
    dates = pd.DatetimeIndex(sorted(set().union(*(bars.index for bars in contracts.values()))))
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(last_open))]
    if len(dates) < 3 or dates[0] != pd.Timestamp(start) or dates[-1] != pd.Timestamp(last_open):
        raise ValueError("ES sample does not span the specified sessions")
    schedule = {pd.Timestamp(date): symbol for date, symbol in rolls.items()}
    if not set(schedule).issubset(dates):
        raise ValueError("A scheduled roll is missing from the source")
    active = pd.Series(schedule, dtype=object).reindex(dates).ffill().fillna(first)
    close_gross = [1.0]
    rows = []
    for i in range(1, len(dates)):
        date, previous_date = dates[i], dates[i - 1]
        symbol = active.iloc[i]
        bars = contracts[symbol]
        required_dates = dates[i - 1 : i + 2]
        required = bars.reindex(required_dates)
        if required[["open", "close", "volume"]].isna().any().any() or (required["volume"] <= 0).any():
            raise ValueError(f"Missing or zero-volume ES bar for {symbol} near {date.date()}")
        close_gross.append(required["close"].iloc[1] / required["close"].iloc[0])
        if i < len(dates) - 1:
            entry, exit_ = required["open"].iloc[1:]
            rows.append({
                "entry_date": date, "exit_date": dates[i + 1], "contract": symbol,
                "roll": symbol != active.iloc[i - 1], "entry_open": entry,
                "exit_open": exit_, "market_return": exit_ / entry - 1,
            })
    close = pd.Series(np.cumprod(close_gross), index=dates)
    intervals = pd.DataFrame(rows, index=dates[:-2])
    return add_features(close, intervals)


def load_es(paths):
    contracts = {}
    for symbol, path in paths.items():
        raw = pd.read_parquet(path)
        if not (raw["contract_symbol"].eq(symbol) & raw["exchange"].eq("CME") & raw["root"].eq("ES")).all():
            raise ValueError(f"Unexpected contract metadata in {path}")
        # Far-deferred zero-volume records outside the declared research window
        # can carry settlement marks outside their OHLC range.
        dates = pd.to_datetime(raw["date"], errors="raise")
        contracts[symbol] = dated_bars(raw[dates.between(ES_START, ES_LAST_OPEN)])
    return build_es(contracts)


def split_spy(frame):
    """Keep every fitted reward/label inside its own chronological partition."""
    cutoffs = pd.to_datetime(["2019-01-01", "2022-01-01", "2024-01-01"])
    train = frame[frame["exit_date"] < cutoffs[0]]
    calibration = frame[(frame.index >= cutoffs[0]) & (frame["exit_date"] < cutoffs[1])]
    validation = frame[(frame.index >= cutoffs[1]) & (frame["exit_date"] < cutoffs[2])]
    test = frame[frame.index >= cutoffs[2]]
    return {"train": train, "calibration": calibration, "validation": validation, "test": test}
