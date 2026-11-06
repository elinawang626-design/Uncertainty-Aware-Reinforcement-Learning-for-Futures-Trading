"""Build trailing features and next-open execution intervals."""

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 60)
FEATURES = ["ret", "log_ret", "vol_20", *(f"close_over_ma_{w}" for w in WINDOWS)]
ES_START = "2024-12-16"
ES_LAST_OPEN = "2026-03-13"
# Trade-date labels excluded by the frozen reference session policy. Shortened
# sessions with their own date (e.g. 2025-01-09) are retained. This bounded policy
# reproduces the old panel and is not a substitute for a verified CME calendar.
ES_REFERENCE_EXCLUSIONS = (
    "2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
    "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16",
)
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


def _check_timestamps(frame, columns):
    for column in columns:
        if (column not in frame or not isinstance(frame[column].dtype, pd.DatetimeTZDtype)
                or frame[column].isna().any()):
            raise ValueError(f"Expected timezone-aware, nonmissing {column} timestamps")


def build_es(contracts, calendar, rolls, first):
    """Build ES from an independent session calendar and timestamped contract bars.

    Index all inputs by timezone-naive CME trade date. The calendar contains
    session_open and session_close. Contract bars contain open_at, close_at
    and close_available_at for the prices used. Timestamp columns are tz-aware.
    """
    _check_timestamps(calendar, ("session_open", "session_close"))
    opens, closes = calendar["session_open"], calendar["session_close"]
    trade_dates = closes.dt.tz_convert("America/Chicago").dt.tz_localize(None).dt.normalize()
    if not calendar.index.equals(pd.DatetimeIndex(trade_dates)):
        raise ValueError("ES trade dates must match the scheduled close dates in America/Chicago")
    if opens.ge(closes).any() or (closes.array[:-1] >= opens.array[1:]).any():
        raise ValueError("Expected nonoverlapping ES sessions in chronological order")
    return _build_es(contracts, calendar.index, rolls, first, calendar)


def _es_prices(contracts, symbol, dates, field, calendar):
    if symbol not in contracts:
        raise ValueError(f"Missing ES contract {symbol}")
    bars = contracts[symbol].reindex(dates)
    prices = bars[[field, "volume"]].to_numpy(dtype=float, na_value=np.nan)
    valid = (np.isfinite(prices) & (prices > 0)).all(axis=1)
    if not valid.all():
        raise ValueError(f"Missing or invalid ES {field} for {symbol} on {dates[~valid][0].date()}")
    if calendar is not None:
        sessions = calendar.loc[dates]
        if field == "open":
            _check_timestamps(bars, ("open_at",))
            deadline = sessions["session_open"] + pd.Timedelta(minutes=5)
            valid = (bars["open_at"].ge(sessions["session_open"])
                     & bars["open_at"].le(deadline) & bars["open_at"].lt(sessions["session_close"]))
        else:
            _check_timestamps(bars, ("open_at", "close_at", "close_available_at"))
            valid = (bars["close_at"].ge(sessions["session_open"])
                     & bars["close_at"].ge(bars["open_at"])
                     & bars["close_available_at"].ge(bars["close_at"])
                     & bars["close_available_at"].le(sessions["session_close"]))
        if not valid.all():
            raise ValueError(f"ES {field} timing invalid for {symbol} on {dates[~valid][0].date()}")
    return bars


def _build_es(contracts, dates, rolls, first, calendar=None):
    if (not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None or len(dates) < 3
            or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())):
        raise ValueError("Expected at least three unique increasing ES trade dates")
    schedule = {pd.Timestamp(date): symbol for date, symbol in rolls.items()}
    if not set(schedule).issubset(dates):
        raise ValueError("A scheduled roll is missing from the ES calendar")
    active = pd.Series(schedule, dtype="string").reindex(dates).ffill().fillna(first)
    close_gross = [1.0]
    rows = []
    for i in range(1, len(dates) - 1):
        symbol = active.iloc[i]
        if i < len(dates) - 2:
            pair = _es_prices(contracts, symbol, dates[i - 1 : i + 1], "close", calendar)
            close_gross.append(pair["close"].iloc[1] / pair["close"].iloc[0])
        fills = _es_prices(contracts, symbol, dates[i : i + 2], "open", calendar)
        entry, exit_ = fills["open"]
        row = {
            "entry_date": dates[i], "exit_date": dates[i + 1], "contract": symbol,
            "roll": symbol != active.iloc[i - 1], "entry_open": entry,
            "exit_open": exit_, "market_return": exit_ / entry - 1,
        }
        if calendar is not None:
            row["decision_at"] = calendar["session_close"].iloc[i - 1]
            row["entry_at"], row["exit_at"] = fills["open_at"]
        rows.append(row)
    close = pd.Series(np.cumprod(close_gross), index=dates[:-2])
    intervals = pd.DataFrame(rows, index=dates[:-2])
    return add_features(close, intervals)


def load_es(paths):
    """Reproduce the pinned, date-only ES diagnostic."""
    contracts = {}
    for symbol, path in paths.items():
        raw = pd.read_parquet(path)
        metadata_ok = raw["contract_symbol"].eq(symbol) & raw["exchange"].eq("CME") & raw["root"].eq("ES")
        if not metadata_ok.fillna(False).all():
            raise ValueError(f"Unexpected contract metadata in {path}")
        # Far-deferred zero-volume records outside the declared research window
        # can carry settlement marks outside their OHLC range.
        dates = pd.to_datetime(raw["date"], errors="raise")
        contracts[symbol] = dated_bars(raw[dates.between(ES_START, ES_LAST_OPEN)])
    sessions = pd.bdate_range(ES_START, ES_LAST_OPEN).difference(pd.DatetimeIndex(ES_REFERENCE_EXCLUSIONS))
    return _build_es(contracts, sessions, ES_ROLLS, "ESH25")


def split_spy(frame):
    """Keep every fitted reward/label inside its own chronological partition."""
    cutoffs = pd.to_datetime(["2019-01-01", "2022-01-01", "2024-01-01"])
    train = frame[frame["exit_date"] < cutoffs[0]]
    calibration = frame[(frame.index >= cutoffs[0]) & (frame["exit_date"] < cutoffs[1])]
    validation = frame[(frame.index >= cutoffs[1]) & (frame["exit_date"] < cutoffs[2])]
    test = frame[frame.index >= cutoffs[2]]
    return {"train": train, "calibration": calibration, "validation": validation, "test": test}
