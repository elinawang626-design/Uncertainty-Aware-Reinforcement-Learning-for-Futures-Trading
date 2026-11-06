"""Audit public CME Time and Sales v2 and FirstRate ES samples without publishing raw data."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .prep_day1 import dated_bars

OHLC = ["open", "high", "low", "close"]
QUARTER_CODES = {"03": "H", "06": "M", "09": "U", "12": "Z"}


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def cme_sample(path):
    raw = pd.read_csv(path, dtype={"Delivery_Date": "string"}, low_memory=False)
    raw = raw.loc[
        raw["Instrument_Symbol"].str.fullmatch(r"ES[HMUZ]\d", na=False)
        & raw["Ticker_Symbol"].eq("ES") & raw["Future_Option_Index_Indicator"].eq("FUT")
        & raw["Exchange_Code"].eq("XCME") & raw["Market_Data_Report_Type"].eq("TRD")
    ].copy()
    delivery = raw["Delivery_Date"]
    expected = "ES" + delivery.str[4:].map(QUARTER_CODES) + delivery.str[3:4]
    wrong_delivery = delivery.notna() & ~expected.eq(raw["Instrument_Symbol"]).fillna(False)
    if wrong_delivery.any():
        raise ValueError("CME sample contains a mismatched delivery code")
    raw["utc"] = pd.to_datetime(raw["Transaction_Time"], format="ISO8601", utc=True)
    local = raw["utc"].dt.tz_convert("America/Chicago").dt.tz_localize(None)
    raw["session"] = (local + pd.Timedelta(hours=7)).dt.normalize()
    raw["price"] = pd.to_numeric(raw["Trade_Price_Formatted"], errors="raise")
    marker_only = raw["Close_Open_Type"].isin(["O", "C"]) & raw["Trade_Quantity"].isna()
    required = ["utc", "price", "Sequence_Number", "Trade_Quantity", "Delivery_Date"]
    missing = raw[required].isna().any(axis=1)
    ticks = raw.loc[~missing & raw["Trade_Quantity"].gt(0)].sort_values(["utc", "Sequence_Number"], kind="stable")
    if ticks.empty or not np.isfinite(ticks["price"]).all() or ticks["price"].le(0).any():
        raise ValueError("Invalid trade prices in CME sample")
    bars = ticks.groupby(["session", "Instrument_Symbol"]).agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("Trade_Quantity", "sum"),
        first_trade=("utc", "first"), last_trade=("utc", "last"),
    )
    marks = raw.loc[raw["Close_Open_Type"].eq("O")]
    groups = []
    for (session, symbol), bar in bars.iterrows():
        matching = marks.loc[marks["session"].eq(session) & marks["Instrument_Symbol"].eq(symbol)]
        groups.append({
            "session": str(session.date()), "symbol": symbol,
            "delivery": str(ticks.loc[ticks["Instrument_Symbol"].eq(symbol), "Delivery_Date"].iloc[0]),
            "first_trade_utc": bar["first_trade"].isoformat(), "last_trade_utc": bar["last_trade"].isoformat(),
            "open_markers": len(matching), "open_matches_marker": bool(matching["price"].eq(bar["open"]).any()),
            "positive_volume_and_valid_ohlc": bool(bar["volume"] > 0 and
                bar["low"] <= min(bar["open"], bar["close"]) <= max(bar["open"], bar["close"]) <= bar["high"]),
        })
    return {
        "sha256": digest(path), "trade_rows": len(ticks), "unique_session_dates": bars.index.get_level_values(0).nunique(),
        "missing_required_trade_fields": int((missing & ~marker_only).sum()),
        "duplicate_trade_keys": int(ticks.duplicated(["Instrument_Symbol", "utc", "Sequence_Number"]).sum()),
        "out_of_regular_session_trades": int((local.loc[ticks.index].dt.hour == 16).sum()),
        "scheduled_roll_2025_03_17_present": bool(pd.Timestamp("2025-03-17") in bars.index.get_level_values(0)),
        "contracts": groups,
    }


def firstrate_sample(path):
    with ZipFile(path) as bundle:
        daily = pd.read_csv(bundle.open("ES_1day_sample.csv")).rename(columns={"timestamp": "date"})
        minute = pd.read_csv(bundle.open("ES_1min_sample.csv"))
    bars = dated_bars(daily)
    times = pd.to_datetime(minute["timestamp"], errors="raise")
    dated_bars(minute.rename(columns={"timestamp": "date"}))
    minutes = minute.assign(
        session=(times + pd.Timedelta(hours=6)).dt.normalize(),
        time=times.dt.strftime("%H:%M"),
    )
    grouped = minutes.groupby("session")
    edges = grouped["time"].agg(["first", "last"])
    complete = edges.index[edges["first"].eq("18:00") & edges["last"].eq("16:59")]
    complete = bars.index.intersection(complete)
    reconstructed = grouped.agg(open=("open", "first"), high=("high", "max"),
                                low=("low", "min"), close=("close", "last"))
    difference = reconstructed.loc[complete, OHLC].ne(bars.loc[complete, OHLC])
    weekdays = pd.bdate_range(bars.index[0], bars.index[-1])
    return {
        "sha256": digest(path), "daily_rows": len(bars), "minute_rows": len(minutes),
        "first_daily_date": str(bars.index[0].date()), "last_daily_date": str(bars.index[-1].date()),
        "missing_weekday_daily_rows": len(weekdays.difference(bars.index)),
        "daily_and_minute_unique_sorted_valid_ohlc": True,
        "nonpositive_daily_volume": int(bars["volume"].le(0).sum()),
        "nonpositive_minute_volume": int(minute["volume"].le(0).sum()),
        "both_boundary_minutes_and_daily_bar": len(complete),
        "daily_vs_session_minute_disagreements": {key: int(value) for key, value in difference.sum().items()},
        "includes_contract_codes": "contract" in bars.columns or "symbol" in bars.columns,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cme-v2", type=Path, required=True)
    parser.add_argument("--firstrate-zip", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"cme_v2": cme_sample(args.cme_v2), "firstrate": firstrate_sample(args.firstrate_zip)}, indent=2))


if __name__ == "__main__":
    main()
