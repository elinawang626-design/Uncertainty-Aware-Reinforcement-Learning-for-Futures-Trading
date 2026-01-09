import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/elina/Desktop/王懿宁的学校/昆山杜克/semester4/Uncertainty-Aware-Reinforcement-Learning-for-Futures-Trading")

RAW = PROJECT_ROOT / "data/raw/sp500_futures_daily.csv"
OUT = PROJECT_ROOT / "data/processed/sp500_futures_daily.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
df.columns = df.columns.str.strip()

df = df.rename(columns={
    "Date": "date",
    "Price": "close",
    "Close": "close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Vol.": "volume",
    "Change %": "change_pct",
})

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

for col in ["open", "high", "low", "close"]:
    df[col] = (
        df[col].astype(str)
        .str.replace(",", "", regex=False)
        .replace("-", np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "volume" in df.columns:
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["close"])

df["ret"] = df["close"].pct_change()
df["log_ret"] = np.log(df["close"]).diff()
df["vol_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)

windows = [5, 10, 20, 60]
for w in windows:
    if len(df) >= w + 5:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()

df = df.dropna(subset=["ret", "log_ret", "vol_20"])

df.to_parquet(OUT)
print("Saved:", OUT)
print("Final shape:", df.shape)
print("Columns:", list(df.columns))
print(df.tail())
