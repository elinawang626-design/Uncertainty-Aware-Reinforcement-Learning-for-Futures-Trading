# src/data_download.py
import os
import pandas as pd
import yfinance as yf

def download_chunk(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,   # 更稳：避免并发触发风控
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

def main():
    symbol = "GC=F"
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "GC=F.csv")

    # 你能拿到 1mo，说明通路 OK；我们就分段拿长历史
    # 从 2010 开始足够做研究（也更不容易触发限制）
    start_year = 2010
    end_year = 2027   # 写到下一年，确保覆盖到现在

    chunks = []
    for y in range(start_year, end_year):
        start = f"{y}-01-01"
        end = f"{y+1}-01-01"
        dfy = download_chunk(symbol, start, end)
        print(f"Chunk {y}: rows={len(dfy)}")
        if not dfy.empty:
            chunks.append(dfy)

    if not chunks:
        raise RuntimeError(
            "All chunks returned empty. "
            "This usually means Yahoo is blocked or rate-limiting. "
            "We can switch to Stooq immediately if needed."
        )

    df = pd.concat(chunks, ignore_index=True)

    # 标准化、去重、排序
    if "date" not in df.columns:
        # 有时 yfinance 返回的日期列名不是 date（极少）
        # 但 reset_index() 一般会生成 Date/date
        possible = [c for c in df.columns if "date" in c]
        if possible:
            df = df.rename(columns={possible[0]: "date"})
        else:
            raise RuntimeError(f"Cannot find date column. Columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    # 保留常用列
    keep = [c for c in ["date","open","high","low","close","adj_close","volume"] if c in df.columns]
    df = df[keep].copy()

    # 去掉没有 close 的行
    df = df.dropna(subset=["close"])

    missing_ratio = df.isna().mean().mean()

    print("=== FINAL Summary ===")
    print("Rows:", len(df))
    print("Start:", str(df["date"].min())[:10])
    print("End  :", str(df["date"].max())[:10])
    print("Overall missing ratio:", f"{missing_ratio:.4%}")
    print("Columns:", list(df.columns))

    df.to_csv(out_path, index=False)
    print("Saved to:", out_path)

if __name__ == "__main__":
    main()
