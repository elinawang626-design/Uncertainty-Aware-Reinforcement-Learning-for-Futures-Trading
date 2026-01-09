import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/Users/elina/Desktop/王懿宁的学校/昆山杜克/semester4/Uncertainty-Aware-Reinforcement-Learning-for-Futures-Trading")

PRED_PATH = PROJECT_ROOT / "data/processed/day3_compare/rf_test_preds_with_intervals.csv"
OUT_DIR = PROJECT_ROOT / "data/processed/day3_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PRED_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

width_col = "width_90"
if width_col not in df.columns:
    width_col = "width_95"

y = df["y_true"].astype(float)
w = df[width_col].astype(float)

# 用等频分组（即使数值重复也能分）
try:
    df["uncert_bucket"] = pd.qcut(
        w,
        q=4,
        labels=["low", "mid_low", "mid_high", "high"],
        duplicates="drop"   # 关键：允许自动减少桶数
    )
except ValueError:
    # 如果 bins 被自动减少，则带 labels 会报错，退回到不带 labels 的 qcut
    df["uncert_bucket"] = pd.qcut(
        w,
        q=4,
        duplicates="drop"
    )

group_stats = (
    df.assign(
        abs_ret=np.abs(y),
        pos=(y > 0).astype(int)
    )
    .groupby("uncert_bucket")
    .agg(
        n=("y_true", "size"),
        mean_ret=("y_true", "mean"),
        std_ret=("y_true", "std"),
        mean_abs_ret=("abs_ret", "mean"),
        pos_rate=("pos", "mean"),
        mean_width=(width_col, "mean"),
    )
    .reset_index()
)

print("\n=== Bucket Stats ===")
print(group_stats)

# 只在 high / low 都存在时才比较
if set(["low", "high"]).issubset(set(group_stats["uncert_bucket"])):
    low = group_stats.loc[group_stats["uncert_bucket"]=="low"].iloc[0]
    high = group_stats.loc[group_stats["uncert_bucket"]=="high"].iloc[0]

    diff = pd.DataFrame([{
        "metric": "mean_ret (high-low)",
        "value": float(high["mean_ret"] - low["mean_ret"])
    },{
        "metric": "std_ret (high-low)",
        "value": float(high["std_ret"] - low["std_ret"])
    },{
        "metric": "mean_abs_ret (high-low)",
        "value": float(high["mean_abs_ret"] - low["mean_abs_ret"])
    },{
        "metric": "pos_rate (high-low)",
        "value": float(high["pos_rate"] - low["pos_rate"])
    },{
        "metric": "mean_width (high-low)",
        "value": float(high["mean_width"] - low["mean_width"])
    }])

    diff.to_csv(OUT_DIR / "high_vs_low_diff.csv", index=False)
    print("\n=== High vs Low Diff ===")
    print(diff)
else:
    print("\n⚠️ high / low bucket not both present — skipping diff comparison")

group_stats.to_csv(OUT_DIR / "uncertainty_bucket_stats.csv", index=False)

# === 鲁棒的不确定性分组（不传 labels，避免 qcut 炸） ===
df["uncert_bucket"] = pd.qcut(
    w,
    q=4,
    duplicates="drop"
)

# 把 Interval 转成字符串，方便 groupby / 输出
df["uncert_bucket"] = df["uncert_bucket"].astype(str)

group_stats = (
    df.assign(
        abs_ret=np.abs(y),
        pos=(y > 0).astype(int)
    )
    .groupby("uncert_bucket")
    .agg(
        n=("y_true", "size"),
        mean_ret=("y_true", "mean"),
        std_ret=("y_true", "std"),
        mean_abs_ret=("abs_ret", "mean"),
        pos_rate=("pos", "mean"),
        mean_width=(width_col, "mean"),
    )
    .reset_index()
)

print("\n=== Bucket Stats ===")
print(group_stats)

# === 自动选择“最低不确定性桶”和“最高不确定性桶” ===
group_stats_sorted = group_stats.sort_values("mean_width")

low = group_stats_sorted.iloc[0]
high = group_stats_sorted.iloc[-1]

diff = pd.DataFrame([{
    "metric": "mean_ret (high-low)",
    "value": float(high["mean_ret"] - low["mean_ret"])
},{
    "metric": "std_ret (high-low)",
    "value": float(high["std_ret"] - low["std_ret"])
},{
    "metric": "mean_abs_ret (high-low)",
    "value": float(high["mean_abs_ret"] - low["mean_abs_ret"])
},{
    "metric": "pos_rate (high-low)",
    "value": float(high["pos_rate"] - low["pos_rate"])
},{
    "metric": "mean_width (high-low)",
    "value": float(high["mean_width"] - low["mean_width"])
}])

group_stats.to_csv(OUT_DIR / "uncertainty_bucket_stats.csv", index=False)
diff.to_csv(OUT_DIR / "high_vs_low_diff.csv", index=False)

print("\n=== High vs Low Diff (auto-selected) ===")
print(diff)

print("\nSaved:")
print(" -", OUT_DIR / "uncertainty_bucket_stats.csv")
print(" -", OUT_DIR / "high_vs_low_diff.csv")


print("\nSaved:")
print(" -", OUT_DIR / "uncertainty_bucket_stats.csv")
print(" -", OUT_DIR / "high_vs_low_diff.csv")