import numpy as np
import pandas as pd
from pathlib import Path

# ===== 项目根路径（写死） =====
PROJECT_ROOT = Path("/Users/elina/Desktop/王懿宁的学校/昆山杜克/semester4/Uncertainty-Aware-Reinforcement-Learning-for-Futures-Trading")

INP = PROJECT_ROOT / "data/processed/day3_compare/rf_test_preds_with_intervals.csv"
OUT_DIR = PROJECT_ROOT / "data/processed/day4_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 读数据 =====
df = pd.read_csv(INP)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

# 真实下一日收益（被预测的目标）
ret = df["y_true"].astype(float)

# 点预测（方向）
signal = np.sign(df["y_pred"].astype(float))  # -1, 0, +1

# 选择不确定性列
width_col = "width_90" if "width_90" in df.columns else "width_95"
width = df[width_col].astype(float)

# ===== 交易成本（可调）=====
cost_per_turnover = 0.0005  # 5 bps

# ===============================
# 策略 A：Baseline（不考虑不确定性）
# ===============================
pos_A = signal.copy()  # 全仓做多/做空
turnover_A = pos_A.diff().abs().fillna(0)
ret_A = pos_A * ret - cost_per_turnover * turnover_A

# ===============================
# 策略 B：Uncertainty-aware（按不确定性调仓）
# 规则：
#  - width 在最低 30%：仓位 = 1
#  - width 在中间 40%：仓位 = 0.5
#  - width 在最高 30%：仓位 = 0
# ===============================
q30, q70 = width.quantile([0.3, 0.7])

size = pd.Series(0.0, index=df.index)
size[width <= q30] = 1.0
size[(width > q30) & (width <= q70)] = 0.5
size[width > q70] = 0.0

pos_B = signal * size
turnover_B = pos_B.diff().abs().fillna(0)
ret_B = pos_B * ret - cost_per_turnover * turnover_B

# ===== 绩效指标函数 =====
def perf_stats(r, freq=252):
    r = r.dropna()
    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    return {
        "total_return": float(total_ret),
        "annual_return": float(ann_ret),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd)
    }

stats_A = perf_stats(ret_A)
stats_B = perf_stats(ret_B)

# ===== 保存结果 =====
out = pd.DataFrame({
    "ret_true": ret,
    "signal": signal,
    "width": width,
    "pos_A": pos_A,
    "ret_A": ret_A,
    "pos_B": pos_B,
    "ret_B": ret_B
})

out.to_csv(OUT_DIR / "daily_pnl.csv")

summary = pd.DataFrame([
    {"strategy": "Baseline (no uncertainty)", **stats_A},
    {"strategy": "Uncertainty-aware", **stats_B},
])

summary.to_csv(OUT_DIR / "performance_summary.csv", index=False)

print("\n=== Performance Summary ===")
print(summary)
print("\nSaved:")
print(" -", OUT_DIR / "daily_pnl.csv")
print(" -", OUT_DIR / "performance_summary.csv")
