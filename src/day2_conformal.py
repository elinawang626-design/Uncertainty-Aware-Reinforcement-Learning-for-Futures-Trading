import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path("/Users/elina/Desktop/王懿宁的学校/昆山杜克/semester4/Uncertainty-Aware-Reinforcement-Learning-for-Futures-Trading")

INP = PROJECT_ROOT / "data/processed/sp500_futures_daily.parquet"
OUT_DIR = PROJECT_ROOT / "data/processed/day2_conformal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INP).copy()

df["next_ret"] = df["ret"].shift(-1)

feature_cols = []
for c in ["ret", "log_ret", "vol_20"]:
    if c in df.columns:
        feature_cols.append(c)

for w in (5, 10, 20, 60):
    ma = f"ma_{w}"
    if ma in df.columns:
        df[f"close_over_{ma}"] = df["close"] / df[ma] - 1.0
        feature_cols.append(f"close_over_{ma}")

df = df.dropna(subset=feature_cols + ["next_ret"]).copy()

X = df[feature_cols].astype(float)
y = df["next_ret"].astype(float)

n = len(df)
train_end = int(n * 0.6)
cal_end = int(n * 0.8)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_cal, y_cal     = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]
X_test, y_test   = X.iloc[cal_end:], y.iloc[cal_end:]

print(f"Rows: total={n}, train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}")
print("Features:", feature_cols)

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=5,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred_cal = model.predict(X_cal)
pred_test = model.predict(X_test)

cal_resid = np.abs(y_cal.values - pred_cal)

def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    n = residuals.shape[0]
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return np.sort(residuals)[k - 1]

alphas = [0.10, 0.05]
qs = {a: conformal_q(cal_resid, a) for a in alphas}
print("Conformal q:", {k: float(v) for k, v in qs.items()})

results = pd.DataFrame(index=X_test.index)
results["y_true"] = y_test.values
results["y_pred"] = pred_test

for a in alphas:
    lvl = int((1 - a) * 100)
    q = qs[a]
    lo = pred_test - q
    hi = pred_test + q
    results[f"lo_{lvl}"] = lo
    results[f"hi_{lvl}"] = hi
    results[f"width_{lvl}"] = hi - lo
    results[f"covered_{lvl}"] = (results["y_true"] >= lo) & (results["y_true"] <= hi)

rmse = np.sqrt(mean_squared_error(y_test, pred_test))
print(f"Test RMSE (point forecast): {rmse:.6f}")

for a in alphas:
    lvl = int((1 - a) * 100)
    coverage = results[f"covered_{lvl}"].mean()
    width = results[f"width_{lvl}"].mean()
    print(f"{lvl}% interval: coverage={coverage:.3f}, avg_width={width:.6f}")

out_csv = OUT_DIR / "sp500_futures_nextret_conformal_test.csv"
results.to_csv(out_csv, index=True)

q_df = pd.DataFrame({"alpha": list(qs.keys()), "q": list(qs.values())})
q_df.to_csv(OUT_DIR / "conformal_q.csv", index=False)

print("✅ Saved:")
print(" -", out_csv)
print(" -", OUT_DIR / "conformal_q.csv")
print(results.tail())
