import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = Path("/Users/elina/Desktop/王懿宁的学校/昆山杜克/semester4/Uncertainty-Aware-Reinforcement-Learning-for-Futures-Trading")

INP = PROJECT_ROOT / "data/processed/sp500_futures_daily.parquet"
OUT_DIR = PROJECT_ROOT / "data/processed/day3_compare"
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

def conformal_q(residuals: np.ndarray, alpha: float) -> float:
    n = residuals.shape[0]
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return np.sort(residuals)[k - 1]

def eval_model(name: str, model):
    model.fit(X_train, y_train)
    pred_cal = model.predict(X_cal)
    pred_test = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    cal_resid = np.abs(y_cal.values - pred_cal)

    alphas = [0.10, 0.05]
    qs = {a: float(conformal_q(cal_resid, a)) for a in alphas}

    out = pd.DataFrame(index=X_test.index)
    out["y_true"] = y_test.values
    out["y_pred"] = pred_test

    rows = []
    for a in alphas:
        lvl = int((1 - a) * 100)
        q = qs[a]
        lo = pred_test - q
        hi = pred_test + q
        covered = ((out["y_true"] >= lo) & (out["y_true"] <= hi)).mean()
        width = float((hi - lo).mean())
        rows.append({
            "model": name, "level": lvl, "q": q,
            "coverage": float(covered), "avg_width": width, "rmse": rmse
        })
        out[f"lo_{lvl}"] = lo
        out[f"hi_{lvl}"] = hi
        out[f"width_{lvl}"] = hi - lo
        out[f"covered_{lvl}"] = (out["y_true"] >= lo) & (out["y_true"] <= hi)

    out.to_csv(OUT_DIR / f"{name}_test_preds_with_intervals.csv")
    return rows

models = [
    ("ridge", Ridge(alpha=1.0, random_state=42)),
    ("rf", RandomForestRegressor(n_estimators=600, random_state=42, min_samples_leaf=5, n_jobs=-1))
]

all_rows = []
for name, m in models:
    all_rows.extend(eval_model(name, m))

summary = pd.DataFrame(all_rows).sort_values(["model", "level"])
summary.to_csv(OUT_DIR / "model_compare_summary.csv", index=False)

print("\n=== Model Compare Summary ===")
print(summary)
print("\nSaved:", OUT_DIR / "model_compare_summary.csv")
