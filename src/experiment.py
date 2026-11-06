"""Run the chronological SPY experiment and the held-out ES contract transfer."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_download import ES_REVISION, ES_SHA256, RAW, ROOT, SPY_REVISION, SPY_SHA256, get_es, get_spy
from .day2_conformal import volatility_scale
from .day3_model_compare import fit_forecasters, forecast, forecast_metrics
from .day3_uncertainty_analysis import bucket_stats, position_sizes, width_edges
from .day4_rule_based_backtest import BASE_COST_BPS, baseline_positions, equity_curves, performance, simulate
from .prep_day1 import ES_REFERENCE_EXCLUSIONS, ES_ROLLS, load_es, load_spy, split_spy
from .rl import TradingEnvironment, fit_q, select_policy
from .round2 import (
    BLOCK_SESSIONS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, LEARNED_SIZED, LONG_SIZED, MYOPIC, MYOPIC_SIZED,
    SCALE_FOLDS, SIZE_QUANTILES, calibrate_error_scale, direction_audit, fit_error_scale, paired_intervals,
    sizing_diagnostic,
)

COST_GRID = (0, BASE_COST_BPS, 10)
CONTRASTS = (
    ("RF sizing", "Random Forest", "Random Forest + sizing"),
    ("RL sizing", "RL", "RL + sizing"),
    ("Adaptive vs static sizing", "RL + static size", "RL + sizing"),
    ("Direction before sizing", "Buy and hold", "RL"),
    ("Direction after sizing", LONG_SIZED, "RL + sizing"),
    ("Discount horizon before sizing", MYOPIC, "RL"),
    ("Discount horizon after sizing", MYOPIC_SIZED, "RL + sizing"),
    ("Myopic sizing", MYOPIC, MYOPIC_SIZED),
    ("Learned error scale vs volatility", "RL + sizing", LEARNED_SIZED),
)


class Controls(NamedTuple):
    myopic_policy: np.ndarray
    scale_model: object
    learned_quantile: float
    learned_edges: np.ndarray


def write_csv(frame, path, index=False):
    frame.to_csv(path, index=index, float_format="%.10g")


def plot_equity(curves, path, title):
    fig, ax = plt.subplots(figsize=(9.2, 4.5), layout="constrained")
    colors = ["#112d4e", "#6b7280", "#a05b19", "#dd934f", "#6554a4", "#068378",
              "#8a6291", "#083b36", "#b33262", "#b56c90", "#236bb2"]
    styles = ["-", "-", "-", "-", "-", "-", "-", "--", "-.", "--", "-"]
    ax.set_prop_cycle(color=colors, linestyle=styles)
    for name in curves:
        ax.plot(curves.index, curves[name], label=name, linewidth=1.65)
    ax.axhline(1, color="#9ca3af", linewidth=0.75, linestyle="--")
    ax.set(title=title, ylabel="Equity (initial value = 1)")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="outside lower center", frameon=False, ncols=4, fontsize=8)
    fig.savefig(path, dpi=125, pil_kwargs={"optimize": True})
    plt.close(fig)


def evaluate(frame, forecasters, policy, rl_edges, size_edges, bucket_edges, static_size, controls, out, title):
    out.mkdir(parents=True, exist_ok=True)
    predictions = {name: forecast(frame, model) for name, model in forecasters.items()}
    for name, prediction in predictions.items():
        write_csv(prediction, out / f"{name}_forecasts.csv", index=True)
    write_csv(pd.DataFrame([
        row for name, pred in predictions.items() for row in forecast_metrics(name, pred)
    ]), out / "forecast_summary.csv")
    write_csv(bucket_stats(predictions["rf"], bucket_edges), out / "uncertainty_buckets.csv", index=True)

    sizes = position_sizes(volatility_scale(frame), size_edges)
    positions = baseline_positions(frame, predictions["rf"], sizes)
    environment = TradingEnvironment(frame, rl_edges)
    rl = environment.policy_positions(policy)
    myopic = environment.policy_positions(controls.myopic_policy)
    learned, learned_sizes, scale_stats = sizing_diagnostic(
        frame, predictions["rf"], sizes, controls.scale_model, controls.learned_quantile, controls.learned_edges,
    )
    positions.update({
        "RL": rl, "RL + sizing": rl * sizes, "RL + static size": rl * static_size,
        LONG_SIZED: sizes, MYOPIC: myopic, MYOPIC_SIZED: myopic * sizes,
        LEARNED_SIZED: rl * learned_sizes,
    })
    ledger = {name: simulate(frame, pos) for name, pos in positions.items()}
    base_summary = pd.DataFrame([{"strategy": name, **performance(run)} for name, run in ledger.items()])
    write_csv(base_summary, out / "performance_summary.csv")
    sensitivity = pd.DataFrame([
        {"cost_bps": cost, "strategy": name, **performance(simulate(frame, pos, cost))}
        for cost in COST_GRID for name, pos in positions.items()
    ])
    write_csv(sensitivity, out / "cost_sensitivity.csv")

    audit, counts = direction_audit(ledger, sizes)
    write_csv(audit, out / "direction_daily.csv", index=True)
    write_csv(pd.DataFrame([counts]), out / "direction_summary.csv")
    write_csv(pd.DataFrame([scale_stats]), out / "scale_comparison.csv")
    risk_rows = []
    for label, prediction, strategy in (
        ("Rolling volatility", predictions["rf"], "RL + sizing"),
        ("Learned error scale", learned, LEARNED_SIZED),
    ):
        risk_rows.append({
            "scale": label, "nominal_coverage": 0.90,
            "coverage": prediction["scaled_90_covered"].mean(),
            "mean_width": prediction["scaled_90_width"].mean(),
            **performance(ledger[strategy]),
        })
    write_csv(pd.DataFrame(risk_rows), out / "scale_performance.csv")

    # Keep decision and execution timestamps, raw positions, and all costs auditable.
    daily = frame[["entry_date", "exit_date", "contract", "roll", "entry_open", "exit_open", "market_return"]].copy()
    daily["rf_scaled_90_width"] = predictions["rf"]["scaled_90_width"]
    daily["size_factor"] = sizes
    daily["learned_90_width"] = learned["scaled_90_width"]
    daily["learned_90_covered"] = learned["scaled_90_covered"]
    daily["learned_size_factor"] = learned_sizes
    for name, run in ledger.items():
        for col in ("position", "turnover", "cost", "net_return", "equity", "drawdown"):
            daily[f"{name} | {col}"] = run[col]
    write_csv(daily, out / "daily_ledger.csv", index=True)

    curves = equity_curves(ledger)
    write_csv(curves, out / "equity_curves.csv", index=True)
    plot_equity(curves, out / "equity_curves.png", f"{title} at {BASE_COST_BPS} bps per side")

    indexed = base_summary.set_index("strategy")
    ablation = []
    for comparison, reference, candidate in CONTRASTS:
        delta = indexed.loc[candidate] - indexed.loc[reference]
        ablation.append({
            "comparison": comparison, "reference": reference, "candidate": candidate,
            "sessions_with_position_change": int(np.count_nonzero(positions[candidate] != positions[reference])),
            **{f"delta_{metric}": delta[metric] for metric in
               ("total_return", "annualized_return", "sharpe", "max_drawdown", "turnover", "mean_abs_exposure")},
        })
    write_csv(pd.DataFrame(ablation), out / "ablation.csv")
    write_csv(paired_intervals(ledger, CONTRASTS), out / "paired_intervals.csv")
    return base_summary


def dates_and_count(frame):
    return {
        "sessions": len(frame), "first_decision": str(frame.index[0].date()),
        "last_decision": str(frame.index[-1].date()), "first_entry": str(frame["entry_date"].iloc[0].date()),
        "last_exit": str(frame["exit_date"].iloc[-1].date()),
    }


def run(raw, out):
    spy = split_spy(load_spy(get_spy(raw)))
    es = load_es(get_es(raw))
    forecasters = fit_forecasters(spy["train"], spy["calibration"])
    policy, rl_edges, tuning = select_policy(spy["train"], spy["validation"])
    calibration = spy["calibration"]
    calibration_rf = forecast(calibration, forecasters["rf"])
    calibration_width = calibration_rf["scaled_90_width"]
    size_edges = width_edges(volatility_scale(calibration), SIZE_QUANTILES)
    myopic_policy, myopic_iterations = fit_q(TradingEnvironment(spy["train"], rl_edges), 0.0)
    scale_model, fold_counts = fit_error_scale(spy["train"], forecasters["rf"][0])
    learned_q, learned_edges = calibrate_error_scale(calibration, calibration_rf, scale_model)
    controls = Controls(myopic_policy, scale_model, learned_q, learned_edges)
    bucket_edges = width_edges(calibration_width, (0.25, 0.50, 0.75))
    validation = spy["validation"]
    validation_rl = np.abs(TradingEnvironment(validation, rl_edges).policy_positions(policy))
    static_size = (
        np.average(position_sizes(volatility_scale(validation), size_edges), weights=validation_rl)
        if validation_rl.sum() else 1.0
    )
    # Freeze all parameters before scoring either holdout.
    out.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(tuning), out / "rl_validation.csv")
    (out / "run_manifest.json").write_text(json.dumps({
        "spy": {key: dates_and_count(frame) for key, frame in spy.items()},
        "es_transfer": {
            **dates_and_count(es), "roll_schedule": ES_ROLLS,
            "session_policy": "fixed_reference_weekdays_with_exclusions",
            "excluded_weekdays": ES_REFERENCE_EXCLUSIONS,
            "price_timing": "unverified_provider_daily_bars",
        },
        "inputs": {
            "spy": {"revision": SPY_REVISION, "sha256": SPY_SHA256},
            "es": {"revision": ES_REVISION, "sha256": ES_SHA256},
        },
        "runtime": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow")},
        },
        "base_cost_bps": BASE_COST_BPS, "cost_grid_bps": COST_GRID,
        "selected_gamma": next(row["gamma"] for row in tuning if row["selected"]),
        "round2": {
            "evaluation": "post_hoc_on_existing_windows",
            "myopic_gamma": 0.0, "myopic_iterations": myopic_iterations,
            "expanding_error_scale_fold_boundaries": SCALE_FOLDS, "oof_fold_sessions": fold_counts,
            "pure_volatility_size_edges": size_edges.tolist(),
            "learned_90_quantile": learned_q, "learned_90_size_edges": learned_edges.tolist(),
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "block_sessions": BLOCK_SESSIONS, "seed": BOOTSTRAP_SEED},
        },
        "rl_volatility_edges": rl_edges.tolist(),
        "rf_size_edges": width_edges(calibration_width, SIZE_QUANTILES).tolist(),
        "rf_bucket_edges": bucket_edges.tolist(), "validation_static_size": float(static_size),
        "conformal_quantiles": {
            name: {level: {"scaled": scaled, "fixed": fixed} for level, (scaled, fixed) in params.items()}
            for name, (_, params) in forecasters.items()
        },
    }, indent=2) + "\n", encoding="utf-8")

    for name, frame, title in (("spy", spy["test"], "SPY holdout"),
                               ("es", es, "CME ES contract transfer")):
        print(f"\n{title}")
        summary = evaluate(frame, forecasters, policy, rl_edges, size_edges, bucket_edges, static_size, controls, out / name, title)
        print(summary.to_string(index=False, float_format=lambda number: f"{number:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
