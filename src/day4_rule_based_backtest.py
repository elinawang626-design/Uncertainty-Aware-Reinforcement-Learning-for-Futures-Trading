"""Common execution model and performance statistics for every strategy."""

import numpy as np
import pandas as pd

SESSIONS_PER_YEAR = 252
BASE_COST_BPS = 5


def trade_step(previous, target, market_return, roll, terminal, cost_rate):
    """Support scalars and arrays so the RL environment shares this reward."""
    open_turnover = np.where(roll, np.abs(previous) + np.abs(target), np.abs(target - previous))
    exit_turnover = np.where(terminal, np.abs(target), 0.0)
    turnover = open_turnover + exit_turnover
    gross_return = target * market_return
    cost = cost_rate * turnover
    return gross_return - cost, turnover, cost


def simulate(frame, positions, cost_bps=BASE_COST_BPS):
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(frame),) or not len(frame) or not np.isfinite(positions).all():
        raise ValueError("Expected one finite position per session")
    if (np.abs(positions) > 1).any() or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Expected exposure in [-1, 1] and nonnegative cost")
    previous = np.r_[0.0, positions[:-1]]
    terminal = np.arange(len(positions)) == len(positions) - 1
    returns, turnover, cost = trade_step(
        previous, positions, frame["market_return"].to_numpy(), frame["roll"].to_numpy(),
        terminal, cost_bps / 10_000,
    )
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid net return or portfolio ruin under the selected cost assumption")
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return frame[["entry_date", "exit_date", "contract", "roll", "entry_open", "exit_open", "market_return"]].assign(
        position=positions, turnover=turnover, cost=cost, net_return=returns,
        equity=equity, drawdown=equity / peak - 1,
    )


def performance(ledger):
    returns = ledger["net_return"]
    n = len(returns)
    ending_equity = ledger["equity"].iloc[-1]
    daily_std = returns.std(ddof=1)
    return {
        "sessions": n,
        "total_return": ending_equity - 1,
        "annualized_return": ending_equity ** (SESSIONS_PER_YEAR / n) - 1,
        "annualized_volatility": daily_std * np.sqrt(SESSIONS_PER_YEAR),
        "sharpe": np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / daily_std if daily_std > 0 else np.nan,
        "max_drawdown": ledger["drawdown"].min(),
        "turnover": ledger["turnover"].sum(),
        "annualized_turnover": ledger["turnover"].mean() * SESSIONS_PER_YEAR,
        "mean_abs_exposure": ledger["position"].abs().mean(),
    }


def baseline_positions(frame, prediction, sizes):
    rf = np.sign(prediction["prediction"].to_numpy())
    return {
        "Buy and hold": np.ones(len(frame)),
        "MA60 trend": np.sign(frame["close_over_ma_60"].to_numpy()),
        "Random Forest": rf,
        "Random Forest + sizing": rf * sizes,
    }


def equity_curves(ledgers):
    first = next(iter(ledgers.values()))
    index = pd.DatetimeIndex([first["entry_date"].iloc[0], *first["exit_date"]], name="date")
    return pd.DataFrame({name: np.r_[1.0, ledger["equity"]] for name, ledger in ledgers.items()}, index=index)
