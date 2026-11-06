"""Offline fitted tabular Q-iteration over a historical trading environment."""

from itertools import product

import numpy as np

from .day4_rule_based_backtest import BASE_COST_BPS, performance, simulate, trade_step

ACTIONS = np.array([-1.0, 0.0, 1.0])
FLAT = 1
GAMMAS = (0.50, 0.90, 0.97)
N_MARKET_STATES = 12


def market_states(frame, volatility_edges):
    fast = frame["close_over_ma_5"].to_numpy() > 0
    slow = frame["close_over_ma_60"].to_numpy() > 0
    vol = np.searchsorted(volatility_edges, frame["vol_20"], side="right")
    return (2 * slow + fast) * 3 + vol


class TradingEnvironment:
    """Exogenous historical bars with actions that control the next inventory."""

    def __init__(self, frame, volatility_edges, cost_bps=BASE_COST_BPS):
        self.states = market_states(frame, volatility_edges)
        self.returns = frame["market_return"].to_numpy()
        self.roll = frame["roll"].to_numpy()
        self.cost_rate = cost_bps / 10_000
        self.terminal = np.arange(len(frame)) == len(frame) - 1

    def reward_samples(self, previous, target):
        return trade_step(previous, target, self.returns, self.roll, self.terminal, self.cost_rate)[0]

    def policy_positions(self, q_values):
        previous = FLAT
        positions = np.empty(len(self.states))
        for t, state in enumerate(self.states):
            values = q_values[state, previous]
            # Prefer flat when tied. An unseen market state also defaults to flat.
            previous = FLAT if values[FLAT] == values.max() else int(np.argmax(values))
            positions[t] = ACTIONS[previous]
        return positions


def fit_q(environment, gamma, tolerance=1e-10, max_iterations=2_000):
    """Average one-step Bellman targets by state. Enumerate all inventory/action pairs."""
    states = environment.states
    counts = np.bincount(states, minlength=N_MARKET_STATES)
    denominator = np.maximum(counts, 1)
    immediate = np.zeros((N_MARKET_STATES, len(ACTIONS), len(ACTIONS)))
    for p, a in product(range(len(ACTIONS)), repeat=2):
        samples = environment.reward_samples(ACTIONS[p], ACTIONS[a])
        immediate[:, p, a] = np.bincount(states, weights=samples, minlength=N_MARKET_STATES) / denominator

    transition = np.zeros((N_MARKET_STATES, N_MARKET_STATES))
    np.add.at(transition, (states[:-1], states[1:]), 1)
    transition /= denominator[:, None]  # Terminal mass has zero continuation value.
    q_values = np.zeros_like(immediate)
    for iteration in range(1, max_iterations + 1):
        continuation = transition @ q_values.max(axis=2)
        updated = immediate + gamma * continuation[:, None, :]
        if np.max(np.abs(updated - q_values)) <= tolerance:
            return updated, iteration
        q_values = updated
    raise RuntimeError(f"Q-iteration failed to converge for gamma={gamma}")


def select_policy(train, validation):
    volatility_edges = train["vol_20"].quantile([1 / 3, 2 / 3]).to_numpy()
    train_env = TradingEnvironment(train, volatility_edges)
    validation_env = TradingEnvironment(validation, volatility_edges)
    candidates = []
    for gamma in GAMMAS:
        q_values, iterations = fit_q(train_env, gamma)
        positions = validation_env.policy_positions(q_values)
        metrics = performance(simulate(validation, positions))
        score = metrics["sharpe"]
        candidates.append((q_values, {
            "gamma": gamma, "iterations": iterations, "validation_sharpe": score,
            "validation_annualized_return": metrics["annualized_return"],
            "validation_exposure": metrics["mean_abs_exposure"],
        }))

    def validation_score(candidate):
        score = candidate[1]["validation_sharpe"]
        return score if np.isfinite(score) else -np.inf

    winner = max(candidates, key=validation_score)
    for candidate in candidates:
        candidate[1]["selected"] = candidate is winner
    return winner[0], volatility_edges, [row for _, row in candidates]
