import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.day2_conformal import add_interval, conformal_q
from src.day3_uncertainty_analysis import bucket_stats, position_sizes, width_edges
from src.day4_rule_based_backtest import performance, simulate
from src.prep_day1 import FEATURES, build_es, load_spy, split_spy
from src.rl import TradingEnvironment, fit_q


class TemporalTests(unittest.TestCase):
    def test_labels_do_not_cross_split_boundaries(self):
        dates = pd.bdate_range("2018-12-26", "2024-01-08", name="date")
        frame = pd.DataFrame({"entry_date": dates[1:-1], "exit_date": dates[2:]}, index=dates[:-2])
        partitions = list(split_spy(frame).values())
        for left, right in zip(partitions, partitions[1:]):
            self.assertLess(left["exit_date"].max(), right.index.min())

    def test_spy_features_end_at_decision_and_labels_start_next_open(self):
        dates = pd.bdate_range("2023-01-02", periods=67, name="date")
        open_ = 100.0 + np.arange(len(dates))
        raw = pd.DataFrame({
            "Date": dates, "Open": open_, "High": open_, "Low": open_, "Close": open_, "Adj Close": open_,
        })
        with TemporaryDirectory() as scratch:
            path = Path(scratch) / "spy.csv"
            raw.to_csv(path, index=False)
            original = load_spy(path)
            raw.loc[61:, ["Open", "High", "Low", "Close", "Adj Close"]] *= 1.2
            raw.to_csv(path, index=False)
            amended = load_spy(path)

        pd.testing.assert_frame_equal(original.loc[:dates[60], FEATURES], amended.loc[:dates[60], FEATURES])
        row = original.loc[dates[60]]
        self.assertEqual(row["entry_date"], dates[61])
        self.assertEqual(row["exit_date"], dates[62])
        self.assertAlmostEqual(row["market_return"], open_[62] / open_[61] - 1)

    def test_es_prices_and_features_do_not_jump_between_contracts(self):
        dates = pd.bdate_range("2025-01-02", periods=67, name="date")
        old_open = 100.0 + np.arange(len(dates))
        new_open = 10 * old_open
        old = pd.DataFrame({"open": old_open, "close": old_open, "volume": 1}, index=dates)
        new = pd.DataFrame({"open": new_open, "close": new_open, "volume": 1}, index=dates)
        roll = dates[61]
        frame = build_es({"old": old, "new": new}, dates[0], dates[-1], {roll: "new"}, "old")
        prior, rolled = frame.loc[dates[59]], frame.loc[dates[60]]
        self.assertEqual(prior["exit_date"], roll)
        self.assertEqual(prior["exit_open"], old.loc[roll, "open"])
        self.assertEqual(rolled["entry_open"], new.loc[roll, "open"])
        self.assertTrue(rolled["roll"])
        self.assertAlmostEqual(rolled["market_return"], new_open[62] / new_open[61] - 1)
        self.assertAlmostEqual(frame.loc[roll, "ret"], new_open[61] / new_open[60] - 1)

        amended = new.copy()
        amended.loc[dates[62]:, ["open", "close"]] *= 1.2
        perturbed = build_es({"old": old, "new": amended}, dates[0], dates[-1], {roll: "new"}, "old")
        pd.testing.assert_frame_equal(frame.loc[:roll, FEATURES], perturbed.loc[:roll, FEATURES])
        self.assertNotEqual(frame.loc[dates[60], "market_return"], perturbed.loc[dates[60], "market_return"])


class UncertaintyTests(unittest.TestCase):
    def test_scaled_intervals_and_finite_sample_rank(self):
        self.assertEqual(conformal_q(np.arange(1, 20), 0.10), 18)
        with self.assertRaises(ValueError):
            conformal_q([1], 0.10)
        frame = pd.DataFrame({"prediction": [0.0, 1.0], "market_return": [0.0, 1.1]})
        add_interval(frame, np.array([0.01, 0.2]), 90)
        np.testing.assert_allclose(frame["scaled_90_width"], [0.02, 0.4])
        self.assertTrue(frame["scaled_90_covered"].all())

    def test_calibration_ties_preserve_real_buckets(self):
        tied = width_edges([0.03, 0.03 + 1e-16, 0.03], (0.3, 0.7))
        np.testing.assert_array_equal(position_sizes([0.02, 0.04], tied), [0.5, 0.5])
        edges = width_edges([1] * 7 + [2] * 3, (0.25, 0.50, 0.75))
        frame = pd.DataFrame({
            "scaled_90_width": [1, 2], "scaled_90_covered": [True, False],
            "market_return": [0.01, -0.02], "prediction": [0.0, 0.0],
        })
        stats = bucket_stats(frame, edges)
        self.assertEqual(stats["sessions"].sum(), 2)
        self.assertEqual(stats.loc["low", "sessions"], 1)
        self.assertEqual(stats.loc["high", "sessions"], 1)
        self.assertEqual(bucket_stats(frame, tied).index.tolist(), ["all"])


def market(returns, roll=None):
    dates = pd.bdate_range("2024-01-02", periods=len(returns) + 2, name="date")
    return pd.DataFrame({
        "entry_date": dates[1:-1], "exit_date": dates[2:], "market_return": returns,
        "roll": np.zeros(len(returns), dtype=bool) if roll is None else roll, "contract": "test",
        "entry_open": 100.0, "exit_open": 100.0, "close_over_ma_5": 0.01,
        "close_over_ma_60": 0.03, "vol_20": 0.01,
    }, index=dates[:-2])


class ExecutionTests(unittest.TestCase):
    def test_entry_flip_roll_and_terminal_costs(self):
        frame = market([0.0, 0.0, 0.0], [False, False, True])
        ledger = simulate(frame, [1.0, -1.0, -0.5], 5)
        np.testing.assert_allclose(ledger["turnover"], [1.0, 2.0, 2.0])
        np.testing.assert_allclose(ledger["net_return"], [-0.0005, -0.001, -0.001])
        self.assertAlmostEqual(performance(ledger)["max_drawdown"], ledger["equity"].iloc[-1] - 1)

    def test_sharpe_uses_mean_and_sample_standard_deviation(self):
        returns = np.array([-0.1, 0.02, 0.03])
        metrics = performance(simulate(market(returns), [1, 1, 1], 0))
        self.assertAlmostEqual(metrics["sharpe"], np.sqrt(252) * returns.mean() / returns.std(ddof=1))
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["annualized_return"], np.prod(1 + returns) ** (252 / 3) - 1)

    def test_q_learning_uses_reward_sign_and_inventory_transition(self):
        edges = np.array([0.005, 0.02])
        bullish = TradingEnvironment(market([0.01] * 30), edges)
        bearish = TradingEnvironment(market([-0.01] * 30), edges)
        long_q, _ = fit_q(bullish, 0.90)
        short_q, _ = fit_q(bearish, 0.90)
        np.testing.assert_array_equal(bullish.policy_positions(long_q), np.ones(30))
        np.testing.assert_array_equal(bullish.policy_positions(short_q), -np.ones(30))
        state = bullish.states[0]
        self.assertGreater(long_q[state, 2, 2], long_q[state, 0, 2])
        self.assertGreater(long_q[state, 1, 2], bullish.reward_samples(0, 1).mean())


if __name__ == "__main__":
    unittest.main()
