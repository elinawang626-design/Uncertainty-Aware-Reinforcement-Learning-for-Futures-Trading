import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.day2_conformal import add_interval, conformal_q
from src.day3_uncertainty_analysis import bucket_stats, position_sizes, width_edges
from src.day4_rule_based_backtest import performance, simulate
from src.es_sample_audit import cme_sample
from src.experiment import Controls, evaluate
from src.prep_day1 import FEATURES, build_es, load_spy, split_spy
from src.rl import N_MARKET_STATES, TradingEnvironment, fit_q
from src.round2 import LONG_SIZED, direction_audit, paired_intervals


def es_market():
    dates = pd.bdate_range("2025-01-02", periods=67, name="date")
    prices = 100.0 + np.arange(len(dates))
    session_close = dates.tz_localize("UTC") + pd.Timedelta(hours=22)
    session_open = session_close - pd.Timedelta(hours=23)
    calendar = pd.DataFrame({"session_open": session_open, "session_close": session_close}, index=dates)
    old = pd.DataFrame({
        "open": prices, "close": prices, "volume": 1, "open_at": session_open,
        "close_at": session_close - pd.Timedelta(seconds=2),
        "close_available_at": session_close - pd.Timedelta(seconds=1),
    }, index=dates)
    new = old.assign(open=10 * prices, close=10 * prices,
                     open_at=session_open + pd.Timedelta(seconds=1))
    return calendar, {"old": old, "new": new}


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
        calendar, contracts = es_market()
        dates = calendar.index
        old, new = contracts["old"], contracts["new"]
        new_open = new["open"].to_numpy()
        roll = dates[61]
        frame = build_es(contracts, calendar, {roll: "new"}, "old")
        prior, rolled = frame.loc[dates[59]], frame.loc[dates[60]]
        self.assertEqual(prior["exit_date"], roll)
        self.assertEqual(prior["exit_open"], old.loc[roll, "open"])
        self.assertEqual(rolled["entry_open"], new.loc[roll, "open"])
        self.assertTrue(rolled["roll"])
        self.assertAlmostEqual(rolled["market_return"], new_open[62] / new_open[61] - 1)
        self.assertAlmostEqual(frame.loc[roll, "ret"], new_open[61] / new_open[60] - 1)
        self.assertEqual(prior["exit_at"], old.loc[roll, "open_at"])
        self.assertEqual(rolled["entry_at"], new.loc[roll, "open_at"])
        self.assertLess(rolled["decision_at"], rolled["entry_at"])
        ledger = simulate(frame, np.ones(len(frame)))
        pd.testing.assert_series_equal(ledger["entry_at"], frame["entry_at"])

        amended = new.copy()
        amended.loc[dates[62]:, ["open", "close"]] *= 1.2
        perturbed = build_es({"old": old, "new": amended}, calendar, {roll: "new"}, "old")
        pd.testing.assert_frame_equal(frame.loc[:roll, FEATURES], perturbed.loc[:roll, FEATURES])
        self.assertNotEqual(frame.loc[dates[60], "market_return"], perturbed.loc[dates[60], "market_return"])

    def test_es_rejects_session_missing_from_every_contract(self):
        calendar, contracts = es_market()
        missing = calendar.index[62]
        incomplete = {name: bars.drop(missing) for name, bars in contracts.items()}
        with self.assertRaisesRegex(ValueError, f"Missing or invalid ES .* on {missing.date()}"):
            build_es(incomplete, calendar, {}, "old")

    def test_es_rejects_delayed_close_and_each_late_roll_leg(self):
        calendar, contracts = es_market()
        roll = calendar.index[61]
        previous = calendar.index[60]
        for name, field, date, delay, error in (
            ("new", "close_available_at", previous, pd.Timedelta(minutes=2), "close timing"),
            ("old", "open_at", roll, pd.Timedelta(minutes=6), "open timing"),
            ("new", "open_at", roll, pd.Timedelta(minutes=6), "open timing"),
        ):
            with self.subTest(contract=name, field=field):
                delayed = contracts[name].copy()
                delayed.loc[date, field] += delay
                with self.assertRaisesRegex(ValueError, f"ES {error} invalid for {name}"):
                    build_es({**contracts, name: delayed}, calendar, {roll: "new"}, "old")

    def test_es_requires_timezone_aware_fill_times(self):
        calendar, contracts = es_market()
        old = contracts["old"]
        with self.assertRaisesRegex(ValueError, "timezone-aware, nonmissing open_at"):
            build_es({"old": old.assign(open_at=old["open_at"].dt.tz_localize(None))},
                     calendar, {}, "old")


class AuditTests(unittest.TestCase):
    def test_cme_counts_missing_trade_size_before_filtering_and_exempts_open_marker(self):
        raw = pd.DataFrame({
            "Delivery_Date": "202503", "Instrument_Symbol": "ESH5", "Ticker_Symbol": "ES",
            "Future_Option_Index_Indicator": "FUT", "Exchange_Code": "XCME", "Market_Data_Report_Type": "TRD",
            "Transaction_Time": "2025-03-11T22:00:00Z", "Trade_Price_Formatted": 5580.0,
            "Sequence_Number": [1, 2, 3], "Trade_Quantity": [1, None, None],
            "Close_Open_Type": [None, None, "O"],
        })
        with TemporaryDirectory() as scratch:
            path = Path(scratch) / "cme.csv"
            raw.to_csv(path, index=False)
            audit = cme_sample(path)
        self.assertEqual(audit["trade_rows"], 1)
        self.assertEqual(audit["missing_required_trade_fields"], 1)
        self.assertTrue(audit["contracts"][0]["open_matches_marker"])


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
    def test_execution_uses_volatility_at_the_rounded_width_cut(self):
        # Reference calibration cutpoints round differently in the two units.
        vol_cut = float.fromhex("0x1.6368cdcf2907dp-7")
        width_cut = float.fromhex("0x1.47187cfaf03c3p-5")
        above = np.nextafter(vol_cut, np.inf)
        q = 1.8406697508993253
        self.assertEqual(2 * q * above, width_cut)
        frame = market([0.01, -0.01, 0.02]).assign(vol_20=[vol_cut, above, 0.012])
        for feature in set(FEATURES).difference(frame.columns):
            frame[feature] = 0.0
        model = Mock()
        model.predict.return_value = np.zeros(len(frame))
        scale = Mock()
        scale.predict.return_value = np.array([0.01, 0.02, 0.03])
        quantiles = {90: (q, 0.01), 95: (2.0, 0.02)}
        policy = np.zeros((N_MARKET_STATES, 3, 3))
        controls = Controls(policy, scale, 1.0, np.array([]))
        with TemporaryDirectory() as scratch, patch("src.experiment.plot_equity"):
            out = Path(scratch)
            evaluate(frame, {"rf": (model, quantiles)}, policy, [0.005, 0.02],
                     [0.005, vol_cut], [], 1.0, controls, out, "test")
            ledger = pd.read_csv(out / "daily_ledger.csv")
        self.assertEqual(ledger["size_factor"].tolist(), [0.5, 0.0, 0.0])

    def test_suppressed_short_runs_and_matched_roll_costs(self):
        frame = market([0.02, -0.03, 0.01, 0.02, -0.02], [False, False, True, False, False])
        raw = np.array([1, -1, -1, 1, -1])
        sizes = np.array([0.5, 0, 0, 1, 0])
        ledger = {"RL": simulate(frame, raw), "RL + sizing": simulate(frame, raw * sizes),
                  LONG_SIZED: simulate(frame, sizes)}
        audit, counts = direction_audit(ledger, sizes)
        self.assertEqual(counts["raw_short_sessions"], 3)
        self.assertEqual(counts["raw_short_entries"], 2)
        self.assertEqual(counts["suppressed_short_sessions"], 3)
        self.assertEqual(counts["sized_short_sessions"], 0)
        self.assertFalse(audit.filter(like="delta_").ne(0).any().any())
        intervals = paired_intervals(ledger, [("direction", LONG_SIZED, "RL + sizing")])
        np.testing.assert_array_equal(intervals[["estimate", "ci_2.5", "ci_97.5"]], 0)

        ledger[LONG_SIZED] = simulate(frame, [0.5, 0.5, 0, 1, 0])
        audit, counts = direction_audit(ledger, sizes)
        self.assertEqual(counts["different_position_sessions"], 1)
        self.assertEqual(counts["different_turnover_sessions"], 2)
        self.assertNotEqual(audit["delta_net_return"].iloc[1], 0)

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

    def test_constant_nonzero_returns_have_undefined_sharpe(self):
        ledger = simulate(market(np.repeat(0.001, 639)), np.ones(639), 0)
        metrics = performance(ledger)
        self.assertEqual(metrics["annualized_volatility"], 0.0)
        self.assertTrue(np.isnan(metrics["sharpe"]))
        intervals = paired_intervals({"a": ledger, "b": ledger}, [("same", "a", "b")]).set_index("metric")
        self.assertEqual(intervals.loc["sharpe", "valid_draws"], 0)
        self.assertTrue(np.isnan(intervals.loc["sharpe", "estimate"]))
        self.assertEqual(intervals.loc["annualized_return", "estimate"], 0.0)

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
