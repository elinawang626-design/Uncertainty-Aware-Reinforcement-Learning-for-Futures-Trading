# Uncertainty Aware Reinforcement Learning for Futures Trading

This study trains a tabular reinforcement learning policy on daily SPY history and tests it on a later SPY holdout and five individual CME E-mini S&P 500 (ES) contracts. A volatility-scaled conformal interval controls the policy's exposure. The paired ablation applies sizing to the same sequence of short, flat and long actions and includes a constant-exposure control.

At 5 bps per unit of one-way turnover, uncertainty sizing raised the RL Sharpe from **0.551 to 0.771 on SPY** and from **0.342 to 0.861 on ES**. It reduced maximum drawdown in both samples. SPY annualized return fell from 7.93% to 5.59%, and ES annualized return rose from 4.94% to 6.21%. Buy and hold earned the highest annualized return in both samples. The ES result measures transfer from SPY over a single year, and the two evaluation windows overlap in calendar time.

## Reproduce the experiment

From the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment
```

The experiment downloads and verifies six pinned inputs on first run. Network access to the GitHub API and Hugging Face is needed for the initial fetch. For offline reruns, reuse the verified files under `data/raw/` or point `--raw-dir` at a cache. The optional `python -m src.data_download` command warms that cache. Use `--output PATH` to choose where results go. The default `runs/` contains separate `spy/` and `es/` outputs, including daily forecasts and an execution ledger with dates, contract IDs, position, turnover, costs, returns and drawdown. The manifest records input revisions, digests, date boundaries, quantiles, chosen discount factor and package versions. CSV results use returns and widths in decimal units.

The [research report](Report.docx), [frozen outputs](results/), [run manifest](results/run_manifest.json) and [validation scores](results/rl_validation.csv) reflect the checked-in reference run. The original report is available in Git history. Original Day 2–4 CSVs and the processed Parquet remain under `data/processed/` for audit.

## Data and chronology

| Sample | Decision dates | Last exit | Observations |
| --- | --- | --- | ---: |
| SPY training | 1993-04-26 to 2018-12-27 | 2018-12-31 | 6,467 |
| SPY conformal calibration | 2019-01-02 to 2021-12-29 | 2021-12-31 | 755 |
| SPY policy validation | 2022-01-03 to 2023-12-27 | 2023-12-29 | 499 |
| SPY holdout | 2024-01-02 to 2026-07-21 | 2026-07-23 | 639 |
| ES contract transfer | 2025-03-13 to 2026-03-11 | 2026-03-13 | 250 |

- **SPY.** [Yahoo Finance historical bars](https://finance.yahoo.com/quote/SPY/history/) via a [pinned public mirror](https://github.com/Setooooo/etf-insight/tree/25eafd054ddb16d5471761312a0c852a672d6431/data). The mirror's source checksum matches CRLF bytes. The loader verifies the LF-normalized Git blob's own digest. SPY tracks S&P 500 exposure. The study scales the daily open by adjusted close divided by close to approximate total-return open-to-open exposure and uses ratios of adjusted closes for features. A common future adjustment factor cancels in those ratios.
- **ES.** [Pinned public third-party daily contract sample](https://huggingface.co/datasets/lynx1231/historical-futures-data-sample/tree/29efdfa21c5a5b2d7aa306397385cf116e011559/data/daily/CME/ES) for ESH25, ESM25, ESU25, ESZ25 and ESH26. The sample identifies the exchange and delivery months. Its upstream vendor, daily session definitions and adjustment conventions are undocumented. The chain uses the [CME customary roll dates](https://www.cmegroup.com/trading/equity-index/rolldates.html) of March 17, June 16, September 15 and December 15, 2025. It starts on December 16, 2024 for warm-up and exits before the March 2026 roll. The execution ledger closes the outgoing contract at its own open and opens the incoming contract at its own open. Holding returns use two opens of the same contract. The close feature chain compounds same-contract close ratios, including the incoming contract's ratio on its roll day.
- **Legacy series.** The 3,758-row Parquet lacks a verifiable instrument, source, expiry mapping and roll method. Its volume is missing throughout. The original downloader requested gold (`GC=F`) while preprocessing expected an untracked S&P 500 CSV. The current experiment uses neither the legacy prices nor the legacy labels.

A signal uses information through session `t` close. The trade enters at the next session's open and earns the return through the following session's open. Labels and rewards that would exit across a partition boundary are dropped. Trailing features can use already observed history from prior partitions. Both holdouts use frozen model weights, thresholds and hyperparameters. Dates in the curves mark the open where the interval P&L is realized.

## Models and uncertainty

- **Baselines.** Buy and hold is always long. The MA60 rule takes the sign of close relative to its trailing 60-session moving average. The original 600-tree Random Forest takes the sign of its predicted return. Ridge and a zero forecast are point-forecast comparators.
- **Adaptive intervals.** For each level, compute the finite-sample order statistic at rank `ceil((n + 1) × (1 − alpha))` of calibration absolute residuals divided by trailing 20-session volatility, floored at 1 bp. Multiply by the decision-day scale for the half-width. The original constant absolute-residual interval remains a comparator. The old implementation's constant width came from its formula. It supplied no volatility-regime information, and fitting its sizing quantiles on test data was forward-looking.
- **Position sizing.** The 30th and 70th percentiles of calibration RF 90% widths assign exposure multipliers of 1, 0.5 and 0. Calibration quartiles define the diagnostic buckets. Duplicate cutpoints collapse without manufacturing a rank difference for tied widths. A constant calibration width yields one bucket and a constant 0.5 sizing factor. Because the calibrated quantile is common to all days, percentile sizing ranks days exactly as the floored volatility scale does. The conformal step calibrates intervals to a target coverage level, and the P&L ablation measures the volatility-based sizing rule.
- **RL.** State consists of the signs of the 5- and 60-session price-to-average features, the trailing volatility tercile fitted on training data, and previous inventory. Action selects a target of −1, 0 or +1. Reward is one open-to-open net return, including the costs of changing inventory and terminal liquidation. Fitted tabular Q-iteration replays the historical market path, enumerates all inventory/action pairs, estimates the empirical state transitions and iterates Bellman targets to convergence. The previous action becomes the next state's inventory. Validation Sharpe selects the discount factor from 0.50, 0.90 and 0.97, with ties going to the first candidate. The reference run selected 0.90.
- **Ablation.** Plain RL, adaptive sizing and constant sizing share the same greedy action sequence. Only the executed position changes. The constant multiplier matches the adaptive policy's average absolute exposure on validation (0.309), so realized test exposures can differ. The sized policy remains an execution overlay on the frozen RL policy.

## Performance at 5 bps per side

| Strategy | SPY annualized | SPY Sharpe | SPY max DD | ES annualized | ES Sharpe | ES max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buy and hold | 20.88% | 1.253 | −19.77% | 16.05% | 0.855 | −14.14% |
| MA60 trend | 5.05% | 0.385 | −16.81% | −8.56% | −0.355 | −19.56% |
| Random Forest | −12.94% | −0.770 | −32.61% | 13.44% | 0.739 | −16.95% |
| Random Forest + sizing | −9.60% | −1.313 | −25.00% | −0.79% | −0.071 | −11.87% |
| RL | 7.93% | 0.551 | −17.32% | 4.94% | 0.342 | −10.44% |
| RL + sizing | 5.59% | 0.771 | −9.51% | 6.21% | 0.861 | −3.61% |
| RL + static size | 2.67% | 0.551 | −5.47% | 1.91% | 0.342 | −3.32% |

- [SPY performance](results/spy/performance_summary.csv) · [Ablation](results/spy/ablation.csv) · [Cost sensitivity](results/spy/cost_sensitivity.csv) · [Forecasts summary](results/spy/forecast_summary.csv) · [Uncertainty buckets](results/spy/uncertainty_buckets.csv)
- [ES performance](results/es/performance_summary.csv) · [Ablation](results/es/ablation.csv) · [Cost sensitivity](results/es/cost_sensitivity.csv) · [Forecasts summary](results/es/forecast_summary.csv) · [Uncertainty buckets](results/es/uncertainty_buckets.csv)

![SPY holdout equity curves for the seven strategies](results/spy/equity_curves.png)

![CME ES contract transfer equity curves for the seven strategies](results/es/equity_curves.png)

Turnover approximates traded notional as a fraction of equity using changes in target exposures, summed across sessions. Mark-to-market weight drift between rebalances is omitted. A flip from −1 to +1 costs two units. On a roll, turnover is the sum of absolute old and new exposure. Both initial entry and final liquidation are charged. Net simple returns compound from 1. Annualized return is ending equity raised to `252 / N` minus 1. Sharpe is `sqrt(252) × mean(daily net return) / sample_std(daily net return)` at a zero risk-free rate. A zero-variance Sharpe is blank. Drawdown includes the initial equity peak. Cost sensitivity reapplies 0 and 10 bps to the frozen positions alongside the reference case.

The simulator uses normalized fractional notional and stylized open fills. ES is modeled using percentage changes in contract price, equivalent to a fractional position in a fully collateralized contract before costs. Exchange margin, integer contract sizing, variable slippage, market impact, collateral yield, and ETF financing and borrow fees are outside the model. Empirical conformal coverage can change under temporal dependence and market shifts. The high uncertainty bucket had larger average absolute realized returns than the low bucket in both samples, but SPY and ES share a market and overlapping dates. Interpret these results as descriptive comparisons over the two historical samples.
