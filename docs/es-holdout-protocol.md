# Protocol for a new ES holdout

Recorded on 2026-11-06 before scoring any new window. Follow this protocol once a multi-year source passes the [data gate](es-data.md). The existing SPY and ES results in this PR are post hoc and use the original split.

## Split and calendar

Use ES data starting with the first verified full July 2010 session. Earlier source bars may warm trailing features. Use expanding training and independent chronological calibration. The endpoints below are calendar dates and are exclusive. Each decision and its exit open must fall inside the same partition. Drop final labels that would cross a boundary. Historical observations may warm features in a later split.

| Fit | Training from July 2010 until | Calibration | Validation |
| --- | --- | --- | --- |
| 1 | 2016-01-01 | 2016-01-01 to 2018-01-01 | 2018-01-01 to 2019-01-01 |
| 2 | 2019-01-01 | 2019-01-01 to 2021-01-01 | 2021-01-01 to 2022-01-01 |
| 3 | 2021-01-01 | 2021-01-01 to 2023-01-01 | 2023-01-01 to 2024-01-01 |
| Final freeze | 2022-01-01 | 2022-01-01 to 2024-01-01 | Use the preceding folds to assess robustness |

January 2024 through July 23, 2026 is quarantined from new fitting, selection and reporting as new evidence. It includes the legacy SPY test decisions (2024-01-02 through 2026-07-21, final exit July 23) and ES transfer decisions (2025-03-13 through 2026-03-11, final exit March 13). Quarantined prices can initialize trailing features. SPY training, calibration and validation overlap the ES development calendar by design. Report overlap using actual decision, entry and exit dates and exact counts in the run manifest.

The new final test is **2026-07-24 through 2027-07-23** by CME trade date. Use decisions on or after July 24, 2026 with exits before July 24, 2027. It has no calendar overlap with prior test decisions or exits. Keep it sealed until the full interval qualifies. As of 2026-11-06, even complete coverage through the latest finished U.S. session, November 5, would allow at most 75 weekday decision dates before purging. Holidays and two unavailable final labels can reduce that count. Any explicitly approved earlier snapshot would be separately dated and labeled interim.

## Comparisons fixed before scoring

- Compare gamma=0.9 against gamma=0 using the same data, state, action space, costs and tie-break. Compare both unsized and with the same rolling-volatility overlay. Include buy and hold and always-long with the identical daily multiplier. Report matched-day exposure, return and turnover differences.
- Use the fixed 600-tree RF specification for the common point forecast. Within each training period, make expanding, past-only yearly RF predictions after three complete years of initial training. Purge boundary-crossing labels from each history and prediction block. Fit the gradient-boosted absolute-error scale on those residuals. Fit the full-training point model separately. Freeze both before calibration. Use the same RF center for both 90% intervals. Use independent calibration for finite-sample residual quantiles and sizing percentiles. Compare floored trailing 20-session volatility against the learned scale under the same fixed gamma=0.9 action sequence and 1, 0.5 and 0 sizing tiers. Report size changes, coverage and width, rank correlation to realized absolute errors, average absolute exposure, and net performance. Retain results of either sign.
- Use CME customary roll sessions, normally the Monday before the third Friday, with published holiday exceptions. Record each dated old and new pair before opening final performance. Same-contract open-to-open returns determine P&L. On roll dates, charge closing old and opening new as separate one-way legs. Charge 5 bps of normalized notional at each entry, rebalance, roll leg and terminal liquidation, with 0 and 10 bps sensitivities. Use a flat initial position, fractional fully collateralized notional and first-trade fill proxies. Log both timestamps on the roll and the prior close of the new contract.
- The primary metric is paired change in net annualized Sharpe at 5 bps for each prespecified contrast. Use 252 sessions, arithmetic mean net daily return, sample standard deviation and zero cash rate. Report annualized return, max drawdown, turnover, average exposure, daily ledger and compounded equity. Use 2,000 paired circular block resamples of 10 consecutive sessions, seed 42, for percentile intervals of Sharpe and annualized return differences. Label these as conditional on realized model paths and note temporal dependence and repeated comparisons. Report forecast coverage and ranking separately from trading returns.
