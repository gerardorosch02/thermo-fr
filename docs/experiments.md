# Experiment log

One row per experiment on the day-ahead price forecast. Written before the results of each experiment are known, so that the rules below cannot bend to them.

## Protocol (pre-registered 2026-10-10, branch fundamentals-v2)

**Selection window.** Strict backtest rows from 2024-02-17 to 2026-06-30: the monthly walk-forward of `forecast/backtest.py` (retrain at the start of every month on all earlier days, forecast the month), scored on the rows whose weather forecasts are as issued (the strict rows). Every feature choice is made on this window only.

**Frozen holdout.** 2026-07-01 onward. It is evaluated once per accepted change and never used to choose between variants. The October 2026 crash (the base price fell from about 200 to about 50 EUR/MWh in the first ten days of the month) lies inside the holdout, so crash-period numbers are reported only at that one evaluation.

**Acceptance rule.** A feature group is accepted if it lowers the overall strict-backtest MAE on the selection window and no slice among weekends, holidays, windiest 10% of days, sunniest 10% of days and top 5% price hours gets worse by more than 1.0 EUR/MWh. Variants that fail the rule are logged and rejected. Among accepted variants the one with the lowest overall MAE is evaluated on the holdout; nothing is tuned on the holdout, on the EEX record or on any single day.

**Slices.** As defined in `forecast/backtest.py`: weekends (Saturday and Sunday), holidays (French public holidays), windiest and sunniest 10% of days by actual national generation, top 5% price hours by actual price. The slice MAE is the gradient boosting model's.

**Holdout evaluation.** Walk-forward with monthly retraining continued through the holdout, reporting the overall MAE, the MAE on 2026-10-01 to 2026-10-10, the lag diagnostic (correlation of the daily mean signed forecast error with the change in the daily mean price over the previous two days, D-1 against D-3) and the EEX record (hit rate, mean P&L per MWh, model against market MAE against the auction, number of windows) computed from the walk-forward forecasts for the delivery days in the local market file.

## Experiments

| Date | Idea | Features changed | Backtest window | Strict MAE all | Weekends | Holidays | Windiest 10% | Sunniest 10% | Top 5% hours | Holdout MAE | EEX record | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-10 | Baseline: the current published honest set (calendar and day types, load forecast, lead-day-2 weather, wind and solar proxies, price lags D-1, D-2, D-7 and same type) | none | selection 2024-02-17 to 2026-06-30, 20,752 strict hours | 16.39 (naive 21.83) | 14.99 | 19.55 | 19.29 | 15.38 | 25.37 | not evaluated here (the published model was fitted through September 2026; its live record is in docs/forecast.md) | 42 windows, hit rate 29%, mean P&L -2.5 EUR/MWh, model MAE 27.0 vs market 5.1 (walk-forward record of 2026-10-10) | reference | the bar every variant must clear |
