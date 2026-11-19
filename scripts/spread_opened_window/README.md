# Working scripts of the opened-window diagnosis, 2026-11-19

Kept as they were run, with the workspace paths they were run with. Exploratory, no registration, no decision. The window
1 July to 10 October 2026 was opened for diagnosis with the owner's approval after the spread test failed its rule.

- `build.py` joins the stored forecasts of H, G and C with their inputs for the backtest and the opened window.
- `d1.py`, `d2.py`, `d3.py` describe the opened window: price levels, inputs against their training range, the sunniest days.
- `variants.py` fits variants of G walk-forward (anchored targets, recency weights, weekly refits).
- `score.py` scores every variant on the backtest and the opened window. `bc.py` tests a correction from recent errors.

The variant kept is frozen as `src/thermo_fr/forecast/spread_w.py`. Results are in `docs/experiments.md`.
