# Pre-registration: how much the German and Spanish price series matter, stage one (measurement)

Written on 2026-11-18, before any fit of this work. Branch `price-seq`, which stacks on `forward-q` (5e1af70). The forward
test of Q and its code are not touched. Nothing public changes. The holdout is not used.

## What is known before any fit

Established from the cached raw ENTSO-E files on the laptop (2021-01-01 to 2026-09-30) and the inputs file, with no model
involved.

- For DE-LU, ENTSO-E returns two price series for every year since 2021: sequence 1 (hourly until 2025-09-30, quarter-hourly
  after) and sequence 2 (quarter-hourly throughout). The parser lets the finer resolution win, so the stored German price
  is sequence 2 on every hour from 2021 to 2025-09-30, and from 2025-10-01, when both are quarter-hourly, a blend of the two
  that changes from one fetch to the next.
- Sequence 1 is the coupled day-ahead auction. It equals the French day-ahead price to the cent in 49% of hours in 2021,
  falling to 13% in 2025, as coupled zones do when no border binds. Sequence 2 equals it in 0.0% to 0.2% of hours. The two
  German series differ by 8 to 9 EUR/MWh on average (19.6 in 2022), with a correlation of 0.95 to 0.98. Which market
  sequence 2 is has not been established.
- For Spain there is one unnumbered series until 2025-09-04, which is the stored price and equals the French price in 27%
  to 37% of hours. From 2025-09-05 three numbered series appear beside it (each equal to the French price in about 1% of
  hours), and the stored price becomes a blend.

So the German price every model has read is not the coupled day-ahead price, and never was. Until October 2025 it was at
least one consistent series. L0 and Q read the German price of D-1 as a block of 24 regressors. honest_v2 reads the D-1
prices of six zones and their spreads to France.

## The corrected inputs

The inputs file with `price_de_lu_eur_mwh` replaced by the hourly mean of sequence 1 and `price_es_eur_mwh` by the hourly
mean of the unnumbered series, on every hour the raw files cover. No other column changes. The timing rule is the same:
the price of D-1 is published about 13:00 Paris on D-2.

## What is measured

On the selection window, delivery days to 2026-06-30, strict hours.

1. honest_v2, monthly walk-forward on the corrected inputs, at research settings (800 trees of 63 leaves) and at production
   settings (300 trees of 31 leaves), on all 763 development days.
2. L0 and Q, refitted through the live entry point (`forward_q.q_live`) on the corrected inputs, on every fourth development
   day starting with the first (191 days).
3. From these, on the 191 days: S1 and S1 with Q at production settings on corrected inputs.

Reported: the error of each model on stored and on corrected inputs, the paired daily difference corrected minus stored
with its interval (moving-block bootstrap, blocks of 7 sampled days, 5000 resamples, seed 0), the two gains (Q over L0, S1
with Q over S1) on both inputs, and the mean absolute change of each forecast.

Stop conditions: the reference figures reproduce; the corrected inputs differ from the stored ones in those two columns
only; on the first sampled day `q_live` on the stored inputs reproduces the stored L0 and Q to 1e-9.

## Rule, written before the run

The question is only whether the forward test should start on 2026-12-01 as registered or wait for a correction.

**Small.** The forward test starts as registered, and a correction is left for after it. Both must hold on the 191 days.
1. On corrected inputs, S1 with Q is better than S1 at production settings, and the upper end of the one-sided 95% interval
   of the paired daily difference is below zero.
2. That gain differs from the gain on stored inputs by less than 0.10 EUR/MWh.

**Large.** Anything else. The owner then decides before 2026-12-01 whether to delay the forward test.

Reported without a rule: whether the correction lowers the error of S1 or of honest_v2 by 0.25 EUR/MWh or more. If it does,
the correction is worth its own registration as an improvement, whatever the forward test does.

## Limits stated in advance

One sample of days, not all of them, for the two LEARs. The stored blend after October 2025 is one draw of a series that
changes between fetches, so the live forecasts have read other draws than the backtest. This stage measures the effect of
reading the right series. It does not measure the effect of the fetch-to-fetch noise. The other four zones return a single
series and are not examined further.
