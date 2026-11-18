# Pre-registration: forward live test of Q, with the checks that come before it

Written on 2026-11-18, before any fit of this work and before any delivery day counts. Branch `forward-q`, which stacks on
`french-rl` (b00b1fb). The candidate is Q exactly as registered in `french_rl_prereg.md` and reported in
`french_rl_results.md`. Nothing about Q is changed here. Nothing public changes during the test.

Status of the dates. The start and end below are fixed in this text. They take effect when the owner approves the
scheduled task that issues the forecasts (section 7). If he chooses other dates, that is a dated amendment written before
the first counted day, and this text stays as it is.

## 1. What is being tested, and against what

| Name | Definition |
|---|---|
| g, the deployed model | The laptop's pre-market honest_v2 version for D, predicted with the published file of the day (`published/model/honest_v2.txt`, 300 trees of 31 leaves, refitted on the first weekday of each month). On a day when the pipeline used the fallback file, g is that fallback forecast, and the day is flagged |
| L0 | The present LEAR. Four windows (364, 728, 1092, 1456 days), refitted for each day, equal average |
| Q | L0 with one added regressor per hourly model, the French residual load (load forecast minus the wind and solar proxies), with the operator's forecast as a training stand-in on delivery days before 2024-05-01 |
| P | The same without the stand-in. Logged, never part of a verdict |
| S1 | (g + L0) / 2 |
| S1-Q | (g + Q) / 2 |

Three comparisons, in this order of importance.

1. **Main.** S1-Q against S1. Same inputs, same issue time, same g. This is the only comparison that carries the verdict.
2. **Deployment.** S1-Q against g. It answers a different question, whether moving the public forecast from honest_v2 to an
   average with Q would have helped. It is reported with the same statistics and has its own three-way outcome.
3. **Diagnostic.** Q against L0. It shows where the main result comes from. No verdict.

## 2. The live entry point

`thermo_fr.forecast.forward_q.q_live(history, day)` issues L0, P and Q for D from one set of inputs.

- The history is `data/forecast/inputs.csv` with every saved morning window up to the pre-market run laid over it, later
  runs winning (`shadow.live_history`, unchanged).
- The history is cut to what the timing rules place before 10:05 Paris on D-1 (`shadow.as_known`), with one addition. The
  operator's wind and solar forecast is kept for delivery days before 2024-05-01 and blanked from that day on. That is
  Q's registered stand-in and nothing more.
- The fits are `neighbour_rl.fit_predict_day`, the function behind the stored research forecasts. No second
  implementation.

## 3. Checks before any day counts

Stop conditions. If one fails, nothing after it is read and no day counts.

- (a) The reference figures reproduce on the 864 days: honest_v2 15.7124, L0 15.3717, S1 14.3687.
- (b) Same path. On the five registered days of the French test (2024-02-18, 2024-05-10, 2024-10-27, 2025-03-30,
  2026-06-24), `q_live` fed the selection inputs reproduces the stored window forecasts of L0, P and Q to 1e-9.
- (c) Look-ahead. On 2024-10-27, 2025-03-30 and 2026-03-29, replacing every input not known at the gate by noise
  (`lear_recipe.scramble_from`) leaves the output of `q_live` unchanged to 1e-9.
- (d) Live replay, unscored. For every delivery day from 2026-10-20 that has a pre-market run on the laptop and a stored
  live LEAR forecast from the S2 shadow, L0 is recomputed through `q_live` from the saved windows of that run and compared
  with the stored live forecast. Q and P must come out for every hour of every such day. These days are an operational
  check only. **No error against a price is computed on them, by this work or its reviewer, and they are never part of
  the forward record.** The laptop ran other library versions, so agreement is reported as found and judged against 0.01
  EUR/MWh, not 1e-9.

Diagnostics. Registered so that they cannot be reshaped after the fact. No decision hangs on them.

- (e) The tree model at production settings. The research S1 uses honest_v2 with 800 trees of 63 leaves. The deployed
  file has 300 trees of 31 leaves. honest_v2 is refitted month by month on the selection window with the production
  settings (`refit.REFIT_PARAMS`), and S1 and S1-Q are recomputed with it on the 763 days. Reported: the two errors, the
  paired daily difference with its interval, and the same for the research settings beside it.
- (f) Load forecast revisions. The saved windows show that the load forecast seen before the gate differs from the value
  the archive holds later. For each delivery day with a run between 10:05 and 11:00 Paris, the ratio of the pre-market
  value to the last value seen, by Paris clock hour, is one revision vector. On every thirteenth day of the 763 days,
  starting with the first, the archive load forecast of D is multiplied by one vector drawn with seed 0, D only, training
  rows untouched. L0 and Q are refitted through `q_live`'s fitting function, and honest_v2 at production settings predicts
  from the changed features. Reported on those days: the change in each error and in the two gains (Q over L0, S1-Q over
  S1). The revisions come from one autumn month and are applied to all seasons. That limit is stated with the result.

## 4. The forward test

**Days.** Delivery days Tuesday to Saturday, the days the weekday morning runs forecast. First counted delivery day
Tuesday 1 December 2026 (issued Monday 30 November). Last counted delivery day Saturday 17 April 2027. That is 20 weeks
and 100 expected days. There is no extension. Days before 1 December are a burn-in of the task and never count.

**A scored day** is an expected day on which g has a pre-market version from a morning run started between 10:05 and
11:00 Paris on D-1, the forward record was written before 11:15 Paris on D-1, and the auction result exists.

**Fallbacks, written down before the first day.**
- Q cannot be issued, L0 can. S1-Q is set to S1 for that day. The day is scored, flagged "Q fallback", and the
  difference is zero by construction.
- L0 cannot be issued either. S1 and S1-Q are both set to g. Scored, flagged "LEAR fallback".
- g came from the fallback file. Scored as issued, flagged "g fallback".
- The record was written at or after 11:15 Paris. Kept with its true time, flagged "late", not scored.
- No pre-market g. Nothing can be scored. The day is "lost", with the reason.
Every expected day ends up in exactly one of scored, late or lost, and the report lists all three with dates.

**Headline.** Pooled hourly mean absolute error of S1-Q and S1 on the scored days, the relative reduction, and the number
of hours. Given twice: with fallback days in (the operational score) and on the days where Q was really issued (the
candidate-only score).

**Inference.** The paired difference of daily mean absolute error, S1-Q minus S1, on the operational score. Moving-block
bootstrap of whole days, block 7, 5000 resamples, seed 0. One-sided 95% ends. The same for blocks 1, 3 and 14 as a fixed
sensitivity set. Pooled-hour and daily figures are reported separately and never mixed.

**Outcome of the main comparison, one of three.**
- **Supported.** The upper end of the one-sided 95% interval is below zero.
- **Contradicted.** The lower end of the one-sided 95% interval is above zero.
- **Inconclusive.** Anything else. Also inconclusive, whatever the numbers, if fewer than 90 days are scored or if Q was
  really issued on fewer than 90% of the scored days.

The bar of 0.25 EUR/MWh from the selection rule is not a pass condition here, because 100 days cannot resolve it. The
estimate and its interval are reported next to the development figure (0.26, about 1.8%) and the report says whether the
interval contains it. On the development days, windows of 100 Tuesday to Saturday days came out "supported" in about
four of five cases, and those days chose the candidate, so the true chance is lower. An inconclusive result is a likely
and acceptable ending.

**What each outcome leads to.** Supported: the owner may consider a release, which would be a larger change than this
test (from honest_v2 alone to an average) and is judged on the deployment comparison as well. Inconclusive: nothing
changes, and any further logging needs a new pre-registration. Contradicted: Q is dropped.

**Watched and reported, not part of the verdict.** Negative-price hours (the development sample showed them worse, with
intervals that include zero), high prices (top 5% of hours), solar hours (10:00 to 16:00), evening ramp (17:00 to
21:00), Saturdays, the windiest and sunniest tenth of days, days of high price range, and Q against P.

**Looks.** Each day the task prints counts only: scored, late, lost, fallbacks. No error of any candidate is computed,
shown or logged before 18 April 2027, when the statistics are computed once. If anyone looks earlier, the look is
written into `docs/experiments.md` with its date and what was seen. The scoring command refuses to run before the end
date unless given a flag whose use it records.

**Refits during the test.** g follows the published monthly refit. L0, P and Q refit daily by construction. The proxy
weights follow the monthly refit. None of these is a change of recipe. Any change to the code of `q_live`, to the timing
rules or to the fitting settings ends the record at that day. The old record is kept, and a new version starts under a
dated amendment.

## 5. The record

For each issue the task writes one file that is never overwritten: the delivery day, the wall-clock issue time, the run
it was built from, the sha256 of that run's saved window and of the model file behind g, the code commit, the library
versions, the 24 or so hourly forecasts of g, L0, P, Q, S1 and S1-Q with the four windows of each LEAR, how many hourly
models kept the regressor, and the fallback flags. A manifest lists every saved window and every record with its sha256,
each line carrying the hash of the line before it, so that a later edit or deletion shows. The window files themselves
are the inputs as fetched that morning. Raw API responses are not kept by the present pipeline. That gap is stated, not
hidden.

## 6. What this test cannot show

Ninety to a hundred winter and early spring days. No summer, few negative prices, so nothing about year-round
behaviour and little about the negative-price concern. Q and P share most of their training history by then, so the
test says nothing about which of the two is better. It tests the recipe as run on this laptop, with its outages.

## 7. Needs the owner's approval before it starts

- One new scheduled task on the laptop that runs the forward issue after the morning run, and nothing else changed.
- The dates above.
- Whether this branch is pushed to the public repository.
