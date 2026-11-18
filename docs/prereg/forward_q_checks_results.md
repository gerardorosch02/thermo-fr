# Results: the checks before the forward test of Q

Pre-registration: `docs/prereg/forward_q_prereg.md` (sha256 761517280f2a45b3f5c8c52f2045b70ac406686d5c8893dc7bf908fb89736402,
commit f67d7c5, written before any fit). Code committed at e34647d before the first fit. Amendment 1:
`docs/prereg/forward_q_amendment_1.md` (commit 8ffd403). Run on 2026-11-18 on branch `forward-q`. Independent review:
`docs/prereg/forward_q_review.md`. Machine-readable figures: `docs/prereg/forward_q_checks.json`.

No forward day has counted. Nothing public has changed. No error against a price was computed on any day after
2026-06-30.

## Outcome

- **The four stop conditions pass.** Q can be produced by the live entry point, from the inputs the morning run saves,
  and it is the same Q as in the research.
- **The headline survives the production settings.** With the tree model as deployed (300 trees of 31 leaves), S1 with Q
  is 0.28 below S1 on the 763 development days, 1.9%, against 0.26 and 1.8% at the research settings.
- **The load forecast in the archive is fresher than the one at the gate, and replaying that costs little.** It takes
  about 0.03 off the ensemble gain on the days sampled.
- **One fault was found and fixed before any day counted.** The live history had a hole in the proxies (amendment 1).

## Stop conditions

| | Check | Result |
|---|---|---|
| (a) | Reference figures on the 864 days | honest_v2 15.7124, L0 15.3717, S1 14.3687. Reproduced, by the project's check and by a separate calculation |
| (b) | Same path. `q_live` fed the selection inputs against the stored window forecasts of L0, P and Q on the five registered days | Largest difference 5.7e-14. Passes |
| (c) | Look-ahead. Every input not known at the gate replaced by noise on three days (136,559 to 894,884 values each) | The forecasts do not move at all (0.0). Passes |
| (d) | Live replay, unscored. L0 through `q_live` from the saved windows against the live LEAR the laptop issued, on every delivery day from 2026-10-20 with a pre-market run and a stored live forecast | 18 days (2026-10-20 to 11-11, and 11-19). Largest difference 1.2e-9 EUR/MWh. L0, P and Q issued for every hour of every day. Passes |

The days between 12 and 18 November are missing from (d) because the laptop has no pre-market shadow forecast for them
(fallback file on two mornings, a run after the window, a morning without runs). An issue of L0, P and Q with the record
written took about 80 seconds here.

## What (d) uncovered, and amendment 1

`data/forecast/inputs.csv` on the laptop has no wind or solar proxy from 2026-08-20 to 2026-10-09, although it has the
point forecasts. L0 does not read the proxies, so the S2 shadow never showed it. Q would have trained with those 51 days
missing and P with a neutral value in their place. The daily issue now fills a missing proxy by the project's own
point-in-time rule, which reproduces every stored proxy value to 7e-12 MW. Values already present are never replaced
and `q_live` is unchanged. With the hole filled, on the 18 replayed days P and Q are identical in windows 364 and 728 and differ in windows 1092 and 1456. Their four-window forecasts differ by 5.6 EUR/MWh on average (1.4 to 13.2 by day), so in live conditions the two have not converged. Q differs from L0 by 12.1 EUR/MWh on average on those days. These are differences between forecasts. No price was read.

## The tree model at production settings (registered diagnostic)

| | 864 days | 763 days |
|---|---|---|
| honest_v2, research settings (800 trees of 63 leaves) | 15.7124 | 15.5424 |
| honest_v2, production settings (300 trees of 31 leaves) | 16.0914 | 15.9199 |
| S1, research settings | 14.3687 | 14.3953 |
| S1, production settings | 14.5693 | 14.6234 |
| S1 with Q, research settings | | 14.1338 |
| S1 with Q, production settings | | 14.3386 |

Paired daily differences on the 763 days, two-sided 95%.
- S1 with Q against S1, research settings: -0.261 [-0.365, -0.170], a reduction of 1.82%.
- S1 with Q against S1, production settings: -0.285 [-0.389, -0.191], a reduction of 1.95%.
- S1 with Q against honest_v2 alone, production settings: -1.581 [-1.947, -1.294], a reduction of 9.9%. Most of that is
  the step from one model to an average of two, which has never been confirmed live. Q is about a fifth of it.
- honest_v2 at production settings against research settings: +0.378 [+0.139, +0.619]. The deployed settings cost about
  0.4 EUR/MWh in the backtest.

## Load forecast revisions replayed (registered diagnostic)

Twenty-one revision vectors from the saved windows (delivery days with a window saved between 10:05 and 11:00 Paris),
mean absolute revision 1.37% of the load forecast. Applied to the load forecast of D on every thirteenth development day,
59 days and 1,416 hours, D only.

| On the 59 days | Archive load | Revised load | Change |
|---|---|---|---|
| L0 | 16.50 | 16.61 | +0.11 |
| Q | 15.70 | 15.87 | +0.17 |
| honest_v2, production settings | 15.81 | 15.83 | +0.02 |
| S1, production settings | 15.28 | 15.33 | +0.04 |
| S1 with Q, production settings | 14.89 | 14.96 | +0.07 |
| Gain of Q over L0 | 0.80 | 0.74 | -0.06 |
| Gain of S1 with Q over S1 | 0.39 | 0.36 | -0.03 |

The forecasts themselves move by 1.4 to 1.5 EUR/MWh an hour for the two LEARs and 0.8 for the tree model. The
intervals of the paired changes all include zero (Q: +0.17 [-0.04, +0.48]). So a load forecast of pre-market quality
makes every model a little worse, Q a little more than L0 because it uses the load twice, and the gains shrink by less
than a tenth. Limits: 59 days, revisions from one autumn month applied to all seasons, and only the forecast day's load
was changed. The 59 days are not a tight sample (their ensemble gain is 0.39 against 0.28 on all 763 days). One of the 21
vectors is nearly the identity, because its last value was saved only two hours after the pre-market one. It was drawn
for 3 of the 59 days and, by the reviewer's count, dropping them changes nothing.

## What the independent review added

Verdict "confirmed with cautions". Every figure above was recomputed with separate code and agrees. The cautions that
bear on the forward test, and what was done.
- The same-path and look-ahead checks do not cover the proxy fill, which sits before `q_live`. Tested afterwards, end to
  end, on development day 2026-06-24: with 51 training days of proxies blanked and filled by the rule, P and Q are
  within 1.9e-9 of the stored research forecasts (up to 15.8 and 21.5 off with the hole left open), and scrambling every
  input not known at the gate before the fill changes nothing.
- A commit hash in a record does not show which code ran. Each record now also carries a hash of the source files
  (amendment 2).
- The German price of D-1, which L0, Q and the tree model all read, comes back different from one fetch to the next
  (94% of hours, 5.3 EUR/MWh on average, by the reviewer's own measurement). Its effect on L0 and Q is not measured.
- Recent mornings were lost often enough to put the count of 90 scored days at risk (amendment 2, point 5).

## Departures from the pre-registration

- Check (d) was run twice. The first pass, before amendment 1, gives the comparison of L0 with the live LEAR. The second
  pass, with the hole filled, gives P and Q as the forward test will issue them. L0 is the same in both.
- "A run between 10:05 and 11:00 Paris" in (f) is read from the time stamp of the saved window, which is when the fetch
  finished, a few minutes after the run started.
- The tree model in (f) was fitted on the unrevised inputs and predicts from the revised load, all sampled days at once.
  The two LEARs were refitted one day at a time with only that day revised.

## Not checked

Whether the laptop, with its own library versions, issues Q within the window every morning. That is what the burn-in
before the first counted day is for. Raw API responses, which the pipeline does not keep. Clock-change days in live
running, of which the saved windows hold none.
