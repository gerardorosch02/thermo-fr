# The proxy gap in the holdout: which earlier conclusions are exposed

Written on 2026-11-19 from the experiment log, the code and the public app. Nothing was refitted or rescored for this
inventory, so it stays inside the owner's one-replay limit on 1 July to 10 October 2026.

## The defect

`data/forecast/inputs.csv` holds no wind or solar proxy after 2026-08-19 23:00 UTC. Every holdout evaluation logged on or
after 2026-10-10 read that file, so from 20 August to 10 October (52 of the 102 holdout days, 1,246 hours) every model that
reads the proxies was scored without them. LightGBM does not fail on a missing input, it routes it down a default branch,
so nothing flagged it.

Measured on 2026-11-19 for the tree model at 800 trees of 63 leaves (the configuration behind the logged figure): 25.81 on
the inputs as they are, 23.56 with the proxies filled by the project's rule; 36.00 and 27.80 on 1 to 10 October.

**The fill's logic uses nothing later than the forecast's cutoff.** Tested on 2026-11-19 on all 52 days of the gap: with
every input not known at the gate, every later day, and all realised wind and solar generation from D-1 onward replaced by
noise, the filled proxies of D do not move (0.0 on every day). That is a test of the rule. It does not establish that the
values the rule reads were in fact available on each of those 52 mornings: it rests on the stored point forecasts being the
ones issued at the time, which the file's flag says for every hour of the period and which cannot be checked further from
here.

## Inventory

| Logged | Conclusion | Configuration scored on the holdout | Did the model read the proxies | Did the holdout decide it | Exposure |
|---|---|---|---|---|---|
| 2026-10-10 | honest_v2 accepted, 25.81 against 26.23 for the previous set; adopted as the public model from 12 October | Tree, 800 of 63, monthly walk-forward, both feature sets | Yes for both. honest_v2 lost three inputs (two proxies and the residual load built from them), the previous set two | No. Accepted on the selection window; the holdout was reported and the merge left to the owner | Both sides handicapped, unequally. The direction of the comparison with inputs is not known. The level (25.81) is wrong as a description of the model |
| 2026-10-10 | Shape model "does not confirm" on the holdout: beats yesterday's shape on one measure, loses to the same-type day, captures less battery value | Shape model against two rule-of-thumb shapes | The model: **yes**. Traced in the code without refitting: `shape.walk_forward_shape` is gradient boosting on the honest_v2 inputs, and the battery valuation adds the level forecast, which reads them too. The two benchmarks: no, they are `price_lag1` and `price_lag_same_type`, past prices only | Yes. It is why the shape model was not merged as a win | **Most exposed.** The model ran without three inputs on 52 days against benchmarks that could not be affected. The effect on the ranking is unresolved. This is finding 1 of the public app |
| 2026-10-11 | Negative-price probability confirmed on the holdout (Brier skill +0.37) and published | Probability models fitted beside the point model | Yes. Traced in the code: `probabilistic.walk_forward_prob` takes the same feature table | Yes, it confirmed publication | Unknown direction. The point forecast improved with the proxies; whether the probability's skill and calibration would rise or fall has not been measured |
| 2026-10-11 | Quantile band: holdout coverage 73.8% logged | Quantile models | Yes, the same feature table | No | The under-coverage on the holdout may be partly the missing inputs. Quoted since, never recomputed |
| 2026-10-15 | Level-lag candidate (C), honest_v2 with change features: 26.19 against 25.81, not adopted | Tree, 800 of 63, both sides | Yes for both | **Yes. The holdout alone rejected it** after it passed the selection rule | Both sides were missing the same inputs. Their errors need not respond alike, so whether 0.38 worse survives with inputs is not known |
| before 2026-10-12, in `refit.py` | 300 trees of 31 leaves chosen for the published file: "within 0.2" of 800 of 63 "on September 2026" | Both tree settings | Yes | Yes, with the file size | **Exposed.** September 2026 lies wholly inside the gap. On 19 November the two settings differ by 0.06 before 20 August and 1.09 after, with the proxies filled |
| 2026-10-10 onward | "On the frozen holdout the model's hourly error was 25.8" on the public app, and `frozen_holdout` in the published model file | As the first row | Yes | Not a decision, a public statement | Overstates the error of the model as it runs with its inputs |

Not exposed: candidates rejected on the selection window and never taken to the holdout (carbon, gas, the LEAR recipe
candidates, neighbours, French residual load), and everything measured on the selection window, whose proxies are complete.

## Prevention

`forecast/coverage.py` (added 2026-11-19) reports, for each model input, the first and last delivery day with a value and
the days missing inside an evaluation window, and `check_coverage` refuses a window in which an input is missing for whole
days unless the caller names it as allowed. Run on the tree's 42 inputs it names the defect at once: the two proxies and the
residual load are missing on 51 whole days and part of one, from 20 August to 10 October 2026, last value 20 August.

It also shows the known start-up gap on the selection window, now visible in one place: the wind proxy and the residual
load begin on 1 May 2024 and the solar proxy on 1 April 2024, so the tree model is scored without them on the first 73 (and
43) of the 864 selection days. That is by construction (the proxies need calibration history, `HONEST_START`), it applies to
every backtest figure of the tree model, and it is the reason the residual-load test used the 763 days from 29 May 2024.
The check is to be called by the corrected rebuild before any score is computed.

## What this inventory cannot say

It cannot say that any verdict would reverse. That needs each candidate rescored on the same filled inputs as its
comparator, which is a new use of the period and needs the owner's approval. The live run is a separate question: the
saved morning data from 19 October carries both proxies, except the first saved window, which has no wind proxy.
