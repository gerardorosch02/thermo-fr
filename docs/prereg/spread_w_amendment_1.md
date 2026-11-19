# Test of W: amendment 1

Dated 2026-11-19, late evening London, and committed that night. Amends `spread_w_prereg.md` (sha256
c9b332fa7387c7a173e551ab377ea274ac095253a8df6f96a25e859a67f32649). Written before any French or German value of a delivery
day after 2026-10-10 was fetched for the test, and before any forecast of such a day was computed.

**Owner's instruction, 2026-11-19 23:13 London.** Keep the 100-day window with one read on or after 19 January. Keep W
frozen. "Before fetching post-10 October German data, please add one matched French-only comparator. Give it W's weekly
refits and 180-day half-life, use the same training and scoring hours, and omit only the German inputs. W versus that
comparator should be the primary test of whether German inputs add forecast value. Keep the two registered comparisons and
document this addition as a dated amendment." Specify the French refetch before running it. Start saving time-stamped
forecasts for all comparators. Revise "the cause is found". Have the diagnosis, the cutoff handling and the scoring code
independently reviewed, with the January results sealed.

## 1. Unchanged

- The window, 2026-10-11 to 2027-01-18, and the single read on or after 2027-01-19. The plan let the owner change the window
  before the fetch; he has confirmed it, and that clause is closed.
- W. `spread_w.py` is byte for byte the file of the plan (sha256 b616ecc9...ae2e).
- H and G as frozen in `spread_eval.py`, the gain rule, the slices, the six stop conditions.

## 2. Added: the matched French-only comparator, HW

HW reads H's inputs and is trained exactly as W is: a refit every seventh day on W's grid, training weights of half-life 180
days, the same training hours (every hour of an earlier day with a realised spread) and the same scored hours. The only
difference from W is that the German block is left out. Code: `src/thermo_fr/forecast/spread_hw.py` (sha256
a00c49dd795f2bfc8ec631b706a73017186530546159b6c167acc9cdd9a9231c).

Two more forecasts are produced by the same file and are reported only: GK (G's inputs, weekly refits, no weights) and GM
(G's inputs, monthly refits, weights). They show W's two changes one at a time.

## 3. Three decisions instead of two

Each by the gain rule on all scored hours, with Bonferroni for three, so the one-sided level is 1 - 0.05/3 for every one of
them. That is stricter than the 0.975 registered for the first two.

1. **W against HW, primary.** Do the German inputs add forecast value when both models are trained the same way?
2. **W against H.** Is W better than the French-only model as first frozen?
3. **W against G.** Did the redesign help?

Reading. The first decides whether the German inputs are worth having. If it passes and the third fails, the German inputs
help and the redesign is not confirmed. If the first fails, the German inputs are not confirmed out of sample, whatever the
other two say. A pass is a reason to keep logging, never proof.

Scoring code, frozen now: `src/thermo_fr/forecast/spread_w_test.py` (sha256
98fb99817b854cef07488b36f71066aa43469c945756ecadcd23d0d8cd166b01). It refuses a frame in which any model has a missing value,
an hour appears twice, the days are not the 100 registered days, or the provenance of a day is not single-valued. The seal in
`read()` is a guard and not a lock; the seal that matters is that the frame of the window is not assembled before the read
date.

## 4. Design figures with the comparator, not test results

Hourly error, EUR/MWh. Stored forecasts sha256 a0a5973ccb1bc836daafe95e64c74f45a266cf840437c1df372d4792dd4b8b5f; figures in
`spread_w_comparator_design.json`. HW passes the scramble check on 2025-10-26, 2026-07-01 and 2026-08-26 (difference 0.0,
inputs identical). GK and GM reproduce the explored variants to 5.7e-14.

| | Backtest, 791 days | Opened window, 102 days |
|---|---|---|
| H, French only, monthly, no weights | 22.16 | 18.53 |
| HW, French only, trained as W | 22.20 | 18.16 |
| G | 18.47 | 16.72 |
| GK, G with weekly refits | 18.00 | 16.18 |
| GM, G with weights | 17.78 | 15.52 |
| W | 17.34 | 14.84 |
| W minus HW, paired, 95% interval | -4.86 [-5.73, -4.08] | -3.32 [-5.31, -1.39] |
| W minus HW on the sunniest tenth of days | -4.17 | **+1.77** |
| HW minus H, paired, 95% interval | +0.04 [-0.46, +0.56] | -0.38 [-1.12, +0.24] |

Two things follow, and both correct what was said earlier.

- **Against the matched comparator the sunny-day loss is still there.** In the opened window W is worse than HW by 1.77 on
  the eleven sunniest days. Had the opened window been a test of W against HW, it would have failed on that slice. The
  statement made to the owner that W "would pass the rule against both other models, including on sunny days" was true only
  against H, whose own error on those days is larger. It is withdrawn as a description of the German inputs.
- **The redesign does nothing for the French-only model.** HW is no better than H. The gain of 1.1 to 1.9 from weekly refits
  and weights exists only when the German block is present.

## 5. The diagnosis, restated

This replaces the section "Why the first model failed, in short" of the plan and the words "the cause is found".

- **What G missed.** G did not lose to H. On 1 July to 10 October 2026 it beat H by 1.81 EUR/MWh (16.72 against 18.53), and
  it met two of the three pass conditions: the gain of at least 0.25 and the one-sided interval below zero. It missed the
  third, no slice worse by more than 1.0, on two slices: public holidays (+1.81 on 3 days) and the sunniest tenth of days
  (+1.70 on 11 days).
- **Where it lost accuracy.** The French discount to Germany closed between mid-June and mid-July 2026 (weekly mean spread
  about -73, -60, -20, -21, -27, then -6), with no advance signal in the inputs. French nuclear output was at its 2025 level
  in July and 3 to 7 GW below it from August to October. The eleven sunniest days all fall between 10 July and 14 August and
  were calm (German wind 2.8 GW against 12.3). On their nights both models forecast the spread too low, H by 9 to 15 and G by
  17 to 20, and G lost to H there by 5.8 while beating it at midday. The wind signal held (windiest tenth -6.99).
- **What is not established.** That slow adaptation caused it. Recency weights did not speed adaptation at the break: over
  the first 20 days the bias was -3.8 for G, -6.7 with weights alone and -6.1 for W. W still loses to H by 4.3 at night on
  those eleven days. The slice result is compatible with an ordinary weak slice combined with a halved gain: random draws
  of 11 calm days from the window are worse by more than 1.0 in about a quarter of cases.
- **What W is.** Weekly refits improve G by 0.47 in the backtest and 0.53 in the window; weights by 0.70 and 1.20; both
  together by 1.13 and 1.88. They are a general improvement chosen with the window in view, not a shown cause or cure.
  W changes two things at once, which is why GK and GM are reported in the test.
- **An untested alternative.** On training days before May 2024 the proxy columns hold ENTSO-E's forecast as a stand-in.
  Those rows are 58% of G's training rows and about 3% of W's weight. The weights may help simply by dropping them. The
  stand-in and the proxy are not interchangeable: for wind the correlation is 0.946 and the proxy sits at 81% of the level.

Corrections of detail to the plan and the log.

- The inputs outside their training range on more than 5% of hours are the French solar proxy (5.0%) and the lagged ENTSO-E
  German solar forecast (5.5%). The German solar proxy is at 4.3%. A range test of single inputs cannot see a regime.
- The half-life was not chosen on the backtest alone. 90 and 180 days tie there (17.774 and 17.778); 180 was kept with the
  opened window in view, and 90 was run after the weekly variant had been picked.
- The plan's table omitted variants that were run: target from the 7-day mean with half-life 365 (18.62, 16.14); the same
  target on H (22.71, 18.61); and the error-correction grid (windows of 7 and 14 days, shrink 0.5 and 1.0, by hour and by
  day, on each model).
- "A holiday slice fails by chance in one window in six" rests almost entirely on Christmas 2024: 102 of the 108 failing
  backtest windows contain it. The new window contains Christmas, with five holiday days.
- "The sunny slice is never worse by more than 0.56 in the backtest" rests on about seven independent windows.

## 6. Provenance of each day, and the refetch

`spread_w_refetch_spec.md` (sha256 a1b26596a4025cf15d98bea3ae6661c309ecf785f161a068d37fdc838f5afae3) is part of this
amendment. It says for every input when it is public, whether the source keeps the 10:05 version, and how missing or revised
values are handled. Each delivery day is one of three classes: saved (forecasts written before the auction), rebuilt with the
French inputs saved before 10:05, and rebuilt. The full 100-day result is kept. The number of days in each class and the
errors by class are reported beside it, with the primary comparison by class where a class has at least 14 days. The classes
are different calendar periods, so a difference between them is not the effect of provenance alone.

## 7. Saving forecasts each morning

A spread logger will write, before the auction, a time-stamped and write-once record of the forecasts of H, HW, G, GK, GM
and W for the next delivery day, with the hashes of its code and inputs. C cannot be saved because its inputs are published
after the gate. It is a separate program in its own folder, so the code hash of the forward test of Q is not touched. It
needs a new scheduled task on the owner's laptop, which he registers himself.

From its first morning the logger fetches German prices every day, because yesterday's spread is an input and a training
target. Those prices are inputs only. No error of any comparator against a price of the window is computed by anyone before
the read date. The logger does not exist yet. The first saved day is the first morning it runs.

## 8. Known weaknesses, stated before the test

- **The third decision has little power.** On design data W against G passes in 72% of 100-day windows and in 31% of those
  starting in September to November. On the same weeks of 2024 it fails all three conditions (gain 0.05, windiest slice
  +4.35). A failure of decision 3 would not be surprising.
- **The scramble tests time stamps, not vintages.** It cannot see a value that was revised after the gate.
- **Two French inputs have five minutes of margin**: the load forecast of D and the nuclear output of D-1 to 08:00, both
  under a 10:00 deadline. ENTSO-E's German load forecast of D has the same deadline and is treated as unavailable. The
  inconsistency is known and is not changed, because W is frozen.
- **The stop conditions compare the refetch with files that are themselves late vintages.**

## 9. Review

An independent reviewer who had not seen the work reviewed the diagnosis, the cutoff handling and the scoring code on
2026-11-19. Its verdicts: the facts of the diagnosis reproduce and the causal reading is not supported; no leak was found
and vintage is untested; the rule is implemented correctly and the scorer lacked input guards. Section 5 is its corrected
reading. Its two errors are fixed here: the scorer now refuses models scored on different hours, and this amendment exists
and is committed with the code. It also ran the scramble with the French proxy fill applied after the scramble instead of
before, on four days, with a difference of 0.0. It could not check HW, GK and GM, which were still being fitted, nor the
code that joins the new data, which is not written.

Figures in sections 5 and 8 that were not in the log before are the reviewer's and have not been recomputed by the builder.

Still to be reviewed independently, before the read date: HW, GK and GM; the join and stop-condition code; the spread
logger. The January results stay sealed until 2027-01-19.
