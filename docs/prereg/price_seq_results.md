# Results: how much the German and Spanish price series matter, stage one

Pre-registration: `docs/prereg/price_seq_prereg.md` (sha256 f7bddb64c539d15890731fefd454939b0f47ae8752e358009a6d831c3fc715dc,
commit 163e4af, written before any fit). Code committed at 7aa6e55 before the first fit. Run on 2026-11-18 on branch
`price-seq`. Independent review: `docs/prereg/price_seq_review.md`. Machine-readable figures:
`docs/prereg/price_seq_results.json`.

The holdout was not used. Nothing public changed. The forward test of Q and its code were not touched.

## Outcome

- **Small, by the rule.** With the coupled German and Spanish prices in place of the stored ones, S1 with Q still beats S1,
  and the gain is the same to a hundredth. The forward test starts on 2026-12-01 as registered.
- **The correction helps every model a little and none measurably.** Errors fall by 0.04 to 0.10 EUR/MWh on the sampled days,
  with every interval including zero. That is below the project's bar of 0.25, so the correction does not earn a
  registration as an improvement. It is still the right series, and it is worth correcting for that reason once the
  forward test is over.

## What was wrong, in one paragraph

ENTSO-E returns two day-ahead price series for DE-LU, and has done since 2021. Sequence 1 is the coupled auction: it equals
the French price to the cent in 49% of hours in 2021 and 13% in 2025, as coupled zones do when no border binds. Sequence
2 equals it in under 0.2% of hours. The parser lets the finer resolution win, so every model has read sequence 2 as "the
German day-ahead price" since the start, 8 to 9 EUR/MWh away from the coupled price on average (19.6 in 2022), with a
correlation of 0.95 to 0.98. From 2025-10-01 both series are quarter-hourly and the stored value is a blend of the two
that changes from fetch to fetch. For Spain the stored price was the right one until 2025-09-04 and a blend of four series
after. Which market sequence 2 is has not been established.

## Stop conditions

The reference figures reproduce (15.7124, 15.3717, 14.3687). The corrected inputs differ from the stored ones in the two
price columns only (49,571 hours for DE-LU, 9,024 for Spain). On the first sampled day the live entry point on the stored
inputs reproduces the stored L0 and Q to 7.1e-15.

## The tree model, all 763 development days

| honest_v2 | Stored inputs | Corrected inputs | Change, 95% interval |
|---|---|---|---|
| Research settings (800 trees of 63 leaves) | 15.54 | 15.66 | +0.12 [+0.01, +0.23] |
| Production settings (300 trees of 31 leaves) | 15.92 | 15.89 | -0.03 [-0.12, +0.05] |

At research settings the tree model is slightly worse with the right series, at production settings unchanged. The series
it had been reading was no worse a regressor than the coupled price.

## The two LEARs and the ensemble, 191 sampled days (4,584 hours)

Every fourth development day. The tree model is at production settings.

| | Stored inputs | Corrected inputs | Change, 95% interval |
|---|---|---|---|
| L0 | 15.81 | 15.71 | -0.10 [-0.24, +0.04] |
| Q | 14.95 | 14.88 | -0.07 [-0.19, +0.09] |
| honest_v2 | 15.78 | 15.72 | -0.06 [-0.25, +0.10] |
| S1 | 14.59 | 14.54 | -0.05 [-0.14, +0.05] |
| S1 with Q | 14.28 | 14.24 | -0.04 [-0.13, +0.06] |
| Gain of Q over L0 | 0.87 | 0.83 | -0.03 |
| Gain of S1 with Q over S1 | 0.31 | 0.30 | -0.01 |

S1 with Q against S1 on corrected inputs, paired daily difference: -0.304 [-0.535, -0.137], one-sided 95% upper end -0.164.
On stored inputs on the same days: -0.315 [-0.558, -0.143].

The forecasts themselves move by about 1.6 EUR/MWh an hour for all three models. The errors barely move because the two
German series carry nearly the same information about the next French price.

## The rule

| Condition | Result |
|---|---|
| 1. On corrected inputs S1 with Q beats S1, one-sided 95% upper end below zero | Holds (-0.164) |
| 2. The gain moves by less than 0.10 | Holds (0.315 to 0.304) |

Small. Reported without a rule: no error falls by 0.25 or more (S1 by 0.05 on the sampled days, honest_v2 at production
settings by 0.03 on all 763 days).

## What this does not show

- It measures reading the right series in place of the wrong one. It does not measure the fetch-to-fetch noise of the
  blend since October 2025, which the live forecasts read every morning. In the saved windows the same German hour came
  back different in 87% of cases, by 5.7 EUR/MWh on average. That noise falls on L0, Q and honest_v2 alike, so on both
  sides of the forward test's main comparison, and it will be in the forward record as it is in production.
- One sample of 191 days for the two LEARs, not all 763. The gain on the sample (0.31) is a little above the gain on all
  days (0.28).
- The selection window, used many times.

## What the independent review added

Verdict "confirmed with cautions". Every score, difference, interval and count was recomputed with separate code and
agrees. Two refits on corrected inputs match the stored ones to 1.4e-14. The reviewer agrees that sequence 1 is the coupled
auction and sees no other reading. Its cautions:
- The stored series may carry information of its own. At research settings the tree model is worse with the coupled price,
  and the interval excludes zero. A correction that simply swaps the series could lose something. Stage two should
  consider keeping sequence 2 as a regressor under its own name beside the coupled price.
- A split the reviewer made after seeing the results, so exploratory: on the 68 sampled days from 2025-10-01, where the
  stored price is a blend, the correction lowers S1 by 0.12 [-0.24, -0.04]. Before that date it does almost nothing. The
  gain of Q is the same on both sides (0.44 against 0.43).
- The gain is concentrated, as before. On the 191 days the mean daily gain of S1 with Q is 0.30, the median 0.17, and 0.12
  without the ten best days.
- The sample holds somewhat harder days for L0 than the full set (15.81 against 15.47). Its gain sits inside the range of
  the other three every-fourth-day samples (0.22 to 0.32).
- The numbered Spanish series look like intraday auctions (the third covers only the second half of each day). Not
  established.

## Departures from the pre-registration

- The check that only two columns differ was made in memory, before the corrected file was written. Read back from disk,
  the wind proxy also differs on 11 hours, by at most 5.7e-14, from the text round trip. The fits read the file.
- The table that identifies the coupled series was first computed on raw prices to 2026-09-30, which includes three months
  of holdout-period prices. No forecast was involved and no figure in this file or in the rule depends on those months.
  The reviewer made the same slip and corrected it, and its figures are cut at 2026-06-30.
- "Never equals sequence 1" in my notes is not literally true. The stored German price equals the hourly sequence 1 in 12
  of 35,064 hours of 2021 to 2024, by chance.

## What follows

The forward test starts on 2026-12-01 as registered, on the production data path as it is. A corrected parser is stage
two. It changes the inputs of L0, Q and the public model, so it needs its own registration and the owner's approval, and
it should land after the forward test ends or at a point where the owner accepts that the record ends.
