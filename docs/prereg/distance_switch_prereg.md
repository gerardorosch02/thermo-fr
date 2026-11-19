# The distance switch: a shadow test on live days, registered before any day counts

Written on 2026-11-19, approved by the owner at about 13:35 London the same day. No delivery day of this test has been
issued. The first counted day is 2026-12-01.

## The question

On 18 live days in October and November 2026 the LEAR forecast too large a fall when yesterday's prices sat far above the
centre of its training windows, and the equal-weight ensemble did worse than the tree model. From 1 July to 10 October 2026
the ensemble did better than the tree model, and that period never reached the same condition. So the condition has been
seen in one stretch of the market only. This test asks whether a rule that drops the LEAR when the condition holds does
better than the fixed ensemble on days it has not seen.

**The threshold of 1.6 is a hypothesis drawn from November.** It is fixed here and is not tuned.

## The three forecasts

All three are written by the forward logger each morning, in the same write-once, hash-chained record, before any price of
the delivery day exists.

| Name | What it is |
|---|---|
| g | The public forecast's pre-market version for the day, as the forward test of Q defines it. Today: honest_v2 inputs, 300 trees of 31 leaves, refitted monthly |
| S1 | (g + L0) / 2, with L0 the LEAR of four windows (364, 728, 1092, 1456 days) refitted every morning |
| **W, the challenger** | S1 when the distance is below 1.6. g when the distance is 1.6 or more |

**The distance**, frozen. The 24 prices of D-1 on the scale each of L0's four windows is fitted on for day D,
asinh((p - m_w) / s_w), with (m_w, s_w) the target scaler of window w from its training days ending D-1, averaged over the
24 hours and the four windows. It is computed from the same known-at-10:05 table the LEAR is fitted on
(`forward_q.distance_from_centre`). It reproduces the distance of the July to October evaluation exactly on the two days
checked (0.7646 on 15 July, 1.4640 on 29 September).

**Missing inputs.** If the LEAR is missing (a LEAR fallback day) or the distance cannot be computed, W is g and the record
is flagged "switch fallback". That day is not an activation. If g is missing the day is lost, as in the forward test of Q.

**What the record holds**, per morning: the unrounded distance and its four per-window values with their medians and
scales, yesterday's mean price, the branch taken (`ensemble`, `tree` or `fallback`), the threshold, the 24 values of g, L0,
S1 and W, the issue time, the code hash.

**If the public model changes** during the test (for example to 800 trees), W keeps following the public forecast. The day
of the change is logged as a version boundary and the two parts are reported separately as well as together.

## Calendar

The calendar of the forward test of Q, unchanged: Tuesday to Saturday delivery days, first counted day 2026-12-01, last
2027-04-17, 100 expected days, late and lost days defined there. **One read, on or after 2027-04-18.** No score of W, S1,
g or L0 on a counted day is computed before then. The scoring code for W is not written yet. It will be committed before
the read and can only use what the records hold.

P and Q are outside this test. Amendment 3 of the forward test stands: no error of P or Q is computed on any live day
before its own statistics date.

## What is reported at the read

1. **W against S1, the primary comparison.** They differ only on activated days. The paired daily difference over all
   scored days, with the moving-block bootstrap (block 7, 5,000 resamples, seed 0) and the one-sided end at 1 - 0.05/2.
2. **W against g.** The same statistics. This is the operational question: would the challenger have beaten the public
   forecast.
3. **Activation.** The number of activated days, the number of distinct spells (a spell is a run of consecutive expected
   delivery days on the "tree" branch, so Saturday to Tuesday counts as consecutive), each spell's dates, and the
   difference of W from S1 inside each spell.
4. Fallback days, late days and lost days, by count.

## Interpretation, written now

- **No activation.** The switch is untested. W equals S1 on every day and nothing is concluded about it.
- **One spell.** The result says what the switch saved or cost in that episode. Nothing is concluded about whether it
  would repeat.
- **Two or more spells.** W is **supported** if the one-sided upper end of W minus S1 is below zero and the difference is
  also negative without the longest spell. It is **contradicted** if the mirrored lower end is above zero. Otherwise
  **undecided**.
- Whatever the outcome, the public forecast changes only by a separate decision of the owner.

## Checks before the first counted day

- The logger's change leaves g, L0, P, Q, S1 and S1-Q of a real morning identical to the record written by the previous
  code (checked on the morning run of 2026-11-18 for delivery day 2026-11-19: largest difference 0.0).
- One actual morning record written on the owner's machine after it pulls the change is read back: the distance, the
  branch and W are present, and W equals the rule.
