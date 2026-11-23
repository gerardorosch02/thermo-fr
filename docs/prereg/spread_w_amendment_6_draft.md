# Test of W, amendment 6, DRAFT for the owner's decision: a day with no scored hour

**Status: a draft. Nothing in it is in force.** Written on 2026-11-23 on the owner's word "go for both, first amendment
draft" (10:24 London). It sets out a decision that is his to take. It becomes an amendment only when he says which parts
he accepts, in his own dated words. The final text is then written, hashed and logged, and the code is committed, before
the read date and before any price of the window is read for scoring.

No forecast of the test window has been scored and no error against any price has been computed. Results stay sealed until
2027-01-19.

## What was known when this was written

Only which days have holes. That comes from the logger's own records and file counts, read each morning. No realised
spread of the window has been looked at, no forecast has been compared with a price, and nobody knows which way any of
the parts below would move a result. That is the reason to decide now and not in January.

One thing is known that is not neutral. A day on which the exchanges' prices never reach ENTSO-E may not be an ordinary
day. If the hole of 20 and 21 November came from a market incident, the spread on those days may have been unusual too.
Leaving such days out could help or hurt any model. This cannot be checked without reading the prices, so it is stated
here and not resolved.

## The problem

The registration asks for 100 days and at least 2,377 hours scored (stop condition 6). The frozen scoring code refuses a
frame that does not hold exactly the 100 registered days. The slack built in was one day's worth of hours.

An hour is scored only if it has a realised spread, a spread of the previous day at that hour, and the German proxies.
The realised spread and the previous day's spread are read from ENTSO-E's refetch in January (amendment 5). So one missing
day of one country's price removes that day, and the day after it because its "previous day" is gone.

On the morning of 2026-11-23, 44 days into the window, ENTSO-E still held no coupled German price for delivery days 20
and 21 November and no French price for 21 November. If that stays so, three days have no scored hour, the test has 97
days at most, and it cannot be read at all as registered.

## What is ruled out already

- **Lengthening the window.** The registration lets the owner change the window only by an amendment written before any
  German price or forecast after 2026-10-10 was fetched. That moment has passed.
- **Estimating a missing price.** A realised price that is modelled or interpolated is not a realised price.
- **Doing nothing and deciding in January.** By then the prices are about to be read, and any rule chosen then is chosen
  under suspicion.

## The parts to decide

Each part stands alone. The owner can accept any, all or none.

### Part A. The test is read on the days that can be scored, above a floor

A day with no scored hour is left out, counted and named in the report with its reason. The test is read if at least
**95 days and 2,280 hours** are scored. Below that, no decision is drawn. The figures are still reported, marked as not
decisive, and the three decisions are recorded as "not readable".

- A day with at least one scored hour is a day, as now. Its daily error is the mean over its scored hours.
- The block bootstrap runs on the scored days in date order, block 7, as registered. A gap is closed up, so a block can
  span a missing day.
- The slices keep their definitions, taken over the scored days.
- The schedule still holds all 2,401 hours of the 100 registered days, so every hour left out is accounted for.

Why 95. The registered slack was one day. Three are already at risk, so one day cannot be kept. On design data W against
G passed in 72% of 100-day windows (amendment 1), and every day lost lowers that. 95 keeps the test close to what was
registered. The alternative is **90 days and 2,160 hours**, which is safer against further holes and weaker as a test.
Without parts B and C the window has lost 3 days in 44, and at that rate it would end near 93, under a floor of 95. One
incident is thin ground for a rate, in either direction.

### Part B. A realised German price may come from the second source

Where ENTSO-E's January refetch holds no coupled German price for an hour, the hourly price of energy-charts.info may be
used as the realised German price, for the realised spread and for the previous day's spread, if all of the following
hold. They are the four conditions of amendment 5, set for one read in January instead of one morning.

1. **ENTSO-E's refetch holds no value for that hour.** An hour it holds is never replaced.
2. **The hour belongs to the window**, 2026-10-11 to 2027-01-18.
3. **All four quarter hours are in the answer**, in EUR per MWh, each between -1000 and 10000. The hourly value is their
   mean.
4. **The two sources agree over the window.** The second source is fetched once in January for the whole window, saved
   untouched and hashed. On the hours both hold, the hourly means must agree to a tenth of a cent on at least 99.9% of
   hours, with at least 2,000 such hours, and every hour that disagrees is listed. If not, nothing is taken.

The report states how many hours and which days were taken from the second source, and gives the three comparisons a
second time with those days left out. The second set of figures is reported and is not a decision.

What this cannot show is the same as in amendment 5. On the hours both hold, the two sources are one and the same data.
The hours taken are those ENTSO-E lacks, and for those energy-charts.info draws on a feed that has not been identified. The
evidence for that feed is 48 hours that agreed with a third route, aWATTar, on 20 and 21 November.

### Part C. An hour without the previous day's spread stays in the three decisions

Today such an hour is dropped for every model, because "yesterday's spread" is a reported baseline and all models are
scored on the same hours. Under this part the seven models (H, HW, G, GK, GM, W, C) are scored on every hour that has a
realised spread, the German proxies and a forecast from each of them. Yesterday's spread is scored on the hours where it
exists, the count is stated, and every figure that compares a model with yesterday's spread is computed on those hours
only.

The reason is not only to save days. The morning after a hole is the morning the forecasts were issued without their
previous-day inputs. Those are real forecasts, issued as they would be in use. Dropping them hides how the models do on
exactly the mornings when the data failed.

Yesterday's spread is in none of the three decisions, so the decisions are still made on hours common to the models they
compare.

### Part D. A second source for the French price

Not proposed here. No source tested held the French price of 21 November. The candidate, RTE's wholesale interface, needs
an account in the owner's name and its terms read first. If it is ever used for scoring it needs its own amendment with
its own agreement check.

## What each choice gives on the three days known today

| Delivery day | As registered | A alone | A and B | A and C | A, B and C |
|---|---|---|---|---|---|
| 2026-11-20, no German price | unscored | left out | scored | left out | scored |
| 2026-11-21, no French price | unscored | left out | left out | left out | left out |
| 2026-11-22, no previous day | unscored | left out | left out | scored | scored |
| Days that can be scored today | 97, not readable | 97 | 98 | 98 | 99 |

Under part B alone the 21st has a previous day again but still no French price. Under part C alone the 21st still has no
realised spread, because its German price is missing too.

## What would change in the code, and what would not

- **The scoring code changes in its checks only.** `spread_w_test.py` would accept between the floor and 100 days, list
  the days left out, and under part C compute yesterday's figures on its own hours. The gain rule, the level, the
  bootstrap, the slices and the three decisions stay as they are. Its registered hash (98fb9981...) would be replaced by a
  new one in the final amendment, and the change goes to an independent reviewer who has not seen it built.
- **The rebuild code** takes the accepted holes of stop condition 3 from this amendment, applies the floor in stop
  condition 6, and under part B reads the second source through one function with the four conditions.
- **Nothing else changes.** W, the comparators, the logger, the window, the read date, the single read and the rule that a
  pass is a reason to test forward and never proof.

## My recommendation

Accept A, B and C, with the floor at 95 days.

- A is needed whatever else is decided. Without it one missing day anywhere makes the whole test unreadable.
- C costs the least and is the most defensible. It only stops a reported baseline from deciding which hours the models
  are judged on.
- B is the weakest of the three, because the feed behind the hours taken is not identified. The second set of figures,
  with those days left out, is what keeps it honest. If the two sets disagree on a decision, the report must say so first.

The case against B is fair. A stricter reading keeps ENTSO-E as the only source of realised prices and accepts 98 days.
That is a sound choice too, and it would not weaken the test much.

## What the owner needs to say

Which of A, B and C he accepts, and the floor, 95 or 90. Until then the registration stands as it is, and as it stands the
test cannot be read unless ENTSO-E fills its holes.
