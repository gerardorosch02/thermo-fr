# Test of W, amendment 6: a day with no scored hour

Written on 2026-11-23. In force from the owner's decision of 2026-11-23, 11:22 London, given in his own words after reading
the draft (`spread_w_amendment_6_draft.md`, sha256 cffa75cd6de9...). His words, in full on the points
decided.

> I accept Parts A and C with a minimum of both 95 eligible delivery days and 2,280 eligible hours. I decline Part B for
> the primary decisions. Use the common eligible hours for the registered model comparisons. Evaluate yesterday's-spread
> baseline against the models on the subset where that baseline exists.
>
> Before putting the amendment in force, please identify the January year, explain which dataset has the gap, and
> distinguish observed coverage from projected coverage through 18 January 2027. At the final read, report the excluded
> dates, hours and known reasons. Keep the original end date and single read.

No forecast of the test window has been scored and no error against any price has been computed. The decision was taken
knowing which days have holes and nothing about any price or error.

## The dates, with their years

- **The window** is delivery days 2026-10-11 to 2027-01-18 inclusive. It is not changed.
- **The read** is one read, on or after 19 January **2027**. It is not changed.
- **"January"** in this amendment, in amendment 5 and in the refetch specification means January 2027.

## Which dataset has the gap

The day-ahead prices on the ENTSO-E Transparency Platform (document type A44), which is the registered source of every
realised price in this test.

| Series | Delivery days with no value | Last checked |
|---|---|---|
| DE-LU, the coupled auction (sequence 1) | 2026-11-20 and 2026-11-21, 0 of 24 hours each | files fetched 2026-11-23 09:01 UTC |
| France | 2026-11-21, 0 of 24 hours | window fetched 2026-11-23 10:33 UTC |

The forecasts, the weather files and the German ENTSO-E forecasts are not the gap. SMARD showed the same holes on
2026-11-21. energy-charts.info holds the German price of both days and not the French price (amendment 5). Why ENTSO-E
lacks these days is not known.

## Coverage observed, and coverage projected

These are two different things and only the first is a fact.

**Observed, delivery days 2026-10-11 to 2026-11-23, 44 days and 1,057 hours.** Counted on 2026-11-23 from the files the
loggers saved, as counts of hours with a value. No price was read as a price.

| | Days with every hour | Days short |
|---|---|---|
| French price, ENTSO-E | 43 of 44 | 2026-11-21 |
| German coupled price, ENTSO-E | 42 of 44 | 2026-11-20, 2026-11-21 |
| Both, so a realised spread exists | 42 of 44, 1,009 hours | 2026-11-20, 2026-11-21 |

This is an upper bound on eligible hours so far, and not the count itself. An hour also needs the German proxies and a
forecast from each model, and on a rebuilt day the weather as issued two days ahead. None of that can be counted before
the rebuild of January 2027. ENTSO-E may also still publish the missing days, or revise others.

**Projected, delivery days 2026-11-24 to 2027-01-18, 56 days.** Nothing is known about them. Two readings, neither a
forecast.

- If no further day is lost, the window ends with at most 98 days and 2,353 hours.
- If days went on being lost at the rate seen so far, 2 in 44, about 2.5 more would go and the window would end near 95.5
  days. That rests on one incident of two consecutive days, so it says little, in either direction.

The floor of 95 days and 2,280 hours therefore leaves room for three more lost days, or 73 lost hours. The hours floor is
95 full days, so it can bind before the day count does. Hours lost on days that are otherwise scored count against it.

## The rule

**1. Eligible hours.** An hour of the window is eligible if it has a realised spread, the German proxies, and a forecast
from each of H, HW, G, GK, GM, W and C. The realised spread is read from ENTSO-E's refetch of January 2027 and from no
other source. A missing spread of the previous day no longer makes an hour ineligible.

**2. Eligible days.** A delivery day with at least one eligible hour. Its daily error is the mean over its eligible hours.

**3. The floor.** The test is read if at least 95 delivery days and at least 2,280 hours are eligible, both. Otherwise no
decision is drawn, the three decisions are recorded as "not readable", and the figures are reported marked as not decisive.
This replaces "100 days and at least 2,377 hours scored" in stop condition 6.

**4. The registered comparisons.** The three decisions (W against HW, W against H, W against G) and every error of a model
are computed on the common eligible hours, the same hours for all seven models. The gain rule, the level, the block
bootstrap (block 7, 5,000 resamples, seed 0, on the eligible days in date order with gaps closed) and the slices (defined
over the eligible days) are as registered.

**5. Yesterday's spread.** It is a reported baseline. It is scored on the eligible hours where the spread of the previous
day exists, read from ENTSO-E's refetch. Every figure that sets a model against yesterday's spread, the direction figure
included, is computed on that subset for both sides, and the number of hours in the subset is stated.

**6. No second source for realised prices.** Part B of the draft is declined for the primary decisions. No realised price
in this test comes from energy-charts.info or any source but ENTSO-E. Amendment 5 stands as it is: the second source may
supply an input on a morning, and never a realised value.

**7. What the read reports.** For every hour of the window that is not eligible, the date, the hour and the reason where
it is known, and for every delivery day with no eligible hour, the date and the reason. The reasons are those of the
schedule: no realised spread (and which country's price is missing), no German proxy, no forecast issued, weather not as
issued two days ahead. The count of eligible hours without yesterday's spread is given separately. The schedule still
holds all 2,401 hours of the 100 registered days.

**8. Stop condition 3.** A whole-day hole in an input stops the rebuild unless an amendment names the input and the day.
This amendment names none. The holes known today are in realised prices and in the lags built from them, and they will be
named, with their days, in a short amendment written when the refetch of January 2027 shows which remain.

## What changes in the code, and what is owed

- `spread_w_test.py` (registered sha256 98fb99817b85...) must change in its checks and in how yesterday's spread is
  scored, to do exactly what sections 1 to 7 say. Nothing else in it changes. **This is not done yet.** The change, its
  tests and its new hash go to an independent reviewer who has not seen it built, and are logged before the read.
- The draft rebuild code must apply the floor in stop condition 6 and drop the previous day's spread from eligibility.
- Until both are done and reviewed the test cannot be read, which is the same position as before this amendment.

## What does not change

W, the comparators, the logger, the window, its end date, the single read on or after 19 January 2027, the three
decisions, the gain rule, and the rule that a pass is a reason to test forward and never proof.

## One point left open

The owner declined part B "for the primary decisions". Whether he wants a second set of figures with a realised German
price from the second source, reported beside the result and deciding nothing, is not settled here and is not done unless
he says so before the read.
