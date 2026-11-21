# Test of W, amendment 5: a second source for the coupled German price

Written on the evening of 2026-11-21, after two unattended records (delivery days 2026-11-21 and 2026-11-22) and before
the third morning. The owner approved building a second source for day-ahead prices at 12:12 Zurich time on 2026-11-21
with the words "yes to both". The design below takes effect only when he installs the build on his laptop. No forecast of
the test window has been scored and no error against any price has been computed. Results stay sealed until 2027-01-19.

Nothing here changes W, the comparators, the scoring code, the window, the three decisions, the gain rule or the read
date. Only the logger changes, in where an input may come from on a morning when ENTSO-E has a gap.

## Why

On both unattended mornings ENTSO-E held no coupled price (sequence 1) of DE-LU for the day before the delivery day. The
forecasts were issued without yesterday's spread and its lags. On the evening of 2026-11-21 ENTSO-E still held no coupled
German price for delivery days 2026-11-20 and 2026-11-21 and no French price for 2026-11-21, though it held both for
2026-11-22. So the gap is not a delay of a few hours. It is a hole that can last days and may never be filled.

## What was checked before choosing

`scripts/price_probe.py` ran on the owner's laptop on 2026-11-21 between 19:49:14 and 19:49:50 UTC. It asked each
candidate, and ENTSO-E at the same moment, for delivery days 16 to 22 November and saved every answer untouched. The
answers were then compared by `scripts/price_probe_read.py`, which prints counts and differences only. A difference
between two published prices is not a forecast error. The saved answers stay on the laptop
(`thermo-fr-weather\price_probe\20261121T194914Z`, `summary.json` sha256
af5362952e56e3cd2b9732186856f64e1b0d88318182fef6d5be90d9256d280b).

| Source | Germany, coupled price | France |
|---|---|---|
| ENTSO-E | lacks 20 and 21 November | lacks 21 November |
| SMARD (Bundesnetzagentur) | the same two days missing | the same day missing |
| energy-charts.info (Fraunhofer ISE) | holds all seven days | the same day missing |
| aWATTar | holds all seven days, hourly only | not offered |

- **energy-charts.info holds the coupled auction, not the early one.** On the five days both hold, its DE-LU price equals
  ENTSO-E's sequence 1 on 480 of 480 quarter hours, to the cent. It equals sequence 2, the early Austrian auction, on 0 of
  672, and the two auctions differ by tens of EUR/MWh at the largest on every day.
- **On the two days ENTSO-E lacks, a third route agrees with it.** The hourly means of energy-charts equal aWATTar's hourly
  prices to half a cent on 48 of 48 hours of 20 and 21 November. ENTSO-E's own sequence 1 does the same against aWATTar
  on the days it holds (120 of 120 hours). Half a cent is what rounding an hourly mean to the cent gives. Against the
  early auction the same hours differ by more than 10 EUR/MWh at the largest on each day.
- **SMARD is not a second source.** It had the same gaps as ENTSO-E in both countries and equals ENTSO-E on every quarter
  hour both hold (480 of 480 for Germany, 576 of 576 for France).
- **No candidate held the French price of 21 November.** energy-charts.info's French series equals ENTSO-E's on 576 of 576
  quarter hours and has the same hole.
- **energy-charts.info limits requests.** Two of seven requests sent within four seconds were refused (HTTP 429).

What is not established.

- **Where energy-charts.info gets the German price on days SMARD lacks.** Its answer names Bundesnetzagentur and SMARD as
  the licensor, and SMARD did not hold those days. It must have another feed. Which one is not known.
- **Whether it holds a price by 10:00 Paris on the morning after the auction.** The probe ran in the evening, a day and a
  half after the first missing auction. The logger's daily fetch will show.
- **How it behaves over more than one week.** The agreement rests on five days in common and two days against a third
  route.

## The change

ENTSO-E stays the first source. energy-charts.info gets one request each run, alongside the other fetches, for the ten
delivery days before D. A refused request is tried again up to three times within the fetch's time budget. The answer is
saved untouched under `live/german_price_second_source/<stamp>.json` and its hash, size and fetch times are written into
the record. An answer that cannot be read is kept too, and the record's error names the file, its hash and its size.

An hour of the German coupled price is taken from the second source only if all of the following hold.

1. **The files fetched from ENTSO-E hold no value for that hour**, in the stored files and in every fetch saved up to that
   moment. An hour they hold is never replaced. On a morning when the price request to ENTSO-E failed, this means the
   files of earlier mornings, and the record carries the failed request among its flags.
2. **The hour belongs to a delivery day before D** and not before 2026-10-11, the first day the stored files do not hold.
   Whatever the answer contains for D or later is counted in the record and never read. The request itself ends at D-1.
3. **All four quarter hours of the hour are in the answer**, in EUR per MWh, each a number between -1000 and 10000. The
   hourly value is their mean, the rule the ENTSO-E path already uses.
4. **The two sources agree that morning.** Over the ten delivery days before D, every hour both hold must agree to a tenth
   of a cent on the hourly mean, and there must be at least 24 such hours. A single quarter hour apart by one cent moves
   the mean by a quarter of a cent and fails. One hour of disagreement and nothing is taken from the second source that
   morning. Fewer than 24 hours in common and nothing is taken either, because there is nothing to check it against.

What condition 4 can and cannot show. It guards against the second source serving another zone or another auction. It
cannot vouch for the hours actually taken. On the hours both hold, energy-charts.info, SMARD and ENTSO-E are one and the
same data as far as the probe shows. The hours taken are by definition those ENTSO-E lacks, and for those energy-charts.info draws on a feed that
has not been identified. The only evidence about that feed is the 48 hours compared with aWATTar above.

The decision is made afresh each morning, from every answer saved so far. Three things follow.

- **An hour stays filled only while the conditions hold.** An hour taken on Monday is taken again on Tuesday only if the
  sources still agree on Tuesday. If they disagree on one hour of the last ten days, nothing is taken that morning, and
  the lags and training rows built from earlier fills are missing again for that morning.
- **After ten days with fewer than 24 hours of ENTSO-E prices, nothing is taken.** With so little in common over the ten
  days before D there is nothing to check against, so every fill is dropped at once. In a hole that long the French price is likely missing
  too.
- **Earlier answers are read as well as the latest.** A later answer wins where it holds a value. An hour a later answer
  no longer holds is still read from an earlier one. The record lists every file read with its hash, including a file a
  given-up fetch finished writing after its run had moved on.

The record states, under `inputs.german_price_second_source`: the files used, how many hours both sources hold over the
ten days, how many of those agree, the largest difference, whether the source was trusted that morning, how many hours it
holds that ENTSO-E lacks, how many were taken and for which delivery days, and how many hours of D or later were in its
answers. A record with hours taken carries the flag "the coupled German price of N hour(s) before D came from the second
source (energy-charts.info) because the files fetched from ENTSO-E held none". A morning on which the second source holds
hours ENTSO-E lacks but fails the agreement rule carries a flag saying so and why. Saved answers that cannot be combined
stop no record: nothing is taken and a flag says so.

Two rules of amendment 4 are narrowed.

- **A failed request to the second source postpones the first run only when it matters**, that is when the German price
  of the eve is missing in the files fetched from ENTSO-E. Otherwise the run issues and the record carries the flag
  "german_price_second_source failed". It postpones in that case even when the French price of the eve is missing too
  and yesterday's spread cannot be built, because the German price of the eve is an input in its own right. The cost is
  a record written by the run of 10:30.
- **A failed German price request to ENTSO-E postpones only if the German price of the eve is then missing.** Before, it
  postponed whenever yesterday's spread was missing for any reason. With the second source in, the German side can be whole
  though the ENTSO-E request failed.

The second source is a fourth fetch and is timed like the others. If it finishes after 10:05 Paris the record says a fetch
finished after 10:05.

## What does not change

- **The French price has no second source.** A morning on which ENTSO-E lacks the French price of D-1 is issued without
  yesterday's spread, as before. A candidate exists (the wholesale market interface of RTE, which needs an account in the
  owner's name and whose data belongs to the exchanges), and is not used.
- **Scoring.** The refetch specification leaves an hour out of scoring when it has no realised spread or no spread of the
  previous day, and both are read from ENTSO-E's refetch in January. That stands. So this amendment gives a morning better
  inputs and does not by itself give the test more scored hours. If ENTSO-E never publishes the German price of
  2026-11-20, the hours of the 20th and of the 21st stay out of scoring whatever the logger read. Whether January's
  scoring may take a realised price from the second source, under the same four conditions, is not decided here. It needs
  the owner's dated decision before the read date and before any price of the window is read.
- **The neighbours' prices.** The Dutch and North Italian prices that the French window lacked on the same mornings come
  through the production fetch, which this amendment does not touch.
- **Rebuilt days.** A day with no saved record is rebuilt in January from ENTSO-E's refetch by the refetch specification.
  The second source plays no part in a rebuild. Rule 3 of that specification, "missing values stay missing, no fill",
  stands for rebuilt days. For saved days it is narrowed by this amendment to the one case described above.
- **The self-test.** It reads the static files only, so it never reads the second source, and the logger still reproduces
  the frozen batch forecasts of 2026-10-10 to 1e-9 on the reference machine.

## What this means for the test

- **All six models read the same inputs**, so a filled hour reaches the lags of H, HW, G, GK, GM and W on the same
  morning. As a training row it reaches the models refitted weekly (W, HW, GK) at their next weekly refit and those
  refitted monthly (H, G, GM) at the next month. That is the difference in refit the test is about, and it applies to
  every training row. Whether the models gain equally from the lagged spread is not examined.
- **Saved days fall into two kinds from here on.** A record with hours from the second source says so. The January
  rebuild code is not written yet. Counting those days separately, beside the days issued with yesterday's spread missing,
  is added to what it must do. The scoring code itself does not change.
- **A filled hour is also a training row.** While the German price of a past day is filled, the spread of that day is a
  target the models are fitted on from the next refit. A January rebuild from ENTSO-E alone lacks those rows if the hole
  is never filled. The registered comparison of rebuilt and saved forecasts will then show a difference on the days after
  a filled hole that is larger than numerical noise. That is expected, and it is reported with the count of filled hours.
- **The records of 2026-11-21 and 2026-11-22 stay as issued**, without yesterday's spread, under the old builds.
- **A later record can differ from a rebuild in the last place.** If ENTSO-E fills the hole afterwards with the same
  prices, a rebuilt input equals the saved one to rounding. By amendment 3 that can still move a forecast by about
  2 EUR/MWh at an hour. If ENTSO-E fills it with different prices, the January comparison of input fingerprints will show
  it for that day.
- **energy-charts.info asks for attribution** (CC BY 4.0). Its values are an input kept on the laptop. Nothing published
  reproduces them.

## Counts

Spread forecasts saved before an auction: 2. Of those, records whose own evidence shows every fetch finished by 10:05
Paris: 0. Mornings on which the coupled German price of D-1 was missing at 10:00 Paris: 2 of 2. Mornings on which the
French price of D-1 was missing: 1 of 2.

## Review

An independent reviewer who had not seen the change being built went through the code, the tests and this document at
commit e714215, and recomputed every count in the section "What was checked before choosing" with its own code. It found
no way for a price of D or later to reach the inputs of D and no departure of the code from the four conditions. It then
looked a second time, at the changes made in answer (commit bd185f8). Its findings and what was changed are in
`spread_w_amendment_5_review.md`. This document is the version after those changes. The code hash of the build to install is recorded in the review document and in `docs/experiments.md`.
