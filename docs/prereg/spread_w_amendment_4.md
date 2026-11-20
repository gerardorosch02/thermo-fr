# Test of W, amendment 4: three changes to the morning logger after its first unattended morning

Written 2026-11-20, after the first unattended record and before the second morning. The owner approved the three changes
at 10:49 Zurich time on 2026-11-20 with the words "I approve all 3 changes". No forecast of the test window has been
scored and no error against any price has been computed. Results stay sealed until 2027-01-19.

Nothing here changes W, the comparators, the scoring code, the window, the three decisions, the gain rule or the read date.
Only the logger changes, in when it fetches, when it issues and how one check is read.

## What the first morning showed

Delivery day 2026-11-21, build ce32053, read back from the laptop on 2026-11-20 between 09:31 and 09:40 UTC.

- **Two runs.** The run of 10:00 Paris fetched the French window by 10:03:13, the German ENTSO-E files by 10:05:07 and the
  German weather points by 10:05:18. It then wrote nothing, because yesterday's spread was missing, and left the morning to
  the run of 10:30. That run fetched again between 10:30 and 10:33 and wrote the record at 10:33:34 Paris.
- **The record was saved before the auction** and is complete: six models, 24 forecasts each, the registered code hash,
  every fetched file on disk under the hash the record names.
- **No fetch of the issuing run finished by 10:05 Paris.** The record says so in its flags.
- **Yesterday's spread was missing and stayed missing.** ENTSO-E held no coupled price (sequence 1) of DE-LU for delivery
  day 2026-11-20 at 10:03 or at 10:32 Paris, though the early-auction series (sequence 2) was there. The production runs
  of the same morning independently found the Dutch and North Italian prices of that day missing. Eight inputs of the day
  were empty for all 24 hours. Nothing was filled in from a later fetch.
- **The scramble check read false.** One input, `price_lag1_mean`, differed by 1.4e-14 at each of the 24 hours when every
  value unknown at the gate was replaced by noise. That is one unit in the last place of a number near 100. No value was
  missing in one table only.

Compared afterwards, outside the record. For every row up to and including the delivery day, the two French windows of the
morning are identical except one nuclear reading at 09:00 Paris. The German prices are identical. The German load, wind and
solar forecasts of days before the delivery day are identical apart from one point on the boundary of the two request
ranges. The two point files are identical. So the values the record used were on the laptop by 10:05:18, but the record's
own evidence does not show it, and 10:05:18 is after the gate.

## Change 1. The three fetches run side by side

Before, the French window, the German ENTSO-E files and the German weather points were fetched one after the other. The
French fetch takes two to three minutes, so the German fetches started at about 10:03 and could not reliably finish by
10:05.

Now the three start together at 10:00 Paris, each in its own thread with its own time budget counted from the common
start (300, 150 and 150 seconds as before). Each fetch records when it began and when it finished, by the laptop's clock,
at the moment its own thread finished. The record carries `fetches_ran: "side by side"`.

Nothing is fetched before 10:00 Paris. That rule stands, though the German inputs the models use are all public earlier,
so that every record has the same starting point.

What this cannot do. The French fetch alone has taken two to three minutes on this laptop. If it takes more than five, or
the laptop wakes late, the fetch finishes after 10:05 and the record says so.

## Change 2. A missing price of D-1 no longer postpones the record

Before, a run that started before 10:25 Paris wrote nothing if yesterday's spread was missing, and left the day to the
run of 10:30. The rule was added on the night of 19 to 20 November after a dry run at 01:33 Paris found the coupled German
price of the 20th missing. That was put down to the hour of the dry run. It was wrong. The price was set at midday on the
19th and was still missing at 10:32 on the 20th. It was a gap at the source, and waiting half an hour gained nothing and
made every fetch late.

Now:

- **A missing price of D-1 does not postpone.** The run of 10:00 issues with those inputs empty. The record lists them
  under `input_hours_missing_for_D`, carries the flag "yesterday's spread is missing for part of D", and nothing is filled
  in later. This is the handling the refetch specification already prescribes for a value that was not available as of the
  gate.
- **A missing French load forecast of D still postpones.** It is due at 10:00 Paris and is sometimes a few minutes late.
  Without it no model has a usable forecast. A run before 10:25 writes nothing and the run of 10:30 issues with what there
  is.
- **A source that fails or hangs still postpones**, as before.

What this means for the test. On a morning with a source gap the saved forecast is made without yesterday's spread and is
scored as issued. The January report already shows `inputs_missing` by day. A rebuilt day uses a later download, in which
such a gap may since have been filled. The refetch specification covers that case for rebuilt days and it is reported by
provenance class.

How often the coupled German price of D-1 is missing at 10:00 Paris is not known. The count stands at one morning out of
one. Amendment 3 said the price of D-1 was present in all 41 production windows saved by 10:05 Paris on 22 mornings. Those
windows blend the coupled and the early-auction series, so they showed that some German price was present and did not show
that the coupled one was. That sentence is superseded.

## Change 3. The scramble check has a tolerance of one part in a billion

Before, the check demanded that every input of the delivery day came out exactly equal when everything unknown at the
gate was replaced by noise. On the laptop one input came out one unit in the last place apart, and the record carried the
flag "an input of D depends on a value not known at the gate".

Now two inputs count as the same when they are within 1e-9 times the larger of 1 and their size. A value missing in one
table only always counts as different. The record states both readings:

- `inputs_identical_when_everything_unknown_at_the_gate_is_scrambled`, read with the tolerance, with `scramble_tolerance`
  beside it;
- `inputs_bit_identical_under_the_scramble`, the exact reading;
- `inputs_that_differ_only_by_rounding_under_the_scramble`, naming each such input and its largest difference.

The flag about a value not known at the gate is raised only beyond the tolerance. A difference inside it raises a
separate, plainly worded flag.

What is and is not established.

- **The noise that replaces unknown values is of the size of the values themselves.** A feature that read a value from
  the delivery day or later would move by an amount of that order, many orders of magnitude above the tolerance. The check
  still catches that.
- **Why one input moves by one unit in the last place on the laptop is not established.** It does not happen on the
  reference machine with the same files. A sum that takes a different path through memory in the scrambled table would
  produce it. So would a calculation that adds later values and subtracts them again. The second would be a dependence on
  unknown values in arithmetic only, with no information carried, and the check as changed would not flag it.
- **By amendment 3 a difference in the last place of an input can move a forecast by about 2 EUR/MWh at an hour.** So this
  is one more source of the numerical noise already described there. It is not a route by which a price of the delivery
  day could inform the forecast.

## What changes in the records

- The logger's code hash changes from 748b61a9f72a... to
  7a8e583141dc69b04c49cd71342bffe8144d97cd1da210792fb4ebc838610849. `spread_live.py` is now sha256
  ec7c45c2b1cc2cf07fda1ffe78db3e6af84f8d566c42e05240b181d17eb94d82. The model path is untouched: the logger still
  reproduces the frozen batch forecasts of 2026-10-10 to 1e-9 on the reference machine, and the self-test reference differs
  from the previous one in the code hash only.
- The record of 2026-11-21 stays as issued, under the old build, with its flags. It counts as saved before the auction and
  as fetched after 10:05.
- The new build takes effect when the owner installs it on his laptop. Until then the old build runs.

## Counts

Spread forecasts saved before an auction: 1. Of those, records whose own evidence shows every fetch finished by 10:05
Paris: 0.

## Review

Not independently reviewed at the time of writing. The review of these three changes, of amendment 3 and of the fixes made
after the second review is recorded in `docs/experiments.md` when it is done.
