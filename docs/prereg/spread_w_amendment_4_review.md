# Test of W, amendment 4: independent review and what it changed

Written 2026-11-20, the same day as amendment 4 and before the amended build ran anywhere. `spread_w_amendment_4.md` is
left as committed. Where this document corrects it, this document stands.

A reviewer that had not seen the change being written read the difference between commits 397e4f0 and 12ad02c, ran the 23
logger tests once and probed the new code with scripts of its own. It worked without the network and computed no error
against any price. Its verdict was that nothing blocked installation, that two things should be fixed first, and that the
tests did not pin the main claim of any of the three changes.

## Findings acted on in the code

1. **A failed price request no longer postponed.** Amendment 4 removed the rule that a missing spread of yesterday
   postpones the record. That rule had also covered a price request that failed outright, as distinct from a price the
   source does not hold. With it gone, a passing ENTSO-E error on the price request at 10:00 would have produced a record
   without yesterday's spread and no second attempt.
   Now a run before 10:25 Paris writes nothing when yesterday's spread is missing and the French or the German price
   request failed that morning. An answer of "no data", or an answer that simply lacks the day, does not postpone. That is
   the case amendment 4 was written for.
2. **An infinite value was read as equal to any other value** in the comparison used by the scramble check, in both the
   exact and the toleranced reading. No input holds an infinite value today. Fixed, with a test.
3. **Whatever ends a fetch is now written down.** A fetch ended by something other than an ordinary error was reported as
   a timeout and its true cause lost.
4. **Fetched files are written whole.** Each is written under a temporary name and renamed. A fetch cut short can no
   longer leave half a file under a name the logger reads on later mornings. This weakness was there before amendment 4.
5. **A record issued without the French load forecast now says so by name.** Before, only the count of missing inputs
   showed it.
6. **Tests.** Added: each fetch is timed by its own finish on a morning with one fetch in time and one late; a failed
   price request postpones while an unanswered day does not; a rounding difference under the scramble reaches the record
   as rounding, is not flagged as a value from the future, and does not fail the check; a load forecast still missing
   after 10:25 is named in the flags; an infinite value is never equal to a finite one; a fetch ended abnormally is
   reported as such; a fetched file is whole or absent. 26 logger tests pass, 50 with the other spread tests.

## Sentences of amendment 4 that are superseded

- **"each in its own thread with its own time budget counted from the common start (300, 150 and 150 seconds as before)."**
  A budget is the least time a fetch is waited for. The fetches are looked at in order, so the two German fetches are in
  practice waited for until the French one is done, and the whole never waits more than 300 seconds. The times written
  into the record are each fetch's own.
- **"A source that fails or hangs still postpones, as before."** Precisely: a fetch that raises or is given up on
  postpones; a failed price request that leaves yesterday's spread missing postpones; any other failed request is flagged
  and does not postpone, as before.
- **"If it takes more than five, or the laptop wakes late, the fetch finishes after 10:05 and the record says so."** A
  French fetch still running after 300 seconds is given up on. Before 10:25 that run then writes nothing, and it is the
  record of the 10:30 run that says the fetch was late.
- **"Without it no model has a usable forecast."** After 10:25 the logger does issue without the load forecast. The six
  forecasts are finite and the record names the missing load forecast.
- **"issues with those inputs empty ... nothing is filled in later."** One input is not left empty. On a morning without
  yesterday's spread, `decoupled_share_l1` reads 0, which means fully coupled, because of how the frozen feature code
  computes it. The record's flag has said so since the first build. This is now the ordinary handling of such a morning.
- **The code hash.** The build that is installed is the corrected one. Its code hash is
  c0caa5f013143fdab9c32fedf1dfc930f6232e07b15979a00e654f3b30f5af2f and `spread_live.py` is sha256
  ca4e9547395ad443957b483457073cb89c123cc49ec0a0d0254093ba3844f339. The hashes printed in amendment 4 belong to the build
  that was reviewed and never ran.

## Sentences of amendment 3 that amendment 4 supersedes and did not name

- Section 5, that the self-test requires the inputs to come out identical under the scramble. The self-test now reads that
  check with the tolerance of amendment 4.
- Section 6, that a missing spread of yesterday postpones the record before 10:25 Paris.

## Found and not changed

- **A fetch that is given up on may still finish and write its files.** They are whole, they carry the stamp of the run,
  and a record built afterwards reads them without naming their hashes. Not fixed.
- **Why one input moves by one unit in the last place on the laptop.** The reviewer read the feature code: the input is a
  mean over the hours of one day, and neither of the two explanations offered in amendment 4 is evident from it. The cause
  stays unestablished.
- **Whether "scored as issued" agrees with the refetch rules when a gap in a price of D-1 is never filled.** The code that
  assembles the window in January is not written. This case has to be settled there, before the read.

## What the review confirmed

- No fetch reads a folder that another fetch writes while both run. Six rounds of the three reading steps side by side
  gave the same results as one at a time.
- The times of each fetch are taken in its own thread, and whether it finished by 10:05 is computed from that.
- The postponement logic does what amendment 4 says for a missing load forecast and for a dry run.
- The tolerance cannot hide a real read of the delivery day: the noise that replaces unknown values is of the size of the
  values, and no input is so small that the floor of the tolerance matters.
- The change to the forecasts function touches the evidence only. The self-test reference differs from the previous one in
  the code hash only.

## What no review could check

- The side-by-side fetch against the live sources. Two requests now go to ENTSO-E on one key at once, and the German
  point requests run beside the French weather requests. This has not run.
- Threads and file renames on Windows.
- The facts of the first morning, which were read from the laptop.

The fixes listed above were made after the review. Whether they were reviewed in turn is recorded in
`docs/experiments.md`.
