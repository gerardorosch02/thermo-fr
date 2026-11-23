# Test of W, amendment 6: how the code applies it, and what it decides that the documents do not

Written on 2026-11-23, on the owner's word "3. yes" (11:30 London) to changing the scoring code and the rebuild draft and
sending both to an independent reviewer. This is a record. It changes no rule. Where it reports a choice the documents
leave open, the choice is listed for the owner and is not settled by being written here.

No forecast of the test window has been scored and no error against any price has been computed. Everything below was
run on design days, to 2026-10-10.

## A correction to the record

Amendment 6 says the frozen scoring file `spread_w_test.py` has sha256 98fb99817b85... That is the hash amendment 1 froze.
The file as it stands is 998d40bf7dcc8ea3522bdd645abe489112b42c16fdcbcc3e2d481fa8772ddf3b, the hash amendment 2
registered, and it has not changed since. The mistake is in the wording of amendment 6 and not in the file.

## The frozen file is not edited

Amendment 6 says `spread_w_test.py` "must change in its checks". It was left as it is. The morning logger's code hash
covers every file in the `forecast` package, so editing the file would have changed the hash of a logger that is running.

A new file takes its place for the one read, `src/thermo_fr/rebuild/spread_w_score.py`. It imports the frozen file's own
functions for the three decisions, the gain rule, the level, the bootstrap and the slices, and replaces only the checks
and the handling of yesterday's spread. On a complete window its three decisions are identical to the frozen file's. The
read refuses if the frozen file is not the registered one.

## What the code does, section by section

1. **Eligible hours.** The frame holds the hours with a realised spread, the German proxies and a forecast from each of
   the seven models. Yesterday's spread may be missing in it.
2. **Eligible days.** A day with at least one eligible hour.
3. **The floor.** 95 days and 2,280 hours, both. Under it the three decisions are "not readable", the three conditions of
   the gain rule are withheld, and the remaining figures are marked not decisive.
4. **The comparisons.** The decisions are computed on a copy of the frame that does not contain yesterday's spread at
   all, so it cannot reach them.
5. **Yesterday's spread.** It has its own block, overall, by month and by provenance class, where both it and the seven
   models are scored on the hours where it exists. It appears in no other table.
6. **No second source.** Nothing reads one. The realised spread comes from the refetched ENTSO-E files only. No second set
   of figures is computed, as the owner decided.
7. **What the read reports.** Every hour left out, by day and reason, every day with no eligible hour, and the count of
   eligible hours without yesterday's spread.

The exception to stop condition 4 is applied by the rebuild and carried into the read. The condition passes only with the
reconstruction file by hash, a self-test run on the spot in the same workspace, and H, G, C and W within 1e-9 on every
stored hour with no forecast missing. HW, GK and GM are compared the same way and reported. The read refuses without the
words "met under the exception of 2026-11-23" and prints them beside the decisions.

## Choices the documents do not settle

For the owner. None is in force by being listed.

1. **The floor does not stop the run.** Amendment 6, section 3, says the floor "replaces" the numbers in stop condition 6,
   and also that under the floor "the figures are reported marked as not decisive". Both cannot hold if a failed stop
   condition means nothing is read. The code follows the second sentence. Stop condition 6 records the floor and stops the
   run only if the scoring code would refuse the frame or the schedule. The consequence is real. Under the floor the single
   read is spent and every error is seen, with no decision. The other reading is that under the floor nothing is read at
   all. **This needs the owner's word.**
2. **Reasons are fixed texts.** "no realised spread: no French price", "no German price" or "no French or German price",
   and "no German proxy", for an hour that had a forecast. "no forecast issued" or "weather not as issued two days ahead"
   for one that had none. Where two apply both are written, so a missing price is always named.
3. **A missing forecast of C** makes the hour "no forecast issued". C is rebuilt on every day, so this should not arise.
4. **A day with one eligible hour counts as a full day** in the paired daily difference and in the slices. This follows
   section 2.
5. **The by-class comparison** keeps the frozen threshold of 14 days, counted on eligible days.
6. **Figures against yesterday's spread** have no minimum number of hours or days, beyond two days for the paired figure.
7. **On a saved day, yesterday's spread in the frame** is the value from the refetch, as amendment 5 says, and not what
   the logger saw that morning.
8. **Stop condition 4 accepts one file only**, the one holding all seven models, so HW, GK and GM cannot go unchecked.

## Review

An independent reviewer who had not seen the code built reviewed commit 3595a83 against amendment 6 and the exception.

- **Arithmetic.** The reviewer recomputed the errors, the gains, the paired differences and a bootstrap of their own on the
  replay of 1 July to 10 October 2026 and on five variants. The largest difference from the code was 2.2e-15. On a complete
  frame the three decisions were identical to the frozen file's.
- **Blocking findings, both fixed.** The read did not state the exception. Stop condition 4 could pass with forecasts
  missing, with HW, GK and GM unchecked, or on a self-test taken on trust from a file.
- **Other findings fixed.** Yesterday's error beside model errors on different hours. Reasons looser than the amendment.
  One reason per hour. Stop condition 6 not running the scoring code's own checks. Reading files to the last bit. Infinite
  and non-numeric values. The wrong hash cited for the frozen file.
- **Left for the owner.** Choice 1 above, and the registration of the hashes of the two new files.

A second independent reviewer, who had seen neither the code built nor the first review at work, checked the fixes in
commit 80da8db.

- **Every finding it was asked to verify is fixed**, and it found no new blocking problem. Its own recomputation on the
  replay and four variants agreed with the code to 1.8e-15.
- **Two further points, both fixed afterwards.** The read took the record of stop condition 4 on its word. It now checks
  the file hash, the self-test and that H, G, C and W were compared on every stored hour. The read did not say whether HW,
  GK and GM reproduced. It now names any that did not.
- **Noted, not changed.** On a saved day a missing forecast of C alone would mark the hour as not issued, though the logger
  issued six forecasts. A window with no eligible hour at all stops the run and cannot be recorded as "not readable",
  which belongs with choice 1 above. The check on the repeated hour of 25 October is for the registered window only.

Those last fixes were made after the second review and have not been seen by a reviewer. They are small and are covered by
tests, but the statement stands.

## The files, as pushed

| File | sha256 |
|---|---|
| `src/thermo_fr/rebuild/spread_w_score.py` | 2775f0e0b3b2115129140b4fb8a94253da7db92e650bc81d392f06d94314292f |
| `src/thermo_fr/rebuild/spread_w_rebuild.py` | 5d3c25fb89ea5dbccadb04e70da957587845bd43f8acee0fdceea65532ab13a9 |
| `tests/test_spread_w_score.py` | 8efa97220b7d258a19010fa63dbc03efcc23a1861f6d7fc952b40fa3f6a9438a |
| `tests/test_spread_w_rebuild.py` | 769d98218699e6ff0cd7f8c0d435ddbbfe2aa205955873f900049bf03b6d342c |
| `src/thermo_fr/forecast/spread_w_test.py`, frozen, unchanged | 998d40bf7dcc8ea3522bdd645abe489112b42c16fdcbcc3e2d481fa8772ddf3b |

These hashes are recorded and not yet registered as the code of the read. Registering them is the owner's step, once
choice 1 is settled, because that choice may change the code.

## What was run, on design days only

- 73 tests pass across the scoring, rebuild and logger suites, with the static files.
- A replay of 1 July to 10 October 2026 through the final rebuild passes all six stop conditions, with the self-test run
  on the spot (2026-11-23 12:33:34 UTC) and all seven models reproducing the reconstruction on 21,430 hours to 9e-14.
- Reading that replay through the final scoring code gives the registered design figures for the opened window, 14.84 for
  W and 18.16 for HW, with W against HW failing and W against H and G passing, as registered.
- The logger's code hash is unchanged.
