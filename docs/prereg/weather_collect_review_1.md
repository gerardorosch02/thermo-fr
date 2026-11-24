# Weather collector, history fetch, pack steps and probes: independent review

Reviewed on 2026-11-24, on the owner's word of 15:39 London. The reviewer had not seen the code built. No network, no fit,
no price read. **No code was changed after this review.** The ECMWF arm is frozen by the owner's decision of the same day,
so these findings wait for it.

## Verdict

The collector is honest about what it knows, and no look-ahead was found in the finished GFS study. It is not yet safe to
lean on for a claim of "available by 10:05 Paris" or for the ECMWF comparison. Three problems must be dealt with before an
ECMWF registration. The Paris clock logic is correct, the point lists are identical to the project's, and the tests pass,
though several cannot fail in the way they are meant to.

`weather_collect.py` at 874c4d4 is byte-identical to the head. `collect()` at 4a07b2a is the same as the head.

## Blocking for the ECMWF registration

**B1. `pack_ecmwf` attributes values to the run that was asked for, not one the answer confirms.** Nothing checks that
the first time stamp equals the run, that lead equals the position, or that the returned coordinates are those asked. A
stand-in answer starting six hours late, at coordinates 0,0, was saved as good and packed with leads 6 to 77. Fix: check
all three in fetch and in pack, and fail loudly.

**B2. Nothing in code or saved metadata separates reruns from runs as issued.** The registration must fix the cut as an
exact run time with its source, and the pack should carry a column derived from it.

**B3. `history()` saves any answer with status 200 as complete.** An answer with 3 of 8 runs and 5 of 10 points was saved
and never asked again. The GFS study did not suffer, since its 5,108 runs are exactly 1,277 days times 4. The GEFS history
has had no such check.

## Should fix

- **S1.** One slow or malformed source can cost the whole morning record. A 200 answer that is not JSON aborts the
  snapshot, and the record is written only at the end.
- **S2.** A `Retry-After` header given as a date raises an error, and a value of 3,600 seconds is honoured three times.
- **S3.** An empty answer, an answer labelled with another run, or a short series raises no flag in a morning snapshot.
- **S4.** The run is chosen by listing order. With an ascending listing the oldest midnight run was taken.
- **S5.** `pack` does not detect duplicates.
- **S6.** GEFS member columns can shift when a member is absent.
- **S7.** The cap of 300 runs is per call, not per day, and no ledger is kept.
- **S8.** A permanently missing ECMWF run is asked again on every call and counts toward the stop at 12 failures.
- **S9.** History envelopes store the parsed answer, not the bytes, so the checksum cannot be re-verified.
- **S10.** The default root is two levels above the script, so a run from a normal clone writes beside other projects.
- **S11.** The only clock is the laptop's. Saving each response's `Date` header would give an independent witness.
- **S12.** No cell selection is set for offshore and coastal points, and the pack drops the returned coordinates.

## The task script and the probes

- The task touches only its own task. Its default Python is the forward test's environment, so it depends on another
  project's folder. The old task is removed before the new one is registered, so a failed install leaves none.
- `price_probe.py` sends the ENTSO-E key only to ENTSO-E and masks it in its summary. Bodies are saved untouched, so the
  key stays out of them only if the server never echoes it.
- `price_probe_read.py` prints no price. It does print the largest daily difference between German and Belgian, Swiss and
  Austrian prices for 16 to 22 November, which are spreads inside the forward window, though not France minus Germany.

## The finished freshness study, consistency only

Arms, variants, stop thresholds, bootstrap settings and the 0.25 bar are as registered, and the tables agree with the
paired contrasts. Leads of the fresh arm are 22 to 46 hours, as the results say; the script's docstring still says 22 to
47. One code change after registration is in the log and not under "Departures". The registration calls the lead gain
for the first hour "about 25 hours"; by the code it is 30.

## What an ECMWF registration must state

1. The exact rerun cut as a run time, its source, and which side carries the primary reading.
2. A weather-only integrity stop before any fit: first stamp equals the run, hourly stamps, 838 distinct runs, no
   duplicates, the returned cell for each point.
3. A check that successive runs differ on overlapping hours, to rule out a merged series.
4. The native time step, tested on weather alone on each side of the cut.
5. The stale arm, defined exactly. Only midnight runs exist, so it cannot mirror the GFS stale arm.
6. Availability. Only mornings from 20 November 2026 are witnessed, all in winter. The one logged availability is 06:55
   UTC against a summer gate of 08:05 UTC. Past availability cannot be shown.
7. German points only were fetched.

## Not checked

Anything live, the laptop's files and installed task, the saved history files and packed tables, and PowerShell behaviour
on Windows.
