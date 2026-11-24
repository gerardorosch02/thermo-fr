# Spread logger: independent review of amendments 3, 4 and 5

Reviewed on 2026-11-24, on the owner's word "proceed with independent reviews owed on amendments 3, 4, and 5 and the
weather collector" (15:39 London). The reviewer had not seen the code built and judged it against the amendments. Tree
13aa621, whose logger code hash is c9621063756d..., the hash of build d1e30ef running on the laptop. No forecast was
compared with any price. **No code was changed after this review.** Any change to the logger needs a new build, an
amendment and the owner's word on the scheduled task.

## Verdict

Nothing blocks the build now running. The reviewer found no route for a price of the delivery day or later into an input
of that day, and the four conditions of amendment 5 held against constructed attacks. The suite gave 34 passed with the
static files. Five things are worth fixing. Two can change forecasts on a bad morning, one can lose a day's record, two
weaken the record.

## Should fix

**S1. An ENTSO-E answer holding only the second auction is read as the coupled price. Affects forecasts.**
`entsoe_rest.coupled_sequence` returns the lowest sequence present, per folder, and a later folder wins. With the first
sequence stripped from a real price file, 192 coupled hours were replaced by early-auction values and the 48 missing hours
were filled with them. No flag fires, because the second source is then distrusted and holds nothing ENTSO-E "lacks". It
needs the first sequence to be absent over the whole request, D-10 to D. Not seen so far, but ENTSO-E has already dropped
the first sequence for single days. Fix: keep only sequence 1 for DE-LU and treat a folder without it as holding no price.

**S2. A first run killed without removing its marker makes the 10:30 run stand back, and the day is lost. Affects
records.** The marker goes stale after 30 minutes and the triggers are 30 minutes apart, so real starts are 29 minutes 46
to 52 seconds apart. A reboot or forced restart between 10:00 and about 10:06 Paris would do it. Fix: staleness of 20
minutes or less, or check that the marker's process is alive.

**S3. HTTP 599 from ENTSO-E is not retried. Affects forecasts, one input missing.** Only 429, 500, 502, 503 and 504 are
retried. On 23 November the German load forecast failed at 10:00:42 Paris and the fetch ended at 10:01:05, with about four
minutes in hand. Fix: retry any 5xx inside the 150-second budget.

**S4. The self-test passes with unregistered code. Affects records.** `code_matches_reference` is computed and `pass`
ignores it. Fix: require the match.

**S5. A disagreement between the two price sources raises no flag unless the second source also holds hours ENTSO-E
lacks.** This matches amendment 5 as written, but a wholesale disagreement is the one visible symptom of S1. Fix: flag
whenever fewer hours agree than both hold.

## Minor

1. `config.py` is outside the code hash, though it supplies the cities, the time zone and the bidding zone.
2. A stray file in `live/german_points` stops every run. The other readers skip such names.
3. The French window and the German points are saved as processed tables, not raw answers, though amendment 2 says raw.
4. The record names the last French window and a count, with no hash of the earlier windows or of the proxy weight files.
5. `upto` filters on the stamp in the file name, which is the run's start and not the write time.
6. No re-check after the early-start sleep.
7. Two exits leave no line in `attempts.jsonl`.
8. Second-source retries are not bounded by the 150-second budget, though amendment 5 says they are.
9. Duplicate stamps in a second-source answer are resolved silently.
10. "missing for part of D" is used when all 24 hours are missing, and the flag does not name the failed item.

## By area

- **Amendment 3.** Clean except S4 and minor 3 and 4. Start time, cutoff per fetch, auction gate, write-once records,
  manifest chain, dry runs kept apart, fingerprints, self-test branches and removal of rows without a wind forecast all
  checked against the code.
- **Amendment 4 and its four late fixes (66a2c77).** All correct. None changes a model computation.
- **Amendment 5 and its late changes (d1e30ef).** Clean on the four conditions. Constructed answers holding values for D
  and later, and for hours ENTSO-E holds, left the German series bit-identical. A difference of 0.0011 on one hour gave
  "not trusted" and nothing taken; 0.0009 was still trusted. Fills are recorded per delivery day. No scoring code reads the
  second source.
- **Look-ahead.** None found. Vintage is recorded and not enforced: a 10:30 run reads what is published by then and says so.
- **The record of 2026-11-24.** Both events were handled as the amendments require and the flags are accurate. The only
  fault is the missing retry (S3).

## Tests

Two tests would pass with the code wrong: the one on items absent for the delivery day re-implements the expressions
without calling the fetch, and the dry-run test stubs the fetches. No test covers the French window fetch and overlay,
the German refetch with live folders, a file with the second auction only, removal of rows without a wind forecast,
takeover of a stale marker, the leak flag firing, `verify` detecting a changed record, or a code hash mismatch in the
self-test.

## Decisions the code makes that the amendments do not state

- A run starting up to five minutes before 10:00 waits and proceeds.
- The German fetch asks for prices through D, and they are dropped afterwards.
- The scramble is vacuous for German prices and ENTSO-E forecasts, because the cut removes them first.
- The French proxy fill runs before the scramble.
- A row with load but no wind forecast loses its load and solar too.
- The second source fills the spread's German leg and not the neighbour column, so `price_de_lu_lag1` can stay missing
  while `de_l1` is filled.
- H, G and GM are refitted every morning on the table as it stands, so an ENTSO-E backfill can change a "monthly" model
  within the month.
- ENTSO-E hourly values are the mean of whatever quarter hours exist. Only the second source requires four.
- The installer has a `-SkipSelfTest` switch.

## Not checked

The network, Windows threads and the scheduled task. The records of 21, 22 and 25 November and the production windows.
The model code, the rebuild and the scoring code. The probe counts of amendment 5.
