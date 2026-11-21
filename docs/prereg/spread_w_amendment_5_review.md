# Independent review of amendment 5 (the second source for the coupled German price)

Written 2026-11-21. One reviewer, who had not seen the change being built, looked twice: at commit e714215, and at the
changes made in answer (commit bd185f8). It ran its own code, did not edit the repository, printed no price and computed
no error of any forecast. Its scratch code is outside the repository.

## Verdict

Nothing serious, both times. It found no way for a price of D or later to reach the inputs of D, no departure of the code
from the four conditions, and every count in the amendment reproduced.

## What it checked and found correct

- **No leak.** 360 random trials on the merge, with values of D and later marked, on ordinary days and around both clock
  changes: no marked value reached the output, no hour ENTSO-E holds was replaced, no partly held hour was used, no hour
  before 2026-10-11 was filled. Through the real table builder and feature code, answers holding D to D+4, files saved by
  later runs and a dry-run file all leave every input up to D bit for bit identical.
- **A morning on which ENTSO-E holds everything is unchanged.** The German series, the input tables and the target are
  identical to a run with no second-source folder, in five states of that folder. The repository's own test confirms the
  frozen forecasts to 1e-9 with the second source down.
- **The agreement rule cannot trip on rounding.** Built by the logger's two real paths, the hourly means are identical on
  3,216 of 3,216 hours of the stored ENTSO-E files (1 July to 11 October 2026, 138 of them with repeated positions left
  out) and on 120 of 120 hours of the probe answers. No clock-change day is in either set.
- **Postponement.** 84 combinations of failed and missing sources run through the real code: 24 postpone, all on the
  first run, none after 10:25 Paris, none against the amendment.
- **On the probe answers of 2026-11-21 the logger would decide:** 120 hours in common, 120 agree, trusted, 48 hours to
  take, 24 each for 20 and 21 November.

Counts of the amendment, recomputed with the reviewer's own code:

| Claim | Amendment | Recomputed |
|---|---|---|
| ENTSO-E DE-LU sequence 1 lacks | 20 and 21 November | the same; sequence 2 holds all seven days |
| ENTSO-E FR, energy-charts FR and SMARD FR lack | 21 November | the same |
| energy-charts DE-LU equals sequence 1 | 480 of 480 quarter hours | 480 of 480, exact |
| energy-charts DE-LU equals sequence 2 | 0 of 672 | 0 of 672 |
| energy-charts hourly against aWATTar, 20 and 21 November | 48 of 48 to half a cent | 48 of 48 |
| Sequence 1 hourly against aWATTar | 120 of 120 to half a cent | 120 of 120 |
| SMARD equals ENTSO-E | 480 of 480 DE-LU, 576 of 576 FR | the same |
| energy-charts FR equals ENTSO-E FR | 576 of 576 | the same |
| Requests refused | 2 of 7 within four seconds | the Dutch and North Italian ones |
| A request by dates returns whole Paris delivery days | yes | yes, in winter time; not seen across a clock change |

## Findings of the first look and what was done

| Finding | Done |
|---|---|
| Fills are decided afresh each morning from every saved answer. The amendment read as one answer and a permanent fill. A fill vanishes after ten days without ENTSO-E prices or on one disagreeing hour; an hour a later answer drops is still read from an earlier one | The amendment now says so, with the three consequences. No code change |
| The flag said "because ENTSO-E held none" also on a morning when the request to ENTSO-E had failed | The flags and the postponement reason now say "the files fetched from ENTSO-E". Condition 1 says what that means on such a morning |
| A saved answer could be on disk with its hash in no record: an unreadable answer, and a file a given-up fetch finished later | The error of an unreadable answer names the file, its hash and its size. Every file read is listed in the record with its hash |
| No test ran the real request range; several cases untested | Tests added: the request range of the real morning (ten days before D, never D); the last run of a morning with an unreadable answer; a failed ENTSO-E price request with the second source filling the eve; an earlier answer still read; the 23 and 24 hour boundary; saved answers that cannot be combined |
| "43 of 48" and "113 of 120" were ties at exactly half a cent counted as misses | The reading script counts a tie as agreeing. The amendment says 48 of 48 and 120 of 120 |
| At half a cent on the hourly mean, one quarter hour two cents apart could pass | The tolerance is a tenth of a cent on the hourly mean. One quarter hour apart by a single cent fails |
| Condition 4 cannot vouch for the hours actually taken, and a line in every record called it "the same auction result" as fact | The amendment has a paragraph on what the rule can and cannot show. The line in the record is reworded |
| The January count of days with second-source hours was promised though no code reads the field | Stated as a requirement on the January rebuild code, which is not written |
| A saved answer with an absurd stamp could stop every later run | The second source is wrapped whole: whatever the saved answers hold, the record is written without them and says so. A price must be a number between -1000 and 10000 with a stamp between 2020 and 2099 |
| A failed second source postpones even when the French price is missing too | Kept, and documented with its cost |
| Wording: "alike", "asked once", the header | Corrected |

## Findings of the second look

| Finding | Done |
|---|---|
| A number hundreds of digits long in the answer made the whole answer unreadable | It is now dropped like any other value outside the range |
| A stamp half a second off the quarter-hour grid was read as on it | Only a whole second on the grid is read |
| No repository test for a whole-number price, for stamps written as floats, or for the "could not be read" flag through a whole run; one test name promised more than it tested | The first two are tested now and the test is renamed. The flag through a whole run is still untested in the repository; the reviewer ran it by hand |
| If ENTSO-E ever served an hourly series rounded to the cent for the coupled auction, most hours would fail the agreement rule and the second source would go unused | Left as it is. It fails on the safe side. No such series is in the stored files or the probe |
| Two sentences of the amendment said more than the probe shows | Corrected |

## What has not been reviewed

The changes made after the second look: two conditions in the filter of `parse_second_source`, one test extended and
renamed, three sentences of the amendment. The 58 spread tests pass with them.

## The build to install

- `src/thermo_fr/forecast/spread_live.py` sha256 0c849bd4dbe6e9f7435f534c559600cdb9527b4d50344dc44cfbd05175fdcd0f.
- Code hash of the logger (the value every record carries as `code_sha256`):
  c9621063756d3c930c0efd24ffd8777a794830d3f2823ddbe3f6e70693543131.
- `docs/prereg/spread_live_reference.json` sha256 cea23bb05acf5b489e37f3cb1516a43381156559e41ef3e8565cbc36dcfd492c. It
  differs from the reference of build 66a2c77 in the code hash only. The logger still reproduces the frozen batch forecasts
  of 2026-10-10 to 1e-9 on the reference machine.
- The commit is recorded in `docs/experiments.md`.
