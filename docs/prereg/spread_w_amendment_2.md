# Test of W: amendment 2

Dated 2026-11-20, in the first hours of the day, London, and committed then. Amends `spread_w_prereg.md` and
`spread_w_amendment_1.md` (sha256 54802facde1a0a7dd4512253c3c24f864566c0c7fe592e9892e14db09e70b06b). Written before any
German price or forecast of a delivery day after 2026-10-10 was fetched and before any forecast of such a day was computed.
`spread_w_README.md` says which statements of the earlier documents are superseded.

**Owner's instruction, 2026-11-19 23:41 London**, sent as a draft headed "I'd send Claude this" and taken as his: keep W
frozen, the 100-day window, the three decision rules and the single read on or after 19 January; W against the matched
French-only model stays the primary comparison; keep the sunny-day result in the record as the reason for that comparator;
reconcile the project note; revise three interpretations; proceed with the morning logger and the data collection it
needs; show scheduled, issued and scored hours with the common-hour error by provenance class in January; arrange the
outstanding review against fixtures or pre-window data with the January scores sealed.

## 1. Unchanged

W (`spread_w.py`, sha256 b616ecc9...ae2e), H and G (`spread_eval.py`, 01507a3c...e5d6), HW, GK and GM (`spread_hw.py`,
a00c49dd...231c), the window 2026-10-11 to 2027-01-18, the single read on or after 2027-01-19, the three decisions and
their one-sided level of 1 - 0.05/3, the gain rule and its slices.

## 2. Kept in the record: why the matched comparator was chosen

On the design data of 1 July to 10 October 2026, W beat the matched comparator HW by 3.32 EUR/MWh overall and lost to it by
1.77 EUR/MWh on the eleven sunniest days, enough to miss the registered slice rule on those design data. That is the reason
W against HW is the primary test: a comparison against H alone would have hidden it.

"French-only" needs one qualification. H and HW read yesterday's German day-ahead price among the neighbours' price lags.
What they lack is the German block: the German wind and solar proxies and ENTSO-E's German load, wind and solar forecasts.

## 3. Wording corrected

- **Amendment 1, section 4**, said the redesign "does nothing for the French-only model" and that the gain "exists only when
  the German block is present". Both are too strong. In July to October, changing the training method improved the
  French-only error by 0.37 EUR/MWh (18.53 to 18.16) and the German-input error by 1.88 (16.72 to 14.84). In the backtest
  the figures are -0.04 (22.16 to 22.20) and 1.13 (18.47 to 17.34). The redesign helped more with German inputs in those
  samples. No test of that difference was registered or made.
- **Amendment 1, section 8**, said a failure of decision 3 "would not be surprising". The figures behind it are historical
  pass frequencies: on design data W against G passed the rule in 72% of 100-day backtest windows and in 31% of those
  starting in September to November. They describe past windows. They cannot say why a future test might fail.
- **The breakdown of W's two changes** (weekly refits about 0.5, weights 0.7 to 1.2, together 1.1 to 1.9) is descriptive.
  GK and GM are reported in January, not tested, and no decision attaches to them.

## 4. The morning logger

Code: `src/thermo_fr/forecast/spread_live.py` (sha256 b85f5e0552820364eee903df675563c63d1a46d275d00dffe3e3da889879bdac),
launcher `scripts/run_spread.py` (bed9a9b0...584f), task script `scripts/spread_task.ps1` (72fffeab...5683). It runs from a
snapshot of this branch in its own folder on the owner's laptop, so the code hash of the forward test of Q is not touched.
The owner registers its scheduled task himself.

What it does each morning, for the next delivery day D:

1. Starts at 10:00 Paris on D-1, the deadline of the French load forecast of D. A run that starts earlier writes nothing.
2. Fetches the French inputs with the production fetch (`jobs.fetch_fresh`), German prices and ENTSO-E forecasts for the
   last ten days, and the German weather points. The raw answers are kept with the time each fetch finished.
3. Drops every German price and ENTSO-E forecast of D or later, whatever the fetch returned.
4. Forecasts D with H, HW, G, GK, GM and W. C cannot be issued: its inputs are published after the gate.
5. Builds the inputs of D a second time with every value that cannot be known at the gate replaced by noise, and records
   whether they came out identical.
6. Writes one record for D, created atomically under a name that can exist once, before the auction closes at 12:00 Paris.
   If a source fails on the 10:00 run, nothing is written and the 10:30 run tries again and issues with what there is.

A record holds the forecasts by hour, the fetch times, a flag for anything fetched after 10:05, the hashes of code and
inputs, how each model was trained, which inputs were missing and left missing, and the flags. It never holds an error
against a price, and no part of the logger computes one.

Limits of what a record proves, stated now.

- **Time stamps are the laptop's own clock.** Nothing outside the laptop witnesses them, and a folder could be deleted and
  written again. The manifest chain detects a changed or removed record, not a rewritten history.
- **The scramble evidence tests positions, not vintages.** It shows that no input of D sits at a time stamp that is unknown
  at the gate. It cannot show that a value was not revised between its publication and the fetch, a few minutes at most
  on a saved day.
- **With a 10:00 start the record is written a few minutes after 10:05.** `issued_by_cutoff` will often be false. What
  matters for the claim is that every fetch finished by 10:05, which the record states, and that the record was written
  before the auction.
- **The frozen feature code reads two absent values as zero**, not as missing: a missing German wind forecast, and the
  share of decoupled hours when yesterday's German price is absent. The logger removes rows without a wind forecast whole
  on days only it holds, so the lag comes out missing, and it flags a missing spread of the previous day. Stored days are
  untouched, so the frozen forecasts still reproduce.
- **A weather point missing at issue can be filled by a later fetch**, so a later rebuild of a saved day can differ from the
  saved forecast. The saved forecast is the one scored.

**Count so far.** Spread forecasts issued before an auction: zero. French input windows saved before 10:05 by the existing
forward logger: 22 delivery days as of 2026-11-18, useful for reconstruction only. Those windows come from the production
runs at 08:00 and 09:00 Paris, before the 10:00 deadline of the load forecast, so the load forecast they hold may be an
earlier version than the one of 10:00.

**German prices are inputs.** From its first run, dry run included, the logger fetches German prices of days after
2026-10-10, because yesterday's spread is an input and a training target. No error of any comparator against them is
computed before the read date.

## 5. The January report: scheduled, issued, scored

Scoring code: `src/thermo_fr/forecast/spread_w_test.py`, now sha256
998d40bf7dcc8ea3522bdd645abe489112b42c16fdcbcc3e2d481fa8772ddf3b (amendment 1 froze 98fb9981...). The decisions, the level
and the rule are unchanged. What changed:

- `report()` takes the schedule, one row per hour of the 100 days, and shows for each provenance class the scheduled hours,
  the issued hours, the scored hours, and the error of every model on the common scored hours.
- Beside each class it states the vintage: on saved days an unavailable value stayed missing and no later revision is used;
  on rebuilt days the latest published version of ENTSO-E's forecasts is used and may have been revised after the gate.
  It counts the hours on which an input was left missing.
- It reports, from the first saved day to the end of the window, on how many days a forecast was in fact saved before the
  auction. That is how often W could have been used.
- An issued hour can be left out of scoring only for one of the three reasons of the first test (no realised spread, no
  spread of the previous day at that hour, no German proxy), and the reason is counted.
- It refuses models scored on different hours, a delivery day that is not the Paris day of its time stamp, a schedule that is
  not exactly the 100 registered days, mixed provenance within a day, and a schedule file other than the one the stop
  conditions were run on.

## 6. What is not written yet

The code of the January rebuild: reading the refetched French file, assigning each day its class, choosing for each day the
window saved on its own eve (rule 2 of the refetch specification), building the frame and the schedule, and running the stop
conditions. The logger's join (`build_data`) takes one cut-off time and cannot do this as it stands. It will be written,
tested on pre-window days and reviewed before the read date.

## 7. Review: what has been checked, by whom, and what has not

All reviews were made by reviewers who had not seen the work, on fixtures and data to 2026-10-10. No error against a price
of the test window was computed by anyone.

| Subject | Checked | How | Outcome |
|---|---|---|---|
| The diagnosis of the first test | yes, 2026-11-19 | Recomputed from data | Facts reproduce. The causal reading was not supported and is corrected in amendment 1 |
| Cutoff handling of the features (H, G, W) | yes, 2026-11-19 | Read, plus a scramble with the proxy fill applied afterwards | No leak found. Vintage cannot be tested |
| The gain rule in the scoring code | yes, twice | Own bootstrap and synthetic frames | Correct, including the level 1 - 0.05/3 |
| HW, GK, GM | yes, 2026-11-20 | Line by line against W, one weekly block refitted independently, stored forecasts on W's hours | HW is exactly H's inputs trained as W. No path into a comparator's own block |
| The logger's model path | yes, 2026-11-20 | Self-test and five replays of pre-window days against the frozen batch, own corruption test on two days | Differences of 1e-14. Forecasts do not move when everything unknown is corrupted |
| The logger's timing, write-once and failure handling | yes, 2026-11-20 | Run with fits stubbed | **Six errors found**: no earliest start, records not write-once under two processes, failures without a trace, a hung source outliving the task, absent values read as zero, unregistered code. All fixed in the code hashed above |
| The scoring report | yes, 2026-11-20 | Synthetic frames and schedules | Decisions correct. Five frames it wrongly accepted and one it wrongly refused. All fixed |
| **The fixes of 2026-11-20** | **no** | Covered by 40 unit and replay tests written by the builder | To be reviewed |
| **The live fetches against the real services** | **no** | Cannot be run without the network | The owner's dry run, then the first unattended record |
| **The Windows task script and launcher** | read only | Cannot be run by a reviewer | The owner's installation, then the first unattended record |
| **The January rebuild and its stop conditions** | **no** | Not written | Before the read date |

The reviewer's remaining risks are accepted and stated in section 4: self-asserted time stamps, the scope of the scramble
evidence, and a point filled by a later fetch.
