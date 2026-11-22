# Test of W: the January rebuild code, first draft, and what it needs decided

Written on 2026-11-22 after the owner's word "yes start rebuild code now" (15:13 London). This is a design note and not an
amendment. Nothing in it is in force. The code is a draft that has not been independently reviewed and has not been run on
any day of the test window. No error against any price of the window has been computed.

Nothing here changes W, the comparators, the scoring code, the logger, the window, the three decisions or the read date.

## What exists

`src/thermo_fr/rebuild/spread_w_rebuild.py` with its tests in `tests/test_spread_w_rebuild.py`. It sits outside the
`forecast` package on purpose, so the code hash of the morning logger does not change.

It does what amendment 2, section 6, said was missing.

- It joins the refetch to the stored files. Everything to 2026-10-10 stays as stored.
- It gives each day one class, saved, rebuilt with French inputs saved, or rebuilt.
- It forecasts with the frozen functions only. No model code is copied.
- On a saved day the scored forecast is the one in the record.
- It builds the frame and the schedule in the form `spread_w_test.read()` expects, with a reason for every unscored hour.
- It runs the six stop conditions and writes `stop_checks.json`.
- It writes a report with no price in it. Days by class, days with second-source hours counted separately, days issued
  without yesterday's spread, saved against refetched French inputs, saved against rebuilt input fingerprints, saved
  against rebuilt forecasts.
- It refuses to assemble the window before 2027-01-19.

## What was tested, on design days only

The workspace of 2026-11-22 passes the logger's self-test with forecasts identical to the reference machine, so it stands
in for it.

- A replay of 4 to 10 October 2026 through the rebuild reproduces the six frozen forecasts of 2026-10-10 in
  `static/reference.json` to 4e-15.
- The files it writes are accepted by `spread_w_test.validate` and `availability` when the window constants are pointed at
  the replay days.
- A saved French window with a different load forecast changes the forecasts of its own day and of no other day.
- A saved record is scored as saved, C stays rebuilt, and realised values never come from a window.
- Nine tests pass. Three need the static files and are skipped without them.

Not tested. Stop condition 4 on the full history, because the stored forecasts were not to hand (point 2 below). Any day
after 2026-10-10, by design. Real production windows, whose file names and contents the code has only met through the
logger's own reader.

## Two things that block the read as registered

**1. A day with no scored hour makes the read impossible.** Stop condition 6 asks for 100 days, and the frozen scoring code
refuses a frame that does not hold exactly 100 days. A day drops out whole when it has no realised spread or no spread of
the previous day on any hour. As ENTSO-E stands today, that is three days.

| Delivery day | Why it has no scored hour today | With a realised German price from the second source |
|---|---|---|
| 2026-11-20 | no German coupled price, so no realised spread | scored |
| 2026-11-21 | no French price, so no realised spread | still unscored |
| 2026-11-22 | no spread of the 21st, so no spread of the previous day | still unscored |

So the scoring decision recovers one day of three. Unless ENTSO-E publishes the French price of 21 November, the test
cannot be read under the registered rule whatever is decided about the second source. This needs an amendment before the
read date and before any price of the window is read. It would have to say what the test does with a day that has no
scored hour, and the scoring code's check on the number of days would have to change with it. Every further hole between
now and 18 January adds to the count.

**2. The stored forecasts for stop condition 4 were not found.** The condition compares the rebuild with the stored
forecasts of H, G, C and W for 2024-05-01 to 2026-10-10 (hashes ab885a4f... and a0a5973c... in the registration). They are
working files, kept out of the repository. They are not in the logger's static folder or in `thermo-fr-handoff` on the
laptop. If they exist only in an earlier workspace they may be gone. They can be regenerated from the static files with
the frozen design commands, and the registered figures (17.34 and 14.84 for W, and the rest) would check the regeneration,
though a regenerated file will not carry the registered file hash. This needs deciding and doing before January.

## Choices the draft makes where the frozen documents leave room

Each of these should be confirmed or changed, then registered, before the read.

1. **What is taken from a saved French window.** The draft takes the load forecast and the nuclear output only. Those are
   the two inputs the refetch specification's table marks "saved window where one exists". Rule 2 of the same document
   reads more widely, "where an input of a day was saved, the saved value is used". The narrow reading keeps the
   production windows' blended German neighbour price (amendment 3, section 7) out of the inputs. Weather is compared with
   the window and not taken from it, as the table says.
2. **Which window counts.** The latest one for the delivery day stamped on its eve at or before 10:05 Paris, from the
   production folder or the logger's own. Production windows carry the start time of their run.
3. **A rebuilt hour whose weather is not the forecast issued two days ahead** is treated as never issued, with its own
   reason, and is not scored. On a saved day every hour of the record counts as issued.
4. **German forecast rows with no wind value** are removed whole on days after 2026-10-10, as the logger does, so the lag
   is missing and not zero.
5. **Stop condition 2 reports and does not stop.** The registration says the stored values are used where the refetch
   differs and the difference is reported. The draft marks it passed once the comparison has run.
6. **Stop condition 3 and accepted holes.** A whole-day hole stops the run unless an amendment names the input and the day.
   The names are passed in a small file. The French price of 21 November would be the first entry.
7. **C on a saved day** is rebuilt from the refetch alone.
8. **Inputs missing on a saved day** are counted from the record, which counts hours by input. An input missing on some
   hours only is counted on every hour of that day, and such days are listed.
9. **Records are not refused by code hash.** The report counts saved days by the hash of the build that wrote them. A list
   of the accepted builds could be registered and checked.
10. **Realised values.** The realised spread and the spread of the previous day come from ENTSO-E's refetch only, as
    amendment 5 says. No path reads a realised price from the second source. If the owner decides to allow it, that is new
    code under its own amendment.

## Still owed

An independent review of this code by a reviewer who has not seen it built. The refetch commands for January, written and
committed beforehand. The amendment for the two blocking points and the choices above.
