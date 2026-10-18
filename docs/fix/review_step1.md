# Review of step 1 (branch solar-align, commit 85b832c)

2026-10-17, reviewer. No price backtest was run, no error figures were computed, and no code was changed. My scripts are in scratchpad/step1rev/.

**Verdict: SIGN.** There are no must-fixes. Two decisions for the designer come before step 2 (at the end).

1. **The default reproduces the logged models bit for bit. Confirmed.** I ran main (5250d26) and the branch on the same inputs.csv (same sha256). These were identical:
   - the honest_v2 feature table (50,616 x 42), its target and info;
   - the LEAR design (2,110 x 302) and its price table;
   - the issue times of every honest_v2 and LEAR timing rule, including the new radiation rule against the old weather rule;
   - a one-month GBM walk-forward (May 2024, 744 hours);
   - two LEAR days (12 and 13 Dec 2024).

2. **Only radiation moves, and nothing arrives late. Confirmed.**
   - In inputs_aligned.csv the only intended changes are the radiation columns (`radiation_fc_wm2`, `radiation_proxy_wm2` and the 21 points), `solar_mw` and `solar_proxy_mw`. They match my own shift exactly.
   - `wind_proxy_mw` differs by 6e-14 from the CSV round trip only. Step 2 should take honest_v2's baseline rows from inputs.csv.
   - The morning path and the history path use the same function. The 13 branch tests pass.
   - My perturbation test ran on start-of-hour labels on 2025-06-21, 2024-12-12 and 2026-03-29. I changed:
     - prices from D;
     - forecast columns after the end of D;
     - actuals and proxies from 08:00 Paris on D-1.

     honest_v2 features, the LEAR design and the targets for every day up to D were unchanged to 0.0. D+1 moved by up to 130,000.

3. **The relabel rule is implemented as written. Confirmed.**
   - My own code flags the same 215 days.
   - The July 2023 days are real: their daily offsets are 0.8 to 1.1 h. They are harmless, because they predate the radiation archive.
   - The rule is not causal. A causal version (median from before July 2023, trailing window) changes one day in the calibration span (73 against 72 days), so this does not matter.

4. **Wind and nuclear are about one hour late too. The finding is real.** Monthly lag of hourly changes, Sep 2023 to Mar 2024 against the other months:

   | Pair | Shift | Months above the outside 90th percentile |
   |---|---|---|
   | Wind against the ENTSO-E wind forecast | +0.93 h | 7 of 7 |
   | Nuclear against the day-ahead price | +0.80 h | 7 of 7 |
   | Control: load forecast against price | -0.03 h | 0 of 7 |

   - A lag measured on levels misses the wind shift because wind output is smooth, so the result rests on hourly changes.
   - This is one fault in the whole A75 document.

5. **The proxy recalibration follows the monthly rule. Confirmed.**
   - Same 365-day window ending two days before the month. The first fit is April 2024, on 71 days.
   - My rebuild matches the aligned proxy to 4e-12. The as-logged case reproduces the stored proxy to 7e-12.
   - MAE goes from 993.9 to 634.7 MW. The midday bias of about +340 MW remains.

6. **The clear-sky test is meaningful. Confirmed.** On the real data it gives:
   - as logged, +0.889 h: fails;
   - fixed, -0.100 h: passes;
   - radiation moved one hour too far, -1.005 h: fails;
   - actuals left one hour late, -1.100 h: fails.

**Decisions for the designer before step 2**
- Relabel wind and nuclear on the same flagged days. It is one fault. `nuclear_d1_early_mw` feeds both models, including strict rows up to 2024-04-01, and wind actuals feed the wind proxy fits until Mar 2025. The expected effect is small. This must be pre-registered either way.
- The replay in step 3 must apply `align_table` to the saved windows and recompute the solar proxy with the corrected weights. It must not reuse the stored proxy column.

## Addendum: amendment 1 (wind and nuclear), commit 3b92475e76cdbf3f9ccd43508b6ead5d2e79c029

Reviewed against fix_prereg.md, sha256 6b9f13952ac7d17d60e72f20db29c446415ae10510ec56bca823cf25af3d61a3.

**Verdict: SIGN.**

**The relabel matches the amendment.** My own code reproduces wind onshore, wind offshore, nuclear and solar exactly on the 215 days.
- There is one refinement on top of the literal rule. The last hour of each flagged block, where h+1 is an unflagged hour that holds its own period, is set to missing (0 for solar).
- This affects 4 hours per series. They fall within the "hours left missing" the pre-registration says will be reported.
- One strict row of `nuclear_d2_mw` becomes newly missing. LEAR has no new missing values.

**The logged default is unchanged.** My bit-for-bit check on main against 3b92475 was rerun, and all of these are identical:
- honest_v2 features, target and info;
- the LEAR design and its prices;
- every issue time;
- the May 2024 GBM walk-forward;
- the two LEAR days.

**The new perturbation tests are meaningful.** They replace values by the true end of each period, not by its label, and they include a control showing that the boundary hour does move the result. All 19 tests pass.

**2024-04-03 is harmless.** Its change comes from `nuclear_d2` only. D-2 is 1 April Paris time, which begins inside the flagged UTC day of 31 March. The rule was applied as written; only the pre-registration's description of the day span was approximate. One clause in the results row is enough, and it is not a deviation.

**`align_window` is fit for the step 3 replay, on two conditions for the replay spec:**
- Each day uses the corrected weights its monthly refit would have published.
- Converted windows are overlaid on `inputs_aligned.csv`, never on the as-logged `inputs.csv`.

The wind proxy MAE is unchanged: 1,399.0 MW before and 1,400.0 MW after.
