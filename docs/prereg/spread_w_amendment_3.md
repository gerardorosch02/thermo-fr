# Test of W: amendment 3

Dated 2026-11-20, about 01:30 London, and committed then. Amends `spread_w_amendment_2.md`. No forecast of a delivery day of
the test window has been saved, and no error against a price of the window has been computed. The dry run of 00:33 UTC
fetched German prices of days after 2026-10-10 as inputs, as amendment 2 allows; they have not been scored.

## 1. What happened

The owner ran the logger's self-test on his laptop (Windows, pandas 3.1.1, numpy 2.5.5, LightGBM 4.7.0). It rebuilds the six
forecasts of 2026-10-10 from the static files and compares them with the frozen forecasts made in the workspace (Linux,
pandas 3.0.5, numpy 2.5.3, the same LightGBM). It failed, and the task script therefore installed nothing.

| On the laptop, against the workspace | Result |
|---|---|
| Static files and code hash | identical |
| Every input column and the target, all 50,614 rows, by count of missing values, sum and sum of squares | identical to one part in a billion |
| LightGBM on a synthetic table that is bit for bit the same on both machines | identical forecasts |
| The six forecasts of 2026-10-10 | differ by 1.2 to 2.3 EUR/MWh on average, by up to 7.8 at one hour |

## 2. The cause that is established

The models answer a change in the last bits of their inputs with a different fit. Shown in the workspace on design data
(`/home/claude/work/numerics`, logged in `docs/experiments.md`): every non-integer input multiplied by 1 + e, with e normal
and of size 1e-15 to 1e-9.

| Perturbation | Forecasts of 2026-10-10: mean move | Largest move |
|---|---|---|
| 1e-15 | 1.7 to 3.6 | 4.6 to 8.4 |
| 1e-13, three draws | 1.3 to 3.9 | 2.8 to 12.3 |
| 1e-11 | 2.4 to 3.3 | 6.2 to 8.8 |
| 1e-9 | 1.4 to 5.6 | 3.2 to 11.6 |

The laptop's differences sit inside these ranges. The size of the move does not depend on the size of the perturbation,
which is the mark of a fit that changes when a value crosses a bin edge. Thread count does not matter: 1, 2 and 8 threads
give identical forecasts in the workspace.

What is established: last-bit differences in the inputs are enough to produce differences of this size, and the laptop's
tables agree with the workspace to 1e-9. What is not yet shown directly: that the laptop's tables differ in their last bits.
The self-test now reports that too, column by column.

## 3. What it means for the numbers of any test

Over the opened window, 1 July to 10 October 2026, with the inputs perturbed by 1e-13 (two draws):

| | Frozen | Draw 1 | Draw 2 |
|---|---|---|---|
| H | 18.53 | 18.46 | 18.62 |
| HW | 18.16 | 17.98 | 18.18 |
| G | 16.72 | 16.66 | 16.68 |
| W | 14.84 | 14.65 | 14.63 |
| W minus HW | -3.32 | -3.33 | -3.55 |
| Hourly forecasts moved by, on average | | 1.9 to 2.5 | 1.9 to 2.6 |

- An hourly forecast of these models is not a single number. Two correct computations differ by about 2 EUR/MWh at an hour.
- A 102-day error moves by up to about 0.2, and a difference between two models by up to about 0.25.
- The gain rule asks for 0.25. A result within about 0.3 of that threshold, or a slice within about that distance of its
  limit of 1.0, is inside the numerical noise of the models. Slices of a few days move more. This applies to the first
  test as well: its gain of 1.81 is far outside it, its two slice results of +1.81 and +1.70 are less so.
- The rule and the decisions are not changed. This is a limit on how finely any outcome can be read, stated before the
  test.

## 4. Consequences for the test of W

1. **A saved forecast is the laptop's number.** On a saved day the scored forecast is the one written that morning, as
   before. It is a forecast of the same frozen model from the same inputs, and it will differ from a rebuild in the
   workspace by about 2 EUR/MWh at an hour.
2. **Rule 6 of the refetch specification is restated.** A saved forecast and its rebuild cannot be expected to agree, so
   their difference cannot show a revised input. Each record now carries the fingerprint of its inputs. In January the
   inputs of every saved day are compared with the rebuilt inputs, column by column, and that comparison is what shows a
   revision.
3. **Stop condition 4 holds on the reference machine only.** "The stored forecasts reproduce to 1e-9" is a statement about
   the workspace in which they were made. The January rebuild runs there.
4. **Rebuilt days and saved days come from different machines.** That is one more reason the two classes are reported
   separately.

## 5. The self-test, as it is now

Code: `src/thermo_fr/forecast/spread_live.py`, sha256 853e3a879b36b6282acbc39bfbe012b13b417e4824bc66b41a732a0b729ee90e
(amendment 2 hashed b85f5e05...). The models, the comparators and the scoring code are unchanged.

A machine passes if all of these hold.

- Its tables agree with the reference machine to 1e-9 in every input column and the target, over all rows.
- LightGBM gives identical forecasts on the synthetic table.
- Its forecasts are identical to the frozen ones, or its tables are not bit for bit identical and its forecasts stay within
  8 EUR/MWh on average and 20 at any hour. Those bands are about one and a half times the largest values in section 2.
- The inputs of the self-test day are identical when everything unknown at the gate is scrambled.

A machine with bit-identical tables and different forecasts fails. So does one whose tables or library differ.

## 6. Other changes since amendment 2, from the dry run

- French items that never exist for the delivery day at the gate (its own price, nuclear output and neighbours' prices,
  and the operator's wind and solar forecast) are no longer counted as failed.
- The scramble evidence compares values and names the inputs that moved. On the laptop it had reported a difference that
  does not reproduce in the workspace from the same fetched files.
- A run that finds the French load forecast of D or yesterday's spread not yet on the sources writes nothing before 10:25
  Paris and leaves it to the next trigger. At 01:33 Paris the German coupled price of the same day was not yet on ENTSO-E.
  In the 41 production windows saved by 10:05 Paris on 22 mornings, the price of D-1 was present every time.

## 7. Known and not fixed here

The production morning windows hold a German neighbour price that is neither auction: the production code on `main` mixes
the coupled and the early-auction series quarter-hour by quarter-hour, and on about half of the windows written after 10:05
it files an early-auction price under the delivery day itself. `inputs.csv` was built the same way. It affects two of H's
inputs (yesterday's German price and its spread to France), in the design data and in the history the logger reads from
production windows. The logger's own fetch uses the coupled series alone, and the spread itself is always built from the
coupled series.

## 8. Review

Not reviewed: everything in this amendment, the fixes made after the second review, the live fetches, the Windows task. The
first unattended record will be checked end to end when it exists.
