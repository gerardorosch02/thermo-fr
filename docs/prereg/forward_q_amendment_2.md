# Amendment 2 to the forward test of Q: what the independent review asked for

Written on 2026-11-18, before any counted day and before any forward record exists. The pre-registration and amendment 1
are unchanged.

1. **Each record now carries a hash of the code that ran.** The reviewer found that a record stamped with a commit says
   nothing about edits made after that commit. Every record now also holds one sha256 over all source files of
   `thermo_fr.forecast` as imported. The value at the freeze is written into `docs/experiments.md`, and a record with
   another value belongs to another version.
2. **One more flag.** "g file differs from the published file": the model file behind the laptop's pre-market forecast
   is not the one on the public branch, which happens when the laptop's checkout is behind a monthly refit. The day is
   scored, because the main comparison uses the same g on both sides, and it is listed in the report, because on such
   a day g is not the deployed forecast and the deployment comparison is weaker.
3. **Amendment 1 is now tested end to end.** On development day 2026-06-24, with the proxies blanked on 51 training days
   and filled by the rule, P and Q are within 1.9e-9 EUR/MWh of the stored research forecasts. With the hole left open
   they are off by up to 15.8 and 21.5. Scrambling every input not known at the gate before the fill changes nothing.
4. **Known and not fixed: the German and Spanish prices.** ENTSO-E returns more than one price series for those zones
   and the parser blends them, differently on each fetch. L0, Q and g all read the German price of D-1. The forward test
   as registered runs on the production data path as it is, defect included, and it affects both sides of the main
   comparison. If the owner has the parser corrected first, that is a change of recipe for L0, Q and the public model,
   it needs its own pre-registration, and the forward test starts after it. If it is corrected during the test, the
   record ends on that day.
5. **A risk to the count of days, stated before the start.** Of the 23 expected delivery days from 2026-10-20 to
   2026-11-19, three had no usable pre-market run on the laptop. At that rate about 87 of 100 expected days would be
   scored, short of the 90 the verdict needs. The dates in the pre-registration stand unless the owner changes them
   before 2026-12-01.
