# Amendment 3 to the forward test of Q: the release rule, the shadow's scores, and the public page

Written on 2026-11-18 at about 20:00 London, before any counted day (the first is 2026-12-01) and before any score of
the S2 shadow was read by this work. The pre-registration text is unchanged (sha256
761517280f2a45b3f5c8c52f2045b70ac406686d5c8893dc7bf908fb89736402). The code of the forward logger is unchanged (code hash
ed330209b0ebc4d46a1c38be0e4ebc520df60b430c6908c09a1e7f47fc7201c0).

These are three decisions of the owner, taken on 2026-11-18 after the second external review
(`Thermo_FR_Next_Steps_Advisory_Review.md`).

## 1. The release rule changes

Until today a pass on the backtest earned a forward test and nothing public changed. From today a model that has passed
its pre-registered backtest rule may be shown on the public dashboard and released as the public forecast, on these
conditions.

- It is labelled as released on backtest evidence, with the window and the number of days stated beside every figure.
- The model it replaces keeps being issued and logged each morning as a comparator, so the live record of the difference
  goes on building.
- A pass is still not proof of a live gain. The live record decides that, and it is published whatever it shows.

The backtest rule itself (the 0.25 bar, the slices, the paired interval, the stop conditions, the independent review) is
unchanged. So is the rule that the holdout from 2026-07-01 is never used to choose.

## 2. The stored scores of the S2 shadow may be read

The pre-registration says of the live replay days from 2026-10-20: "No error against a price is computed on them, by this
work or its reviewer". That sentence was written about the replayed L0, P and Q, and I had applied it to the shadow's
stored scores of S1, S2 and LEAR as well. The owner saw 17 days of those scores on 2026-11-13 and stopped the trial on
them, so the period is already exposed at the level of the project. From today:

- The shadow's stored scores of S1, S2 and LEAR (tables `shadow_scores` and `shadow_errors` of the laptop store) may be
  read, compared with honest_v2 on the same days, and published as aggregates.
- Every day before the corrected model is frozen is development and operating evidence, not a holdout. It is described
  as seen.
- **Q and P stay unscored on every live day** until the statistics date of the forward test. No error of Q or P against
  a price is computed on a replay day or a counted day, by this work or its reviewer.

## 3. The public page gains the ensemble, and what that does to the test

The pre-registration says "Nothing public changes during the test". The public dashboard will gain a page for the S1
ensemble (backtest results and the live record of the shadow). This does not touch the test, for these reasons.

- honest_v2 keeps being issued each morning by the same workflow from the same stored file. It is the "g" of the test, so
  the main comparison (S1 with Q against S1, same g) and the deployment comparison (S1 with Q against g) are unchanged.
- The forward logger reads the store and the saved windows only. It does not read the dashboard.
- The days, the minimum of 90 scored days, the statistics date and the three outcomes are unchanged.

If the public headline forecast itself later becomes the ensemble, honest_v2 is still issued and logged as the comparator
under section 1, and the same holds.
