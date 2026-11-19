# Amendment 4 to the forward test of Q: the logger also writes the distance switch

Written on 2026-11-19, before any counted day (the first is 2026-12-01). The pre-registration text is unchanged (sha256
761517280f2a45b3f5c8c52f2045b70ac406686d5c8893dc7bf908fb89736402).

## What changes

`forward_q.py` gains two functions, `distance_from_centre` and `switch_forecast`, and `issue` writes four more things into
each record: `distance`, `switch_branch`, `switch_threshold` and a forecast `W`. They belong to a separate registered test,
`distance_switch_prereg.md`. A day on which the switch falls back carries the flag "switch fallback".

## What does not change

- L0, P, Q, S1 and S1-Q. On the morning run of 2026-11-18 for delivery day 2026-11-19 the new code writes the same 24
  values of g, L0, P, Q, S1 and S1-Q as the previous code (largest difference 0.0).
- The calendar, the comparisons, the stop conditions, the lock on scoring and every rule of the registration.
- Amendment 3: Q and P stay unscored on every live day.

## The code hash

The hash over `thermo_fr/forecast/*.py` changes, as it must. Before: ed330209b0ebc4d46a1c38be0e4ebc520df60b430c6908c09a1e7f47fc7201c0.
After, on branch `forward-q`: 1381db67b436d2fbf8aef3e3d540b1f3405a557a2e401e31b50d9b19cf7dd2ee. Records written before the owner's machine pulls the change carry the old hash and
no distance. All of them fall in the burn-in, before the first counted day, provided the pull happens before 2026-12-01.
If it does not, the counted days written by the old code have no W and are reported as such in the switch test; the test
of Q is unaffected.
