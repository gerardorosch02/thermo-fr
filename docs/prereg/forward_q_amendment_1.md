# Amendment 1 to the forward test of Q: the hole in the proxies of the live history

Written on 2026-11-18, before any counted day, after check (d) of `forward_q_prereg.md` and before any forward record
exists. The pre-registration text is unchanged (sha256 761517280f2a45b3f5c8c52f2045b70ac406686d5c8893dc7bf908fb89736402).

**What was found.** `data/forecast/inputs.csv` on the laptop holds the wind and solar point forecasts up to 2026-10-10 but
no wind or solar proxy from 2026-08-20 to 2026-10-09. The saved morning windows carry the proxies from 2026-10-10. So the
live history has 51 delivery days on which the French residual load cannot be formed. L0 does not read the proxies and is
not affected, which is why the S2 shadow never showed it. Q drops those days from its training set and P keeps them with
a neutral value, so in live running the two would differ even in the two short windows, where the research has them
identical. The research itself ends on 2026-06-30 and is not touched by this.

**What changes.** Before `q_live` is called, the daily issue fills a missing proxy value, where the point forecasts
exist, with the project's own point-in-time rule (`gen_proxy.rolling_proxy`: weights refitted each month on actual
generation up to two days before the month). A value that is already there is never replaced. `q_live` itself is
unchanged. The record of each issue states how many hours were filled and for which days.

**Why this is the registered Q and not a new one.** On the 20,186 and 20,906 hours where the inputs file does hold a wind
or solar proxy, the same rule reproduces the stored value to 7e-12 MW. So the filled values are what the file would
have held had the proxies been computed to its end.

**Checked.** With the hole filled, on the inputs of 2026-11-18 for delivery day 2026-11-19, P and Q are identical in
windows 364 and 728 and differ in windows 1092 and 1456, as the research describes. No price was read.

**Still open.** The same rule needs actual wind and solar generation, which the morning windows do not carry. The file
holds it to 2026-10-10, enough for the weights of October and November. It does not matter while the saved windows carry
the proxies themselves, which they do from 2026-10-10 on. If a morning run ever fails to store a proxy, that day's
regressor is missing and the day is a "Q fallback" by the rule already written.
