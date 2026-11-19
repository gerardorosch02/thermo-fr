"""Exploratory, second pass with the right information set. EXAA holds its 10:15 auction on Austrian working days only: Friday's
auction covers Saturday, Sunday and Monday, and the day before a holiday covers the holiday. A cross-auction position for
delivery day D is taken at 10:15 on A(D), the last working day before D, so a rule may use only spreads of delivery days up to
A(D) itself (its coupled auction cleared the day before). Data to 2026-06-30 only."""
import sys, numpy as np, pandas as pd, holidays

from thermo_fr.forecast.shape import block_bootstrap_mean
d = pd.read_csv(sys.argv[1])  # the per-sequence hourly extract of the raw ENTSO-E price documents (price_seq.coupled_series); d['t'] = pd.to_datetime(d.timestamp_utc, utc=True); d = d.set_index('t').sort_index()
sdac = d['de_lu__PT60M__1'].where(d['de_lu__PT60M__1'].notna(), d['de_lu__PT15M__1'])
f = pd.DataFrame({'sdac': sdac, 'exaa': d['de_lu__PT15M__2']}).dropna()
loc = f.index.tz_convert('Europe/Paris'); f['day'] = loc.normalize().tz_localize(None); f['hour'] = loc.hour
f = f[(f.day >= '2021-01-02') & (f.day <= '2026-06-30')]; f['spread'] = f.sdac - f.exaa
piv = f.pivot_table(index='day', columns='hour', values='spread').sort_index().asfreq('D')
at = holidays.Austria(years=range(2020, 2027))
work = pd.DatetimeIndex([x for x in pd.date_range('2020-12-20', '2026-07-05') if x.dayofweek < 5 and x not in at])
def auction_day(D): return work[work.searchsorted(D, side='left') - 1]            # last working day strictly before D
A = pd.Series([auction_day(D) for D in piv.index], index=piv.index)
gap = (piv.index - pd.DatetimeIndex(A.to_numpy())).days.to_numpy(); print('delivery days by days since the EXAA auction:', dict(pd.Series(gap).value_counts().sort_index()))
absd = piv.abs().mean(axis=1); print('mean absolute spread by that gap:', {int(k): round(v, 2) for k, v in absd.groupby(gap).mean().items()})
def known(table, shift_days=0):
    """table re-indexed so that row D holds the row of delivery day A(D) minus shift_days working positions."""
    src = A if shift_days == 0 else A.map(lambda a: work[work.searchsorted(a) - shift_days])
    return pd.DataFrame(table.reindex(src.to_numpy()).to_numpy(), index=table.index, columns=table.columns)
def report(name, signal, target=piv, first='2022-01-01', mask=None, quiet=False):
    sig = np.sign(signal); pnl = (sig * target)
    keep = (target.index >= first) & (mask if mask is not None else True)
    dd = pnl[keep].mean(axis=1).dropna(); lo, hi = block_bootstrap_mean(dd.to_numpy(), block=7, resamples=5000, seed=0, level=0.95)
    ok = (sig != 0) & (target != 0) & target.notna(); hit = ((np.sign(target) == sig) & ok)[keep].to_numpy().sum() / ok[keep].to_numpy().sum()
    by = dd.groupby(dd.index.year).mean()
    if not quiet: print('  %-52s %+.3f [%+.3f, %+.3f] | sign right %.1f%% | days %d | by year %s' % (name, dd.mean(), lo, hi, 100 * hit, len(dd), ' '.join('%d:%+.2f' % (y, v) for y, v in by.items())))
    return dd
print('\nRULES with the right information set (P&L per MWh before costs; position = sign of the signal)')
last = known(piv); base = pd.DataFrame(np.repeat(last.mean(axis=1).to_numpy()[:, None], 24, axis=1), index=piv.index, columns=piv.columns)
roll7 = piv.rolling(7, min_periods=4).mean(); roll28 = piv.rolling(28, min_periods=14).mean()
a = report('latest known day, same hour', last)
b = report('latest known day, base spread, all hours one side', base)
c = report('mean of the 7 days up to the latest known, same hour', known(roll7))
e = report('mean of the 28 days up to the latest known, same hour', known(roll28))
g = report('one working day older (decay check), same hour', known(piv, 1))
h = report('two working days older, same hour', known(piv, 2))
print('\nby days since the EXAA auction (rule: latest known day, base spread):')
for k in sorted(set(gap)):
    if (gap == k).sum() > 30: report('  gap %d day(s)' % k, base, mask=(gap == k))
print('\nrobustness of "latest known day, base spread":')
w = piv.clip(piv.stack().quantile(0.01), piv.stack().quantile(0.99)); report('spreads capped at their 1st and 99th percentiles', base, target=w)
dd = b; top = dd.abs().nlargest(int(len(dd) * 0.01)).index; r = dd.drop(top)
lo, hi = block_bootstrap_mean(r.to_numpy(), block=7, resamples=5000, seed=0, level=0.95); print('  without the 1%% largest days (either sign): %+.3f [%+.3f, %+.3f]; share of days positive %.1f%%; median day %+.3f' % (r.mean(), lo, hi, 100 * (dd > 0).mean(), dd.median()))
m = dd.groupby([dd.index.year, dd.index.month]).mean(); print('  months positive: %d of %d; worst month %+.2f, best %+.2f' % ((m > 0).sum(), len(m), m.min(), m.max()))
q = dd.groupby([dd.index.year, (dd.index.month - 1) // 3 + 1]).mean(); print('  by quarter:', ' '.join('%dQ%d:%+.1f' % (y, k, v) for (y, k), v in q.items()))
print('  half-years 2023 onward only (after the 2022 crisis): %+.3f, days %d' % (dd[dd.index >= '2023-01-01'].mean(), (dd.index >= '2023-01-01').sum()))
lo, hi = block_bootstrap_mean(dd[dd.index >= '2023-01-01'].to_numpy(), block=7, resamples=5000, seed=0, level=0.95); print('     interval from 2023: [%+.3f, %+.3f]' % (lo, hi))
for cost in (0.1, 0.2, 0.4): print('  net of %.1f EUR/MWh for the round trip: %+.3f' % (cost, dd.mean() - cost), end=' |')
print('\n  spread volatility for scale: std of daily base spread %.2f; rule mean / std of its daily P&L = %.3f (daily), x sqrt(252) = %.2f' % (piv.mean(axis=1).std(), dd.mean() / dd.std(), dd.mean() / dd.std() * np.sqrt(252)))

print('\nHOUR-OF-DAY BIAS, walk-forward: signal = mean spread of that hour over every day known so far (at least 365 days)')
exp = piv.expanding(min_periods=365).mean(); report('expanding mean by hour, all days', known(exp), first='2022-01-03')
for k in (1, 2, 3):
    sub = piv.where(pd.Series(gap == k, index=piv.index), axis=0); ex = sub.expanding(min_periods=60).mean().ffill()
    report('  expanding mean by hour, gap-%d days only' % k, known(ex), mask=(gap == k), first='2022-06-01')
yr = piv.groupby(piv.index.year).mean(); print('mean spread by year for selected hours:'); print(yr[[6, 7, 10, 11, 19, 20, 21]].round(1).to_string())
print('\nEXAA as a forecast of the coupled price, hourly MAE by year, days auctioned the day before only (gap 1), and the price level:')
ff = f.assign(gap=pd.Series(gap, index=piv.index).reindex(f.day).to_numpy()); n1 = ff[ff.gap == 1]
print(pd.DataFrame({'mae': n1.spread.abs().groupby(n1.day.dt.year).mean(), 'price': n1.sdac.groupby(n1.day.dt.year).mean(), 'yesterday_same_hour_mae': (piv - piv.shift(1) * 0 - 0).abs().mean(axis=1).groupby(piv.index.year).mean() * 0 + n1.assign(nv=(n1.sdac - n1.groupby('hour').sdac.shift(1)).abs()).groupby(n1.day.dt.year).nv.mean()}).round(2).to_string())
