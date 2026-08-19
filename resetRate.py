# -*- coding: utf-8 -*-
"""
Post-determined last reset rate -- calculation and blotter-driven check.

`last_reset_rate` computes the reset rate independently (numpy only); the pricer
imports it. Run this file to read the float book (A_PSwapsFloat.csv) and, for
each deal, pull its Historical Fixings and Curve Index curves from the tabs of
curves.xlsx (tabs share the column names) and report the reset rate.

@author: E42656
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from linearInterpolation import load_curve_set


# ----------------------------------------------------------------------------
# The calculation (numpy only; imported by simpleSwapPricing)
# ----------------------------------------------------------------------------
def _sorted(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs, kind='mergesort')
    return xs[order], ys[order]


def floor_lookup(offsets, rates, query):
    """Rate of the most recent node at or before `query` (LOCF, no interp)."""
    xs, ys = _sorted(offsets, rates)
    i = int(np.searchsorted(xs, query, side='right')) - 1
    if i < 0:
        i = 0
    return float(ys[i])


def observed_accumulation(valuation, accrual_start, fix_offsets, fix_rates):

    valuation = pd.Timestamp(valuation)
    start = pd.Timestamp(accrual_start)

    # convert nanosecond timestamps → datetime
    xs = pd.to_datetime(np.asarray(fix_offsets))
    ys = np.asarray(fix_rates, dtype=float)

    fix_series = pd.Series(ys, index=xs).sort_index()

    acc = 1.0
    d = start

    while d < valuation:

        # LOCF by actual date
        valid = fix_series.loc[:d]

        if len(valid) == 0:
            rate = fix_series.iloc[0]
        else:
            rate = valid.iloc[-1]

        acc *= 1.0 + rate / 360.0
        d += pd.Timedelta(days=1)

    return acc


def forward_accumulation(valuation, accrual_end, index_days, index_rates):
    """Index-curve growth from valuation to accrual end: 1 / DF(accrual_end)
    = (1 + z) ** (days / 365), z read off the index curve."""
    off = (pd.Timestamp(accrual_end) - pd.Timestamp(valuation)).days
    z = float(np.interp(off, np.asarray(index_days, dtype=float),
                        np.asarray(index_rates, dtype=float)))
    return (1.0 + z) ** (off / 365.0)


def year_fraction_act360(start, end):
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 360.0


def last_reset_rate(valuation, accrual_start, accrual_end,
                    fix_offsets, fix_rates, index_days, index_rates, accrual=None):
    """Post-determined last reset rate (bare). Returns (reset, observed, forward).
    Pass `accrual` (the leg year fraction) so rate * accrual reproduces the
    interest factor; defaults to ACT/360 over [accrual_start, accrual_end]."""
    observed = observed_accumulation(valuation, accrual_start, fix_offsets, fix_rates)
    forward = forward_accumulation(valuation, accrual_end, index_days, index_rates)
    if accrual is None:
        accrual = year_fraction_act360(accrual_start, accrual_end)
    if accrual <= 0:
        return 0.0, observed, forward
    return (observed * forward - 1.0) / accrual, observed, forward


# ----------------------------------------------------------------------------
# Historical-fixings workbook (separate from the zero-curve workbook)
# ----------------------------------------------------------------------------
def load_fixing_curves(path):
    """Read every tab of the historical-fixings workbook into

        {tab_name: (dates, rates)}

    Each tab is one fixings series, named after the curve: column 1 holds the
    fixing DATES (Excel dates or m/d/yyyy text), column 2 the rates (decimals).
    Rows that fail to parse are dropped; each series is returned sorted by
    date. These are OBSERVED fixings keyed on calendar dates -- unlike the
    zero-curve tabs of curves.xlsx, whose first column is day offsets -- and
    they feed observed_accumulation directly."""
    xls = pd.ExcelFile(path)
    out = {}
    for name in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=name, header=None)
        if raw.shape[1] < 2:
            continue
        dates = raw.iloc[:, 0].apply(
            lambda v: pd.to_datetime(v, dayfirst=False, errors='coerce'))
        rates = pd.to_numeric(raw.iloc[:, 1], errors='coerce')
        keep = dates.notna() & rates.notna()
        sub = pd.DataFrame({'d': dates[keep], 'r': rates[keep]}).sort_values('d')
        if len(sub):
            out[name.strip()] = (sub['d'].values, sub['r'].values.astype(float))
    return out


def resolve_fixing_curve(fixing_curves, name, aliases=None):
    """(dates, rates) for a fixings-series name, or None when `name` is blank.

    Resolution order: the exact tab name; an explicit alias (e.g. the Float
    tab says 'EUR-STR' while the workbook tab is 'EUR_STR_ON'); a normalised
    match ignoring case and -/_ separators. An unresolvable non-blank name
    raises, listing the tabs, so a misnamed series fails loudly instead of
    silently mispricing the reset."""
    s = u'{0}'.format(name).strip()
    if not s or s.lower() in ('nan', 'none'):
        return None
    if s in fixing_curves:
        return fixing_curves[s]
    if aliases:
        target = u'{0}'.format(aliases.get(s, '')).strip()
        if target and target in fixing_curves:
            return fixing_curves[target]

    def _norm(v):
        return u'{0}'.format(v).strip().upper().replace('-', '').replace('_', '')
    for tab, nodes in fixing_curves.items():
        if _norm(tab) == _norm(s):
            return nodes
    raise KeyError(
        'Historical-fixings series {0!r} not found. Workbook tabs: {1}. '
        'Map it in SWAP_HIST_FIXINGS_ALIASES (main.py).'.format(
            s, sorted(fixing_curves)))


# ----------------------------------------------------------------------------
# Blotter-driven check: curves come from curves.xlsx tabs named by the columns
# ----------------------------------------------------------------------------
XLSX           = 'curves.xlsx'          # curve workbook; each tab is one curve
BLOTTER        = 'A_PSwapsFloat.csv'    # float book (semicolon-separated)
VALUATION_DATE = '31/12/2025'

HISTFIX_COL = 'Historical Fixings'      # column naming the observed-fixings tab
INDEX_COL   = 'Curve Index'             # column naming the forward/index tab


def _term_years(raw):
    s = u'{0}'.format(raw).strip().lower()
    n = float(''.join(c for c in s if c.isdigit() or c == '.') or '0')
    return n / 12.0 if 'month' in s else n


def _maturity(raw):
    """'{2028/10/20 340000000}' -> maturity Timestamp."""
    return pd.to_datetime(u'{0}'.format(raw).strip().strip('{}').split()[0],
                          format='%Y/%m/%d')


def _first_period(effective, maturity, term_years, valuation):
    """(start, end) of the first live period, stepping forward from effective."""
    months = int(round(term_years * 12))
    prev = effective
    i = 1
    while True:
        nxt = effective + DateOffset(months=months * i)
        if nxt >= maturity or nxt > valuation:
            break
        prev = nxt
        i += 1
    return prev, prev + DateOffset(months=months)


def main():
    curves = load_curve_set(XLSX, method='linear')
    df = pd.read_csv(BLOTTER, sep=';', dtype=str, encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    val = pd.Timestamp(pd.to_datetime(VALUATION_DATE, dayfirst=True))

    print("valuation {0}   curves {1} ({2} tabs)   blotter {3} ({4} deals)".format(
        VALUATION_DATE, XLSX, len(curves.curves), BLOTTER, len(df)))
    print("=" * 92)
    for _, r in df.iterrows():
        deal = u'{0}'.format(r['DealNum']).strip()
        effective = pd.to_datetime(u'{0}'.format(r['Effective Date']).strip(), dayfirst=True)
        term = _term_years(r['Term'])
        maturity = _maturity(r['Swap Notional'])
        start, end = _first_period(effective, maturity, term, val)

        hist = u'{0}'.format(r[HISTFIX_COL]).strip()
        index = u'{0}'.format(r[INDEX_COL]).strip()
        missing = [c for c in (hist, index) if c not in curves.curves]
        if missing:
            print("{0}: curve tab(s) not in {1}: {2}".format(deal, XLSX, missing))
            print("-" * 92)
            continue

        fixings = curves.curves[hist]     # Interpolation nodes for the fixings tab
        idx = curves.curves[index]        # Interpolation nodes for the index tab
        accrual = year_fraction_act360(start, end)
        reset, observed, forward = last_reset_rate(
            val, start, end, fixings.x, fixings.y, idx.x, idx.y, accrual=accrual)

        print("{0}   fixings={1}   index={2}".format(deal, hist, index))
        print("   period {0:%Y-%m-%d} -> {1:%Y-%m-%d}   accrual {2:.6f}".format(
            pd.Timestamp(start), pd.Timestamp(end), accrual))
        print("   Observed {0:.12f}   Forward {1:.12f}".format(observed, forward))
        print("   LAST RESET RATE (UAT) = {0:.12f}".format(reset))
        print("-" * 92)


if __name__ == "__main__":
    main()