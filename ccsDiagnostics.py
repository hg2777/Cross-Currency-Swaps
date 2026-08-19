# -*- coding: utf-8 -*-
"""
Per-swap diagnostics workbook for the cross-currency swap pricer.

Mirrors simpleSwapDiagnostics: a second workbook (alongside the results
workbook) whose two tabs let an MtM / GIRR difference against RiskWatch be
traced swap by swap and curve by curve:

    'MtM Analytics'   : for each deal, the priced table of EACH floating leg
                    (schedule dates, accruals, rates, discount factors, PVs)
                    with a header line, plus a small summary table carrying
                    the per-leg PVs (own currency and reporting), the MtM and
                    the FX rate used.
    'Scenario Curves' : every curve that was altered to produce the deltas --
                    each deal's IR curves AND the derived XCCY basis curves,
                    de-duplicated -- one block per curve: the unchanged curve
                    ('BaseRate'), a parallel 1bp column and one column per
                    GIRR tenor showing the curve after that tenor's tent
                    shock.

Targets Python 2.7 (no f-strings, .format(), object base classes).

@author: E42656
"""

import pandas as pd

from ccsPricing import CrossCurrencySwap
from ccsSensitivity import ccs_risk_factors
from sensitivity import (ONE_BP, build_sensitivity_table, get_curve_nodes,
                         tenors_from_days)

# Per-leg columns shown in the analytics tables, in order. Filtered to those
# the pricer actually produced, so it is safe if a column is absent.
LEG_COLS = ['period_start', 'period_end', 'accrual_start', 'payment_date',
            'accr_start_used', 'accr_end_used', 'accrual', 'rate',
            'disc_rate', 'discount_factor', 'cash_flow', 'pv']

DATE_FMT = 'YYYY-MM-DD'


def _dates_only(frame):
    '''Return a copy with any datetime column reduced to a plain date.'''
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    return out


def _put(xl, sheet, row, frame, header=True):
    '''Write a frame at `row` (date-only) and return the next free row index.'''
    _dates_only(frame).to_excel(xl, sheet_name=sheet, startrow=row,
                                index=False, header=header)
    return row + len(frame) + (1 if header else 0)


def ccs_leg_tables(curves, params):
    '''Price one deal and return ([(leg_params, leg df), ...], priced swap).'''
    swap = CrossCurrencySwap(curves, params)

    tables = []
    for leg in params['legs']:
        df = swap.legs[leg['name']]
        if df is None or len(df) == 0:
            tables.append((leg, pd.DataFrame()))
            continue
        cols = [c for c in LEG_COLS if c in df.columns]
        tables.append((leg, df[cols].copy()))
    return tables, swap


def altered_curves(swap_specs):
    '''
    De-duplicated list of every curve that gets shocked across the book, in
    first-seen order: each deal's IR curves then its XCCY basis curves.
    '''
    names = []
    for spec in swap_specs:
        girr, basis = ccs_risk_factors(spec['params'])
        for nm in girr + basis:
            nm = u'{0}'.format(nm).strip()
            if nm and nm not in names:
                names.append(nm)
    return names


def write_ccs_diagnostics(path, curves, swap_specs, valuation_date,
                          tenor_days, tenor_labels,
                          method='linear', shock=ONE_BP, verbose=True):
    '''
    Build the two-tab diagnostics workbook for the priced cross-currency
    book.

    swap_specs : list of {'id', 'params'} (build_ccs_specs output).
    tenor_days / tenor_labels come from main.py, so the Scenario Curves tab
    shocks exactly the same grid as the GIRR sensitivity run.
    '''
    if not swap_specs:
        if verbose:
            print('[ccsDiagnostics] no swaps to write; skipped {0}'.format(path))
        return path

    tenor_table = tenors_from_days(tenor_days, tenor_labels, valuation_date)

    with pd.ExcelWriter(path, engine='openpyxl',
                        date_format=DATE_FMT, datetime_format=DATE_FMT) as xl:
        # --- Analytics tab : every leg + summary, per deal ------------------
        r = 0
        for spec in swap_specs:
            params = spec['params']
            sid = u'{0}'.format(spec.get('id', params.get('id', '')))
            tables, swap = ccs_leg_tables(curves, params)

            header = ('Swap {0}  ({1} {2})   reporting={3}'.format(
                sid, params.get('pair', ''),
                params.get('instrument_type', ''),
                params.get('reporting_currency', '')))
            r = _put(xl, 'MtM Analytics', r,
                     pd.DataFrame([[header]]), header=False)

            summary_rows = []
            for leg, table in tables:
                disc = leg['discount_curve']
                disc_s = ' + '.join(disc) if isinstance(disc, (list, tuple)) \
                    else u'{0}'.format(disc)
                # fixed legs show their coupon; float legs their projection
                # curve and basis spread
                if leg.get('coupon_rate') is not None:
                    rate_s = 'coupon={0}'.format(leg['coupon_rate'])
                else:
                    rate_s = 'forecast={0}  spread={1}'.format(
                        leg.get('forecast_curve', ''), leg.get('spread', 0.0))
                leg_hdr = ('{0}  position={1}  notional={2:,.2f}  '
                           'discount={3}  {4}'.format(
                               leg['name'].upper(), leg.get('position', ''),
                               float(leg['notional']), disc_s, rate_s))
                r = _put(xl, 'MtM Analytics', r,
                         pd.DataFrame([[leg_hdr]]), header=False)
                if len(table):
                    r = _put(xl, 'MtM Analytics', r, table)
                r += 1
                summary_rows.append(
                    [leg['name'], leg['currency'],
                     round(swap.leg_pv(leg['name']), 2),
                     round(swap.leg_pv_reporting(leg['name']), 2),
                     leg.get('position', '')])

            summary = pd.DataFrame(
                summary_rows,
                columns=['Leg', 'Currency', 'Leg PV (ccy)',
                         'Leg PV ({0})'.format(
                             params.get('reporting_currency', '')),
                         'Position'])
            r = _put(xl, 'MtM Analytics', r, summary)

            fx = params.get('fx', {})
            fx_s = '; '.join('{0}={1}'.format(k, v.get('rate'))
                             for k, v in fx.items())
            mtm_line = pd.DataFrame(
                [['MtM ({0})'.format(params.get('reporting_currency', '')),
                  round(swap.npv(), 2), 'FX: {0}'.format(fx_s)]])
            r = _put(xl, 'MtM Analytics', r, mtm_line, header=False)
            r += 2

        # --- Sensitivity tab : one block per altered curve ------------------
        r = 0
        for name in altered_curves(swap_specs):
            nd, nr = get_curve_nodes(curves, name)
            table = build_sensitivity_table(valuation_date, nd, nr, tenor_table,
                                            method=method, shock=shock)
            r = _put(xl, 'Scenario Curves', r,
                     pd.DataFrame([['Curve: {0}'.format(name)]]), header=False)
            r = _put(xl, 'Scenario Curves', r, table)
            r += 2

    if verbose:
        print('[ccsDiagnostics] wrote {0}  (tabs: MtM Analytics, Scenario '
              'Curves) for {1} swaps, {2} curves'.format(
                  path, len(swap_specs), len(altered_curves(swap_specs))))
    return path