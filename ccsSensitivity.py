# -*- coding: utf-8 -*-
"""
GIRR delta sensitivities for cross-currency swaps, per PHYSICAL curve -- the
RiskWatch risk factor.

Mirrors simpleSwapSensitivity: the tent shock and tenor grid come from the
shared sensitivity module (main.py configures the vertices as day counts +
labels; nothing here hardcodes them), the shock is applied per physical curve
through a thin CurveShockedSet wrapper -- the curve moves in EVERY role that
reads it (a currency's IR curve is at once a forecast curve and the IR
component of the composite discount) -- and Delta = (V_shocked - V_base) /
shock with a single one-sided bump (one-sided matches RiskWatch here; a
central difference introduces a ~0.01% convexity error).

What is CCS-specific:
  * two risk classes, matching the discount decomposition
    foreign discount = IR + XCCY:
        GIRR : per-tenor tent shocks on each IR curve the deal reads
               (e.g. EUR-STR, USD-STR);
        XCCY : a single parallel 1bp shift of the derived basis curve
               (discount only), reported as Tenor='XCCY';
  * the RiskWatch join is curve-aware: GIRR cells match on
    (DealNum, currency token, tenor) -- the report's Risk Factor Bucket is a
    currency tag, so EUR-STR vs USD-STR never collide and our OIS naming
    reconciles with the report's labels -- and the basis matches the GIRR
    rows tagged 'Basis' on (DealNum, currency token).

GIRR_Delta output is long-format:  ID | Tenor | Curve | Delta

@author: E42656
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from sensitivity import ONE_BP, _tent_shock_fn, tenors_from_days
from ccsPricing import CrossCurrencySwap

XCCY_TENOR_LABEL = 'XCCY'


class CurveShockedSet(object):
    '''
    Wrap a CurveSet and add a shock (callable of time-in-days) to ONE physical
    curve, matched by name. Every .rate() call on that curve -- discounting,
    the forward projection and the post-determined reset alike -- sees the
    shocked rates; every other curve passes through to the underlying
    (unshocked) set. This mirrors the RiskWatch risk factor: the shock is
    applied to the curve itself, once, wherever the swap reads it.
    '''
    def __init__(self, base, shock_fn, shocked_curve):
        self._base = base
        self._shock_fn = shock_fn
        self._shocked = str(shocked_curve).strip()

    def rate(self, curve_name, t_days):
        z = self._base.rate(curve_name, t_days)
        if str(curve_name).strip() == self._shocked:
            return np.asarray(z, dtype=float) + self._shock_fn(t_days)
        return z

    def __getattr__(self, name):
        return getattr(self._base, name)


def _parallel_shock_fn(shock=ONE_BP):
    '''Callable f(t_days) -> flat parallel shock at every date.'''
    return lambda td: np.full(np.shape(td), shock, dtype=float)


def ccs_risk_factors(params):
    '''Partition the curves a deal references into GIRR (IR) and XCCY basis
    risk factors, so sensitivities work for any currency pair without naming
    curves up front. Every curve a leg reads (its forecast curve and each
    component of its discount curve) is a risk factor; those materialised by
    the portfolio's derived-curve step carry the XCCY basis and get the
    parallel shock, the rest are IR curves and get per-tenor GIRR tents.

    Returns (girr_curves, basis_curves), each unique in first-appearance
    order.'''
    derived = set(s['name'] for s in params.get('derived_curves', []))
    girr, basis, seen = [], [], set()
    for leg in params['legs']:
        disc = leg['discount_curve']
        names = list(disc) if isinstance(disc, (list, tuple)) else [disc]
        # a FIXED leg has no forecast curve; it still carries the full
        # discount risk (its IR curve and the XCCY basis), just no projection
        # curve to shock.
        if leg.get('forecast_curve'):
            names.append(leg['forecast_curve'])
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            (basis if n in derived else girr).append(n)
    return girr, basis


def ccs_girr_delta_long(curves, swap_specs, tenor_table,
                        shock=ONE_BP, value_fn=None):
    '''
    Long-format delta per swap and PHYSICAL curve:

        ID | Tenor | Curve | Delta          (Delta = (V_shocked - V_base)/shock)

    For each swap: every IR curve it reads gets one tent shock per tenor
    (the curve moves in every role -- forecast, reset and the IR component of
    the composite discount at once), and every derived XCCY basis curve gets
    ONE parallel shock, emitted as Tenor='XCCY'. Shocking the roles
    separately and summing is NOT equivalent -- it drops the discount x
    forward cross-term of the bump (see simpleSwapSensitivity).
    '''
    if value_fn is None:
        value_fn = lambda npv: npv          # delta on MtM, no position scaling

    tenor_days = tenor_table['days'].values.astype(float)
    tenor_labels = list(tenor_table['tenor'])

    rows = []
    for spec in swap_specs:
        params = spec['params']
        girr_curves, basis_curves = ccs_risk_factors(params)

        v_base = value_fn(CrossCurrencySwap(curves, params).npv())

        for t, label in enumerate(tenor_labels):
            fn = _tent_shock_fn(tenor_days, t, shock)
            for cname in girr_curves:
                shocked = CurveShockedSet(curves, fn, cname)
                v = value_fn(CrossCurrencySwap(shocked, params).npv())
                rows.append(OrderedDict([
                    ('ID', spec['id']),
                    ('Tenor', label),
                    ('Curve', cname),
                    ('Delta', (v - v_base) / shock),
                ]))

        for cname in basis_curves:
            shocked = CurveShockedSet(curves, _parallel_shock_fn(shock), cname)
            v = value_fn(CrossCurrencySwap(shocked, params).npv())
            rows.append(OrderedDict([
                ('ID', spec['id']),
                ('Tenor', XCCY_TENOR_LABEL),
                ('Curve', cname),
                ('Delta', (v - v_base) / shock),
            ]))

    out = pd.DataFrame(rows, columns=['ID', 'Tenor', 'Curve', 'Delta'])
    if not out.empty:
        out = out.sort_values('ID', kind='mergesort').reset_index(drop=True)
    return out


def build_ccs_specs(port):
    '''Priced swaps in a CCSPortfolio -> delta specs (id + params). Deals
    whose curves are not loaded are skipped, matching the pricing pass.'''
    specs = []
    for deal_num, recs in port.load_pairs():
        try:
            p = port._build_params(deal_num, recs)
        except (KeyError, ValueError):
            continue
        needed = []
        for leg in p['legs']:
            disc = leg['discount_curve']
            needed.extend(list(disc) if isinstance(disc, (list, tuple))
                          else [disc])
            if leg.get('forecast_curve'):      # fixed legs have none
                needed.append(leg['forecast_curve'])
        # membership against the LIVE CurveSet (derived XCCY curves are
        # materialised by _build_params just above)
        if any(c not in port.curves.curves for c in needed):
            continue
        specs.append({'id': p['deal_num'], 'params': p})
    return specs


def ccs_girr_for_portfolio(port, tenor_days, tenor_labels, shock=ONE_BP):
    '''
    GIRR + XCCY delta table (long, curve-split) for an already-constructed
    CCSPortfolio: the pricing pass and the GIRR pass share one portfolio
    (curves, FX and workbook loaded once, derived basis curves materialised).

    tenor_days / tenor_labels come from main.py: the GIRR vertices as day
    counts from the portfolio's valuation date, paired with their labels.
    '''
    specs = build_ccs_specs(port)
    tenor_table = tenors_from_days(tenor_days, tenor_labels,
                                   port.valuation_date)
    return ccs_girr_delta_long(port.curves, specs, tenor_table, shock=shock)


# ---------------------------------------------------------------------------
# RiskWatch GIRR reconciliation (matched on DealNum + currency + tenor)
# ---------------------------------------------------------------------------
def _norm_id(s):
    '''int id 1001 and float-string '1001.0' both reconcile to '1001'.'''
    s = str(s).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s


def label_to_years(label):
    '''".25Y" / "0.25Y" / "6M" / "10Y" -> years.'''
    s = str(label).strip().upper()
    if s.endswith('M'):
        return float(s[:-1]) / 12.0
    if s.endswith('Y'):
        body = s[:-1]
        if body.startswith('.'):
            body = '0' + body
        return float(body)
    return float(s)


def pct_diff(our, rw):
    '''(our / rw - 1) * 100, or NaN if either side is missing or rw is 0.'''
    if our is None or rw is None:
        return float('nan')
    try:
        if pd.isna(our) or pd.isna(rw):
            return float('nan')
    except (TypeError, ValueError):
        pass
    if rw == 0:
        return float('nan')
    return (our / rw - 1.0) * 100.0


def _ccs_rw_id(s, tag='CCS'):
    '''
    Reconcile a swap id to its DealNum. RiskWatch tags cross-currency rows as
    "CCS <DealNum>" / "CCS 'P1180855'"; strip the tag and any quotes. Our
    delta table is already keyed on DealNum, so this is a no-op for our ids.
    '''
    s = u'{0}'.format(s).strip()
    tag_u = u'{0}'.format(tag).strip().upper()
    if s.upper().startswith(tag_u):
        s = s[len(tag_u):].strip().strip('\'"').strip()
    return _norm_id(s)


def _ccy_of(curve_name):
    '''Currency token of a curve name: 'USD-STR' -> 'USD', 'USD-XCCY' ->
    'USD'. The curve-aware join key: the report's Risk Factor Bucket is a
    currency tag, so matching on the token reconciles our OIS naming
    (EUR-STR, CHF-OIS) with whatever RiskWatch labels that currency's
    curve.'''
    return u'{0}'.format(curve_name).strip().split('-')[0].upper()


def load_rw_ccs_girr(path, id_col='Instrument ID',
                     class_col='Risk Factor Class',
                     bucket_col='Risk Factor Bucket',
                     girr_type_col='GIRR Risk Factor Type',
                     tenor_col='Risk Factor Vertex 1',
                     value_col='Sensitivity Value (Reporting Currency)',
                     type_col='Sensitivity Type', tag='CCS',
                     basis_types=('basis',), verbose=True):
    '''
    RiskWatch GIRR + XCCY deltas from the FRTB SA report, keyed by currency:

        girr  : {DealNum: {(ccy, tenor_years): value}}
        basis : {DealNum: {ccy: value}}

    Only GIRR Delta rows tagged "CCS <DealNum>" are read. A row is the XCCY
    basis when its GIRR Risk Factor Type is in `basis_types` (the report
    carries the cross-currency basis as a GIRR row tagged 'Basis', e.g.
    'USD-CSA-Spread', with a blank vertex); otherwise it is an interest-rate
    vertex. The Risk Factor Bucket (EUR / USD / CHF) is the currency key.
    '''
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [u'{0}'.format(c).strip() for c in raw.columns]
    tag_u = u'{0}'.format(tag).strip().upper()
    basis_set = set(u'{0}'.format(b).strip().lower() for b in basis_types)

    girr = OrderedDict()
    basis = OrderedDict()
    n_g = n_b = 0
    for _, r in raw.iterrows():
        if u'GIRR' not in u'{0}'.format(r.get(class_col)).upper():
            continue
        if u'delta' not in u'{0}'.format(r.get(type_col)).strip().lower():
            continue
        inst = u'{0}'.format(r.get(id_col)).strip()
        if not inst.upper().startswith(tag_u):
            continue
        deal = _ccs_rw_id(inst, tag)
        ccy = u'{0}'.format(r.get(bucket_col)).strip().upper()
        v = pd.to_numeric(r.get(value_col), errors='coerce')
        if not deal or not ccy or pd.isna(v):
            continue
        rtype = u'{0}'.format(r.get(girr_type_col)).strip().lower()
        if rtype in basis_set:
            basis.setdefault(deal, OrderedDict())[ccy] = float(v)
            n_b += 1
        else:
            y = pd.to_numeric(r.get(tenor_col), errors='coerce')
            if pd.isna(y):
                continue
            girr.setdefault(deal, OrderedDict())[
                (ccy, round(float(y), 6))] = float(v)
            n_g += 1

    if verbose:
        print("[ccsSensitivity] RW GIRR {0!r}: {1} deals, {2} (ccy,tenor) "
              "cells, {3} basis cells".format(
                  path, len(set(girr) | set(basis)), n_g, n_b))
    return girr, basis


def ccs_girr_delta_with_riskwatch(girr_long, frtb_report_csv=None, tag='CCS',
                                  sens_round=6, pct_round=4, verbose=True):
    '''
    Aggregate the per-curve deltas (identity groupby kept for safety) and
    attach the RiskWatch comparison, matched on DealNum + currency + tenor
    for the IR cells and DealNum + currency for the XCCY basis rows.

    Returns long format:
        ID | Tenor | Curve | Delta-UAT [| Delta-RiskWatch | (Delta-UAT/RW-1)%]

    The error column is (Delta-UAT / Delta-RiskWatch - 1) * 100. When a
    report is supplied, only rows that reconcile to a numeric RiskWatch delta
    are kept (matching the single-currency book's output).
    '''
    if girr_long is None or len(girr_long) == 0:
        return girr_long

    agg = (girr_long.groupby(['ID', 'Tenor', 'Curve'], sort=False)['Delta']
           .sum().reset_index())
    agg = agg.rename(columns={'Delta': 'Delta-UAT'})
    agg['Delta-UAT'] = agg['Delta-UAT'].round(sens_round)

    if frtb_report_csv:
        rw_girr, rw_basis = load_rw_ccs_girr(frtb_report_csv, tag=tag,
                                             verbose=verbose)

        def _rw(did, curve, tenor):
            deal = _ccs_rw_id(did, tag)
            ccy = _ccy_of(curve)
            if u'{0}'.format(tenor).strip().upper() == XCCY_TENOR_LABEL:
                return rw_basis.get(deal, {}).get(ccy)
            try:
                yrs = round(label_to_years(tenor), 6)
            except (ValueError, TypeError):
                return None
            return rw_girr.get(deal, {}).get((ccy, yrs))

        rw_vals = [_rw(d, c, t)
                   for d, c, t in zip(agg['ID'], agg['Curve'], agg['Tenor'])]
        agg['Delta-RiskWatch'] = [round(v, sens_round) if v is not None else None
                                  for v in rw_vals]
        agg['(Delta-UAT/RW-1)%'] = [
            round(pct_diff(u, v), pct_round)
            for u, v in zip(agg['Delta-UAT'], agg['Delta-RiskWatch'])]
        # Keep only rows that reconcile to a numeric RiskWatch delta.
        agg = agg[pd.to_numeric(agg['Delta-RiskWatch'],
                                errors='coerce').notna()]

    # Order by ID, then curve, then tenor (numeric years; the XCCY basis row
    # sorts last). mergesort keeps the sort stable.
    def _tenor_yrs(t):
        try:
            return label_to_years(t)
        except (ValueError, TypeError):
            return float('inf')

    agg['_tenor_yrs'] = agg['Tenor'].apply(_tenor_yrs)
    agg = (agg.sort_values(['ID', 'Curve', '_tenor_yrs'], kind='mergesort')
              .drop(columns='_tenor_yrs')
              .reset_index(drop=True))
    return agg