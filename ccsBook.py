# -*- coding: utf-8 -*-
"""
Cross-currency swap BOOK runner.

Reads every cross-currency swap in a KSwap blotter, prices each one, computes
its FRTB GIRR (per tenor, for every IR curve it touches) and its XCCY basis
delta, and lines all of it up against RiskWatch (the frtb_sa_report.csv):

    MtM   : MtM-UAT vs MtM-RiskWatch, per deal
    GIRR  : per-tenor delta vs RiskWatch, per curve, INCLUDING the XCCY basis

Built on the single-deal engine (ccsPricing / ccsSensitivity) and the blotter
loader (ccsPortfolio); nothing there changes. The only new ideas here:

  * currency auto-detection -- one leg is always the reporting currency (EUR),
    the other is read off the blotter, so the curve set is assembled per deal
    without naming currencies up front;
  * the IR (OIS) curve per currency is discovered from the workbook, so
    CHF-SWP -> CHF-OIS happens automatically (CHF-OIS is what CHF-SWP would be
    had it existed) -- see IR_SUFFIXES;
  * the foreign discount is decomposed into IR + XCCY basis exactly as the USD
    leg was (XXX-XCCY = XXX-CSA-EUR - XXX-IR), so the basis is its own shockable
    risk factor for every currency pair;
  * FX comes from the curves workbook's FX tab, not the trade config.

@author: E42656
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from ccsPricing import CrossCurrencySwap, build_derived_curves
from ccsPortfolio import (load_ccs_params, list_deal_nums, _read_blotter, _COL,
                          _term_to_years)
from ccsSensitivity import (ccs_girr_delta_table, ccs_basis_delta,
                            ccs_risk_factors, _tenor_to_months)

REPORTING = 'EUR'

# IR/OIS curve suffix preference. The blotter quotes <CCY>-SWP for every leg,
# but the reconciled pricing uses each currency's OIS curve. CHF has no -STR or
# -SWP tab, only CHF-OIS, so -OIS must be in the list: this is the
# "CHF-OIS == CHF-SWP had it existed" rule, applied generally.
IR_SUFFIXES = ('-STR', '-OIS', '-SARON', '-SONIA', '-SOFR', '-ESTR', '-SWP')


# ----------------------------------------------------------------------------
# Curve / FX discovery
# ----------------------------------------------------------------------------
def ir_curve_name(curves, ccy):
    '''The IR (OIS) curve tab for a currency, by suffix preference.'''
    for suf in IR_SUFFIXES:
        name = '{0}{1}'.format(ccy, suf)
        if name in curves.curves:
            return name
    raise KeyError(
        "No IR/OIS curve for {0!r}; looked for {1}. Add the tab or extend "
        "IR_SUFFIXES.".format(ccy, [ccy + s for s in IR_SUFFIXES]))


def csa_curve_name(ccy, reporting=REPORTING):
    '''The foreign-under-reporting-collateral discount curve, e.g. USD-CSA-EUR.'''
    return '{0}-CSA-{1}'.format(ccy, reporting)


def load_fx_table(path, tab, reporting=REPORTING, fx_scale=None):
    '''Read the FX tab into {ccy: rate}, rate = foreign units per 1 reporting
    (an EUR<CCY> quote: USD -> 1.17425). The reporting currency maps to 1.

    The tab is read headerless (the curve tabs are too): the currency column is
    the one whose cells look like 3-letter codes; the rate column is the first
    numeric column. A 6-letter 'pair' cell (EURUSD) is also accepted.

    fx_scale : optional {ccy: multiplier} applied after reading, for a source
    that stores a currency on a different scale (here the tab holds USD at 100x,
    so fx_scale={'USD': 0.01} brings 117.425 -> 1.17425). RiskWatch is the
    arbiter that pins these.
    '''
    fx_scale = fx_scale or {}
    df = pd.read_excel(path, sheet_name=tab, header=None)
    df.columns = list(range(df.shape[1]))

    def _looks_ccy(series):
        vals = [u'{0}'.format(v).strip().upper() for v in series.dropna()]
        good = [v for v in vals if len(v) in (3, 6) and v.isalpha()]
        return bool(vals) and len(good) >= max(1, len(vals) // 2)

    ccy_col = None
    for c in df.columns:
        if _looks_ccy(df[c]):
            ccy_col = c
            break
    rate_col = None
    for c in df.columns:
        if c == ccy_col:
            continue
        if pd.to_numeric(df[c], errors='coerce').notna().any():
            rate_col = c
            break
    if ccy_col is None or rate_col is None:
        raise ValueError(
            "Could not find currency/rate columns on FX tab {0!r}; "
            "columns were {1}.".format(tab, list(df.columns)))

    out = {str(reporting).strip().upper(): 1.0}
    for _, r in df.iterrows():
        code = u'{0}'.format(r[ccy_col]).strip().upper()
        if len(code) == 6:                       # a pair like EURUSD
            code = code.replace(reporting.upper(), '') or code[:3]
        rate = pd.to_numeric(r[rate_col], errors='coerce')
        if code and code.lower() != 'nan' and pd.notna(rate):
            out[code] = float(rate) * float(fx_scale.get(code, 1.0))
    return out


def fx_quote_for(fx_table, ccy, reporting=REPORTING):
    '''Build the params['fx'] entry for a foreign currency from the FX table.

    The table holds foreign-per-reporting (EUR<CCY>), so the quote orientation
    is "<reporting><ccy>" and the pricer divides by it -- the same convention
    the single-deal engine already uses.
    '''
    ccy = str(ccy).strip().upper()
    if ccy not in fx_table:
        raise KeyError(
            "FX rate for {0} not on the FX tab (have {1}).".format(
                ccy, sorted(fx_table)))
    return {'rate': float(fx_table[ccy]),
            'quote': '{0}{1}'.format(reporting, ccy)}


# ----------------------------------------------------------------------------
# Per-deal assembly
# ----------------------------------------------------------------------------
def deal_currencies(frame, deal_num):
    '''(reporting_ccy, foreign_ccy) for a deal -- one leg is always EUR.'''
    sub = frame[frame[_COL['deal_num']].astype(str).str.strip() == str(deal_num).strip()]
    ccys = [u'{0}'.format(c).strip().upper() for c in sub[_COL['currency']]]
    foreign = [c for c in ccys if c != REPORTING]
    if REPORTING not in ccys or not foreign:
        raise ValueError(
            "Deal {0} is not a {1}-vs-foreign pair (currencies {2}).".format(
                deal_num, REPORTING, ccys))
    return REPORTING, foreign[0]


def build_overlay(curves, foreign_ccy, fx_table, valuation_date,
                  fixings=None, previous_pay_date=None):
    '''Assemble the market overlay for one deal from discovered curves + FX.

    EUR leg discounts/forecasts on EUR-STR. The foreign leg discounts on
    [<CCY>-IR, <CCY>-XCCY] and forecasts on <CCY>-IR, with the XCCY basis
    derived as <CCY>-CSA-EUR - <CCY>-IR. Fixings (first-period reset rates) are
    looked up per currency if provided.
    '''
    fixings = fixings or {}
    ir_rep = ir_curve_name(curves, REPORTING)
    ir_for = ir_curve_name(curves, foreign_ccy)
    xccy = '{0}-XCCY'.format(foreign_ccy)

    derived = [{'name': xccy, 'base': csa_curve_name(foreign_ccy),
                'subtract': ir_for, 'grid_curve': ir_for}]

    rep_leg = {'discount_curve': ir_rep, 'forecast_curve': ir_rep}
    if REPORTING in fixings:
        rep_leg['last_reset_rate'] = float(fixings[REPORTING])

    for_leg = {'discount_curve': [ir_for, xccy], 'forecast_curve': ir_for}
    if foreign_ccy in fixings:
        for_leg['last_reset_rate'] = float(fixings[foreign_ccy])

    overlay = {
        'valuation_date': valuation_date,
        'reporting_currency': REPORTING,
        'fx': {foreign_ccy: fx_quote_for(fx_table, foreign_ccy)},
        'derived_curves': derived,
        'legs': {REPORTING: rep_leg, foreign_ccy: for_leg},
    }
    if previous_pay_date is not None:
        overlay['previous_pay_date'] = previous_pay_date
    return overlay


def build_book(blotter, curves, fx_table, valuation_date,
               fixings=None, previous_pay_dates=None, deal_nums=None):
    '''Build {deal_num: params} for every deal (or a chosen subset).

    Also materialises every derived XCCY curve into `curves` once, so pricing
    and shocking can reference them. fixings / previous_pay_dates are optional
    dicts keyed by deal_num (each a {ccy: rate} / date).
    '''
    fixings = fixings or {}
    previous_pay_dates = previous_pay_dates or {}
    frame = _read_blotter(blotter)
    if deal_nums is None:
        deal_nums = list_deal_nums(blotter)

    book = OrderedDict()
    all_derived = OrderedDict()
    missing_fix = []
    for dn in deal_nums:
        _, foreign = deal_currencies(frame, dn)
        overlay = build_overlay(curves, foreign, fx_table, valuation_date,
                                fixings=fixings.get(dn),
                                previous_pay_date=previous_pay_dates.get(dn))
        for spec in overlay['derived_curves']:
            all_derived[spec['name']] = spec
        params = load_ccs_params(blotter, dn, overlay)
        params['_pair'] = '{0}/{1}'.format(REPORTING, foreign)
        book[dn] = params
        for leg in params['legs']:
            if leg.get('hist_fixings_nodes') is not None:
                continue                      # post-determined reset available
            if not (fixings.get(dn) or {}).get(leg['currency']):
                if float(leg.get('last_reset_rate', 0.0)) == 0.0:
                    missing_fix.append('{0}:{1}'.format(dn, leg['currency']))

    build_derived_curves(curves, list(all_derived.values()))
    if missing_fix:
        print("[ccsBook] WARNING no first-period reset for {0} -- those legs "
              "use a zero reset; add the leg's fixings tab to the "
              "historical-fixings workbook (or populate Last Reset Rate) "
              "for an exact current-period MtM.".format(
                  ', '.join(missing_fix)))
    return book


# ----------------------------------------------------------------------------
# Pricing + sensitivity over the book
# ----------------------------------------------------------------------------
def price_book(curves, book):
    '''MtM (reporting ccy) per deal. Returns a DataFrame: Deal | Pair | MtM_UAT.'''
    rows = []
    for dn, params in book.items():
        mtm = CrossCurrencySwap(curves, params).price()
        rows.append({'Deal': dn, 'Pair': params.get('_pair', ''),
                     'MtM_UAT': mtm})
    return pd.DataFrame(rows, columns=['Deal', 'Pair', 'MtM_UAT'])


def girr_book(curves, book, tenors):
    '''Per-tenor GIRR for every curve, plus the XCCY basis (single parallel
    figure), for every deal. Returns a long DataFrame:

        Deal | Pair | Curve | RiskClass | Tenor | Delta_UAT
    '''
    rows = []
    for dn, params in book.items():
        pair = params.get('_pair', '')
        derived_names = [s['name'] for s in params.get('derived_curves', [])]
        girr_curves, basis_curves = ccs_risk_factors(params, derived_names)

        g = ccs_girr_delta_table(curves, params, tenors, girr_curves)
        for _, r in g.iterrows():
            rows.append({'Deal': dn, 'Pair': pair, 'Curve': r['Curve'],
                         'RiskClass': 'GIRR', 'Tenor': r['Tenor'],
                         'Delta_UAT': r['Delta']})
        for name in basis_curves:
            rows.append({'Deal': dn, 'Pair': pair, 'Curve': name,
                         'RiskClass': 'XCCY', 'Tenor': 'XCCY',
                         'Delta_UAT': ccs_basis_delta(curves, params, name)})
    return pd.DataFrame(
        rows, columns=['Deal', 'Pair', 'Curve', 'RiskClass', 'Tenor',
                       'Delta_UAT'])


# ----------------------------------------------------------------------------
# RiskWatch reconciliation
# ----------------------------------------------------------------------------
def _ccs_rw_id(deal_num, tag='CCS'):
    '''Reconcile our DealNum to the RiskWatch instrument id. RiskWatch tags
    cross-currency rows with a product prefix (e.g. "CCS <DealNum>"), mirroring
    the swap book's "IRS <DealNum>". Strip the tag and any quotes.'''
    from sensitivityComparison import _norm_id
    s = u'{0}'.format(deal_num).strip()
    return _norm_id(s)


def _rw_id_variants(deal_num, tag='CCS'):
    '''The id strings RiskWatch might carry for a deal, normalised.'''
    from sensitivityComparison import _norm_id
    dn = u'{0}'.format(deal_num).strip()
    cands = [dn, '{0} {1}'.format(tag, dn), "{0} '{1}'".format(tag, dn)]
    return set(_norm_id(c) for c in cands) | {_norm_id(dn)}


def _ccy_of(curve_name):
    '''Currency token of a curve name: 'USD-STR' -> 'USD', 'USD-XCCY' -> 'USD'.
    Used as the curve-aware join key so the EUR and USD GIRR legs never collide,
    and so our OIS naming (EUR-STR) reconciles with whatever RiskWatch labels the
    same currency's curve (EUR / EUR-OIS / EUR-STR).'''
    return u'{0}'.format(curve_name).strip().split('-')[0].upper()


def _pct_diff(uat, rw):
    if rw is None or pd.isna(rw) or float(rw) == 0.0:
        return np.nan
    return (float(uat) / float(rw) - 1.0) * 100.0


def _load_mtm_map(rw_csv, id_col='Instrument ID', mtm_col='Mark To Market'):
    '''One MtM per instrument from the FRTB SA report (first non-null seen).
    Self-contained here because sensitivityComparison only exposes _norm_id;
    ids are normalised with it so 'CCS P1180855' reconciles consistently.'''
    from sensitivityComparison import _norm_id
    raw = pd.read_csv(rw_csv)
    raw.columns = [str(c).strip() for c in raw.columns]
    if mtm_col not in raw.columns:
        raise KeyError("RiskWatch report missing MtM column {0!r}. Found: {1}"
                       .format(mtm_col, list(raw.columns)))
    out = OrderedDict()
    for _, r in raw.iterrows():
        rid = _norm_id(r[id_col])
        if rid.lower() in ('', 'nan', 'none') or rid in out:
            continue
        v = pd.to_numeric(r[mtm_col], errors='coerce')
        if pd.notna(v):
            out[rid] = float(v)
    return out


def mtm_vs_riskwatch(mtm_df, rw_csv, tag='CCS', mtm_col='Mark To Market'):
    '''Add MtM_RW and (UAT/RW-1)% to the MtM frame, joining on the RiskWatch id.'''
    from sensitivityComparison import _norm_id
    rw = _load_mtm_map(rw_csv, mtm_col=mtm_col)
    rw_norm = {_norm_id(k): v for k, v in rw.items()}

    out = mtm_df.copy()
    rw_vals = []
    for dn in out['Deal']:
        hit = np.nan
        for vid in _rw_id_variants(dn, tag):
            if vid in rw_norm:
                hit = rw_norm[vid]
                break
        rw_vals.append(hit)
    out['MtM_RW'] = rw_vals
    out['Diff_%'] = [_pct_diff(u, r) for u, r in zip(out['MtM_UAT'], out['MtM_RW'])]
    return out


def _vertex_to_months(v):
    '''Parse a RiskWatch vertex into whole months. The report stores decimal
    YEARS ('0.2500000000' -> 3, '2.0' -> 24); a labelled tenor ('6M', '0.5Y')
    is also accepted. Blank/NaN -> None (used by the basis rows).'''
    s = u'{0}'.format(v).strip()
    if s == '' or s.lower() == 'nan':
        return None
    try:
        return int(round(float(s) * 12.0))          # decimal years
    except ValueError:
        return _tenor_to_months(s)                   # labelled tenor


def load_rw_girr_long(rw_csv, id_col='Instrument ID',
                      class_col='Risk Factor Class',
                      tenor_col='Risk Factor Vertex 1',
                      value_col='Sensitivity Value (Reporting Currency)',
                      curve_col=None, type_col='GIRR Risk Factor Type',
                      girr_classes=('GIRR',),
                      basis_types=('Basis',),
                      xccy_classes=('XCCY', 'CCY Basis', 'Cross Currency Basis')):
    '''Long-format RiskWatch GIRR/XCCY rows -> list of dicts:
        {id, curve, risk_class, tenor_months, value}

    Column names default to the frtb_sa_report.csv schema. A row is the XCCY
    basis when its GIRR Risk Factor Type is in `basis_types` (the report carries
    the cross-currency basis as a GIRR row tagged 'Basis', e.g.
    'USD-CSA-Spread'); otherwise it is an interest-rate GIRR vertex. The
    standalone `xccy_classes` check is kept so a report that instead puts the
    basis in its own Risk Factor Class still works.

    `curve_col` is the curve/currency tag (here 'Risk Factor Bucket' = EUR /
    USD / CHF); matching reduces it to a currency token, so EUR-STR vs USD-STR
    never collide and our OIS naming reconciles with the report's labels.
    '''
    from sensitivityComparison import _norm_id
    raw = pd.read_csv(rw_csv)
    raw.columns = [str(c).strip() for c in raw.columns]
    girr_classes = set(s.lower() for s in girr_classes)
    xccy_classes = set(s.lower() for s in xccy_classes)
    basis_types = set(s.lower() for s in basis_types)

    out = []
    for _, r in raw.iterrows():
        rcls = u'{0}'.format(r.get(class_col, '')).strip().lower()
        if rcls not in girr_classes and rcls not in xccy_classes:
            continue
        rtype = u'{0}'.format(r.get(type_col, '')).strip().lower()
        is_basis = (rcls in xccy_classes) or (rtype in basis_types)

        val = pd.to_numeric(r.get(value_col), errors='coerce')
        if pd.isna(val):
            continue
        if is_basis:
            risk, months = 'XCCY', None
        else:
            risk = 'GIRR'
            months = _vertex_to_months(r.get(tenor_col))
            if months is None:
                continue
        out.append({
            'id': _norm_id(r.get(id_col)),
            'curve': (u'{0}'.format(r.get(curve_col)).strip()
                      if curve_col else None),
            'risk_class': risk,
            'tenor_months': months,
            'value': float(val),
        })
    return out


def girr_vs_riskwatch(girr_df, rw_csv, tag='CCS', curve_col=None, **rw_kwargs):
    '''Add Delta_RW and (UAT/RW-1)% to the GIRR frame.

    Match key is (deal -> RiskWatch id, tenor-in-months); when `curve_col` is
    given it is also (curve). XCCY rows match on (id, risk_class=XCCY).
    '''
    rw_rows = load_rw_girr_long(rw_csv, curve_col=curve_col, **rw_kwargs)

    # index RiskWatch rows for lookup. When a curve/currency column is given we
    # key on the CURRENCY token (USD, EUR), so EUR-STR vs USD-STR never collide
    # and our OIS naming reconciles with RiskWatch's curve labels.
    girr_idx = {}     # (id, months[, ccy]) -> value
    xccy_idx = {}     # (id[, ccy]) -> value
    for row in rw_rows:
        ccy = _ccy_of(row['curve']) if (curve_col and row['curve']) else None
        if row['risk_class'] == 'XCCY':
            key = (row['id'], ccy) if curve_col else (row['id'],)
            xccy_idx[key] = row['value']
        else:
            key = ((row['id'], row['tenor_months'], ccy) if curve_col
                   else (row['id'], row['tenor_months']))
            girr_idx[key] = row['value']

    if not curve_col:
        print("[ccsBook] NOTE girr_vs_riskwatch curve_col=None: GIRR rows for a "
              "deal are matched on (id, tenor) only, so a two-curve CCS will "
              "collide. Set RW_CURVE_COL to the report's curve/currency column.")

    out = girr_df.copy()
    rw_vals = []
    for _, r in out.iterrows():
        ids = _rw_id_variants(r['Deal'], tag)
        ccy = _ccy_of(r['Curve'])
        hit = np.nan
        if r['RiskClass'] == 'XCCY':
            for vid in ids:
                key = (vid, ccy) if curve_col else (vid,)
                if key in xccy_idx:
                    hit = xccy_idx[key]
                    break
        else:
            months = _tenor_to_months(r['Tenor'])
            for vid in ids:
                key = ((vid, months, ccy) if curve_col
                       else (vid, months))
                if key in girr_idx:
                    hit = girr_idx[key]
                    break
        rw_vals.append(hit)
    out['Delta_RW'] = rw_vals
    out['Diff_%'] = [_pct_diff(u, w)
                     for u, w in zip(out['Delta_UAT'], out['Delta_RW'])]
    return out


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_results(path, mtm_df, girr_df, na_rep='N/A'):
    '''Write the MtM and GIRR comparison frames to a two-tab workbook.'''
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        mtm_df.to_excel(writer, sheet_name='MtM', index=False, na_rep=na_rep)
        girr_df.to_excel(writer, sheet_name='GIRR', index=False, na_rep=na_rep)
    return path