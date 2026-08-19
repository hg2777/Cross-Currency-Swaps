# -*- coding: utf-8 -*-
"""
Cross-currency swap book from the transposed population workbook, reconciled
against the RiskWatch FRTB SA report.

Mirrors simpleSwapPortfolio.SimpleSwapPortfolio: one Portfolio class owns the
workbook read, the scope filter, the leg matching, the params translation and
the (optional) MtM comparison to the FRTB SA report, where cross-currency
swaps appear in 'Instrument ID' as "CCS <DealNum>".

The workbook holds TWO transposed tabs ('Float' and 'Fixed'): field labels
down column A, one LEG per column; the two legs of a deal share the DealNum
in the Name row ("Float Leg of P1180855" / "Fixed Leg of P1450028"). Legs are
matched across BOTH tabs on that DealNum, so a deal may be float/float,
fixed/float or fixed/fixed.

Float tab fields:

    Name, Type, THEO/Value, First Accrual Date, Discount Curve, Currency,
    Variable Notional, Term, Underlying Curve Index, Last Reset Rate,
    Spread, Day Count Basis, Maturity Date, Historical Fixings

Fixed tab fields (same layout; 'Effective Date' is the schedule anchor the
Float tab calls 'First Accrual Date', and 'Coupon Rate' replaces the
index/reset/spread fields):

    Name, Type, THEO/Value, Discount Curve, Currency, Effective Date,
    Variable Notional, Term, Coupon Rate, Day Count Basis, Maturity Date

Conventions of this format:
  * 'First Accrual Date' is the date the FIRST-EVER accrual begins -- the
    swap's effective date (the schedule anchors on it).
  * 'Spread' and 'Coupon Rate' are quoted in PERCENT (-0.2575 ->
    -25.75bp -> -0.002575; 2.069 -> 0.02069).
  * a FIXED leg accrues on payment-adjusted dates (calendar_adjustment=True),
    unlike a float leg, which accrues on the unadjusted schedule. This is the
    RiskWatch convention and is what reproduces the Fixed tab's THEO/Value.
  * Pay/receive is NOT carried: the reporting-currency (EUR) leg defaults to
    RECEIVE and the foreign leg to DEFAULT_FOREIGN_POSITION; POSITIONS
    overrides the FOREIGN side per deal.
  * 'Historical Fixings' names a tab of the historical-fixings workbook; the
    post-determined first-period reset compounds those observed fixings, then
    grows on the index curve (spread added by the pricer).

What is CCS-specific relative to the single-currency portfolio:
  * currency auto-detection -- one leg is always the reporting currency
    (EUR), the other is read off the workbook, so the curve set is assembled
    per deal without naming currencies up front;
  * the IR (OIS) curve per currency is discovered from the loaded curve tabs
    by suffix preference (CHF-SWP -> CHF-OIS automatically, see IR_SUFFIXES);
  * the foreign discount is decomposed into IR + XCCY basis
    (XXX-XCCY = XXX-CSA-EUR - XXX-IR), materialised once into the CurveSet,
    so the basis is its own shockable risk factor for every currency pair;
  * FX comes from the curves workbook's FX tab, not the trade config.

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

import re
from collections import OrderedDict

import pandas as pd
from pandas.tseries.offsets import DateOffset

from ccsPricing import (CrossCurrencySwap, build_derived_curves,
                        parse_business_day_rule)
import resetRate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

INPUT_XLSX = 'Post_Determined_Input1.xlsx'
SHEET = 'Float'                      # the transposed FLOAT leg tab
FIXED_SHEET = 'Fixed'                # the transposed FIXED leg tab (or None)
VALUATION_DATE = '2025-12-31'

# RiskWatch FRTB SA report (set to None to skip the MtM comparison).
RW_MTM_CSV = None
RW_INSTRUMENT_COL = 'Instrument ID'
RW_MTM_COL = 'Mark To Market'
RW_TAG = 'CCS'                       # RiskWatch product prefix on the deal id

# Reporting currency: its leg defaults to RECEIVE; the foreign leg takes
# DEFAULT_FOREIGN_POSITION ('pay'/'receive'); POSITIONS overrides the FOREIGN
# side per deal ({DealNum -> 'pay'/'receive'}), set in main.py.
REPORTING_CCY = 'EUR'
DEFAULT_FOREIGN_POSITION = 'pay'
POSITIONS = {}

# The blotter's Last Reset Rate is already a DECIMAL; no /100.
RESET_IS_PERCENT = False

# Map a workbook curve name to the curve-tab / RiskWatch risk-factor name.
# The CCS discount and index names ('EUR-STR', 'USD-CSA-EUR') already match
# the curve tabs, so no automatic renaming is applied; unmapped names pass
# through unchanged. Set in main.py.
CURVE_ALIASES = {}

# Optional reset-rate override file (set to None to skip). Two columns:
#   deal id | last reset rate
# When supplied, it supersedes the Float tab's Last Reset Rate for that deal.
# The first accrual date is an input on the Float tab and is never overridden.
RESET_CSV = None

# Historical-fixings workbook: one tab per fixings series (column 1 dates,
# column 2 rates). Feeds the post-determined reset's observed accumulation.
# None -> legs fall back to the flat Last Reset Rate.
HIST_FIXINGS_XLSX = None
# Float-tab series name -> workbook tab name (e.g. 'EUR-STR' -> 'EUR_STR_ON').
HIST_FIXINGS_ALIASES = {}

# The FX tab of the curves workbook: foreign units per 1 reporting (EUR<CCY>).
FX_TAB = 'FX rates'

# IR/OIS curve suffix preference. The workbook quotes an index per leg, but
# the reconciled pricing uses each currency's OIS curve; CHF has only a
# CHF-OIS tab, so -OIS must be in the list ("CHF-OIS is what CHF-SWP would be
# had it existed", applied generally).
IR_SUFFIXES = ('-STR', '-OIS', '-SARON', '-SONIA', '-SOFR', '-ESTR', '-SWP')

# Strings that mean "no value" (blank cells, Excel error tokens).
_NA_TOKENS = ('', 'nan', 'none', 'na', 'n/a', 'nat', '#value!', '#n/a',
              '#ref!', '#div/0!', '#name?', '#num!', '#null!')


def _u(v):
    '''Coerce a header or cell to unicode without tripping Python 2.7's
    implicit ASCII decode. utf-8 first, latin-1 as a never-fail fallback.'''
    if isinstance(v, bytes):
        try:
            return v.decode('utf-8')
        except UnicodeDecodeError:
            return v.decode('latin-1')
    return u'{0}'.format(v)


def _norm_header(col):
    return re.sub(r'[^a-z0-9]', '', _u(col).lower())


def _norm_id(v):
    s = u'{0}'.format(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def _blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) \
        or _u(v).strip().lower() in _NA_TOKENS


def _resolve_col(df, wanted):
    '''Find a column by normalised name (tolerant of case / trailing spaces).'''
    target = _norm_header(wanted)
    for c in df.columns:
        if _norm_header(c) == target:
            return c
    return None


# Field parsers
def _clean_num(x):
    '''Parse a numeric cell; blanks / Excel error tokens -> None.'''
    s = u'{0}'.format(x).replace('%', '').replace(',', '.').strip()
    if s.lower() in _NA_TOKENS:
        return None
    return float(s)


def to_reset_rate(x, rate_is_percent=RESET_IS_PERCENT):
    v = _clean_num(x)
    if v is None:
        return float('nan')
    return v / 100.0 if rate_is_percent else v


def pct_spread_to_decimal(value):
    '''Spread in PERCENT -> decimal. -0.2575 -> -0.002575 ; blank -> 0.0.'''
    if _blank(value):
        return 0.0
    return float(_u(value).replace('%', '').strip()) / 100.0


def to_coupon_rate(value):
    '''Fixed-leg coupon in PERCENT -> decimal. 2.069 -> 0.02069.

    A fixed leg with no coupon is a data error, not a zero-coupon leg, so a
    blank raises rather than silently pricing the leg at 0%.'''
    if _blank(value):
        raise ValueError('Fixed leg has no Coupon Rate.')
    return float(_u(value).replace('%', '').replace(',', '').strip()) / 100.0


def to_notional(value):
    '''Plain notional cell (commas as thousands separators tolerated).'''
    return float(_u(value).replace(',', '').strip())


def term_to_years(term):
    '''"3 Months"/"3-Months"/"6M" -> 0.25 / 0.25 / 0.5 (years).'''
    digits = re.findall(r'\d+', u'{0}'.format(term))
    if not digits:
        raise ValueError('Cannot read a term from {0!r}'.format(term))
    return int(digits[0]) / 12.0


# ---------------------------------------------------------------------------
# Transposed population workbook reader (field per row, one LEG per column)
# ---------------------------------------------------------------------------
_SHEET_CANON = {
    'name':                 'name',
    'type':                 'leg_type',
    'theovalue':            'theo_value',
    'firstaccrualdate':     'first_accrual_date',
    'effectivedate':        'effective_date',
    'discountcurve':        'discount_curve',
    'currency':             'currency',
    'variablenotional':     'notional_field',
    'term':                 'term',
    'underlyingcurveindex': 'curve_index',
    'lastresetrate':        'last_reset_rate',
    'couponrate':           'coupon_rate',
    'spread':               'spread',
    'daycountbasis':        'day_count_basis',
    'maturitydate':         'maturity_date',
    'calendaradjustment':   'calendar_adjustment',
    'businessdayrule':      'business_day_rule',
    'historicalfixings':    'hist_fixings',
    # pay/receive, if the workbook is ever extended to carry it
    'payreceiveindicator':  'pay_receive',
    'payreceive':           'pay_receive',
    'position':             'pay_receive',
    'direction':            'pay_receive',
}


def _deal_num_from_name(name):
    '''"Float Leg of P1180855" / "P1180855" -> "P1180855".'''
    s = _u(name).strip()
    idx = s.lower().rfind('leg of ')
    if idx != -1:
        s = s[idx + len('leg of '):].strip()
    return _norm_id(s)


def _read_leg_sheet(path, sheet, leg_kind='float'):
    '''
    Read a transposed leg tab into a list of per-LEG dict records keyed by
    canonical field names. Column A holds the field labels; each further
    column is one leg. The top index row and any column without a Name are
    ignored. Every record is tagged with `leg_kind` ('float' / 'fixed') so
    the params builder knows which tab it came from.
    '''
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if raw.shape[1] < 2:
        return []

    keys = []
    for lab in raw.iloc[:, 0]:
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            keys.append(None)
        else:
            keys.append(_SHEET_CANON.get(_norm_header(lab)))

    records = []
    for j in range(1, raw.shape[1]):
        rec = {}
        for i, key in enumerate(keys):
            if key is not None:
                rec[key] = raw.iat[i, j]
        name = rec.get('name')
        if name is None or _u(name).strip().lower() in _NA_TOKENS:
            continue                       # empty trailing column
        rec['deal_num'] = _deal_num_from_name(name)
        rec['leg_kind'] = leg_kind
        records.append(rec)
    return records


def _effective_of(rec):
    '''The schedule anchor for a leg record. The Float tab calls it 'First
    Accrual Date', the Fixed tab 'Effective Date'; both name the date the
    first-ever accrual begins.'''
    for key in ('first_accrual_date', 'effective_date'):
        if not _blank(rec.get(key)):
            return pd.to_datetime(rec.get(key))
    raise KeyError(
        'Leg {0!r} has neither a First Accrual Date nor an Effective '
        'Date.'.format(_u(rec.get('name', '')).strip()))


def _leg_name_for(params, ccy):
    '''The built leg name for a currency -- float and fixed legs are named
    differently, so the reporting layer looks them up rather than assuming.'''
    for leg in params['legs']:
        if leg['currency'] == ccy:
            return leg['name']
    raise KeyError('No {0} leg on deal {1}.'.format(ccy, params['deal_num']))


# ---------------------------------------------------------------------------
# RiskWatch MtM + reset overrides
# ---------------------------------------------------------------------------
def load_rw_ccs_mtm(path, instrument_col=RW_INSTRUMENT_COL,
                    mtm_col=RW_MTM_COL, tag=RW_TAG):
    '''
    {DealNum -> Mark to Market} from the FRTB SA report.

    Cross-currency rows carry Instrument ID like "CCS P1180855" /
    "CCS 'P1180855'"; the leading tag (and any quotes) are stripped to
    recover the DealNum. Non-CCS rows (bonds, IRS etc.) are ignored.
    '''
    df = pd.read_csv(path, dtype=str)
    inst_c = _resolve_col(df, instrument_col)
    mtm_c = _resolve_col(df, mtm_col)
    if inst_c is None or mtm_c is None:
        raise KeyError(
            'FRTB report missing columns {0!r}/{1!r}. Found: {2}'.format(
                instrument_col, mtm_col, list(df.columns)))

    tag_u = u'{0}'.format(tag).strip().upper()
    out = {}
    for _, r in df.iterrows():
        inst = u'{0}'.format(r[inst_c]).strip()
        if not inst.upper().startswith(tag_u):
            continue
        deal = _norm_id(inst[len(tag_u):].strip().strip('\'"').strip())
        mtm = _clean_num(r[mtm_c])
        if deal and mtm is not None and deal not in out:
            out[deal] = mtm
    return out


_RESET_ID_ALIASES = ('dealnum', 'dealnumber', 'dealid', 'swapid', 'swapnum',
                     'id', 'instrumentid')
_RESET_RATE_ALIASES = ('lastresetrate', 'resetrate')


def _resolve_any(df, aliases):
    '''First column whose normalised header matches one of `aliases`.'''
    for c in df.columns:
        if _norm_header(c) in aliases:
            return c
    return None


def load_reset_rates(path, rate_is_percent=RESET_IS_PERCENT):
    '''
    Read the reset-rate override file into {DealNum -> last_reset_rate}.

    Two columns are expected (case / punctuation tolerant): a deal id and a
    last reset rate. Rate units follow rate_is_percent. Rows without a usable
    deal id or rate are skipped. The first accrual date is an input on the
    Float tab of the population workbook and cannot be overridden here.
    '''
    df = pd.read_csv(path, dtype=str)
    id_c = _resolve_any(df, _RESET_ID_ALIASES)
    rate_c = _resolve_any(df, _RESET_RATE_ALIASES)
    if id_c is None or rate_c is None:
        raise KeyError(
            'Reset file {0!r} needs a deal-id and a reset-rate column. '
            'Found: {1}'.format(path, [_u(c) for c in df.columns]))

    out = {}
    for _, r in df.iterrows():
        deal = _norm_id(r[id_c])
        if deal.lower() in _NA_TOKENS:
            continue
        v = to_reset_rate(r[rate_c], rate_is_percent)
        if not pd.isna(v):
            out[deal] = float(v)
    return out


# ---------------------------------------------------------------------------
# Curve / FX discovery
# ---------------------------------------------------------------------------
def ir_curve_name(curves, ccy):
    '''The IR (OIS) curve tab for a currency, by suffix preference.'''
    for suf in IR_SUFFIXES:
        name = '{0}{1}'.format(ccy, suf)
        if name in curves.curves:
            return name
    raise KeyError(
        "No IR/OIS curve for {0!r}; looked for {1}. Add the tab or extend "
        "IR_SUFFIXES.".format(ccy, [ccy + s for s in IR_SUFFIXES]))


def csa_curve_name(ccy, reporting=REPORTING_CCY):
    '''The foreign-under-reporting-collateral discount curve, e.g. USD-CSA-EUR.'''
    return '{0}-CSA-{1}'.format(ccy, reporting)


def xccy_curve_name(ccy):
    '''The derived cross-currency basis curve, e.g. USD-XCCY.'''
    return '{0}-XCCY'.format(ccy)


def load_fx_table(path, tab=FX_TAB, reporting=REPORTING_CCY, fx_scale=None):
    '''Read the FX tab into {ccy: rate}, rate = foreign units per 1 reporting
    (an EUR<CCY> quote: USD -> 1.17425). The reporting currency maps to 1.

    The tab is read headerless (the curve tabs are too): the currency column
    is the one whose cells look like 3-letter codes; the rate column is the
    first numeric column. A 6-letter 'pair' cell (EURUSD) is also accepted.

    fx_scale : optional {ccy: multiplier} applied after reading, for a source
    that stores a currency on a different scale. RiskWatch is the arbiter
    that pins these.
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


def fx_quote_for(fx_table, ccy, reporting=REPORTING_CCY):
    '''Build the params['fx'] entry for a foreign currency from the FX table.

    The table holds foreign-per-reporting (EUR<CCY>), so the quote
    orientation is "<reporting><ccy>" and the pricer divides by it.
    '''
    ccy = str(ccy).strip().upper()
    if ccy not in fx_table:
        raise KeyError(
            "FX rate for {0} not on the FX tab (have {1}).".format(
                ccy, sorted(fx_table)))
    return {'rate': float(fx_table[ccy]),
            'quote': '{0}{1}'.format(reporting, ccy)}


# Portfolio: load -> match the two legs per DealNum -> derive curves -> price
class CCSPortfolio(object):

    def __init__(self, curves, input_xlsx=INPUT_XLSX, sheet=SHEET,
                 valuation_date=VALUATION_DATE, fixed_sheet=FIXED_SHEET,
                 fx_xlsx=None, fx_tab=FX_TAB,
                 reset_is_percent=RESET_IS_PERCENT,
                 rw_mtm_csv=RW_MTM_CSV, rw_instrument_col=RW_INSTRUMENT_COL,
                 rw_mtm_col=RW_MTM_COL, rw_tag=RW_TAG, only_ids=None,
                 reset_csv=RESET_CSV, positions=POSITIONS,
                 default_foreign_position=DEFAULT_FOREIGN_POSITION,
                 curve_aliases=CURVE_ALIASES,
                 hist_fixings_xlsx=HIST_FIXINGS_XLSX,
                 hist_fixings_aliases=HIST_FIXINGS_ALIASES,
                 previous_pay_dates=None, fx_scale=None):
        self.curves = curves
        self.input_xlsx = input_xlsx
        self.sheet = sheet
        self.fixed_sheet = fixed_sheet
        self.valuation_date = valuation_date
        self.reporting = REPORTING_CCY
        self.reset_is_percent = reset_is_percent

        # FX from the curves workbook's FX tab; fx_xlsx defaults to the
        # curve workbook path when the caller loads both from one file.
        self.fx_table = load_fx_table(fx_xlsx or input_xlsx, fx_tab,
                                      self.reporting, fx_scale)

        # foreign side per deal: explicit override wins, else the default;
        # the reporting leg always takes the opposite side.
        self.positions = {_norm_id(k): u'{0}'.format(v).strip().lower()
                          for k, v in (positions or {}).items()}
        self.default_foreign_position = default_foreign_position
        self._defaulted_position = []

        # workbook curve name -> curve-tab / RiskWatch name (explicit only)
        self.curve_aliases = dict(curve_aliases) if curve_aliases else {}

        # optional reset-rate override file, loaded once
        self.reset_csv = reset_csv
        self.reset_lookup = (load_reset_rates(reset_csv, reset_is_percent)
                             if reset_csv else {})

        # historical-fixings workbook, loaded once: {tab: (dates, rates)}
        self.hist_fixings_aliases = dict(hist_fixings_aliases or {})
        self.hist_fixings = (resetRate.load_fixing_curves(hist_fixings_xlsx)
                             if hist_fixings_xlsx else {})

        self.rw_mtm_csv = rw_mtm_csv
        self.rw_instrument_col = rw_instrument_col
        self.rw_mtm_col = rw_mtm_col
        self.rw_tag = rw_tag

        # optional per-deal previous-pay dates; omitted -> computed from the
        # schedule (last reset on/before valuation).
        self.previous_pay_dates = dict(previous_pay_dates or {})

        # optional troubleshooting filter: restrict the book to these DealNums
        self.only_ids = (set(_norm_id(i) for i in only_ids)
                         if only_ids else None)

        self._curve_notes = set()             # (deal, cell, expected) warned
        self.derived_curves = OrderedDict()   # xccy name -> spec (materialised)
        self.swaps = OrderedDict()            # deal_num -> CrossCurrencySwap
        self.skipped = []                     # (deal_num, reason)
        self.rw_mtm = {}                      # deal_num -> RiskWatch MtM
        self.results = None                   # DataFrame, set by price()

    # -- helpers -------------------------------------------------------------
    def _alias_curve(self, name):
        '''Resolve a workbook curve name to a curve-tab name. An explicit
        curve_aliases entry wins; everything else passes through.'''
        key = _u(name).strip()
        return self.curve_aliases.get(key, key)

    def _resolve_foreign_position(self, recs, deal_num):
        '''The FOREIGN leg side, in priority order: the per-deal override
        dict from main.py, then an explicit workbook field on either leg,
        then self.default_foreign_position (recorded for one warning).'''
        ov = self.positions.get(deal_num, '')
        if ov.startswith('p'):
            return 'pay'
        if ov.startswith('r'):
            return 'receive'
        for rec in recs:
            if _u(rec.get('currency', '')).strip().upper() != self.reporting:
                s = _u(rec.get('pay_receive')).strip().upper()
                if s.startswith('P'):
                    return 'pay'
                if s.startswith('R'):
                    return 'receive'
        self._defaulted_position.append(deal_num)
        return self.default_foreign_position

    def _fixings_nodes(self, name):
        '''(dates, rates) for a leg's fixings series, or None. A named series
        with no matching tab is reported by the caller and falls back to the
        flat Last Reset Rate for that leg.'''
        if _blank(name) or not self.hist_fixings:
            return None
        return resetRate.resolve_fixing_curve(self.hist_fixings, name,
                                              self.hist_fixings_aliases)

    @staticmethod
    def _prev_reset(effective, term_years, valuation):
        '''Last schedule boundary on/before the valuation date, stepped the
        same effective + k*term way the engine builds its schedule -- the
        first live period's start, so the first live accrual spans exactly
        one period. Forward-starting deals return the effective date.'''
        eff = pd.Timestamp(effective)
        val = pd.Timestamp(valuation)
        months = int(round(float(term_years) * 12))
        prev = eff
        i = 1
        while True:
            nxt = eff + DateOffset(months=months * i)
            if nxt > val:
                break
            prev = nxt
            i += 1
        return prev

    # -- inputs --------------------------------------------------------------
    def load_pairs(self):
        '''[(deal_num, [reporting_leg_rec, foreign_leg_rec]), ...] for deals
        whose two legs pair up as reporting-vs-foreign.

        Legs are read from BOTH the float and the fixed tab and matched on
        DealNum, so float/float, fixed/float and fixed/fixed deals all pair
        up the same way.

        Each DealNum is loaded ONCE: duplicate columns beyond the two legs
        are reported as skipped (a duplicated deal would double the GIRR
        aggregation, a flat +100% against RiskWatch).'''
        records = _read_leg_sheet(self.input_xlsx, self.sheet, 'float')
        if self.fixed_sheet:
            try:
                tabs = pd.ExcelFile(self.input_xlsx).sheet_names
            except Exception:
                tabs = []
            if self.fixed_sheet in tabs:
                records = records + _read_leg_sheet(
                    self.input_xlsx, self.fixed_sheet, 'fixed')
            else:
                print('NOTE: fixed-leg tab {0!r} not in {1}; float legs '
                      'only.'.format(self.fixed_sheet, self.input_xlsx))

        by_num = OrderedDict()
        for rec in records:
            by_num.setdefault(rec['deal_num'], []).append(rec)

        pairs = []
        self.skipped = []
        for num, recs in by_num.items():
            if self.only_ids is not None and num not in self.only_ids:
                continue                      # outside the troubleshooting scope
            if len(recs) > 2:
                self.skipped.append(
                    (num, 'duplicate leg column(s) in workbook '
                          '(first two used)'))
                recs = recs[:2]
            if len(recs) < 2:
                self.skipped.append((num, 'only one leg in workbook'))
                continue
            ccys = [_u(r.get('currency', '')).strip().upper() for r in recs]
            if self.reporting not in ccys or len(set(ccys)) != 2:
                self.skipped.append(
                    (num, 'not a {0}-vs-foreign pair (currencies {1})'.format(
                        self.reporting, ccys)))
                continue
            # reporting leg first (stable ordering for reporting/specs)
            recs = sorted(recs, key=lambda r: 0 if _u(r.get('currency', ''))
                          .strip().upper() == self.reporting else 1)
            pairs.append((num, recs))
        return pairs

    # -- curve derivation ----------------------------------------------------
    def ensure_derived_curve(self, foreign_ccy):
        '''Materialise the foreign currency's XCCY basis curve
        (XXX-XCCY = XXX-CSA-EUR - XXX-IR on the IR grid) into the CurveSet,
        once per currency, and return its spec.'''
        xccy = xccy_curve_name(foreign_ccy)
        if xccy not in self.derived_curves:
            ir_for = ir_curve_name(self.curves, foreign_ccy)
            spec = {'name': xccy, 'base': csa_curve_name(foreign_ccy,
                                                         self.reporting),
                    'subtract': ir_for, 'grid_curve': ir_for}
            build_derived_curves(self.curves, [spec])
            self.derived_curves[xccy] = spec
        return self.derived_curves[xccy]

    def _check_discount_curve(self, deal_num, rec, ccy, is_rep,
                              ir_rep, ir_for):
        '''Warn when the workbook's 'Discount Curve' cell is not the curve
        the currency-driven derivation reproduces.

        The engine discounts the reporting leg on its IR curve and the foreign
        leg on [IR, XCCY], whose sum is <CCY>-CSA-<REPORTING>. That is
        deliberate -- it is what makes the basis a separate risk factor -- but
        it means the workbook cell is not read. Anything else in that cell is
        reported ONCE per deal so a genuinely different discount basis is
        never silently replaced.'''
        cell = _u(rec.get('discount_curve', '')).strip()
        if _blank(cell):
            return
        expected = ir_rep if is_rep else csa_curve_name(ccy, self.reporting)
        if self._alias_curve(cell) == expected:
            return
        key = (deal_num, cell, expected)
        if key in self._curve_notes:
            return
        self._curve_notes.add(key)
        print("NOTE: {0} {1} leg names discount curve {2!r}; priced on {3} "
              "(= {4}). Map it in CCS_CURVE_ALIASES if that is wrong.".format(
                  deal_num, ccy, cell, expected,
                  ir_rep if is_rep else '[{0}, {1}]'.format(
                      ir_for, xccy_curve_name(ccy))))

    # -- params --------------------------------------------------------------
    def _build_params(self, deal_num, recs):
        '''Translate a matched leg pair into a CrossCurrencySwap dict.

        The reporting (EUR) leg discounts and forecasts on the reporting IR
        curve. The foreign leg forecasts on its IR curve and discounts on
        [IR, XCCY] -- the CSA discount decomposed into independently
        shockable risk factors. A FIXED leg is built the same way minus the
        projection: no forecast curve, no reset, its blotter coupon instead,
        and accrual on payment-adjusted dates.

        NOTE: the discount curve is DERIVED from the leg's currency, not read
        from the workbook's 'Discount Curve' cell -- that decomposition is
        what makes the XCCY basis a separate risk factor. The cell is used
        only to warn when it names a curve the derivation does not
        reproduce.'''
        valuation = pd.Timestamp(self.valuation_date)
        foreign_ccy = [
            _u(r.get('currency', '')).strip().upper() for r in recs
            if _u(r.get('currency', '')).strip().upper() != self.reporting][0]

        ir_rep = ir_curve_name(self.curves, self.reporting)
        ir_for = ir_curve_name(self.curves, foreign_ccy)
        xccy_spec = self.ensure_derived_curve(foreign_ccy)

        foreign_side = self._resolve_foreign_position(recs, deal_num)
        reporting_side = 'receive' if foreign_side == 'pay' else 'pay'

        # all legs share one effective (the tab's First Accrual Date) and
        # maturity; take them across the legs
        effective = min(_effective_of(r) for r in recs)
        maturity = max(pd.to_datetime(r['maturity_date']) for r in recs)

        # An explicit previous-pay override is deal-level and authoritative;
        # otherwise each leg gets its own, stepped on ITS OWN frequency (a 6M
        # fixed leg and a 3M float leg do not share boundaries).
        prev_pay = self.previous_pay_dates.get(deal_num)
        explicit_prev_pay = prev_pay is not None

        legs = []
        for rec in recs:
            ccy = _u(rec.get('currency', '')).strip().upper()
            is_rep = ccy == self.reporting
            fixed = _u(rec.get('leg_kind', 'float')).strip().lower() == 'fixed'
            term_years = term_to_years(rec.get('term'))

            self._check_discount_curve(deal_num, rec, ccy, is_rep,
                                       ir_rep, ir_for)

            leg = {
                'currency':        ccy,
                'leg_kind':        'fixed' if fixed else 'float',
                'position':        reporting_side if is_rep else foreign_side,
                'notional':        to_notional(rec.get('notional_field')),
                'maturity_date':   pd.to_datetime(rec.get('maturity_date')),
                'term_years':      term_years,
                'basis':           _u(rec.get('day_count_basis', '')).strip(),
                # per-leg Business Day Rule (Modified/Regular Following, x-day
                # offset, own Cal). None when the cell is blank/absent -> the
                # pricer falls back to the per-leg-currency Following roll.
                'business_day_rule': parse_business_day_rule(
                    rec.get('business_day_rule')),
                'discount_curve':  ir_rep if is_rep
                                   else [ir_for, xccy_spec['name']],
                'first_accrual_date': None,
                'previous_pay_date': None if explicit_prev_pay else
                                     self._prev_reset(effective, term_years,
                                                      valuation),
            }

            if fixed:
                # Fixed leg: the blotter coupon in every period. No projection
                # curve and no reset, and the accrual endpoints roll with the
                # payment calendar -- the RiskWatch fixed-leg convention.
                leg.update({
                    'name':            '{0} Fixed Leg'.format(ccy),
                    'forecast_curve':  None,
                    'coupon_rate':     to_coupon_rate(rec.get('coupon_rate')),
                    'spread':          0.0,
                    'calendar_adjustment': True,
                    'last_reset_rate': 0.0,
                    'hist_fixings_curve': None,
                    'hist_fixings_nodes': None,
                })
            else:
                # last reset: reset-override file wins, then the workbook cell
                last_reset = to_reset_rate(rec.get('last_reset_rate'),
                                           self.reset_is_percent)
                if deal_num in self.reset_lookup:
                    last_reset = self.reset_lookup[deal_num]
                if pd.isna(last_reset):
                    last_reset = 0.0

                hist_name = _u(rec.get('hist_fixings', '')).strip() or None
                leg.update({
                    'name':            '{0} Floating Leg'.format(ccy),
                    'forecast_curve':  ir_rep if is_rep else ir_for,
                    'coupon_rate':     None,
                    'spread':          pct_spread_to_decimal(rec.get('spread')),
                    'calendar_adjustment': False,  # RiskWatch float convention
                    'last_reset_rate': last_reset,
                    'hist_fixings_curve': hist_name,
                    # (dates, rates) from the fixings workbook -- the observed
                    # leg of the post-determined reset; None falls back to the
                    # flat last reset rate.
                    'hist_fixings_nodes': self._fixings_nodes(hist_name),
                })
            legs.append(leg)

        if prev_pay is None:
            prev_pay = self._prev_reset(effective, legs[0]['term_years'],
                                        valuation)

        return {
            'id':                 deal_num,
            'deal_num':           deal_num,
            'instrument_type':    _u(recs[0].get('leg_type',
                                                 'Cross-Currency Swap')).strip()
                                  or 'Cross-Currency Swap',
            'pair':               '{0}/{1}'.format(self.reporting, foreign_ccy),
            'foreign_currency':   foreign_ccy,
            'valuation_date':     self.valuation_date,
            'effective_date':     effective,
            'maturity_date':      maturity,
            'previous_pay_date':  pd.Timestamp(prev_pay),
            'reporting_currency': self.reporting,
            'fx':                 {foreign_ccy: fx_quote_for(
                                       self.fx_table, foreign_ccy,
                                       self.reporting)},
            'derived_curves':     [xccy_spec],
            'legs':               legs,
        }

    # -- pricing -------------------------------------------------------------
    def price(self):
        '''Price every cross-currency swap; return the summary.'''
        pairs = self.load_pairs()
        self._defaulted_position = []

        self.rw_mtm = {}
        if self.rw_mtm_csv:
            self.rw_mtm = load_rw_ccs_mtm(self.rw_mtm_csv,
                                          self.rw_instrument_col,
                                          self.rw_mtm_col, self.rw_tag)
        compare = bool(self.rw_mtm_csv)

        self.swaps = OrderedDict()
        rows = []
        for deal_num, recs in pairs:
            try:
                params = self._build_params(deal_num, recs)
            except (KeyError, ValueError) as e:
                self.skipped.append((deal_num, str(e)))
                continue

            # skip cleanly when a referenced curve is not loaded
            needed = []
            for leg in params['legs']:
                disc = leg['discount_curve']
                needed.extend(list(disc) if isinstance(disc, (list, tuple))
                              else [disc])
                if leg.get('forecast_curve'):   # fixed legs have none
                    needed.append(leg['forecast_curve'])
            # membership is checked against the LIVE CurveSet: _build_params
            # has just materialised any derived XCCY curve into it.
            missing = [c for c in needed if c not in self.curves.curves]
            if missing:
                self.skipped.append((deal_num,
                                     'curve(s) not loaded: {0}'.format(missing)))
                continue

            try:
                swap = CrossCurrencySwap(self.curves, params)
                self.swaps[deal_num] = swap
                rep_name = _leg_name_for(params, self.reporting)
                for_name = _leg_name_for(params, params['foreign_currency'])
                rep_pv = swap.leg_pv(rep_name)
                for_pv = swap.leg_pv(for_name)
                for_pv_rep = swap.leg_pv_reporting(for_name)
                mtm, err = swap.npv(), ''
            except Exception as e:
                rep_pv = for_pv = for_pv_rep = mtm = float('nan')
                err = str(e)

            foreign_side = [lg['position'] for lg in params['legs']
                            if lg['currency'] == params['foreign_currency']][0]
            row = OrderedDict([
                ('DealNum',    deal_num),
                ('ID',         params['id']),
                ('Swap type',  params['instrument_type']),
                ('Legs',       '/'.join(lg['leg_kind'].upper()
                                        for lg in params['legs'])),
                ('Pair',       params['pair']),
                ('Position',   'RECEIVE {0} / PAY {1}'.format(
                    self.reporting, params['foreign_currency'])
                    if foreign_side == 'pay' else
                    'PAY {0} / RECEIVE {1}'.format(
                        self.reporting, params['foreign_currency'])),
                ('{0} Leg PV'.format(self.reporting), round(rep_pv, 2)),
                ('Foreign Leg PV (ccy)', round(for_pv, 2)),
                ('Foreign Leg PV ({0})'.format(self.reporting),
                 round(for_pv_rep, 2)),
            ])

            if compare:
                rw = self.rw_mtm.get(deal_num, float('nan'))
                if pd.notna(mtm) and pd.notna(rw) and rw != 0:
                    pct = (mtm / rw - 1.0) * 100.0
                else:
                    pct = float('nan')
                row['MtM-UAT'] = round(mtm, 2)
                row['MtM-RiskWatch'] = round(rw, 2) if pd.notna(rw) else float('nan')
                row['(MtM-UAT/RW-1)%'] = round(pct, 4) if pd.notna(pct) else float('nan')
            else:
                row['MtM'] = round(mtm, 2)

            row['Error'] = err
            rows.append(row)

        out = pd.DataFrame(rows)
        if 'Error' in out.columns and (out['Error'] == '').all():
            out = out.drop(columns=['Error'])
        self.results = out

        if self._defaulted_position:
            print('NOTE: pay/receive not in the workbook; defaulted the '
                  'FOREIGN side of {0} deal(s) to {1!r}: {2}'.format(
                      len(self._defaulted_position),
                      self.default_foreign_position,
                      self._defaulted_position))
        if self.skipped:
            from collections import Counter
            reasons = Counter(r for _, r in self.skipped)
            print('Skipped {0} deal(s):'.format(len(self.skipped)))
            for reason, n in reasons.most_common():
                print('  {0:>4d}  {1}'.format(n, reason))

        return out

    def summary(self):
        '''The per-swap results DataFrame (prices on first call if needed).'''
        if self.results is None:
            self.price()
        return self.results