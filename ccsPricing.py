# -*- coding: utf-8 -*-
"""
Cross-currency post-determined swap pricing (float and/or fixed legs).

Mirrors simpleSwapPricing.SimpleSwap one-for-one -- the same payment-date
holiday roll (per-deal currency calendars, Following), the same COMPOUNDING switch, the
same backward schedule generation, the same accr_start_used/accr_end_used
convention (discounting always on the ADJUSTED payment date; the float
accrual, the forward projection and the post-determined reset on the
UNADJUSTED schedule unless a leg sets calendar_adjustment=True) and the same
post-determined first-period reset via resetRate.last_reset_rate.

What is CCS-specific:
  * a deal is a LIST of float legs (params['legs']), one per currency, each
    priced in its own currency and converted to the reporting currency
    through params['fx'] for the deal MtM;
  * a leg's discount curve may be a LIST of curve names whose zero rates are
    SUMMED -- the foreign discount decomposed as IR + XCCY basis, each an
    independently shockable risk factor (build_derived_curves materialises
    the basis curve as XXX-XCCY = XXX-CSA-EUR - XXX-IR on the IR grid);
  * every FLOAT period's coupon is index PLUS the leg's basis spread --
    including the post-determined first-period reset (the blotter carries
    BARE index resets; the engine adds the spread consistently across all
    periods);
  * a FIXED leg (the workbook's Fixed tab) pays its blotter coupon in every
    period: no forecast curve and no reset, but the same schedule,
    discounting, notional redemption and FX conversion as a float leg. Its
    accrual endpoints roll with the payment calendar
    (calendar_adjustment=True) -- the RiskWatch FIXED-leg convention, unlike
    the float legs, which accrue on the unadjusted schedule.

Targets Python 2.7 (no f-strings, .format(), object base classes).

@author: E42656
"""

import re

import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.offsets import DateOffset
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday,
                                    GoodFriday, EasterMonday, nearest_workday,
                                    next_monday, next_monday_or_tuesday)

from linearInterpolation import Interpolation

import resetRate

pd.set_option('display.max_columns', None)

# 'annual'     -> DF = 1 / (1 + z) ** (days / 365)
# 'continuous' -> DF = exp(-z * days / 365)
COMPOUNDING = 'annual'


# ---------------------------------------------------------------------------
# Payment-date holiday calendars: US federal, UK bank and EU TARGET holidays.
# Payment dates roll Following over weekends and the holidays of the
# currency of the LEG being paid (USD -> US federal, GBP -> UK bank,
# EUR -> EU TARGET; a currency without a mapped calendar, e.g. CHF,
# contributes weekends only) -- the RiskWatch convention. Rolling a leg
# over the union of the DEAL's calendars instead pushed the EUR leg of an
# EUR/USD deal one business day past US-only holidays (e.g. the EUR leg
# of P1422733 paying its final flow on 2027-11-26, the day after
# Thanksgiving, while RiskWatch pays on 2027-11-25) -- a one-day discount
# gap on coupon-plus-notional that broke both the MtM and the GIRR
# reconciliation of that leg's currency.
# ---------------------------------------------------------------------------
class _USHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('MLK Day', month=1, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday('Presidents Day', month=2, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday('Memorial Day', month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Juneteenth', month=6, day=19, start_date='2021-06-19',
                observance=nearest_workday),
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Columbus Day', month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday('Veterans Day', month=11, day=11, observance=nearest_workday),
        Holiday('Thanksgiving', month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
    ]


class _UKHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May Bank Holiday', month=5, day=1,
                offset=DateOffset(weekday=MO(1))),
        Holiday('Spring Bank Holiday', month=5, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer Bank Holiday', month=8, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day', month=12, day=26,
                observance=next_monday_or_tuesday),
    ]


class _EUHolidays(AbstractHolidayCalendar):
    '''EU TARGET closing days (fixed calendar days, no weekend observance).'''
    rules = [
        Holiday('New Year', month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday('Labour Day', month=5, day=1),
        Holiday('Christmas', month=12, day=25),
        Holiday('Goodwill Day', month=12, day=26),
    ]


_CCY_CALENDARS = {'USD': _USHolidays, 'GBP': _UKHolidays, 'EUR': _EUHolidays}
_HOLIDAY_RANGE = ('1990-01-01', '2099-12-31')
_HOLIDAYS = {}


def _holiday_set(currencies=None):
    '''Union of the holiday calendars mapped to `currencies` (an iterable
    of ISO codes) over the working range, built once per combination.
    Unmapped currencies contribute no holidays; None -> every mapped
    calendar (the legacy US/UK/EU union).'''
    if currencies is None:
        ccys = tuple(sorted(_CCY_CALENDARS))
    else:
        ccys = tuple(sorted(
            set(u'{0}'.format(c).strip().upper() for c in currencies)
            & set(_CCY_CALENDARS)))
    if ccys not in _HOLIDAYS:
        days = set()
        for c in ccys:
            cal = _CCY_CALENDARS[c]()
            for d in cal.holidays(pd.Timestamp(_HOLIDAY_RANGE[0]),
                                  pd.Timestamp(_HOLIDAY_RANGE[1])):
                days.add(pd.Timestamp(d).normalize())
        _HOLIDAYS[ccys] = frozenset(days)
    return _HOLIDAYS[ccys]


# ---------------------------------------------------------------------------
# 'Business Day Rule' field: "Regular|Modified Following x-day (CalEUR)".
# The Cal suffix (CalEUR/CalUSD/CalGBP) selects the holiday calendar HERE,
# superseding the leg's Currency field; Modified rolls BACK when the forward
# roll crosses into a later month; 'x-day' steps x further business days.
# Reuses the currency->calendar classes above, keyed by the Cal code.
# ---------------------------------------------------------------------------
_HOLIDAYS_BY_CAL = {}


def _holiday_set_for(cal):
    '''Holiday set for a single Business Day Rule calendar (EUR/USD/GBP),
    built once and cached. EUR -> EU TARGET, USD -> US federal, GBP -> UK
    bank. An unknown code contributes weekends only (empty holiday set).'''
    key = u'{0}'.format(cal).strip().upper()
    if key not in _HOLIDAYS_BY_CAL:
        cls = _CCY_CALENDARS.get(key)
        if cls is None:
            _HOLIDAYS_BY_CAL[key] = frozenset()
        else:
            days = set()
            for d in cls().holidays(pd.Timestamp(_HOLIDAY_RANGE[0]),
                                    pd.Timestamp(_HOLIDAY_RANGE[1])):
                days.add(pd.Timestamp(d).normalize())
            _HOLIDAYS_BY_CAL[key] = frozenset(days)
    return _HOLIDAYS_BY_CAL[key]


_BDR_RE = re.compile(
    r'(regular|modified)\s+following\s+(\d+)\s*-?\s*day\s*'
    r'\(\s*cal\s*(eur|usd|gbp)\s*\)', re.IGNORECASE)


def parse_business_day_rule(text):
    '''Parse a Business Day Rule cell of the form

        "Regular/Modified Following x-day (CalEUR/CalUSD/CalGBP)"

    into {'modified': bool, 'offset': int, 'calendar': 'EUR'|'USD'|'GBP'}.
    A blank / unparseable cell returns None; the pricer then falls back to the
    legacy per-leg-currency Following roll (unchanged CCS behaviour).'''
    s = u'{0}'.format(text).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'):
        return None
    m = _BDR_RE.search(s)
    if not m:
        return None
    return {'modified': m.group(1).lower() == 'modified',
            'offset': int(m.group(2)),
            'calendar': m.group(3).upper()}


def is_fixed_leg(leg):
    '''True when a leg pays a fixed coupon rather than a projected index.

    Driven by the leg_kind tag the portfolio sets from the source tab; a leg
    carrying a coupon rate but no forecast curve is treated as fixed too, so
    hand-built params behave the same way.'''
    kind = u'{0}'.format(leg.get('leg_kind', '')).strip().lower()
    if kind:
        return kind == 'fixed'
    return leg.get('coupon_rate') is not None


def _act_act_isda(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    if end <= start:
        return 0
    total = 0.0
    cursor = start
    while cursor < end:
        next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1)
        seg_end = min(end, next_year)
        days_in_year = 366 if cursor.is_leap_year else 365
        total += (seg_end - cursor).days / float(days_in_year)
        cursor = seg_end
    return total


def _thirty_360_us(start, end):
    '''US (NASD) 30/360 day-count, expressed in years.

        if D1 == 31              -> D1 = 30
        if D2 == 31 and D1 == 30 -> D2 = 30

    (D1 == 30 already covers the case where D1 was 31 and clamped above.)
    '''
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (end.year - start.year)
            + 30 * (end.month - start.month)
            + (d2 - d1)) / 360.0


def swap_year_fraction(start, end, basis):
    '''Year fraction between two dates for a swap leg basis.'''
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    days = (end - start).days
    b = str(basis).strip().lower().replace('smp', '').strip()

    if b in ('actual/360', 'act/360', 'actual360'):
        return days / 360.0
    elif b in ('actual/365', 'act/365', 'actual365'):
        return days / 365.0
    elif b in ('30/360', '30u/360', 'us 30/360', '360/360', 'bond',
               '30e/360', 'european 30/360'):
        return _thirty_360_us(start, end)
    elif b in ('actual/actual', 'act/act', 'actualactual'):
        return _act_act_isda(start, end)
    return days / 365.0


# ----------------------------------------------------------------------------
# Derived (basis) curves
# ----------------------------------------------------------------------------
def build_derived_curves(curves, specs, method='linear'):
    """
    Materialise basis curves of the form (base - subtract) into a CurveSet,
    so a discount curve can be expressed as the SUM of an IR curve and a basis
    curve, each shockable as its own risk factor.

    Each spec is a dict:
        name        : name of the new curve, e.g. 'USD-XCCY'
        base        : curve subtracted from,  e.g. 'USD-CSA-EUR'
        subtract    : curve subtracted,       e.g. 'USD-STR'
        grid_curve  : whose tenor nodes to build the new curve on (the basis
                      is observed on this grid), e.g. 'USD-STR'

    At each node t of grid_curve: value = base(t) - subtract(t). Summing the
    new curve back with `subtract` reproduces `base` (exactly at the nodes;
    to machine precision at points the grid brackets cleanly).
    """
    for spec in specs:
        grid = curves.curves[spec['grid_curve']].x
        vals = np.array([float(curves.rate(spec['base'], t))
                         - float(curves.rate(spec['subtract'], t))
                         for t in grid])
        curves.curves[spec['name']] = Interpolation(grid, vals, method=method)
    return curves


# ----------------------------------------------------------------------------
# Cross-currency swap
# ----------------------------------------------------------------------------
class CrossCurrencySwap(object):

    def __init__(self, curves, params):
        self.curves = curves
        self.params = params
        self.fx = params['fx']

        self.valuation = pd.Timestamp(params['valuation_date'])


        # full working frames, one per leg (helper columns retained)
        self.legs = {}
        for leg in params['legs']:
            self.legs[leg['name']] = self._value_leg(leg)

    # ----------------------------------------------------------- curve / DF
    def _adjust_payment_date(self, dt, holidays, adjust=True):
        '''Following roll: move forward past weekends and the holidays of
        the paying LEG's currency. adjust False -> leave the unadjusted
        schedule date.'''
        d = pd.Timestamp(dt).normalize()
        if not adjust:
            return d
        while d.weekday() >= 5 or d in holidays:
            d = d + pd.Timedelta(days=1)
        return d

    def _roll_business_day(self, dt, bdr, fallback_holidays):
        '''Business Day Rule roll for a PAYMENT date.

        x=0 gives the next business day (Following); 'Modified' rolls back to
        the previous business day when that next business day falls in a new
        month, while 'Regular' keeps the new-month date. The 'x-day' offset
        then steps x further business days forward from that adjusted date
        (x=0 -> no step; x is never negative). The calendar is the rule's own
        Cal (EUR/USD/GBP), NOT the leg currency.

        bdr None -> the legacy per-leg Following roll on `fallback_holidays`
        (this leg's own currency calendar), i.e. unchanged CCS behaviour.'''
        d = pd.Timestamp(dt).normalize()
        if bdr is None:
            return self._adjust_payment_date(d, fallback_holidays, True)

        holidays = _holiday_set_for(bdr['calendar'])

        def is_bus(x):
            return x.weekday() < 5 and x not in holidays

        def following(x):
            while not is_bus(x):
                x = x + pd.Timedelta(days=1)
            return x

        def preceding(x):
            while not is_bus(x):
                x = x - pd.Timedelta(days=1)
            return x

        base = following(d)
        if bdr['modified'] and (base.year, base.month) != (d.year, d.month):
            base = preceding(d)              # last business day within d's month

        res = base
        for _ in range(int(bdr['offset'])):  # 'x-day' forward offset
            res = following(res + pd.Timedelta(days=1))
        return res

    def _days(self, dt):
        return (pd.Timestamp(dt) - self.valuation).days

    def _curve_rate(self, spec, days):
        '''Zero rate for a single curve name, or the SUM of several.

        A composite discount curve (e.g. ['USD-STR', 'USD-XCCY']) returns the
        summed rate, so it can be decomposed into independently-shockable risk
        factors while reproducing the original curve.
        '''
        if isinstance(spec, (list, tuple)):
            return sum(float(self.curves.rate(c, days)) for c in spec)
        return float(self.curves.rate(spec, days))

    def _zero_and_df(self, curve_spec, dt):
        '''Zero rate and discount factor at a date (COMPOUNDING convention).'''
        d = self._days(dt)
        if d <= 0:
            return 0.0, 1.0
        z = self._curve_rate(curve_spec, d)
        if str(COMPOUNDING).strip().lower().startswith('cont'):
            return z, float(np.exp(-z * d / 365.0))
        return z, 1.0 / (1.0 + z) ** (d / 365.0)

    def _forward_rate(self, forecast_curve, start, end, spread):
        '''Simple forward implied by the FORECAST curve over [start, end]
        (ACT/360 on the interval), plus the leg's basis spread. The interval
        is the leg's accrual interval (accr_start_used / accr_end_used), so
        the projected rate and the accrual always cover the same dates.
        '''
        tau = (pd.Timestamp(end) - pd.Timestamp(start)).days / 360.0
        if tau <= 0:
            return 0.0
        _, df_start = self._zero_and_df(forecast_curve, start)
        _, df_end = self._zero_and_df(forecast_curve, end)
        return (df_start / df_end - 1.0) / tau + float(spread)

    def _first_period_is_reset(self, first_accrual):
        '''First live period carries a known fixing (the reset) rather than
        a projected forward. With a leg first-accrual date (reset override)
        it is the authoritative boundary: reset iff the valuation date is
        strictly PAST it. With no override the legacy rule stands -- the
        first live period is always the reset.'''
        if first_accrual is None:
            return True
        return self.valuation > pd.Timestamp(first_accrual)

    def _compounded_reset_rate(self, leg, start, end, accrual):
        '''Post-determined first-period reset rate (BARE) via the shared
        resetRate module: observed historical fixings compounded to the
        valuation date, then the index curve's forward growth to the period
        end, over the period accrual. The index rate is read through
        _curve_rate (composite / shocked curves resolve); the observed leg
        reads the DATED fixings series from leg['hist_fixings_nodes'] (loaded
        from the historical-fixings workbook by the portfolio) -- observed
        history, never shocked. The leg's basis spread is added by the caller,
        so every period is index-plus-spread consistently.'''
        fix_dates, fix_rates = leg['hist_fixings_nodes']
        off = (pd.Timestamp(end) - self.valuation).days
        z = self._curve_rate(leg['forecast_curve'], off)
        reset, _observed, _forward = resetRate.last_reset_rate(
            self.valuation, start, end,
            fix_dates, fix_rates, [off], [z], accrual=accrual)
        return reset

    # -------------------------------------------------------------- schedule
    def _schedule_boundaries(self, term_years):
        '''Period boundaries, generated BACKWARD from maturity (any stub is
        the first, possibly short, period) -- same as the single-currency
        post-determined pricer.'''
        effective = pd.Timestamp(self.params['effective_date'])
        maturity = pd.Timestamp(self.params['maturity_date'])
        months = int(round(float(term_years) * 12))

        boundaries = [maturity]
        i = 1
        while True:
            prev = maturity - DateOffset(months=months * i)
            if prev <= effective:
                break
            boundaries.append(prev)
            i += 1
        boundaries.append(effective)     # first (possibly short) period
        boundaries.reverse()
        return boundaries

    def _build_leg(self, leg):
        '''Remaining schedule + accrual for one leg. The discount/payment
        date always rolls to the next business day (Following, holiday-aware);
        the accrual endpoints -- and hence the forward projection and the
        post-determined reset -- roll with this leg's calendar_adjustment
        flag: rolled when True, left on the unadjusted schedule when False
        (the RiskWatch float-leg convention, the default).'''
        boundaries = self._schedule_boundaries(leg['term_years'])
        adjust = bool(leg.get('calendar_adjustment', False))
        # optional per-leg Business Day Rule (Modified/Regular Following,
        # x-day offset, own Cal). None -> the legacy per-leg Following roll.
        bdr = leg.get('business_day_rule')
        # payment-roll holidays for THIS leg: its own currency's calendar
        # (the fallback used when no Business Day Rule is set)
        holidays = _holiday_set([leg['currency']])

        rows = []
        for i in range(len(boundaries) - 1):
            rows.append({
                'period_start':  boundaries[i],        # unadjusted schedule start
                'period_end':    boundaries[i + 1],    # unadjusted schedule end
                'accrual_start': boundaries[i],        # unadjusted base
                # discount/payment date ALWAYS rolls: under this leg's
                # Business Day Rule when set (Modified/Regular Following,
                # x-day, own Cal), else the legacy per-leg Following roll.
                'payment_date':  self._roll_business_day(boundaries[i + 1],
                                                         bdr, holidays),
            })
        df = pd.DataFrame(rows)

        # keep only cash flows settling after the valuation date
        df = df[df['period_end'] > self.valuation].reset_index(drop=True)
        if df.empty:
            return df

        # first live period accrues from the previous payment date; a leg
        # first-accrual date (reset override) takes precedence as that anchor.
        # A per-LEG previous pay date wins: the legs of one deal can pay on
        # different frequencies (a 6M fixed leg against a 3M float leg), so a
        # single deal-level anchor would put the shorter leg's first accrual
        # on the wrong boundary. Deal-level stays the fallback (and carries
        # any explicit previous_pay_dates override from main.py).
        anchor = (leg.get('first_accrual_date') or leg.get('previous_pay_date')
                  or self.params.get('previous_pay_date',
                                     self.params['effective_date']))
        df.loc[df.index[0], 'accrual_start'] = pd.Timestamp(anchor)

        basis = leg.get('basis', 'actual/360')

        # Accrual endpoints actually used. CCS convention: a float leg accrues
        # on the UNADJUSTED schedule (calendar_adjustment False) regardless of
        # any Business Day Rule; a fixed leg (calendar_adjustment True) rolls
        # its accrual with the SAME rule as its payment date, so it accrues
        # over exactly the interval it is discounted on. The float forward in
        # _value_leg is projected over these SAME endpoints.
        if adjust:
            roll_accr = lambda d: self._roll_business_day(d, bdr, holidays)
        else:
            roll_accr = lambda d: self._adjust_payment_date(d, holidays, False)
        df['accr_start_used'] = df['accrual_start'].apply(roll_accr)
        df['accr_end_used'] = df['period_end'].apply(roll_accr)
        df['accrual'] = df.apply(
            lambda r: swap_year_fraction(
                r['accr_start_used'], r['accr_end_used'], basis),
            axis=1)
        return df

    # ----------------------------------------------------------------- value
    def _value_leg(self, leg):
        df = self._build_leg(leg)
        if df.empty:
            return df

        notional = float(leg['notional'])
        disc_curve = leg['discount_curve']
        fcast_curve = leg.get('forecast_curve')
        spread = float(leg.get('spread', 0.0))

        # discounting off the adjusted payment date
        df['days'] = df['payment_date'].apply(self._days)
        zd = df['payment_date'].apply(lambda d: self._zero_and_df(disc_curve, d))
        df['disc_rate'] = [z for z, _ in zd]
        df['discount_factor'] = [f for _, f in zd]

        # A FIXED leg pays its blotter coupon in every period -- no forecast
        # curve, no reset. Everything after this (notional redemption,
        # discounting, FX) is shared with the float legs.
        if is_fixed_leg(leg):
            df['rate'] = float(leg['coupon_rate'])
        else:
            # floating rate per period. The first live period carries the
            # reset when _first_period_is_reset (post-determined if a
            # Historical Fixings series is set, else the static/overridden
            # last reset); a first-accrual date still in the future projects
            # it forward instead. Every branch adds the leg's basis spread, so
            # all periods are consistently index-plus-spread.
            first_accrual = leg.get('first_accrual_date')
            rates = []
            for pos, (_, r) in enumerate(df.iterrows()):
                if pos == 0 and self._first_period_is_reset(first_accrual):
                    if leg.get('hist_fixings_nodes') is not None:
                        rates.append(self._compounded_reset_rate(
                            leg, r['accr_start_used'], r['accr_end_used'],
                            r['accrual']) + spread)
                    else:
                        # no fixings series available -> flat last reset rate
                        rates.append(float(leg['last_reset_rate']) + spread)
                else:
                    rates.append(self._forward_rate(
                        fcast_curve, r['accr_start_used'], r['accr_end_used'],
                        spread))
            df['rate'] = rates

        df['cash_flow'] = notional * df['rate'] * df['accrual']
        # notional redemption on the final cash flow
        df.loc[df.index[-1], 'cash_flow'] += notional

        df['pv'] = df['cash_flow'] * df['discount_factor']

        return df

    # ----------------------------------------------------------- aggregation
    def _to_reporting(self, currency, amount):
        '''
        Convert an amount in `currency` into the reporting currency.

        The FX quote orientation is taken from params['fx'][currency]['quote'],
        a pair written in market order (price = units of the 2nd ccy per 1 unit
        of the 1st). The user picks the input direction by choosing the pair:
            reporting-then-foreign (e.g. 'EURUSD' = USD per 1 EUR) -> divide
            foreign-then-reporting (e.g. 'USDEUR' = EUR per 1 USD) -> multiply
        '''
        reporting = str(self.params.get('reporting_currency', 'EUR')).strip().upper()
        ccy = str(currency).strip().upper()
        if ccy == reporting:
            return amount

        spec = self.fx[currency]
        rate = float(spec['rate'])
        quote = str(spec['quote']).upper().replace('/', '').replace(' ', '').strip()
        base, term = quote[:3], quote[3:]

        if base == reporting and term == ccy:
            return amount / rate          # rate is foreign per 1 reporting
        if base == ccy and term == reporting:
            return amount * rate          # rate is reporting per 1 foreign
        raise ValueError(
            "FX quote {0!r} does not connect {1} and {2}.".format(
                quote, ccy, reporting))

    def _leg_params(self, name):
        for leg in self.params['legs']:
            if leg['name'] == name:
                return leg
        raise KeyError('Leg {0} not defined.'.format(name))

    # ------------------------------------------------------------------- pvs
    def leg_pv(self, name):
        '''Leg PV in its own currency.'''
        df = self.legs[name]
        return float(df['pv'].sum()) if not df.empty else 0.0

    def leg_pv_reporting(self, name):
        '''Leg PV converted to the reporting currency.'''
        leg = self._leg_params(name)
        return self._to_reporting(leg['currency'], self.leg_pv(name))

    def npv(self):
        '''
        Dirty MtM in the reporting currency: receive legs minus pay legs,
        each converted to the reporting currency.
        '''
        total = 0.0
        for leg in self.params['legs']:
            pv = self.leg_pv_reporting(leg['name'])
            side = str(leg.get('position', 'receive')).strip().lower()
            sign = 1.0 if side in ('receive', 'rec', 'long') else -1.0
            total += sign * pv
        return total

    def price(self):
        '''MtM of the cross-currency swap.'''
        return self.npv()