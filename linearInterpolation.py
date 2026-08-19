# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:20:11 2026

@author: E42656
"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

LINEAR = 'linear'
CUBIC = 'cubic'
VALID_METHODS = (LINEAR, CUBIC)


class Interpolation(object):

    def __init__(self, tenors_days, rates, method=LINEAR):
        x = np.asarray(tenors_days, dtype=float)
        y = np.asarray(rates, dtype=float)
        if x.size == 0 or y.size == 0:
            raise ValueError("Cannot build a curve from empty tenors/rates.")

        method = str(method).strip().lower()
        if method not in VALID_METHODS:
            raise ValueError(
                "Unknown interpolation method {0!r}. Use one of {1}.".format(
                    method, VALID_METHODS))
        self.method = method

        # sort by tenor and drop duplicate tenors
        order = np.argsort(x)
        x, y = x[order], y[order]
        keep = np.concatenate(([True], np.diff(x) > 0))
        self.x = x[keep]
        self.y = y[keep]

        self._spline = None
        if self.method == CUBIC:
            if self.x.size < 2:
                raise ValueError(
                    "Cubic interpolation needs at least 2 distinct tenors; "
                    "got {0}.".format(self.x.size))
            from scipy.interpolate import CubicSpline
            self._spline = CubicSpline(self.x, self.y, bc_type='natural')

    def __call__(self, t_days):
        if self._spline is None:
            return np.interp(t_days, self.x, self.y)

        t = np.clip(np.asarray(t_days, dtype=float), self.x[0], self.x[-1])
        result = self._spline(t)
        if np.isscalar(t_days):
            return float(result)
        return result


def curve_from_frame(df, tenor_col=0, rate_col=1, method=LINEAR):
    sub = df.iloc[:, [tenor_col, rate_col]].copy()
    sub.columns = ['tenor', 'rate']
    sub['tenor'] = pd.to_numeric(sub['tenor'], errors='coerce')
    sub['rate'] = pd.to_numeric(sub['rate'], errors='coerce')
    sub = sub.dropna()
    return Interpolation(sub['tenor'].values, sub['rate'].values, method=method)


class CurveSet(object):
    def __init__(self, curves):
        self.curves = curves
        

    def rate(self, curve_name, t_days):
        key = str(curve_name).strip()
        
        if key not in self.curves:
            raise KeyError('Curve {0} not in curves. Current curves: {1}'.format(
                    key, list(self.curves)))
        
        return self.curves[key](t_days)
    
    def rf(self, discount_curve, t_days):
        return self.rate(discount_curve, t_days)
    
    def gov(self, discount_curve, t_days):
        return self.rate(discount_curve, t_days)
    
    def spread_at(self, spread_curve, t_days, base = None):
        name = str(spread_curve).strip()
        
        # CDS already spreads
        if name.upper().endswith('CDS'):
            return self.rate(name, t_days)
        
        # -Spread = -GOV minus -SWAP
        if name.endswith('-Spread'):
            parts = name.split('-')
            gov_curve = "{0}-GOV".format(parts[0])
            base_curve = base if base is not None else '{0}-SWP'.format(parts[1])
            
            gov = self.rate(gov_curve, t_days)
            base = self.rate(base_curve, t_days)
            
            return gov - base
        
        #fallback
        return self.rate(name, t_days)
    def get_spread(self, spread_curve, t_days):
        return self.spread_at(spread_curve, t_days)
    
    def zero_coupon_rate(self, discount_curve, spread_curve, t_days, spread_over_yield = 0.0):
        name = str(spread_curve).strip()
        spread = self.spread_at(spread_curve, t_days)
        
        if name.endswith('-Spread'):
            rf_curve = '{0}-SWP'.format(name.split('-')[1])
        else:
            rf_curve = discount_curve
            
        return self.rf(rf_curve, t_days) + spread + spread_over_yield


def _filter_tabs(sheet_names, exclude):
    exclude_lower = set(str(e).strip().lower() for e in exclude)
    return [s for s in sheet_names if s.strip().lower() not in exclude_lower]


def list_curve_tabs(path, exclude=()):
    return _filter_tabs(pd.ExcelFile(path).sheet_names, exclude)


def load_curve_set(path, curve_tabs=None, exclude=(), tenor_col=0, rate_col=1,
                   header=None, method=LINEAR):
    """
    Load every curve tab in an Excel workbook into a CurveSet.
    """
    xls = pd.ExcelFile(path)
    if curve_tabs is None:
        curve_tabs = _filter_tabs(xls.sheet_names, exclude)

    curves = {}
    for name in curve_tabs:
        tab = pd.read_excel(path, sheet_name=name, header=header)
        curves[name.strip()] = curve_from_frame(tab, tenor_col, rate_col,
                                                method=method)
    return CurveSet(curves)


if __name__ == '__main__':
    XLSX = 'Bond_Pricing.xlsx'
    NON_CURVE_TABS = ['Input_Transactional_Data_CSV', 'Input_Mkt_Data_CSV',
                      'RiskWatch-Results']

    curves = load_curve_set(XLSX, exclude=NON_CURVE_TABS, method='linear')
    print('GRD-GOV at 10962d:', round(float(curves.rf('GRD-GOV', 10962)), 6))