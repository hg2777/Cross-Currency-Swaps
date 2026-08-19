# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:02:25 2026

Tent-shock machinery for the GIRR delta sensitivities: the tenor grid, the
triangle (tent) shock, and the shocked-curve table shown in the diagnostics
workbook. The tenor grid itself (days + labels) is configured in main.py.

@author: E42656
"""

from collections import OrderedDict

import pandas as pd
import numpy as np

from linearInterpolation import Interpolation

ONE_BP = 0.0001


def tenors_from_days(days, labels, valuation_date):
    '''
    Build a tenor table (tenor / date / days) from hardcoded day counts from
    the valuation date, paired positionally with their display labels. Pass
    the days ascending (the tent shock assumes monotonic tenors).
    '''
    valuation = pd.Timestamp(valuation_date)
    rows = []
    for label, d in zip(labels, days):
        d = int(d)
        rows.append((u'{0}'.format(label).strip(),
                     valuation + pd.Timedelta(days=d), d))
    return pd.DataFrame(rows, columns=['tenor', 'date', 'days'])


def _tent_knots(tenor_days, tenor, shock):
    '''
    Creates triangle structure
    '''
    n = len(tenor_days)
    if n == 1:
        return np.array([tenor_days[0]]), np.array([shock])
    if tenor == 0:
        xk = [tenor_days[0], tenor_days[1]]
        yk = [shock, 0.0]
    elif tenor == n - 1:
        xk = [tenor_days[tenor - 1], tenor_days[tenor]]
        yk = [0.0, shock]
    else:
        xk = [tenor_days[tenor - 1], tenor_days[tenor], tenor_days[tenor + 1]]
        yk = [0.0, shock, 0.0]
    return np.asarray(xk, dtype=float), np.asarray(yk, dtype=float)


def tent_shock(grid_days, tenor_days, tenor, shock=ONE_BP):
    '''
    Shock to add at each grid point for a tenor
    '''
    xk, yk = _tent_knots(np.asarray(tenor_days, dtype=float), tenor, shock)
    if len(xk) == 1:
        return np.full(np.shape(grid_days), yk[0], dtype=float)
    return np.interp(np.asarray(grid_days, dtype=float), xk, yk)


def _tent_shock_fn(tenor_days, tenor, shock=ONE_BP):
    '''
    To be used when repricing.
    Returns the shock for tenor.
    '''
    xk, yk = _tent_knots(np.asarray(tenor_days, dtype=float), tenor, shock)
    if len(xk) == 1:
        return lambda td: np.full(np.shape(td), yk[0], dtype=float)
    return lambda td: np.interp(np.asarray(td, dtype=float), xk, yk)


def build_sensitivity_table(valuation_date, node_days, node_rates, tenor_table,
                            method='linear', shock=ONE_BP,
                            include_parallel=True):
    '''
    Returns table: Date, Days, BaseRate, BaseRate + 1BP, one col per tenor for
    shocked rate.
    '''
    valuation = pd.Timestamp(valuation_date)
    node_days = np.asarray(node_days, dtype=float)
    node_rates = np.asarray(node_rates, dtype=float)

    curve = Interpolation(node_days, node_rates, method=method)

    tenor_days = tenor_table['days'].values.astype(float)
    tenor_labels = list(tenor_table['tenor'])

    grid_days = np.union1d(node_days, tenor_days)

    base = np.asarray(curve(grid_days), dtype=float)

    cols = OrderedDict()
    cols['Date'] = [valuation + pd.Timedelta(days=int(d)) for d in grid_days]
    cols['Days'] = grid_days.astype(int)
    cols['BaseRate'] = base
    if include_parallel:
        cols['ParallelUp1bp'] = base + shock

    df = pd.DataFrame(cols)

    for tenor, label in enumerate(tenor_labels):
        df[label] = base + tent_shock(grid_days, tenor_days, tenor, shock)

    # Sanity Check - Last column should be 1bp
    df['SanityCheck'] = (
            df[list(tenor_labels)].sum(axis=1) - len(tenor_labels) * base
            )

    return df


def get_curve_nodes(curve_set, name):
    '''
    Return days and rates for a curve in CurveSet.
    '''
    key = str(name).strip()
    if key in curve_set.curves:
        c = curve_set.curves[key]
        return np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float)
    raise KeyError("Curve '{0}' not found in the curve set.".format(key))