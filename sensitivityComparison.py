# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:32:35 2026

@author: E42656
"""

from collections import OrderedDict
 
import pandas as pd
 
 
def _norm_id(s):
    '''int id 1001 and float-string '1001.0' both reconcile to '1001'.'''
    s = str(s).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s
 
def _risk_class(raw_class):
    '''Canonical bucket for a RiskWatch risk-factor class, case/label tolerant.
    'GIRR' -> 'GIRR'; 'CSR (non-sec)' / 'CSR-N' / 'CSR Non-Sec' -> 'CSR'.'''
    s = u'{0}'.format(raw_class).strip().upper()
    if u'GIRR' in s:
        return 'GIRR'
    if u'CSR' in s:
        return 'CSR'
    return None
 

def year_label(y):
    '''0.25 -> ".25Y", 1 -> "1Y", 1.5 -> "1.5Y", 10 -> "10Y".'''
    s = '{0:g}'.format(y)
    if s.startswith('0.'):
        s = s[1:]
    return s + 'Y'
 
 
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
 

# ---------------------------------------------------------------------------
# 1. RiskWatch sensitivities (long-format CSV)
# ---------------------------------------------------------------------------
def load_rw_sensitivities_csv(path, id_col='Instrument ID',
                              class_col='Risk Factor Class',
                              tenor_col='Risk Factor Vertex 1',
                              value_col='Sensitivity Value (Reporting Currency)',
                              type_col='Sensitivity Type',
                              keep_types=('delta',), verbose=True):
    '''
    One row per instrument / risk-factor class / tenor. Returns
        {id: {'GIRR': {tenor_years: value}, 'CSR': {tenor_years: value}}}.
    Only GIRR and CSR classes are kept. keep_types filters the Sensitivity Type
    column by substring (so 'GIRR Delta' / 'CSR Delta' / 'Delta' all pass);
    set keep_types=None to keep every type.
    '''
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
 
    missing = [c for c in (id_col, class_col, tenor_col, value_col)
               if c not in raw.columns]
    if missing:
        raise KeyError('CSV missing columns {0}. Found: {1}'
                       .format(missing, list(raw.columns)))
 
    bucket = raw[class_col].apply(_risk_class)   # 'GIRR' / 'CSR' / None
    keep = bucket.notna()
    n_class_rows = int(keep.sum())
 
    types_seen = []
    if keep_types and type_col in raw.columns:
        ks = [str(k).strip().lower() for k in keep_types]
        t = raw[type_col].astype(str).str.strip().str.lower()
        types_seen = sorted(set(t[keep]))
        keep = keep & t.apply(lambda s: any(k in s for k in ks))
 
    out = OrderedDict()
    n_g = n_c = 0
    for _, r in raw[keep].iterrows():
        rid = _norm_id(r[id_col])
        if rid.lower() in ('', 'nan', 'none'):
            continue
        rc = _risk_class(r[class_col])
        if rc is None:
            continue
        y = pd.to_numeric(r[tenor_col], errors='coerce')
        v = pd.to_numeric(r[value_col], errors='coerce')
        if pd.isna(y) or pd.isna(v):
            continue
        out.setdefault(rid, {'GIRR': {}, 'CSR': {}})[rc][round(float(y), 6)] = float(v)
        if rc == 'GIRR':
            n_g += 1
        else:
            n_c += 1
 
    if verbose:
        print("[sensitivityComparison] CSV {0!r}: {1} ids; GIRR={2}, CSR={3}"
              .format(path, len(out), n_g, n_c))
        if types_seen:
            print("[sensitivityComparison] sensitivity types seen={0}; kept~{1}"
                  .format(types_seen, list(keep_types)))
        if (n_g + n_c) == 0 and n_class_rows > 0:
            print("[sensitivityComparison] WARNING: keep_types dropped all {0} "
                  "GIRR/CSR rows -- pass keep_types=None.".format(n_class_rows))
    return out
 
 

# ---------------------------------------------------------------------------
# 2. Our delta tables -> {id: {tenor_years: value}}
# ---------------------------------------------------------------------------
def delta_table_to_maps(delta_df, id_col='ID', skip=('V_base',)):
    if delta_df is None or len(delta_df) == 0:
        return {}
    col_years = {}
    for c in delta_df.columns:
        if c == id_col or c in skip:
            continue
        try:
            col_years[c] = round(label_to_years(c), 6)
        except (ValueError, TypeError):
            continue
    maps = {}
    for _, r in delta_df.iterrows():
        m = {}
        for c, y in col_years.items():
            if pd.notna(r[c]):
                m[y] = float(r[c])
        maps[str(r[id_col]).strip()] = m
    return maps
 
 
# ---------------------------------------------------------------------------
# 3. Final results frame
# ---------------------------------------------------------------------------
BASE_COLS = ['ID', 'Product Type', 'MtM-UAT', 'MtM-RiskWatch',
             '(MtM-UAT/RW-1)%', 'Position Units', 'Position Value']
 
 

def build_results_sheets(mtm_results, girr_delta, csr_delta, rw_sens,
                         id_col='ID', sensitivity_source='MtM-UAT',
                         sens_round=2, pct_round=4,
                         id_normalizer=_norm_id, verbose=True):
    '''
    Split the results into three frames, one per Excel tab:
        'MtM'
        'GIRR'
        'CSR'
    '''
    nrm = id_normalizer
    our = {'GIRR': {nrm(k): v for k, v in delta_table_to_maps(girr_delta).items()},
           'CSR': {nrm(k): v for k, v in delta_table_to_maps(csr_delta).items()}}
    rw_n = {nrm(k): v for k, v in rw_sens.items()}
    use_rw = str(sensitivity_source).upper() == 'RW'
 
    res = mtm_results[mtm_results[id_col].notna()].copy()
    res[id_col] = res[id_col].apply(nrm)
    res_ids = list(res[id_col])
 
    tenors = {}
    for rc in ('GIRR', 'CSR'):
        ts = set()
        for key in res_ids:
            ts |= set(our[rc].get(key, {}))
            ts |= set(rw_n.get(key, {}).get(rc, {}))
        tenors[rc] = sorted(ts)
 
    def _round(v, n):
        return round(v, n) if (v is not None and pd.notna(v)) else v
 
    sheets = OrderedDict()
 
    # MtM 
    mtm_cols = [c for c in BASE_COLS if c in res.columns]
    sheets['MtM'] = res[mtm_cols].reset_index(drop=True)
 
    # GIRR and CSR 
    id_cols = [c for c in ('ID', 'Product Type') if c in res.columns]
    for rc in ('GIRR', 'CSR'):
        columns = list(id_cols)
        for y in tenors[rc]:
            lbl = year_label(y)
            columns.append('{0} {1} sensitivity'.format(rc, lbl))
            columns.append('(MtM-UAT/RW-1)% for {0} {1}'.format(rc, lbl))
 
        records = []
        for _, row in res.iterrows():
            key = nrm(row[id_col])
            rec = OrderedDict()
            for c in id_cols:
                rec[c] = row[c]
            our_map = our[rc].get(key, {})
            rw_map = rw_n.get(key, {}).get(rc, {})
            for y in tenors[rc]:
                lbl = year_label(y)
                our_v = our_map.get(y)
                rw_v = rw_map.get(y)
                sens = rw_v if use_rw else our_v
                # deltas are now on the bond MtM, so compare directly to RW
                rec['{0} {1} sensitivity'.format(rc, lbl)] = _round(sens, sens_round)
                rec['(MtM-UAT/RW-1)% for {0} {1}'.format(rc, lbl)] = \
                    _round(pct_diff(our_v, rw_v), pct_round)
            records.append(rec)
        sheets[rc] = pd.DataFrame(records, columns=columns)
 
    return sheets