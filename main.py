# -*- coding: utf-8 -*-
"""
Cross-currency swap pricing and GIRR sensitivity (post-determined float
legs and/or fixed legs).

Prices the cross-currency swap book from the transposed population workbook
and reconciles MtM and GIRR/XCCY deltas to RiskWatch. Mirrors the
post-determined single-currency runner (main.py of the simpleswap book):
STEP 1 prices the book and compares MtM; STEP 2 runs the per-curve GIRR
deltas plus the XCCY basis delta. Both run off one CCSPortfolio and write to
a single workbook ('CCS_MtM' and 'GIRR_Delta' tabs).

@author: E42656
"""

import pandas as pd

from linearInterpolation import load_curve_set
from ccsPortfolio import CCSPortfolio
from ccsSensitivity import (ccs_girr_for_portfolio,
                            ccs_girr_delta_with_riskwatch, build_ccs_specs)
from ccsDiagnostics import write_ccs_diagnostics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
XLSX = 'curves.xlsx'                 # workbook holding the curve tabs + FX tab

FX_TAB = 'FX rates'                  # FX tab (foreign per EUR); not a curve
NON_CURVE_TABS = [FX_TAB]

INTERPOLATION_METHOD = 'linear'

# --- cross-currency book inputs ---------------------------------------------
# ONE transposed tab, one LEG per column; the two legs of a deal share the
# DealNum in the Name row ('Float Leg of P1180855' x2). 'First Accrual Date'
# is the swap's effective date; 'Spread' is in PERCENT.
CCS_INPUT_XLSX        = 'Cross_Currency_Swaps_Input.xlsx'
CCS_SHEET             = 'Float'      # the transposed FLOAT leg tab
# The transposed FIXED leg tab, same layout but carrying 'Coupon Rate' (in
# PERCENT) and 'Effective Date' in place of the index/reset/spread fields.
# Legs are matched to their float counterpart on the DealNum in the Name row,
# so a deal may be float/float, fixed/float or fixed/fixed. Set to None for a
# workbook with no fixed legs; a named-but-absent tab is reported and skipped.
CCS_FIXED_SHEET       = 'Fixed'
# Curve-name resolution. The discount and index names ('EUR-STR',
# 'USD-CSA-EUR') match the curve tabs and the RiskWatch Risk Factor IDs
# as-is, so no automatic renaming is applied. This dict is only for any
# additional explicit overrides (none needed here).
CCS_CURVE_ALIASES     = {}
CCS_VALUATION_DATE    = '2025-12-31'
CCS_RW_MTM_CSV        = 'frtb_sa_report.csv'  # FRTB SA report; None to skip
CCS_RW_INSTRUMENT_COL = 'Instrument ID'
CCS_RW_MTM_COL        = 'Mark To Market'
CCS_RW_TAG            = 'CCS'        # RiskWatch product prefix on the deal id
# Pay/receive is NOT in the workbook. The EUR (reporting) leg takes the
# opposite side of the foreign leg; deals without an explicit direction fall
# back to this FOREIGN side and are listed once in the run log.
# CCS_POSITIONS overrides the FOREIGN side per deal ({'P1180855': 'receive'}).
CCS_DEFAULT_FOREIGN_POSITION = 'pay'
CCS_POSITIONS         = {}
# Reset-rate override file: 2 cols (deal id, last reset rate); supersedes the
# Float tab's Last Reset Rate for that deal. The first accrual date is an
# input on the Float tab and is never overridden. None to skip.
CCS_RESET_CSV         = None
# Historical-fixings workbook for the post-determined reset: one tab per
# fixings series, column 1 = fixing dates, column 2 = rates (decimals).
# The Float tab's 'Historical Fixings' cell names the series; the aliases
# dict maps that name to the workbook tab when they differ.
CCS_HIST_FIXINGS_XLSX    = 'Historical Fixing Curves.xlsx'
CCS_HIST_FIXINGS_ALIASES = None
# Optional explicit previous-pay dates per deal; omitted -> computed from the
# schedule (last reset on/before valuation).
CCS_PREV_PAYS         = {}
CCS_OUT               = 'ccs_results.xlsx'      # single output workbook
CCS_DIAGNOSTICS_OUT   = 'ccs_diagnostics.xlsx'  # per-swap analytics + curves

# --- GIRR sensitivity grid ---------------------------------------------------
# FRTB GIRR vertices, hardcoded as DAYS counted from the valuation date and
# paired positionally with their display labels: 0.25Y <-> 90, ..., 30Y <->
# 10957. Change the two tuples together (same length, ascending days -- the
# tent shock assumes a monotonic grid). The same grid drives the GIRR deltas,
# the RiskWatch reconciliation labels and the diagnostics 'Scenario Curves'.
GIRR_TENOR_LABELS = ('0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y',
                     '10Y', '15Y', '20Y', '30Y')
GIRR_TENOR_DAYS   = (90, 181, 365, 730, 1096, 1826, 3652, 5479, 7305, 10957)
GIRR_SHOCK        = 0.0001   # 1bp bump-and-reprice shock

# Troubleshooting: restrict the run to these DealNums. None -> whole book.
CCS_ONLY_IDS = None


# ----------------------------------------------------------------------------
# Reconciliation summaries
# ----------------------------------------------------------------------------
def _print_rw_reconciliation(mtm):
    """Short RiskWatch reconciliation summary for the priced book."""
    if 'MtM-RiskWatch' in mtm.columns:
        m = mtm[mtm['MtM-UAT'].notna() & mtm['MtM-RiskWatch'].notna()
                & (mtm['MtM-RiskWatch'] != 0)]
        rel = (m['MtM-UAT'] / m['MtM-RiskWatch'] - 1.0).abs() * 100.0
        print("  priced            : {0}".format(int(mtm['MtM-UAT'].notna().sum())))
        print("  compared to RW    : {0}".format(len(m)))
        if len(m):
            print("  median |UAT/RW-1| : {0:.4f}%".format(rel.median()))
            print("  within 5% / 10%   : {0:.1f}% / {1:.1f}%".format(
                (rel <= 5).mean() * 100.0, (rel <= 10).mean() * 100.0))
    else:
        n = int(mtm['MtM'].notna().sum()) if 'MtM' in mtm.columns else len(mtm)
        print("  priced : {0}   (no RiskWatch report supplied -> MtM only)".format(n))


def _print_girr_reconciliation(girr):
    """Aggregate GIRR vs RiskWatch over all (swap, tenor, curve) cells."""
    col = '(Delta-UAT/RW-1)%'
    if col not in girr.columns:
        print("  no GIRR rows matched RiskWatch")
        return
    vals = pd.to_numeric(girr[col], errors='coerce').dropna().abs()
    print("  GIRR cells compared to RW : {0}".format(len(vals)))
    if len(vals):
        print("  median |UAT/RW-1|         : {0:.4f}%".format(vals.median()))
        print("  within 5% / 10%           : {0:.1f}% / {1:.1f}%".format(
            (vals <= 5).mean() * 100.0, (vals <= 10).mean() * 100.0))
    # Material UAT deltas with no RiskWatch match flag a curve-name mismatch
    # (our forecast/discount curve name != the report's Risk Factor ID).
    if 'Delta-RiskWatch' in girr.columns:
        uat = pd.to_numeric(girr['Delta-UAT'], errors='coerce').abs()
        unmatched = girr['Delta-RiskWatch'].isna() & (uat > 1e-6)
        n_un = int(unmatched.sum())
        if n_un:
            curves = sorted(girr.loc[unmatched, 'Curve'].unique())
            print("  WARNING: {0} nonzero UAT cell(s) had no RW match on "
                  "curve(s) {1}".format(n_un, curves))


def price_ccs(curves):
    """Cross-currency book run, in two steps off one CCSPortfolio:

        Step 1 : price the book and compare MtM to RiskWatch
        Step 2 : GIRR delta sensitivities per curve + the XCCY basis delta

    Both write to a single workbook ('CCS_MtM' and 'GIRR_Delta' tabs).
    """
    port = CCSPortfolio(curves, CCS_INPUT_XLSX, CCS_SHEET,
                        CCS_VALUATION_DATE,
                        fixed_sheet=CCS_FIXED_SHEET,
                        fx_xlsx=XLSX, fx_tab=FX_TAB,
                        rw_mtm_csv=CCS_RW_MTM_CSV,
                        rw_instrument_col=CCS_RW_INSTRUMENT_COL,
                        rw_mtm_col=CCS_RW_MTM_COL, rw_tag=CCS_RW_TAG,
                        only_ids=CCS_ONLY_IDS,
                        reset_csv=CCS_RESET_CSV,
                        positions=CCS_POSITIONS,
                        default_foreign_position=CCS_DEFAULT_FOREIGN_POSITION,
                        curve_aliases=CCS_CURVE_ALIASES,
                        hist_fixings_xlsx=CCS_HIST_FIXINGS_XLSX,
                        hist_fixings_aliases=CCS_HIST_FIXINGS_ALIASES,
                        previous_pay_dates=CCS_PREV_PAYS)

    # -- Step 1 : pricing + RiskWatch comparison -----------------------------
    print("=" * 95)
    print("CCS STEP 1/2 : pricing + RiskWatch MtM comparison")
    print("  workbook  : {0}   tabs: {1} / {2}".format(
        CCS_INPUT_XLSX, CCS_SHEET, CCS_FIXED_SHEET))
    print("  valuation : {0}   RiskWatch report: {1}".format(
        CCS_VALUATION_DATE, CCS_RW_MTM_CSV))
    print("  FX table  : {0}".format(port.fx_table))
    print("-" * 95)
    mtm = port.summary()                 # prices the book (+ RW reconciliation)
    _print_rw_reconciliation(mtm)

    # -- Step 2 : GIRR + XCCY sensitivities + RiskWatch comparison -----------
    print("=" * 95)
    print("CCS STEP 2/2 : GIRR + XCCY delta sensitivities vs RiskWatch")
    print("  GIRR shock: per physical curve (single 1bp tent, all roles); "
          "XCCY: parallel 1bp")
    print("-" * 95)
    girr_long = ccs_girr_for_portfolio(port, GIRR_TENOR_DAYS,
                                       GIRR_TENOR_LABELS, shock=GIRR_SHOCK)
    # Attach the RiskWatch comparison, matched on DealNum + currency + tenor
    # (basis rows on DealNum + currency).
    girr = ccs_girr_delta_with_riskwatch(girr_long, CCS_RW_MTM_CSV,
                                         tag=CCS_RW_TAG)
    print("  GIRR rows : {0}   curves: {1}   tenors: {2}".format(
        len(girr), sorted(girr['Curve'].unique()),
        list(pd.unique(girr['Tenor']))))
    if CCS_RW_MTM_CSV:
        _print_girr_reconciliation(girr)

    with pd.ExcelWriter(CCS_OUT) as xl:
        mtm.to_excel(xl, sheet_name='CCS_MtM', index=False, na_rep='N/A')
        girr.to_excel(xl, sheet_name='GIRR_Delta', index=False, na_rep='N/A')

    print("=" * 95)
    print("written to {0}  (tabs: CCS_MtM, GIRR_Delta)".format(CCS_OUT))
    print("=" * 95)

    # Per-swap diagnostics workbook (per-leg analytics + altered-curve
    # sensitivity), to trace any MtM / GIRR differences against RiskWatch.
    write_ccs_diagnostics(CCS_DIAGNOSTICS_OUT, port.curves,
                          build_ccs_specs(port),
                          port.valuation_date,
                          GIRR_TENOR_DAYS, GIRR_TENOR_LABELS,
                          method=INTERPOLATION_METHOD,
                          shock=GIRR_SHOCK)

    return mtm, girr


def main():
    curves = load_curve_set(XLSX, exclude=NON_CURVE_TABS,
                            method=INTERPOLATION_METHOD)
    print("Curves loaded from: {0}".format(XLSX))
    print("Interpolation method: {0}\n".format(INTERPOLATION_METHOD))
    return price_ccs(curves)


if __name__ == "__main__":
    main()