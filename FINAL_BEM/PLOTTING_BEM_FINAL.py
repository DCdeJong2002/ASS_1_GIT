"""
plot_results.py  —  Standalone plotting script
================================================
Loads full_bem_results.npz (produced by assignment.py) and
reproduces all assignment plots without re-running any BEM.

Usage
-----
    python plot_results.py                           # uses full_bem_results.npz in cwd
    python plot_results.py path/to/results.npz       # explicit path

Plots saved to ./plotting_plots_assignment/
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# MENU — choose which plots to produce and whether to display them
# =============================================================================

SHOW_PLOTS = False   # False -> save and close immediately (non-interactive)
                     # True  -> plt.show() after each save

# ── Plots ─────────────────────────────────────────────────────────────────────
PLOT_4_1  = True   # alpha and inflow angle vs r/R          (needs span sweep in npz)
PLOT_4_2  = True   # axial and tangential induction vs r/R  (needs span sweep in npz)
PLOT_4_3  = True   # Cn and Ct loading vs r/R               (needs span sweep in npz)
PLOT_4_4  = True   # CT, CQ, CP vs TSR  (broad perf sweep)  (needs perf sweep in npz)
PLOT_5    = True   # tip correction influence                (needs res_nc in npz)
PLOT_6    = True   # annuli count, spacing, convergence      (needs annuli data in npz)
PLOT_7    = True   # stagnation pressure                     (needs results_tsr8 in npz)
PLOT_8    = True   # all designs — chord, twist, induction, loading, alpha, performance
                   #   sub-plots produced when PLOT_8=True:
                   #     8a   chord distribution
                   #     8b   twist distribution
                   #     8ab  chord and twist combined
                   #     8c   axial induction
                   #     8c2  tangential induction  (NEW)
                   #     8d   normal loading
                   #     8e   angle of attack
                   #     8f   performance bar chart
PLOT_9    = True   # Cl and chord relation  (analytical optimum)
PLOT_9_CMP = True  # NEW: chord + Cl comparison analytical vs quartic on one figure
PLOT_10   = True   # Cl/Cd polar with operating points

# =============================================================================
# 1.  LOAD RESULTS
# =============================================================================

# =============================================================================
# 1.  LOAD RESULTS — two separate npz files
# =============================================================================
#
# bem_results.npz   — BEM sweeps, tip correction, convergence, section-6 data
# opt_results.npz   — baseline + all optimised designs (analytical, cubic, quartic)
#
# Both files are written next to assignment.py by default.
# You can override either path via command-line:
#   python plot_results.py bem_results.npz opt_results.npz
# Or override just the first:
#   python plot_results.py path/to/bem_results.npz
#
# If opt_results.npz is absent, all PLOT_8/9/10 plots are skipped gracefully.
# =============================================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) >= 3:
    bem_path = sys.argv[1]
    opt_path = sys.argv[2]
elif len(sys.argv) == 2:
    bem_path = sys.argv[1]
    opt_path = os.path.join(_script_dir, "opt_results.npz")
else:
    bem_path = os.path.join(_script_dir, "bem_results.npz")
    opt_path = os.path.join(_script_dir, "opt_results.npz")

if not os.path.exists(bem_path):
    raise FileNotFoundError(
        f"Cannot find BEM results file: '{bem_path}'\n"
        "Run assignment.py with SAVE_BEM_RESULTS=True first.")

_opt_available = os.path.exists(opt_path)
if not _opt_available:
    print(f"  [INFO] opt_results.npz not found at '{opt_path}' — "
          "PLOT_8/9/10 will be skipped.")

print(f"Loading BEM results from : {bem_path}")
B = np.load(bem_path, allow_pickle=False)

if _opt_available:
    print(f"Loading opt results from : {opt_path}")
    O = np.load(opt_path, allow_pickle=False)
else:
    O = {}   # empty fallback so _oget always returns None/default

def _bget(key, default=None):
    """Get from BEM file."""
    try:    return B[key]
    except KeyError: return default

def _oget(key, default=None):
    """Get from opt file."""
    try:    return O[key]
    except (KeyError, TypeError): return default

def _getf(D, key, default=None):
    """Get scalar float from either dict/npz."""
    try:
        v = D[key]
        return float(v)
    except (KeyError, TypeError):
        return default

def _arr_or_none(D, key):
    """Return array if present and non-empty, else None."""
    try:
        v = D[key]
        return v if v.size > 0 else None
    except (KeyError, TypeError):
        return None

# ── Configuration scalars — prefer opt file, fall back to BEM file ────────────
_cfg = O if _opt_available else B
Radius         = _getf(_cfg, "cfg_Radius",         50.0)
NBlades        = int(_getf(_cfg, "cfg_NBlades",    3))
U0             = _getf(_cfg, "cfg_U0",             10.0)
rho            = _getf(_cfg, "cfg_rho",            1.0)      # matches assignment.py
RootLocation_R = _getf(_cfg, "cfg_RootLocation_R", 0.2)
TipLocation_R  = _getf(_cfg, "cfg_TipLocation_R",  1.0)
Pitch          = _getf(_cfg, "cfg_Pitch",          -2.0)
CHORD_ROOT     = _getf(_cfg, "cfg_CHORD_ROOT",     3.4)
CHORD_MIN      = _getf(_cfg, "cfg_CHORD_MIN",      0.3)
CT_TARGET      = _getf(_cfg, "cfg_CT_TARGET",      0.75)
TSR_DESIGN     = _getf(_cfg, "cfg_TSR_DESIGN",     8.0)
DELTA_R_R      = _getf(_cfg, "cfg_DELTA_R_R",      0.005)

# ── Polar — from BEM file (always present) ────────────────────────────────────
polar_alpha = B["polar_alpha"]
polar_cl    = B["polar_cl"]
polar_cd    = B["polar_cd"]

# ── Spanwise sweep (6, 8, 10) — BEM file ─────────────────────────────────────
# Only keep TSRs that have an actual per-TSR result array saved.
# tsr_CT / tsr_CP are aligned to match.
_all_span_tsrs = [int(t) for t in B["sweep_tsrs"]]
_span_mask      = [f"sweep_res_{int(t)}" in B.files for t in _all_span_tsrs]
TSR_SWEEP_SPAN  = [t for t, ok in zip(_all_span_tsrs, _span_mask) if ok]
tsr_CT_span     = B["tsr_CT"][_span_mask]
tsr_CP_span     = B["tsr_CP"][_span_mask]
sweep_data_span = {}
for t in TSR_SWEEP_SPAN:
    sweep_data_span[t] = B[f"sweep_res_{t}"]

# ── Performance sweep (wide range) — BEM file ────────────────────────────────
_perf_tsrs = _arr_or_none(B, "sweep_tsrs_perf")
if _perf_tsrs is not None and len(_perf_tsrs) > 0:
    _raw_tsrs = [float(t) for t in _perf_tsrs]
    _raw_ct   = list(B["tsr_CT_perf"])
    _raw_cp   = list(B["tsr_CP_perf"])
    _seen = {}
    for t, ct, cp in zip(_raw_tsrs, _raw_ct, _raw_cp):
        _seen[round(t, 6)] = (t, ct, cp)
    _sorted = sorted(_seen.values(), key=lambda x: x[0])
    TSR_SWEEP_PERF = [t   for t, ct, cp in _sorted]
    tsr_CT_perf    = np.array([ct for t, ct, cp in _sorted])
    tsr_CP_perf    = np.array([cp for t, ct, cp in _sorted])
else:
    TSR_SWEEP_PERF = TSR_SWEEP_SPAN
    tsr_CT_perf    = tsr_CT_span
    tsr_CP_perf    = tsr_CP_span

# ── TSR=8 specific — BEM file ────────────────────────────────────────────────
results_tsr8 = _arr_or_none(B, "results_tsr8")
res_nc       = _arr_or_none(B, "res_nc")
ct_hist_tsr8 = _arr_or_none(B, "ct_hist_tsr8")
F_tsr8       = _arr_or_none(B, "F_tsr8")

# ── Section-6: annuli sensitivity + spacing comparison — BEM file ─────────────
_annuli_N_arr   = _arr_or_none(B, "annuli_N_list")
_annuli_CT_arr  = _arr_or_none(B, "annuli_CT_list")
_annuli_CP_arr  = _arr_or_none(B, "annuli_CP_list")

# Build dicts keyed by integer N
_ANNULI_N_KEYS = [4, 8, 16, 32, 64, 160]
annuli_results = {}
for n in _ANNULI_N_KEYS:
    v = _arr_or_none(B, f"annuli_N{n}")
    if v is not None:
        annuli_results[n] = v

# Scalar CT/CP per N (for convergence curve plots)
if _annuli_N_arr is not None and _annuli_CT_arr is not None:
    annuli_CT_scalar = {int(n): float(ct) for n, ct in zip(_annuli_N_arr, _annuli_CT_arr)}
    annuli_CP_scalar = {int(n): float(cp) for n, cp in zip(_annuli_N_arr, _annuli_CP_arr)}
else:
    annuli_CT_scalar = {}
    annuli_CP_scalar = {}

# Spacing N (read from file, fallback to 20)
_spacing_N_arr = _arr_or_none(B, "spacing_N")
N_SPACING = int(_spacing_N_arr[0]) if _spacing_N_arr is not None else 20

spacing_results = {k: v for k, v in
                   [("Constant", _arr_or_none(B, "spacing_constant")),
                    ("Cosine",   _arr_or_none(B, "spacing_cosine"))] if v is not None}

def _first(*arrays):
    """Return the first non-None array from the arguments."""
    for a in arrays:
        if a is not None:
            return a
    return None

# ── Geometry nodes — opt file (with BEM file fallback for baseline) ───────────
r_base   = _first(_arr_or_none(O, "r_base"),   _arr_or_none(B, "r_base"))
c_base   = _first(_arr_or_none(O, "c_base"),   _arr_or_none(B, "c_base"))
tw_base  = _first(_arr_or_none(O, "tw_base"),  _arr_or_none(B, "tw_base"))
r_anal   = _arr_or_none(O, "r_anal");   c_anal   = _arr_or_none(O, "c_anal");   tw_anal   = _arr_or_none(O, "tw_anal")
r_cubic  = _arr_or_none(O, "r_cubic");  c_cubic  = _arr_or_none(O, "c_cubic");  tw_cubic  = _arr_or_none(O, "tw_cubic")
r_qrt    = _arr_or_none(O, "r_qrt");    c_qrt    = _arr_or_none(O, "c_qrt");    tw_qrt    = _arr_or_none(O, "tw_qrt")

# ── BEM results — opt file (with BEM file fallback for baseline) ──────────────
res_base  = _first(_arr_or_none(O, "res_base"), _arr_or_none(B, "res_base"))
res_anal  = _arr_or_none(O, "res_anal")
res_cubic = _arr_or_none(O, "res_cubic")
res_qrt   = _arr_or_none(O, "res_qrt")

# ── Scalar performance — opt file (with BEM file fallback for baseline) ───────
def _first_f(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None

CT_base  = _first_f(_getf(O, "CT_base"),  _getf(B, "CT_base"))
CP_base  = _first_f(_getf(O, "CP_base"),  _getf(B, "CP_base"))
CT_anal  = _getf(O, "CT_anal");  CP_anal  = _getf(O, "CP_anal")
CT_cubic = _getf(O, "CT_cubic"); CP_cubic = _getf(O, "CP_cubic")
CT_qrt   = _getf(O, "CT_qrt");   CP_qrt   = _getf(O, "CP_qrt")
CP_ad    = _getf(O, "CP_ad")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"  Span TSR sweep : {TSR_SWEEP_SPAN}")
print(f"  Perf TSR sweep : {TSR_SWEEP_PERF}")
print(f"  Baseline       : CT={CT_base:.4f}  CP={CP_base:.4f}" if CT_base else "  Baseline       : not in file")
print(f"  Analytical     : CT={CT_anal:.4f}  CP={CP_anal:.4f}" if CT_anal else "  Analytical     : not in file")
print(f"  Cubic poly     : CT={CT_cubic:.4f}  CP={CP_cubic:.4f}" if CT_cubic else "  Cubic poly     : not in file")
print(f"  Quartic poly   : CT={CT_qrt:.4f}  CP={CP_qrt:.4f}" if CT_qrt else "  Quartic poly   : not in file")

# =============================================================================
# 2.  COLOR SCHEME
# =============================================================================

# TSR sweep lines: 3 values -> blue, green, red  |  more -> extended palette
_TSR_3_COLORS    = ["#0000ff", "#2ca02c", "#d62728"]
_TSR_MANY_COLORS = ["#000000", "#2ca02c", "#d62728", "#0000ff",
                    "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

def _tsr_color(idx, n):
    return _TSR_3_COLORS[idx % 3] if n <= 3 else _TSR_MANY_COLORS[idx % len(_TSR_MANY_COLORS)]

# Design comparison: black=Baseline, green=Analytical, red=Cubic, blue=Quartic
# Same palette for 3 or 4 designs
_DESIGN_COLORS = {
    "Baseline":     "#000000",   # black
    "Analytical":   "#2ca02c",   # green
    "Cubic poly":   "#d62728",   # red
    "Quartic poly": "#0000FF",   # blue
}

def _design_color(label, n=None):   # n kept for API compatibility, unused
    return _DESIGN_COLORS.get(label, "#888888")

# annuli and spacing section colors
_ANNULI_COLS  = ["#000000", "#2ca02c", "#d62728"]
_SPACING_COLS = {"Constant": "#000000", "Cosine": "#2ca02c"}

# =============================================================================
# 3.  HELPER FUNCTIONS
# =============================================================================

def find_optimal_alpha():
    alphas = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_v   = np.interp(alphas, polar_alpha, polar_cl)
    cd_v   = np.interp(alphas, polar_alpha, polar_cd)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cd_v > 1e-6, cl_v / cd_v, 0.0)
    idx = int(np.argmax(ratio))
    return float(alphas[idx]), float(cl_v[idx]), float(cd_v[idx])


def _skip(name, reason):
    print(f"  [SKIP] {name} — {reason}")
    plt.close("all")


def _build_designs(r_R_dense):
    """
    Assemble available designs for comparison plots.
    Returns list of (label, c_dense, tw_dense, res_arr, CT, CP).
    Chord/twist interpolated directly from saved node arrays.
    Absent designs are silently omitted with an [INFO] message.
    """
    designs = []
    if r_base is not None:
        designs.append(("Baseline",
                         np.interp(r_R_dense, r_base/Radius, c_base),
                         np.interp(r_R_dense, r_base/Radius, tw_base),
                         res_base, CT_base, CP_base))
    if r_anal is not None and res_anal is not None:
        designs.append(("Analytical",
                         np.interp(r_R_dense, r_anal/Radius, c_anal),
                         np.interp(r_R_dense, r_anal/Radius, tw_anal),
                         res_anal, CT_anal, CP_anal))
    else:
        print("  [INFO] analytical design absent — skipping that curve")
    if r_cubic is not None and res_cubic is not None:
        designs.append(("Cubic poly",
                         np.interp(r_R_dense, r_cubic/Radius, c_cubic),
                         np.interp(r_R_dense, r_cubic/Radius, tw_cubic),
                         res_cubic, CT_cubic, CP_cubic))
    else:
        print("  [INFO] cubic poly absent — skipping that curve")
    if r_qrt is not None and res_qrt is not None:
        designs.append(("Quartic poly",
                         np.interp(r_R_dense, r_qrt/Radius, c_qrt),
                         np.interp(r_R_dense, r_qrt/Radius, tw_qrt),
                         res_qrt, CT_qrt, CP_qrt))
    else:
        print("  [INFO] quartic poly absent — skipping that curve")
    return designs


def _tangential_induction_comparison(designs):
    """
    Plot tangential (azimuthal) induction factor a' vs r/R for all designs.
    res_arr column 1 = a'.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,1], color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$a'$ [-]")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def _chord_cl_comparison(r_R_dense, designs):
    """
    Two-panel figure in the style of plot 9a (twin y-axes per panel).

    Left panel  — Chord [m] on left y-axis, for Analytical (green) and Quartic (blue).
                  A second right y-axis shows Cl for the same two designs using
                  dashed lines, so chord and Cl for each design share the same panel
                  and color.

    Right panel — Same information with Cl on the left y-axis and Chord on the right,
                  giving a view focused on the aerodynamic quantity.

    This mirrors the 9a convention (solid = primary left quantity,
    dashed = secondary right quantity, same color per design).
    """
    targets = {"Analytical", "Quartic poly"}
    subset  = [(lbl, c_d, tw_d, res, CT, CP)
               for (lbl, c_d, tw_d, res, CT, CP) in designs
               if lbl in targets]

    if len(subset) == 0:
        print("  [INFO] chord/Cl comparison skipped — neither analytical nor quartic available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: primary = Chord, secondary = Cl ──────────────────────────
    ax_left   = axes[0]
    ax_left_r = ax_left.twinx()
    for lbl, c_d, _, res, *_ in subset:
        col = _design_color(lbl)
        ax_left.plot(r_R_dense, c_d,   color=col, lw=2,       label=f"{lbl} — chord")
        ax_left_r.plot(res[:,2], res[:,8], color=col, lw=2, ls="--", label=f"{lbl} — $C_l$")
    ax_left.axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,  label=f"Min chord {CHORD_MIN} m")
    ax_left.axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root chord {CHORD_ROOT} m")
    ax_left.set_xlabel("r/R"); ax_left.set_ylabel("Chord [m]")
    ax_left_r.set_ylabel(r"$C_l$ [-]")
    ax_left.set_zorder(ax_left_r.get_zorder() + 1); ax_left.patch.set_visible(False)
    # Combine legends from both axes
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_left_r.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, fontsize=8)
    ax_left.grid(True)

    # ── Right panel: primary = Cl, secondary = Chord ─────────────────────────
    ax_right   = axes[1]
    ax_right_r = ax_right.twinx()
    for lbl, c_d, _, res, *_ in subset:
        col = _design_color(lbl)
        ax_right.plot(res[:,2], res[:,8], color=col, lw=2,       label=f"{lbl} — $C_l$")
        ax_right_r.plot(r_R_dense, c_d,  color=col, lw=2, ls="--", label=f"{lbl} — chord")
    ax_right.set_xlabel("r/R"); ax_right.set_ylabel(r"$C_l$ [-]")
    ax_right_r.set_ylabel("Chord [m]")
    ax_right.set_zorder(ax_right_r.get_zorder() + 1); ax_right.patch.set_visible(False)
    h1, l1 = ax_right.get_legend_handles_labels()
    h2, l2 = ax_right_r.get_legend_handles_labels()
    ax_right.legend(h1 + h2, l1 + l2, fontsize=8)
    ax_right.grid(True)

    fig.tight_layout()
    return fig

# =============================================================================
# 4.  SAVE / SHOW HELPER
# =============================================================================

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "plotting_plots_assignment")
os.makedirs(save_folder, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(save_folder, name), dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}")
    if SHOW_PLOTS: plt.show()
    else:          plt.close()

norm_val = 0.5 * U0**2 * Radius
n_span   = len(TSR_SWEEP_SPAN)

# =============================================================================
# 5.  PLOTS — SECTION 4.1  (alpha and inflow angle)
# =============================================================================

if PLOT_4_1 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (6, r"$\alpha$ [deg]", "4_1a_angle_of_attack_vs_rR.png"),
            (7, r"$\phi$ [deg]",   "4_1b_inflow_angle_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_1:
    _skip("PLOT_4_1", "span sweep data missing from npz")

# =============================================================================
# 6.  PLOTS — SECTION 4.2  (induction factors)
# =============================================================================

if PLOT_4_2 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",  "4_2a_axial_induction_vs_rR.png"),
            (1, r"$a'$ [-]", "4_2b_tangential_induction_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col], color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_2:
    _skip("PLOT_4_2", "span sweep data missing from npz")

# =============================================================================
# 7.  PLOTS — SECTION 4.3  (loading)
# =============================================================================

if PLOT_4_3 and sweep_data_span:
    for qty_col, ylabel, fname in [
            (3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$", "4_3a_normal_loading_Cn_vs_rR.png"),
            (4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$", "4_3b_azimuthal_loading_Ct_vs_rR.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, TSR in enumerate(TSR_SWEEP_SPAN):
            res = sweep_data_span[TSR]
            ax.plot(res[:,2], res[:,qty_col]/norm_val, color=_tsr_color(k, n_span), lw=2,
                    label=rf"$\lambda={TSR}$")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_3:
    _skip("PLOT_4_3", "span sweep data missing from npz")

# =============================================================================
# 8.  PLOTS — SECTION 4.4  (CT, CQ, CP vs TSR — broad performance sweep)
# =============================================================================

if PLOT_4_4 and len(TSR_SWEEP_PERF) > 0:
    CQ_perf = tsr_CP_perf / np.array(TSR_SWEEP_PERF, dtype=float)
    for vals, ylabel, fname, color_i in [
            (tsr_CT_perf, r"$C_T$ [-]", "4_4a_thrust_coefficient_CT_vs_TSR.png",  "blue"),
            (CQ_perf,     r"$C_Q$ [-]", "4_4b_torque_coefficient_CQ_vs_TSR.png",  "red"),
            (tsr_CP_perf, r"$C_P$ [-]", "4_4c_power_coefficient_CP_vs_TSR.png",   "green")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(TSR_SWEEP_PERF, vals, "o-", color=color_i, lw=2)
        ax.set_xlabel(r"Tip-speed ratio $\lambda$ [-]"); ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_4_4:
    _skip("PLOT_4_4", "performance sweep data missing from npz")

# =============================================================================
# 9.  PLOTS — SECTION 5  (tip correction)
# =============================================================================

if PLOT_5 and results_tsr8 is not None and res_nc is not None:
    r_R8 = results_tsr8[:,2]
    for qty_col, ylabel, fname in [
            (0, r"$a$ [-]",   "5a_axial_induction_tip_correction_comparison.png"),
            (3, r"$C_n$ [-]", "5b_normal_loading_tip_correction_comparison.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        yc  = results_tsr8[:,qty_col] if qty_col == 0 else results_tsr8[:,qty_col]/norm_val
        ync = res_nc[:,0]             if qty_col == 0 else res_nc[:,3]/norm_val
        ax.plot(r_R8,        yc,  "b-",  lw=2, label="With Prandtl correction")
        ax.plot(res_nc[:,2], ync, "r--", lw=2, label="No correction (F=1)")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)
elif PLOT_5:
    _skip("PLOT_5", "res_nc or results_tsr8 missing (run with RUN_NO_CORRECTION=True)")

# =============================================================================
# 10.  PLOTS — SECTION 6  (annuli sensitivity + spacing study)
# =============================================================================

if PLOT_6 and results_tsr8 is not None and annuli_results and spacing_results:

    _ANNULI_COLS  = {4:"#08306b", 8:"#2171b5", 16:"#6baed6",
                     32:"#bdd7e7", 64:"#fd8d3c", 160:"#d62728"}
    _SPACING_COLS = {"Constant":"#000000", "Cosine":"#2ca02c"}

    def _annuli_marker(N):
        return ("o", 4) if N <= 16 else (None, None)

    def _plot_annuli_quantity(col_idx, ylabel, fname, xlim=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for N, res_N in sorted(annuli_results.items()):
            mk, ms = _annuli_marker(N)
            ax.plot(res_N[:,2],
                    res_N[:,col_idx]/norm_val if col_idx in (3, 4) else res_N[:,col_idx],
                    color=_ANNULI_COLS.get(N, "#888888"), lw=2,
                    marker=mk, markersize=ms, label=f"N={N}")
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(*xlim)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    def _plot_spacing_quantity(col_idx, ylabel, fname, xlim=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl, res_sp in spacing_results.items():
            ax.plot(res_sp[:,2],
                    res_sp[:,col_idx]/norm_val if col_idx in (3, 4) else res_sp[:,col_idx],
                    "-o", markersize=5, color=_SPACING_COLS.get(lbl, "#888888"),
                    lw=2, label=lbl)
        ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
        if xlim: ax.set_xlim(*xlim)
        ax.legend(); ax.grid(True)
        fig.tight_layout(); save_fig(fname)

    # ══ ANNULI SENSITIVITY ════════════════════════════════════════════════════
    _plot_annuli_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                          "6a1_Cn_vs_rR_annuli_sensitivity.png")
    _plot_annuli_quantity(4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$",
                          "6a2_Ct_vs_rR_annuli_sensitivity.png")
    _plot_annuli_quantity(0, r"$a$ [-]",
                          "6a3_axial_induction_vs_rR_annuli_sensitivity.png")
    _plot_annuli_quantity(6, r"$\alpha$ [deg]",
                          "6a4_alpha_vs_rR_annuli_sensitivity.png")

    # 6a5 — CT vs N  (global convergence)
    if annuli_CT_scalar:
        _N_list = sorted(annuli_CT_scalar.keys())
        ref_CT  = annuli_CT_scalar.get(160, None)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(_N_list, [annuli_CT_scalar[n] for n in _N_list], "o-", color="#d62728", lw=2)
        if ref_CT: ax.axhline(ref_CT, color="k", ls="--", lw=0.8,
                               label=f"N=160 reference ({ref_CT:.4f})"); ax.legend()
        ax.set_xlabel("Number of annuli N"); ax.set_ylabel(r"$C_T$ [-]")
        ax.grid(True); fig.tight_layout(); save_fig("6a5_CT_vs_N_annuli_convergence.png")

    # 6a6 — CP vs N  (global convergence)
    if annuli_CP_scalar:
        _N_list = sorted(annuli_CP_scalar.keys())
        ref_CP  = annuli_CP_scalar.get(160, None)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(_N_list, [annuli_CP_scalar[n] for n in _N_list], "o-", color="#2ca02c", lw=2)
        if ref_CP: ax.axhline(ref_CP, color="k", ls="--", lw=0.8,
                               label=f"N=160 reference ({ref_CP:.4f})"); ax.legend()
        ax.set_xlabel("Number of annuli N"); ax.set_ylabel(r"$C_P$ [-]")
        ax.grid(True); fig.tight_layout(); save_fig("6a6_CP_vs_N_annuli_convergence.png")

    # 6a7 — Cn tip zoom
    _plot_annuli_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                          "6a7_Cn_tip_zoom_annuli_sensitivity.png", xlim=(0.8, 1.0))

    # ══ SPACING COMPARISON ════════════════════════════════════════════════════
    _plot_spacing_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                           "6b1_Cn_vs_rR_spacing_comparison.png")
    _plot_spacing_quantity(3, r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$",
                           "6b2_Cn_tip_zoom_spacing_comparison.png", xlim=(0.8, 1.0))
    _plot_spacing_quantity(4, r"$C_t = F_t\,/\,(½\rho U_\infty^2 R)$",
                           "6b3_Ct_vs_rR_spacing_comparison.png")
    _plot_spacing_quantity(0, r"$a$ [-]",
                           "6b4_axial_induction_vs_rR_spacing_comparison.png")
    _plot_spacing_quantity(6, r"$\alpha$ [deg]",
                           "6b5_alpha_vs_rR_spacing_comparison.png")

    # ══ ITERATION CONVERGENCE ════════════════════════════════════════════════
    if ct_hist_tsr8 is not None:
        n_show = min(60, len(ct_hist_tsr8))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(ct_hist_tsr8)+1), ct_hist_tsr8, "b-", lw=2)
        ax.set_xlim(1, n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$C_T$ [-]")
        ax.grid(True); fig.tight_layout(); save_fig("6c_CT_convergence_history.png")

        resid = np.abs(np.diff(ct_hist_tsr8))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(range(2, len(ct_hist_tsr8)+1), resid, "r-", lw=2,
                    label=r"$|C_{T,i}-C_{T,i-1}|$")
        ax.axhline(1e-5, color="k", ls="--", lw=0.8, label="Tolerance = 1e-5")
        ax.set_xlim(1, n_show); ax.set_xlabel("Iteration"); ax.set_ylabel(r"$|\Delta C_T|$")
        ax.legend(); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("6d_CT_convergence_residuals_log_scale.png")

elif PLOT_6:
    _skip("PLOT_6", "annuli/spacing data missing (run with RUN_TSR_SWEEP_SPAN=True)")

# =============================================================================
# 11.  PLOTS — SECTION 7  (stagnation pressure)
# =============================================================================

if PLOT_7 and results_tsr8 is not None:
    r_R8 = results_tsr8[:,2]
    a_R8 = results_tsr8[:,0]

    # Freestream dynamic pressure
    q_inf = 0.5 * rho * U0**2

    # Normalised stagnation pressures
    P0_12 = np.ones(len(r_R8))
    P0_34 = (1.0 - 2.0*a_R8)**2

    # Dimensional
    P0_up = q_inf * P0_12
    P0_down = q_inf * P0_34

    # Small offset to visually separate overlapping curves
    eps = 0.003 * q_inf

    # -------------------------------------------------------
    # LEFT FIGURE: four stations with small offset trick
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(r_R8, P0_up, color="#0000FF", lw=2.5,
        label=r"$P_0^{\infty,\uparrow}$ (infinity upwind)")

    ax.plot(r_R8, P0_down, color="#FF0000", lw=2.5,
            label=r"$P_0^{\infty,\downarrow}$ (infinity downwind)")

    ax.plot(r_R8, P0_up+eps, color="#000000", lw=1.8, linestyle="--", alpha=0.95,
            label=r"$P_0^{+}$ (rotor upwind)")

    ax.plot(r_R8, P0_down+eps, color="#00AA00", lw=1.8, linestyle="--", alpha=0.95,
            label=r"$P_0^{-}$ (rotor downwind)")

    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$P_0$ [Pa]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig("7_stagnation_pressure_four_stations.png")

    # -------------------------------------------------------
    # RIGHT FIGURE: paper-style stagnation pressure drop plot
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(r_R8, P0_12, color="#0000FF", lw=2.5,
            label=r"Upstream $P_0/q_\infty = 1$")

    ax.plot(r_R8, P0_34, color="#FF0000", lw=2.5,
            label=r"Downstream $P_0/q_\infty = (1-2a)^2$")

    # Shade stagnation pressure drop (energy extracted)
    ax.fill_between(
        r_R8,
        P0_34,
        P0_12,
        color="#B0B0B0",
        alpha=0.35,
        label=r"$\Delta P_0$"
    )

    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$P_0/q_\infty$ [-]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig("7_stagnation_pressure_drop.png")

elif PLOT_7:
    _skip("PLOT_7", "results_tsr8 missing (run with RUN_TSR_SWEEP_SPAN=True)")

# =============================================================================
# 12.  PLOTS — SECTION 8  (design comparison)
# =============================================================================

if PLOT_8 and res_base is not None:
    r_R_dense = np.linspace(RootLocation_R, TipLocation_R, 400)
    designs   = _build_designs(r_R_dense)
    n_d       = len(designs)

    # 8a — chord
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, c_d, *_ in designs:
        ax.plot(r_R_dense, c_d, color=_design_color(lbl), lw=2, label=lbl)
    ax.axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,  label=f"Min chord  {CHORD_MIN} m")
    ax.axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root chord  {CHORD_ROOT} m")
    ax.set_xlabel("r/R"); ax.set_ylabel("Chord [m]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8a_chord_distribution_design_comparison.png")

    # 8b — twist
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, tw_d, *_ in designs:
        ax.plot(r_R_dense, tw_d, color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel("Twist [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8b_twist_distribution_design_comparison.png")

    # 8ab — chord and twist side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, c_d, tw_d, *_ in designs:
        col = _design_color(lbl)
        axes[0].plot(r_R_dense, c_d,  color=col, lw=2, label=lbl)
        axes[1].plot(r_R_dense, tw_d, color=col, lw=2, label=lbl)
    axes[0].axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,  label=f"Min  {CHORD_MIN} m")
    axes[0].axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root  {CHORD_ROOT} m")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
    axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); save_fig("8ab_chord_and_twist_design_comparison.png")

    # 8c — axial induction
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,0], color=_design_color(lbl), lw=2, label=lbl)
    ax.axhline(1/3, color="grey", ls=":", lw=0.8, label="a = 1/3  (Betz)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8c_axial_induction_design_comparison.png")

    # 8c2 — tangential induction  (NEW)
    fig = _tangential_induction_comparison(designs)
    save_fig("8c2_tangential_induction_design_comparison.png")

    # 8d — normal loading
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,3]/norm_val, color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8d_normal_loading_design_comparison.png")

    # 8e — angle of attack
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,6], color=_design_color(lbl), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$\alpha$ [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8e_angle_of_attack_design_comparison.png")

    # 8f — performance bar chart
    a_ad_  = 0.5 * (1.0 - np.sqrt(1.0 - CT_TARGET))
    cp_ad_ = CP_ad if CP_ad is not None else 4.0*a_ad_*(1.0-a_ad_)**2
    labels_b = [d[0] for d in designs] + ["Actuator disk"]
    cp_vals  = [d[5] for d in designs] + [cp_ad_]
    ct_vals  = [d[4] for d in designs] + [CT_TARGET]
    eff_vals = [v / cp_ad_ for v in cp_vals]
    bar_cols = [_design_color(d[0]) for d in designs] + ["#aaaaaa"]
    x = np.arange(len(labels_b)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x-w, cp_vals,  w, label=r"$C_P$",
                color=[c+"cc" for c in bar_cols])
    b2 = ax.bar(x,   ct_vals,  w, label=r"$C_T$",
                color=bar_cols, alpha=0.6)
    b3 = ax.bar(x+w, eff_vals, w, label=r"$C_P/C_{P,\mathrm{AD}}$",
                color=bar_cols, hatch="//", alpha=0.85)
    for bars in [b1, b2, b3]:
        for bar in bars: bar.set_edgecolor("white"); bar.set_linewidth(0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels_b, rotation=15, ha="right")
    ax.set_ylabel("Coefficient [-]"); ax.legend()
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=7.5)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=7.5)
    ax.bar_label(b3, fmt="%.3f", padding=3, fontsize=7.5)
    ax.grid(True, axis="y")
    fig.tight_layout(); save_fig("8f_performance_comparison_all_designs.png")

elif PLOT_8:
    _skip("PLOT_8", "res_base missing from npz")

# =============================================================================
# 13.  PLOTS — SECTION 9  (Cl, chord and circulation)
# =============================================================================
#
# 9a_combined : Cl (solid) + chord (dashed) on twin axis for all three
#               optimised designs (Analytical, Cubic poly, Quartic poly)
#               overlaid on a single figure.  Chord lines use a lighter
#               (desaturated) variant of each design's color.
# 9a_<design> : same twin-axis plot produced individually for each design.
# 9b          : Cl·c circulation proxy, all optimised designs overlaid.

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def _lighten(hex_color, amount=0.45):
    """Return a lighter version of hex_color by blending toward white."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    r2 = r + (1 - r) * amount
    g2 = g + (1 - g) * amount
    b2 = b + (1 - b) * amount
    return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"

def _make_9a_axes(ax, r_mid, cl, chord, col, lbl):
    """Draw Cl (solid) + chord (dashed, lighter) on twin-axis for one design."""
    col_chord = _lighten(col, 0.45)
    ax.plot(r_mid, cl, color=col, lw=2, ls="-",  label=rf"{lbl} — $C_l$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(r_mid, chord, color=col_chord, lw=2, ls="--",
             label=f"{lbl} — chord")
    ax2.set_ylabel("Chord [m]")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)

if PLOT_9:
    # Only the three optimised designs (no baseline)
    _opt_lbls = ["Analytical", "Cubic poly", "Quartic poly"]
    _opt_data  = []   # (lbl, r_mid, cl, chord)
    for lbl, r_arr, c_arr, res_arr in [
            ("Analytical",   r_anal,  c_anal,  res_anal),
            ("Cubic poly",   r_cubic, c_cubic, res_cubic),
            ("Quartic poly", r_qrt,   c_qrt,   res_qrt)]:
        if res_arr is not None and r_arr is not None and res_arr.shape[1] >= 9:
            r_mid = res_arr[:,2]
            cl    = res_arr[:,8]
            chord = np.interp(r_mid, r_arr/Radius, c_arr)
            _opt_data.append((lbl, r_mid, cl, chord))

    if _opt_data:
        # ── 9a combined — all three optimised designs on one figure ──────────
        fig, ax = plt.subplots(figsize=(9, 5))
        ax2_combined = ax.twinx()
        for lbl, r_mid, cl, chord in _opt_data:
            col       = _design_color(lbl)
            col_chord = _lighten(col, 0.45)
            ax.plot(r_mid, cl,    color=col,       lw=2, ls="-",
                    label=rf"{lbl} — $C_l$")
            ax2_combined.plot(r_mid, chord, color=col_chord, lw=2, ls="--",
                    label=f"{lbl} — chord")
        ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
        ax2_combined.set_ylabel("Chord [m]")
        ax.set_zorder(ax2_combined.get_zorder() + 1); ax.patch.set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2_combined.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)
        fig.tight_layout()
        save_fig("9a_combined_Cl_and_chord_optimised_designs.png")

        # ── 9a individual — one file per optimised design ────────────────────
        for lbl, r_mid, cl, chord in _opt_data:
            col  = _design_color(lbl)
            fig, ax = plt.subplots(figsize=(9, 5))
            _make_9a_axes(ax, r_mid, cl, chord, col, lbl)
            fig.tight_layout()
            fname = "9a_" + lbl.lower().replace(" ", "_") + "_Cl_and_chord.png"
            save_fig(fname)

        # ── 9b circulation proxy — all optimised designs overlaid ────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        for lbl, r_mid, cl, chord in _opt_data:
            ax.plot(r_mid, cl*chord, color=_design_color(lbl), lw=2, label=lbl)
        ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l \cdot c$  [m]"); ax.grid(True)
        ax.legend()
        fig.tight_layout()
        save_fig("9b_circulation_proxy_Cl_times_chord_optimised_designs.png")

    else:
        _skip("PLOT_9", "no optimised design results with Cl data available in npz")

# 9c — chord + Cl side-by-side, analytical vs quartic (twin-axis style)
if PLOT_9_CMP:
    if not (PLOT_8 and res_base is not None):
        r_R_dense = np.linspace(RootLocation_R, TipLocation_R, 400)
        designs   = _build_designs(r_R_dense)
    if len(designs) > 0:
        fig = _chord_cl_comparison(r_R_dense, designs)
        if fig is not None:
            save_fig("9c_chord_and_Cl_analytical_vs_quartic_comparison.png")
    else:
        _skip("PLOT_9_CMP", "no design data available in npz")

# =============================================================================
# 15.  PLOTS — SECTION 10  (Cl/Cd polar and glide ratio)
# =============================================================================
#
# For each plot type (10a polar, 10b glide ratio):
#   • one combined figure showing all available designs together
#   • one individual figure per optimised design (Analytical, Cubic, Quartic)
#
# Scatter points are colored by r/R via viridis.  Each design uses its
# predefined color as the edge color and a unique marker shape.

if PLOT_10:
    _pol_all = []    # all designs incl. baseline for combined plot
    _pol_opt = []    # optimised only for individual plots
    for lbl, res_arr in [("Baseline",     res_base),
                          ("Analytical",   res_anal),
                          ("Cubic poly",   res_cubic),
                          ("Quartic poly", res_qrt)]:
        if res_arr is not None and res_arr.shape[1] >= 10:
            _pol_all.append((lbl, res_arr))
            if lbl != "Baseline":
                _pol_opt.append((lbl, res_arr))

    if _pol_all:
        alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
        alphas_d = np.linspace(polar_alpha[0], polar_alpha[-1], 500)
        cl_d  = np.interp(alphas_d, polar_alpha, polar_cl)
        cd_d  = np.interp(alphas_d, polar_alpha, polar_cd)
        ld_d  = cl_d / np.maximum(cd_d, 1e-8)
        _norm = mcolors.Normalize(vmin=RootLocation_R, vmax=TipLocation_R)
        _cmap = cm.viridis
        _MARKERS = {"Baseline":"o", "Analytical":"s",
                    "Cubic poly":"^", "Quartic poly":"D"}

        def _draw_polar(ax, designs, title_lbl=None):
            """Draw DU95W180 polar + operating point scatter on ax."""
            ax.plot(polar_cd, polar_cl, "k-", lw=1.5,
                    label="DU95W180 polar", zorder=1)
            for lbl, res_arr in designs:
                sc = ax.scatter(
                    res_arr[:,9], res_arr[:,8],
                    c=res_arr[:,2], cmap=_cmap, norm=_norm,
                    s=30, marker=_MARKERS.get(lbl, "o"), zorder=5,
                    label=lbl)
            plt.colorbar(sc, ax=ax, label="r/R")
            ax.set_xlabel(r"$C_d$ [-]"); ax.set_ylabel(r"$C_l$ [-]")
            ax.legend(fontsize=8); ax.grid(True)

        def _draw_glide(ax, designs, alpha_opt, cl_opt, cd_opt):
            """Draw glide-ratio curve + operating point scatter on ax."""
            ax.plot(alphas_d, ld_d, "k-", lw=1.5,
                    label=r"$C_l/C_d$  DU95W180", zorder=1)
            ax.axvline(alpha_opt, color="k", ls="--", lw=0.8,
                       label=rf"$\alpha_{{opt}}={alpha_opt:.1f}°$,"
                             rf"  $(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
            for lbl, res_arr in designs:
                ld_ops = (np.interp(res_arr[:,6], polar_alpha, polar_cl)
                          / np.maximum(
                              np.interp(res_arr[:,6], polar_alpha, polar_cd), 1e-8))
                ax.scatter(
                    res_arr[:,6], ld_ops,
                    c=res_arr[:,2], cmap=_cmap, norm=_norm,
                    s=30, marker=_MARKERS.get(lbl, "o"), zorder=5,
                    label=lbl)
            plt.colorbar(ax.collections[-1], ax=ax, label="r/R")
            ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_l/C_d$ [-]")
            ax.legend(fontsize=8); ax.grid(True)

        # ── 10a combined — all designs ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_polar(ax, _pol_all)
        fig.tight_layout()
        save_fig("10a_combined_Cl_Cd_polar_all_designs.png")

        # ── 10a individual — one per optimised design ─────────────────────────
        for lbl, res_arr in _pol_opt:
            fig, ax = plt.subplots(figsize=(9, 5))
            _draw_polar(ax, [(lbl, res_arr)])
            fig.tight_layout()
            fname = "10a_" + lbl.lower().replace(" ", "_") + "_Cl_Cd_polar.png"
            save_fig(fname)

        # ── 10b combined — all designs ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_glide(ax, _pol_all, alpha_opt, cl_opt, cd_opt)
        fig.tight_layout()
        save_fig("10b_combined_glide_ratio_vs_alpha_all_designs.png")

        # ── 10b individual — one per optimised design ─────────────────────────
        for lbl, res_arr in _pol_opt:
            fig, ax = plt.subplots(figsize=(9, 5))
            _draw_glide(ax, [(lbl, res_arr)], alpha_opt, cl_opt, cd_opt)
            fig.tight_layout()
            fname = "10b_" + lbl.lower().replace(" ", "_") + "_glide_ratio_vs_alpha.png"
            save_fig(fname)

    else:
        _skip("PLOT_10", "no design BEM results with Cd data available in npz")

# =============================================================================
# 16.  EXTRA — STAGNATION PRESSURE FOR OPTIMISED DESIGNS
# =============================================================================
#
# Produces the same two Section-7 style plots for:
#   • Analytical
#   • Cubic poly
#   • Quartic poly
#
# Saved files:
#   7_opt_analytical_stagnation_pressure_four_stations.png
#   7_opt_analytical_stagnation_pressure_drop.png
#   7_opt_cubic_poly_stagnation_pressure_four_stations.png
#   7_opt_cubic_poly_stagnation_pressure_drop.png
#   7_opt_quartic_poly_stagnation_pressure_four_stations.png
#   7_opt_quartic_poly_stagnation_pressure_drop.png
# =============================================================================

def _plot_stagnation_pressure_for_design(res_arr, design_name, file_tag):
    """
    Make the two stagnation-pressure plots for a single design result array.

    Parameters
    ----------
    res_arr : ndarray
        BEM result array with:
        col 0 = axial induction a
        col 2 = r/R
    design_name : str
        Name used in legend labels only.
    file_tag : str
        Safe filename tag, e.g. 'analytical', 'cubic_poly', 'quartic_poly'.
    """
    if res_arr is None:
        print(f"  [INFO] stagnation-pressure plot skipped for {design_name} — result array missing")
        return

    r_R = res_arr[:, 2]
    a_R = res_arr[:, 0]

    # Freestream dynamic pressure
    q_inf = 0.5 * rho * U0**2

    # Normalised stagnation pressures
    P0_12 = np.ones(len(r_R))
    P0_34 = (1.0 - 2.0 * a_R) ** 2

    # Dimensional
    P0_up = q_inf * P0_12
    P0_down = q_inf * P0_34

    # Small offset to visually separate overlapping curves
    eps = 0.003 * q_inf

    # -------------------------------------------------------------------------
    # Figure 1: four stations
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(r_R, P0_up, color="#0000FF", lw=2.5,
            label=rf"$P_0^{{\infty,\uparrow}}$ ({design_name}, infinity upwind)")

    ax.plot(r_R, P0_down, color="#FF0000", lw=2.5,
            label=rf"$P_0^{{\infty,\downarrow}}$ ({design_name}, infinity downwind)")

    ax.plot(r_R, P0_up + eps, color="#000000", lw=1.8, linestyle="--", alpha=0.95,
            label=rf"$P_0^{{+}}$ ({design_name}, rotor upwind)")

    ax.plot(r_R, P0_down + eps, color="#00AA00", lw=1.8, linestyle="--", alpha=0.95,
            label=rf"$P_0^{{-}}$ ({design_name}, rotor downwind)")

    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$P_0$ [Pa]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(f"7_opt_{file_tag}_stagnation_pressure_four_stations.png")

    # -------------------------------------------------------------------------
    # Figure 2: paper-style stagnation pressure drop
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(r_R, P0_12, color="#0000FF", lw=2.5,
            label=r"Upstream $P_0/q_\infty = 1$")

    ax.plot(r_R, P0_34, color="#FF0000", lw=2.5,
            label=r"Downstream $P_0/q_\infty = (1-2a)^2$")

    ax.fill_between(
        r_R,
        P0_34,
        P0_12,
        color="#B0B0B0",
        alpha=0.35,
        label=r"$\Delta P_0$"
    )

    ax.set_xlabel("r/R")
    ax.set_ylabel(r"$P_0/q_\infty$ [-]")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(f"7_opt_{file_tag}_stagnation_pressure_drop.png")


# Run for the three optimised designs
_plot_stagnation_pressure_for_design(res_anal,  "Analytical",   "analytical")
_plot_stagnation_pressure_for_design(res_cubic, "Cubic poly",   "cubic_poly")
_plot_stagnation_pressure_for_design(res_qrt,   "Quartic poly", "quartic_poly")

# =============================================================================
print("\n" + "="*60)
print(f"ALL PLOTS SAVED TO: {save_folder}")
print("="*60)
print("\nFile list:")
for f in sorted(os.listdir(save_folder)):
    if f.endswith(".png"):
        print(f"  {f}")



