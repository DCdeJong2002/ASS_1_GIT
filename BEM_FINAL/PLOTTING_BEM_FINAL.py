"""
plot_results.py  —  Standalone plotting script
================================================
Loads full_bem_results.npz (produced by assignment.py) and
reproduces all assignment plots without re-running any BEM.

Usage
-----
    python plot_results.py                           # uses full_bem_results.npz in cwd
    python plot_results.py path/to/results.npz       # explicit path

Plots saved to ./plots_assignment/
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
PLOT_4_1 = True   # alpha and inflow angle vs r/R        (needs span sweep in npz)
PLOT_4_2 = True   # axial and tangential induction vs r/R (needs span sweep in npz)
PLOT_4_3 = True   # Cn and Ct loading vs r/R              (needs span sweep in npz)
PLOT_4_4 = True   # CT, CQ, CP vs TSR  (broad perf sweep) (needs perf sweep in npz)
PLOT_5   = True   # tip correction influence               (needs res_nc in npz)
PLOT_6   = True   # annuli count, spacing, convergence     (needs annuli data in npz)
PLOT_7   = True   # stagnation pressure                    (needs results_tsr8 in npz)
PLOT_8   = True   # all designs — chord, twist, induction, loading, alpha, performance
                  #   sub-plots produced when PLOT_8=True:
                  #     8a  chord distribution
                  #     8b  twist distribution
                  #     8ab chord and twist combined
                  #     8c  axial induction
                  #     8d  normal loading
                  #     8e  angle of attack
                  #     8f  performance bar chart
PLOT_9   = True   # Cl and chord relation (analytical optimum)
PLOT_10  = True   # Cl/Cd polar with operating points

# =============================================================================
# 1.  LOAD RESULTS
# =============================================================================

npz_path = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\douwe\AE4135-Rotor-wake\ASS_1_GIT\BEM_FINAL\full_bem_results.npz"

if not os.path.exists(npz_path):
    raise FileNotFoundError(
        f"Cannot find '{npz_path}'.\n"
        "Run assignment.py first to generate full_bem_results.npz.")

print(f"Loading results from: {npz_path}")
D = np.load(npz_path, allow_pickle=False)

def _get(key, default=None):
    try:    return D[key]
    except KeyError: return default

def _getf(key, default=None):
    v = _get(key)
    return float(v) if v is not None else default

def _arr_or_none(key):
    """Return array if present and non-empty, else None."""
    v = _get(key)
    return v if (v is not None and v.size > 0) else None

# ── Configuration scalars ────────────────────────────────────────────────────
Radius         = _getf("cfg_Radius",         50.0)
NBlades        = int(_getf("cfg_NBlades",    3))
U0             = _getf("cfg_U0",             10.0)
rho            = _getf("cfg_rho",            1.0)      # matches assignment.py
RootLocation_R = _getf("cfg_RootLocation_R", 0.2)
TipLocation_R  = _getf("cfg_TipLocation_R",  1.0)
Pitch          = _getf("cfg_Pitch",          -2.0)
CHORD_ROOT     = _getf("cfg_CHORD_ROOT",     3.4)
CHORD_MIN      = _getf("cfg_CHORD_MIN",      0.3)
CT_TARGET      = _getf("cfg_CT_TARGET",      0.75)
TSR_DESIGN     = _getf("cfg_TSR_DESIGN",     8.0)
DELTA_R_R      = _getf("cfg_DELTA_R_R",      0.005)

# ── Polar ────────────────────────────────────────────────────────────────────
polar_alpha = D["polar_alpha"]
polar_cl    = D["polar_cl"]
polar_cd    = D["polar_cd"]

# ── Spanwise sweep (6, 8, 10) — used for all r/R plots ───────────────────────
TSR_SWEEP_SPAN = [int(t) for t in D["sweep_tsrs"]]
tsr_CT_span    = D["tsr_CT"]
tsr_CP_span    = D["tsr_CP"]
sweep_data_span = {int(D["sweep_tsrs"][i]): D[f"sweep_res_{int(D['sweep_tsrs'][i])}"]
                   for i in range(len(D["sweep_tsrs"]))}

# ── Performance sweep (wide range) — used for CT/CQ/CP vs TSR ────────────────
_perf_tsrs = _arr_or_none("sweep_tsrs_perf")
if _perf_tsrs is not None and len(_perf_tsrs) > 0:
    TSR_SWEEP_PERF = [int(t) for t in _perf_tsrs]
    tsr_CT_perf    = D["tsr_CT_perf"]
    tsr_CP_perf    = D["tsr_CP_perf"]
else:
    # Fall back to span sweep if perf sweep was not saved
    TSR_SWEEP_PERF = TSR_SWEEP_SPAN
    tsr_CT_perf    = tsr_CT_span
    tsr_CP_perf    = tsr_CP_span

# ── TSR=8 specific ───────────────────────────────────────────────────────────
results_tsr8 = _arr_or_none("results_tsr8")
res_nc       = _arr_or_none("res_nc")
ct_hist_tsr8 = _arr_or_none("ct_hist_tsr8")
F_tsr8       = _arr_or_none("F_tsr8")

# ── Section-6 pre-computed ───────────────────────────────────────────────────
annuli_results  = {k: v for k, v in
                   [(8,   _arr_or_none("annuli_N8")),
                    (20,  _arr_or_none("annuli_N20")),
                    (100, _arr_or_none("annuli_N100"))] if v is not None}
spacing_results = {k: v for k, v in
                   [("Constant", _arr_or_none("spacing_constant")),
                    ("Cosine",   _arr_or_none("spacing_cosine"))] if v is not None}

# ── Geometry nodes ───────────────────────────────────────────────────────────
r_base   = _arr_or_none("r_base");   c_base   = _arr_or_none("c_base");   tw_base   = _arr_or_none("tw_base")
r_anal   = _arr_or_none("r_anal");   c_anal   = _arr_or_none("c_anal");   tw_anal   = _arr_or_none("tw_anal")
r_cubic  = _arr_or_none("r_cubic");  c_cubic  = _arr_or_none("c_cubic");  tw_cubic  = _arr_or_none("tw_cubic")
r_qrt    = _arr_or_none("r_qrt");    c_qrt    = _arr_or_none("c_qrt");    tw_qrt    = _arr_or_none("tw_qrt")

# ── BEM results ──────────────────────────────────────────────────────────────
res_base  = _arr_or_none("res_base")
res_anal  = _arr_or_none("res_anal")
res_cubic = _arr_or_none("res_cubic")
res_qrt   = _arr_or_none("res_qrt")

# ── Scalar performance ───────────────────────────────────────────────────────
CT_base  = _getf("CT_base");  CP_base  = _getf("CP_base")
CT_anal  = _getf("CT_anal");  CP_anal  = _getf("CP_anal")
CT_cubic = _getf("CT_cubic"); CP_cubic = _getf("CP_cubic")
CT_qrt   = _getf("CT_qrt");   CP_qrt   = _getf("CP_qrt")
CP_ad    = _getf("CP_ad")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"  Span TSR sweep : {TSR_SWEEP_SPAN}")
print(f"  Perf TSR sweep : {TSR_SWEEP_PERF}")
print(f"  Baseline       : CT={CT_base:.4f}  CP={CP_base:.4f}" if CT_base else "  Baseline       : not in file")
print(f"  Analytical     : CT={CT_anal:.4f}  CP={CP_anal:.4f}" if CT_anal else "  Analytical     : not in file")
print(f"  Cubic poly     : CT={CT_cubic:.4f}  CP={CP_cubic:.4f}" if CT_cubic else "  Cubic poly     : not in file")
print(f"  Quartic poly   : CT={CT_qrt:.4f}  CP={CP_qrt:.4f}" if CT_qrt else "  Quartic poly   : not in file")

# =============================================================================
# 2.  COLOR SCHEME  (identical to assignment.py)
# =============================================================================

_TSR_3_COLORS    = ["#0000ff", "#2ca02c", "#d62728"]
_TSR_MANY_COLORS = ["#000000", "#2ca02c", "#d62728", "#0000ff",
                    "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

def _tsr_color(idx, n):
    return _TSR_3_COLORS[idx % 3] if n <= 3 else _TSR_MANY_COLORS[idx % len(_TSR_MANY_COLORS)]

_DESIGN_COLORS_3 = {"Baseline":"#2ca02c","Analytical":"#1f77b4",
                    "Cubic poly":"#d62728","Quartic poly":"#ff7f0e"}
_DESIGN_COLORS_4 = {"Baseline":"#1f77b4","Analytical":"#2ca02c",
                    "Cubic poly":"#d62728","Quartic poly":"#ff7f0e"}

def _design_color(label, n):
    d = _DESIGN_COLORS_3 if n <= 3 else _DESIGN_COLORS_4
    return d.get(label, "#888888")

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

# =============================================================================
# 4.  SAVE / SHOW HELPER  (folder matches assignment.py)
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
    for vals, ylabel, fname in [
            (tsr_CT_perf, r"$C_T$ [-]", "4_4a_thrust_coefficient_CT_vs_TSR.png"),
            (CQ_perf,     r"$C_Q$ [-]", "4_4b_torque_coefficient_CQ_vs_TSR.png"),
            (tsr_CP_perf, r"$C_P$ [-]", "4_4c_power_coefficient_CP_vs_TSR.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(TSR_SWEEP_PERF, vals, "o-", color="#1f77b4", lw=2)
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
# 10.  PLOTS — SECTION 6  (annuli, spacing, convergence)
# =============================================================================

if PLOT_6 and results_tsr8 is not None and annuli_results and spacing_results:
    annuli_cols = ["#000000", "#2ca02c", "#d62728"]

    # 6a — number of annuli
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (N, res_N) in enumerate(sorted(annuli_results.items())):
        ax.plot(res_N[:,2], res_N[:,3]/norm_val, "-o", markersize=4,
                color=annuli_cols[idx % 3], lw=2, label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6a_normal_loading_influence_of_number_of_annuli.png")

    # 6b — spacing method
    spacing_cols = {"Constant": "#000000", "Cosine": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, res_sp in spacing_results.items():
        ax.plot(res_sp[:,2], res_sp[:,3]/norm_val, "-o", markersize=5,
                color=spacing_cols.get(lbl, "#888888"), lw=2, label=lbl)
    ax.set_xlim(0.85, 1.01); ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("6b_normal_loading_spacing_method_comparison.png")

    # 6c — CT convergence history
    if ct_hist_tsr8 is not None:
        n_show = min(60, len(ct_hist_tsr8))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(ct_hist_tsr8)+1), ct_hist_tsr8, "b-", lw=2)
        ax.set_xlim(1, n_show)
        ax.set_xlabel("Iteration"); ax.set_ylabel(r"$C_T$ [-]")
        ax.grid(True)
        fig.tight_layout(); save_fig("6c_CT_convergence_history.png")

        # 6d — residuals (log scale)
        resid = np.abs(np.diff(ct_hist_tsr8))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(range(2, len(ct_hist_tsr8)+1), resid, "r-", lw=2,
                    label=r"$|C_{T,i}-C_{T,i-1}|$")
        ax.axhline(1e-5, color="k", ls="--", lw=0.8, label="Tolerance = 1e-5")
        ax.set_xlim(1, n_show)
        ax.set_xlabel("Iteration"); ax.set_ylabel(r"$|\Delta C_T|$")
        ax.legend(); ax.grid(True, which="both")
        fig.tight_layout(); save_fig("6d_CT_convergence_residuals_log_scale.png")

elif PLOT_6:
    _skip("PLOT_6", "annuli/spacing data missing (run with RUN_TSR_SWEEP_SPAN=True)")

# =============================================================================
# 11.  PLOTS — SECTION 7  (stagnation pressure)
# =============================================================================

if PLOT_7 and results_tsr8 is not None and F_tsr8 is not None:
    r_R8    = results_tsr8[:,2]
    P0_up   = 0.5 * rho * U0**2 * np.ones(len(r_R8))
    dP0     = 2.0*rho*U0**2 * results_tsr8[:,0] * F_tsr8 * (
               1.0 - results_tsr8[:,0] * F_tsr8)
    P0_down = P0_up - dP0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r_R8, P0_up,   "b-",  lw=2,   label=r"$P_0^{\infty,\mathrm{up}}$  (stat. 1)")
    ax.plot(r_R8, P0_up,   "b--", lw=1.5, alpha=0.6,
            label=r"$P_0^+$  rotor upwind  (stat. 2)")
    ax.plot(r_R8, P0_down, "r--", lw=1.5, alpha=0.6,
            label=r"$P_0^-$  rotor downwind  (stat. 3)")
    ax.plot(r_R8, P0_down, "r-",  lw=2,
            label=r"$P_0^{\infty,\mathrm{down}}$  (stat. 4)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"Stagnation pressure $P_0$ [Pa]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("7_stagnation_pressure_four_stations.png")
elif PLOT_7:
    _skip("PLOT_7", "results_tsr8 or F_tsr8 missing (run with RUN_TSR_SWEEP_SPAN=True)")

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
        ax.plot(r_R_dense, c_d, color=_design_color(lbl, n_d), lw=2, label=lbl)
    ax.axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,   label=f"Min chord  {CHORD_MIN} m")
    ax.axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8,  label=f"Root chord  {CHORD_ROOT} m")
    ax.set_xlabel("r/R"); ax.set_ylabel("Chord [m]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8a_chord_distribution_design_comparison.png")

    # 8b — twist
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, tw_d, *_ in designs:
        ax.plot(r_R_dense, tw_d, color=_design_color(lbl, n_d), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel("Twist [deg]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8b_twist_distribution_design_comparison.png")

    # 8ab — chord and twist side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, c_d, tw_d, *_ in designs:
        col = _design_color(lbl, n_d)
        axes[0].plot(r_R_dense, c_d,  color=col, lw=2, label=lbl)
        axes[1].plot(r_R_dense, tw_d, color=col, lw=2, label=lbl)
    axes[0].axhline(CHORD_MIN,  color="grey", ls=":",  lw=1,   label=f"Min  {CHORD_MIN} m")
    axes[0].axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8,  label=f"Root  {CHORD_ROOT} m")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
    axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); save_fig("8ab_chord_and_twist_design_comparison.png")

    # 8c — axial induction
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,0], color=_design_color(lbl, n_d), lw=2, label=lbl)
    ax.axhline(1/3, color="grey", ls=":", lw=0.8, label="a = 1/3  (Betz)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8c_axial_induction_design_comparison.png")

    # 8d — normal loading
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,3]/norm_val, color=_design_color(lbl, n_d), lw=2, label=lbl)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½\rho U_\infty^2 R)$")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("8d_normal_loading_design_comparison.png")

    # 8e — angle of attack
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:,2], res[:,6], color=_design_color(lbl, n_d), lw=2, label=lbl)
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
    bar_cols = [_design_color(d[0], n_d) for d in designs] + ["#aaaaaa"]
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
# 13.  PLOTS — SECTION 9  (Cl and chord relation)
# =============================================================================

if PLOT_9 and res_anal is not None and r_anal is not None:
    r_R_am = res_anal[:,2]
    cl_am  = res_anal[:,8]
    c_am   = np.interp(r_R_am, r_anal/Radius, c_anal)

    # 9a — Cl with chord on twin axis
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r_R_am, cl_am, "b-", lw=2, label=r"$C_l$")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l$ [-]"); ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(r_R_am, c_am, "r--", lw=2, label="Chord [m]")
    ax2.set_ylabel("Chord [m]", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)
    fig.tight_layout(); save_fig("9a_lift_coefficient_and_chord_vs_rR.png")

    # 9b — circulation proxy Cl·c
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r_R_am, cl_am*c_am, color="#2ca02c", lw=2)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_l \cdot c$  [m]"); ax.grid(True)
    fig.tight_layout(); save_fig("9b_circulation_proxy_Cl_times_chord_vs_rR.png")

elif PLOT_9:
    _skip("PLOT_9", "analytical result missing (run with RUN_ANALYTICAL=True)")

# =============================================================================
# 14.  PLOTS — SECTION 10  (Cl/Cd polar)
# =============================================================================

if PLOT_10 and res_anal is not None:
    alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
    alphas_d = np.linspace(polar_alpha[0], polar_alpha[-1], 500)
    cl_d = np.interp(alphas_d, polar_alpha, polar_cl)
    cd_d = np.interp(alphas_d, polar_alpha, polar_cd)
    ld_d = cl_d / np.maximum(cd_d, 1e-8)
    r_R_am = res_anal[:,2]

    # 10a — Cl vs Cd polar
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(polar_cd, polar_cl, "k-", lw=1.5, label="DU95W180 polar")
    sc = ax.scatter(res_anal[:,9], res_anal[:,8],
                    c=r_R_am, cmap="viridis", s=30, zorder=5,
                    label="Analytical opt — operating points")
    plt.colorbar(sc, ax=ax, label="r/R")
    ax.set_xlabel(r"$C_d$ [-]"); ax.set_ylabel(r"$C_l$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("10a_Cl_Cd_polar_with_operating_points.png")

    # 10b — Cl/Cd vs alpha
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas_d, ld_d, "k-", lw=1.5, label=r"$C_l/C_d$")
    ax.axvline(alpha_opt, color="r", ls="--",
               label=rf"$\alpha_{{opt}}={alpha_opt:.1f}°$"
                     rf",  $(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
    ld_ops = (np.interp(res_anal[:,6], polar_alpha, polar_cl)
              / np.maximum(np.interp(res_anal[:,6], polar_alpha, polar_cd), 1e-8))
    sc2 = ax.scatter(res_anal[:,6], ld_ops, c=r_R_am, cmap="viridis", s=30, zorder=5,
                     label="Operating points")
    plt.colorbar(sc2, ax=ax, label="r/R")
    ax.set_xlabel(r"$\alpha$ [deg]"); ax.set_ylabel(r"$C_l/C_d$ [-]")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); save_fig("10b_glide_ratio_vs_alpha_with_operating_points.png")

elif PLOT_10:
    _skip("PLOT_10", "analytical result missing (run with RUN_ANALYTICAL=True)")

# =============================================================================
print("\n" + "="*60)
print(f"ALL PLOTS SAVED TO: {save_folder}")
print("="*60)
print("\nFile list:")
for f in sorted(os.listdir(save_folder)):
    if f.endswith(".png"):
        print(f"  {f}")