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
# MENU — choose which plots to produce
# =============================================================================
#
# Set any flag to False to skip that figure entirely.
# If the underlying data was not saved in the .npz (because the corresponding
# RUN_* flag was False in assignment.py), the plot is skipped automatically
# with a printed warning — no crash.
# =============================================================================

PLOT_4_1 = True   # alpha and inflow angle vs r/R
PLOT_4_2 = True   # axial and tangential induction vs r/R
PLOT_4_3 = True   # thrust and azimuthal loading vs r/R
PLOT_4_4 = True   # CT and CQ vs TSR
PLOT_5   = True   # tip correction influence      (needs res_nc in npz)
PLOT_6   = True   # annuli count, spacing, convergence history
PLOT_7   = True   # stagnation pressure
PLOT_8   = True   # all designs — chord, twist, induction, loading, performance
                  #   sub-plots: 8a chord, 8b twist, 8ab combined,
                  #              8c induction, 8d loading, 8e alpha, 8f performance
PLOT_9   = True   # Cl and chord relation         (needs analytical result in npz)
PLOT_10  = True   # Cl/Cd polar with operating points (needs analytical result)

# =============================================================================
# 1.  LOAD RESULTS
# =============================================================================

npz_path = sys.argv[1] if len(sys.argv) > 1 else "full_bem_results.npz"
if not os.path.exists(npz_path):
    raise FileNotFoundError(
        f"Cannot find '{npz_path}'.\n"
        "Run assignment.py first to generate full_bem_results.npz.")

print(f"Loading results from: {npz_path}")
D = np.load(npz_path, allow_pickle=False)

def _get(key, default=None):
    try:
        return D[key]
    except KeyError:
        return default

def _getf(key, default=None):
    v = _get(key)
    return float(v) if v is not None else default

# ── Configuration scalars ────────────────────────────────────────────────────
Radius         = _getf("cfg_Radius",         50.0)
NBlades        = int(_getf("cfg_NBlades",    3))
U0             = _getf("cfg_U0",             10.0)
rho            = _getf("cfg_rho",            1.225)
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

# ── TSR sweep ────────────────────────────────────────────────────────────────
TSR_SWEEP  = [int(t) for t in D["sweep_tsrs"]]
tsr_CT     = D["tsr_CT"]
tsr_CP     = D["tsr_CP"]
sweep_data = {int(D["sweep_tsrs"][i]): D[f"sweep_res_{int(D['sweep_tsrs'][i])}"]
              for i in range(len(D["sweep_tsrs"]))}

# ── TSR=8 specific ───────────────────────────────────────────────────────────
results_tsr8 = _get("results_tsr8")
res_nc       = _get("res_nc")
ct_hist_tsr8 = _get("ct_hist_tsr8")
F_tsr8       = _get("F_tsr8")

# ── Section-6 pre-computed ───────────────────────────────────────────────────
annuli_results  = {k: v for k, v in
                   [(8,   _get("annuli_N8")),
                    (20,  _get("annuli_N20")),
                    (100, _get("annuli_N100"))] if v is not None}
spacing_results = {k: v for k, v in
                   [("Constant", _get("spacing_constant")),
                    ("Cosine",   _get("spacing_cosine"))] if v is not None}

# ── Geometry nodes ───────────────────────────────────────────────────────────
r_base   = _get("r_base");   c_base   = _get("c_base");   tw_base   = _get("tw_base")
r_anal   = _get("r_anal");   c_anal   = _get("c_anal");   tw_anal   = _get("tw_anal")
r_cubic  = _get("r_cubic");  c_cubic  = _get("c_cubic");  tw_cubic  = _get("tw_cubic")
r_qrt    = _get("r_qrt");    c_qrt    = _get("c_qrt");    tw_qrt    = _get("tw_qrt")

# ── BEM results ──────────────────────────────────────────────────────────────
res_base  = _get("res_base")
res_anal  = _get("res_anal")
res_cubic = _get("res_cubic")
res_qrt   = _get("res_qrt")

# ── Scalar performance ───────────────────────────────────────────────────────
CT_base  = _getf("CT_base");  CP_base  = _getf("CP_base")
CT_anal  = _getf("CT_anal");  CP_anal  = _getf("CP_anal")
CT_cubic = _getf("CT_cubic"); CP_cubic = _getf("CP_cubic")
CT_qrt   = _getf("CT_qrt");   CP_qrt   = _getf("CP_qrt")
CP_ad    = _getf("CP_ad")

# ── Polynomial parameters ────────────────────────────────────────────────────
p_cubic = _get("p_cubic")
p_qrt   = _get("p_qrt")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"  TSR sweep    : {TSR_SWEEP}")
print(f"  Baseline     : CT={CT_base:.4f}  CP={CP_base:.4f}" if CT_base else "  Baseline     : not in file")
print(f"  Analytical   : CT={CT_anal:.4f}  CP={CP_anal:.4f}" if CT_anal else "  Analytical   : not in file")
print(f"  Cubic poly   : CT={CT_cubic:.4f}  CP={CP_cubic:.4f}" if CT_cubic else "  Cubic poly   : not in file")
print(f"  Quartic poly : CT={CT_qrt:.4f}  CP={CP_qrt:.4f}" if CT_qrt else "  Quartic poly : not in file")

# =============================================================================
# 2.  COLOR SCHEME
# =============================================================================
#
# Design comparison plots assign one fixed color per design.
# With 3 designs present: green, blue, red (as requested).
# With 4 designs present: blue, green, red, orange
# — all colorblind-distinguishable and print-safe.
#
# Colors are resolved once here so every plot 8 sub-figure is consistent.
# =============================================================================

# Master color map — always the same color for the same design label
_DESIGN_COLORS = {
    "Baseline":     "#1f77b4",   # blue
    "Analytical":   "#2ca02c",   # green
    "Cubic poly":   "#d62728",   # red
    "Quartic poly": "#ff7f0e",   # orange
}

# If exactly 3 designs are present (baseline + 2 optimisers) reorder to
# green / blue / red as requested.  With 4 designs keep the 4-color palette.
_THREE_COLOR_OVERRIDE = {
    "Baseline":   "#2ca02c",   # green
    "Analytical": "#1f77b4",   # blue
    "Cubic poly": "#d62728",   # red
    "Quartic poly": "#ff7f0e", # orange (only used when 4 present)
}

def _design_color(label, n_designs):
    """Return the color for a design label given how many designs are present."""
    if n_designs <= 3:
        return _THREE_COLOR_OVERRIDE.get(label, "#888888")
    return _DESIGN_COLORS.get(label, "#888888")

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
    Each entry: (label, c_dense, tw_dense, res_arr, CT, CP)
    Dense chord/twist are interpolated from the saved node arrays.
    """
    designs = []

    if r_base is not None:
        designs.append(("Baseline",
                         np.interp(r_R_dense, r_base / Radius, c_base),
                         np.interp(r_R_dense, r_base / Radius, tw_base),
                         res_base, CT_base, CP_base))

    if r_anal is not None and res_anal is not None:
        designs.append(("Analytical",
                         np.interp(r_R_dense, r_anal / Radius, c_anal),
                         np.interp(r_R_dense, r_anal / Radius, tw_anal),
                         res_anal, CT_anal, CP_anal))
    else:
        print("  [INFO] analytical design absent — skipping that curve")

    if r_cubic is not None and res_cubic is not None:
        designs.append(("Cubic poly",
                         np.interp(r_R_dense, r_cubic / Radius, c_cubic),
                         np.interp(r_R_dense, r_cubic / Radius, tw_cubic),
                         res_cubic, CT_cubic, CP_cubic))
    else:
        print("  [INFO] cubic poly absent — skipping that curve")

    if r_qrt is not None and res_qrt is not None:
        designs.append(("Quartic poly",
                         np.interp(r_R_dense, r_qrt / Radius, c_qrt),
                         np.interp(r_R_dense, r_qrt / Radius, tw_qrt),
                         res_qrt, CT_qrt, CP_qrt))
    else:
        print("  [INFO] quartic poly absent — skipping that curve")

    return designs

# =============================================================================
# 4.  GEOMETRY COMPARISON FUNCTIONS  (used by PLOT_8)
# =============================================================================

def plot_chord_comparison(designs, r_R_dense):
    """8a — Chord distribution, one line per design."""
    n = len(designs)
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, c_d, *_ in designs:
        ax.plot(r_R_dense, c_d, label=lbl, color=_design_color(lbl, n), lw=2)
    ax.axhline(CHORD_MIN,  color="grey", ls=":",  lw=1, label=f"Min chord  {CHORD_MIN} m")
    ax.axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root chord  {CHORD_ROOT} m")
    ax.set_xlabel("r/R"); ax.set_ylabel("Chord [m]")
    ax.set_title("Chord distribution — design comparison")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def plot_twist_comparison(designs, r_R_dense):
    """8b — Twist distribution, one line per design."""
    n = len(designs)
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, _, tw_d, *_ in designs:
        ax.plot(r_R_dense, tw_d, label=lbl, color=_design_color(lbl, n), lw=2)
    ax.set_xlabel("r/R"); ax.set_ylabel("Twist [deg]")
    ax.set_title(r"Twist distribution — design comparison"
                 "\n" r"(convention: $\alpha = \mathrm{twist} + \phi$)")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def plot_chord_and_twist(designs, r_R_dense):
    """8ab — Chord and twist side-by-side."""
    n = len(designs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, c_d, tw_d, *_ in designs:
        col = _design_color(lbl, n)
        axes[0].plot(r_R_dense, c_d,  label=lbl, color=col, lw=2)
        axes[1].plot(r_R_dense, tw_d, label=lbl, color=col, lw=2)
    axes[0].axhline(CHORD_MIN,  color="grey", ls=":",  lw=1, label=f"Min  {CHORD_MIN} m")
    axes[0].axhline(CHORD_ROOT, color="k",   ls="--", lw=0.8, label=f"Root  {CHORD_ROOT} m")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
    axes[0].set_title("Chord distribution"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
    axes[1].set_title(r"Twist  ($\alpha = \mathrm{twist} + \phi$)")
    axes[1].legend(); axes[1].grid(True)
    fig.suptitle("Geometry comparison — all designs")
    fig.tight_layout()
    return fig


def plot_induction_comparison(designs):
    """8c — Axial induction a vs r/R."""
    n = len(designs)
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:, 2], res[:, 0], label=lbl, color=_design_color(lbl, n), lw=2)
    ax.axhline(1/3, color="grey", ls=":", lw=0.8, label="a = 1/3  (Betz optimum)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$a$ [-]")
    ax.set_title("Axial induction — design comparison")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def plot_loading_comparison(designs):
    """8d — Normal loading Cn vs r/R."""
    n = len(designs)
    norm_val = 0.5 * U0 ** 2 * Radius
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:, 2], res[:, 3] / norm_val,
                label=lbl, color=_design_color(lbl, n), lw=2)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n = F_n\,/\,(½U_\infty^2 R)$")
    ax.set_title("Normal (thrust) loading — design comparison")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def plot_alpha_comparison(designs):
    """8e — Angle of attack vs r/R."""
    n = len(designs)
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, _, _, res, *_ in designs:
        ax.plot(res[:, 2], res[:, 6], label=lbl, color=_design_color(lbl, n), lw=2)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$\alpha$ [deg]")
    ax.set_title("Angle of attack — design comparison")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def plot_performance_bar(designs):
    """8f — CT, CP and CP/CP_AD grouped bar chart."""
    a_ad_  = 0.5 * (1.0 - np.sqrt(1.0 - CT_TARGET))
    cp_ad_ = CP_ad if CP_ad is not None else 4.0 * a_ad_ * (1.0 - a_ad_) ** 2

    labels  = [d[0] for d in designs] + ["Actuator disk"]
    cp_vals = [d[5] for d in designs] + [cp_ad_]
    ct_vals = [d[4] for d in designs] + [CT_TARGET]
    eff_vals = [v / cp_ad_ for v in cp_vals]
    bar_colors = [_design_color(d[0], len(designs)) for d in designs] + ["#aaaaaa"]

    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, cp_vals,  w, label="CP",
                color=[c + "cc" for c in bar_colors])   # slight transparency via alpha hex
    b2 = ax.bar(x,     ct_vals,  w, label="CT",
                color=bar_colors, alpha=0.6)
    b3 = ax.bar(x + w, eff_vals, w, label=r"$C_P\,/\,C_{P,AD}$",
                color=bar_colors, hatch="//", alpha=0.85)

    # Solid bar edges for clarity
    for bars in [b1, b2, b3]:
        for bar in bars:
            bar.set_edgecolor("white")
            bar.set_linewidth(0.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Coefficient [-]")
    ax.set_title(f"Performance summary  (target $C_T$ = {CT_TARGET})")
    ax.legend()
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=7.5)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=7.5)
    ax.bar_label(b3, fmt="%.3f", padding=3, fontsize=7.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig

# =============================================================================
# 5.  PLOTTING
# =============================================================================

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "plots_assignment")
os.makedirs(save_folder, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(save_folder, name), dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}")
    plt.show()

norm_val = 0.5 * U0 ** 2 * Radius

# ── 4.1  Alpha and inflow angle ───────────────────────────────────────────────
if PLOT_4_1:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for TSR in TSR_SWEEP:
        res = sweep_data[TSR]
        axes[0].plot(res[:,2], res[:,6], label=rf"$\lambda$={TSR}")
        axes[1].plot(res[:,2], res[:,7], label=rf"$\lambda$={TSR}")
    for ax, yl, tl in zip(axes, [r"$\alpha$ [deg]", r"$\phi$ [deg]"],
                           ["Angle of attack", "Inflow angle"]):
        ax.set_xlabel("r/R"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.grid(True); ax.legend()
    fig.suptitle("Baseline geometry — angle distributions")
    fig.tight_layout(); save_fig("4_1_alpha_phi.png")

# ── 4.2  Induction factors ────────────────────────────────────────────────────
if PLOT_4_2:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for TSR in TSR_SWEEP:
        res = sweep_data[TSR]
        axes[0].plot(res[:,2], res[:,0], label=rf"$\lambda$={TSR}")
        axes[1].plot(res[:,2], res[:,1], label=rf"$\lambda$={TSR}")
    for ax, yl, tl in zip(axes, [r"$a$ [-]", r"$a'$ [-]"],
                           ["Axial induction", "Tangential induction"]):
        ax.set_xlabel("r/R"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.grid(True); ax.legend()
    fig.suptitle("Baseline geometry — induction factors")
    fig.tight_layout(); save_fig("4_2_induction.png")

# ── 4.3  Thrust and azimuthal loading ────────────────────────────────────────
if PLOT_4_3:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for TSR in TSR_SWEEP:
        res = sweep_data[TSR]
        axes[0].plot(res[:,2], res[:,3]/norm_val, label=rf"$\lambda$={TSR}")
        axes[1].plot(res[:,2], res[:,4]/norm_val, label=rf"$\lambda$={TSR}")
    for ax, yl, tl in zip(axes,
            [r"$C_n = F_n/(½U_\infty^2 R)$", r"$C_t = F_t/(½U_\infty^2 R)$"],
            ["Normal (thrust) loading", "Tangential (torque) loading"]):
        ax.set_xlabel("r/R"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.grid(True); ax.legend()
    fig.suptitle("Baseline geometry — spanwise loading")
    fig.tight_layout(); save_fig("4_3_loading.png")

# ── 4.4  CT and CQ vs TSR ────────────────────────────────────────────────────
if PLOT_4_4:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(TSR_SWEEP, tsr_CT, "bo-")
    axes[0].set_xlabel(r"$\lambda$"); axes[0].set_ylabel(r"$C_T$")
    axes[0].set_title(r"$C_T$ vs TSR"); axes[0].grid(True)
    axes[1].plot(TSR_SWEEP, tsr_CP / np.array(TSR_SWEEP, dtype=float), "ro-")
    axes[1].set_xlabel(r"$\lambda$"); axes[1].set_ylabel(r"$C_Q$")
    axes[1].set_title(r"$C_Q$ vs TSR"); axes[1].grid(True)
    fig.tight_layout(); save_fig("4_4_CT_CQ_TSR.png")

# ── 5  Tip correction ─────────────────────────────────────────────────────────
if PLOT_5 and results_tsr8 is not None and res_nc is not None:
    r_R8 = results_tsr8[:,2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(r_R8, results_tsr8[:,0], "b-",  lw=2, label="With Prandtl correction")
    axes[0].plot(res_nc[:,2], res_nc[:,0], "r--", lw=2, label="No correction (F=1)")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$a$ [-]")
    axes[0].set_title("Axial induction (TSR=8)"); axes[0].grid(True); axes[0].legend()
    axes[1].plot(r_R8, results_tsr8[:,3]/norm_val, "b-",  lw=2, label="With Prandtl correction")
    axes[1].plot(res_nc[:,2], res_nc[:,3]/norm_val, "r--", lw=2, label="No correction (F=1)")
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_n$ [-]")
    axes[1].set_title("Normal loading (TSR=8)"); axes[1].grid(True); axes[1].legend()
    fig.suptitle("Influence of Prandtl tip/root correction")
    fig.tight_layout(); save_fig("5_tip_correction.png")
elif PLOT_5:
    _skip("PLOT_5", "res_nc or results_tsr8 missing (run with RUN_NO_CORRECTION=True)")

# ── 6  Annuli, spacing, convergence ──────────────────────────────────────────
if PLOT_6 and results_tsr8 is not None and annuli_results and spacing_results:

    # 6a — number of annuli
    fig, ax = plt.subplots(figsize=(9, 5))
    for N, res_N in annuli_results.items():
        ax.plot(res_N[:,2], res_N[:,3]/norm_val, "-o", markersize=4, label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.set_title("Normal loading — influence of N annuli (TSR=8)")
    ax.grid(True); ax.legend(); save_fig("6a_annuli.png")

    # 6b — spacing method
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, res_sp in spacing_results.items():
        ax.plot(res_sp[:,2], res_sp[:,3]/norm_val, "-o", markersize=5, label=lbl)
    ax.set_xlim(0.85, 1.01); ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.set_title("Spacing method near tip (N=40, TSR=8)")
    ax.grid(True); ax.legend(); save_fig("6b_spacing.png")

    # 6c — convergence history
    if ct_hist_tsr8 is not None:
        n_show = min(60, len(ct_hist_tsr8))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(range(1, len(ct_hist_tsr8)+1), ct_hist_tsr8, "b-", lw=2)
        axes[0].set_xlim(1, n_show); axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(r"$C_T$"); axes[0].set_title("CT convergence (TSR=8)")
        axes[0].grid(True)
        resid = np.abs(np.diff(ct_hist_tsr8))
        axes[1].semilogy(range(2, len(ct_hist_tsr8)+1), resid, "r-", lw=2,
                         label=r"$|C_{T,i}-C_{T,i-1}|$")
        axes[1].axhline(1e-5, color="k", ls="--", lw=0.8, label="tol = 1e-5")
        axes[1].set_xlim(1, n_show); axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(r"$|\Delta C_T|$"); axes[1].set_title("Residuals (log scale)")
        axes[1].grid(True, which="both"); axes[1].legend()
        fig.tight_layout(); save_fig("6c_convergence.png")

elif PLOT_6:
    _skip("PLOT_6", "annuli/spacing data missing (run with RUN_TSR_SWEEP=True)")

# ── 7  Stagnation pressure ────────────────────────────────────────────────────
if PLOT_7 and results_tsr8 is not None and F_tsr8 is not None:
    r_R8    = results_tsr8[:,2]
    P0_up   = 0.5 * rho * U0**2 * np.ones(len(r_R8))
    dP0     = 2.0*rho*U0**2 * results_tsr8[:,0] * F_tsr8 * (
               1.0 - results_tsr8[:,0] * F_tsr8)
    P0_down = P0_up - dP0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_R8, P0_up,   "b-",  lw=2,   label=r"$P_0$ far upstream  (stat. 1)")
    ax.plot(r_R8, P0_up,   "b--", lw=1.5, alpha=0.6, label=r"$P_0$ rotor upwind  (stat. 2)")
    ax.plot(r_R8, P0_down, "r--", lw=1.5, alpha=0.6, label=r"$P_0$ rotor downwind (stat. 3)")
    ax.plot(r_R8, P0_down, "r-",  lw=2,   label=r"$P_0$ far downstream (stat. 4)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$P_0$ [Pa]")
    ax.set_title("Stagnation pressure at four streamwise stations (TSR=8)")
    ax.legend(); ax.grid(True); save_fig("7_stagnation_pressure.png")
elif PLOT_7:
    _skip("PLOT_7", "results_tsr8 or F_tsr8 missing (run with RUN_TSR_SWEEP=True)")

# ── 8  All designs ────────────────────────────────────────────────────────────
if PLOT_8 and res_base is not None:
    r_R_dense = np.linspace(RootLocation_R, TipLocation_R, 400)
    designs   = _build_designs(r_R_dense)

    fig = plot_chord_comparison(designs, r_R_dense);   save_fig("8a_chord.png")
    fig = plot_twist_comparison(designs, r_R_dense);   save_fig("8b_twist.png")
    fig = plot_chord_and_twist(designs, r_R_dense);    save_fig("8ab_chord_twist.png")
    fig = plot_induction_comparison(designs);           save_fig("8c_induction.png")
    fig = plot_loading_comparison(designs);             save_fig("8d_loading.png")
    fig = plot_alpha_comparison(designs);               save_fig("8e_alpha.png")
    fig = plot_performance_bar(designs);                save_fig("8f_performance.png")

elif PLOT_8:
    _skip("PLOT_8", "res_base missing from npz")

# ── 9  Cl and chord (analytical optimum) ─────────────────────────────────────
if PLOT_9 and res_anal is not None and r_anal is not None:
    r_R_am = res_anal[:,2]
    cl_am  = res_anal[:,8]
    c_am   = np.interp(r_R_am, r_anal / Radius, c_anal)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(r_R_am, cl_am, "b-", lw=2, label=r"$C_l$")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$C_l$ [-]"); axes[0].grid(True)
    ax2 = axes[0].twinx()
    ax2.plot(r_R_am, c_am, "r--", lw=2, label="Chord [m]")
    ax2.set_ylabel("Chord [m]", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    axes[0].set_zorder(ax2.get_zorder() + 1); axes[0].patch.set_visible(False)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1+h2, l1+l2)
    axes[0].set_title(r"$C_l$ and chord (analytical optimum)")
    axes[1].plot(r_R_am, cl_am * c_am, "g-", lw=2)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_l \cdot c$  [m]")
    axes[1].set_title(r"Circulation proxy  $\Gamma \propto C_l \cdot c$")
    axes[1].grid(True)
    fig.tight_layout(); save_fig("9_cl_chord.png")
elif PLOT_9:
    _skip("PLOT_9", "analytical result missing (run with RUN_ANALYTICAL=True)")

# ── 10  Polar with operating points ──────────────────────────────────────────
if PLOT_10 and res_anal is not None:
    alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
    alphas_d = np.linspace(polar_alpha[0], polar_alpha[-1], 500)
    cl_d = np.interp(alphas_d, polar_alpha, polar_cl)
    cd_d = np.interp(alphas_d, polar_alpha, polar_cd)
    ld_d = cl_d / np.maximum(cd_d, 1e-8)
    r_R_am = res_anal[:,2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(polar_cd, polar_cl, "k-", lw=1.5, label="Polar")
    sc = axes[0].scatter(res_anal[:,9], res_anal[:,8],
                         c=r_R_am, cmap="viridis", s=25, zorder=5,
                         label="Analytical opt")
    plt.colorbar(sc, ax=axes[0], label="r/R")
    axes[0].set_xlabel(r"$C_d$"); axes[0].set_ylabel(r"$C_l$")
    axes[0].set_title(r"$C_l$–$C_d$ polar"); axes[0].grid(True); axes[0].legend()
    axes[1].plot(alphas_d, ld_d, "k-", lw=1.5, label=r"$C_l/C_d$")
    axes[1].axvline(alpha_opt, color="r", ls="--",
                    label=rf"$\alpha_{{opt}}={alpha_opt:.1f}°$, "
                          rf"$(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
    ld_ops = (np.interp(res_anal[:,6], polar_alpha, polar_cl)
              / np.maximum(np.interp(res_anal[:,6], polar_alpha, polar_cd), 1e-8))
    sc2 = axes[1].scatter(res_anal[:,6], ld_ops,
                          c=r_R_am, cmap="viridis", s=25, zorder=5)
    plt.colorbar(sc2, ax=axes[1], label="r/R")
    axes[1].set_xlabel(r"$\alpha$ [deg]"); axes[1].set_ylabel(r"$C_l/C_d$")
    axes[1].set_title(r"$C_l/C_d$ vs $\alpha$"); axes[1].grid(True); axes[1].legend()
    fig.tight_layout(); save_fig("10_polar.png")
elif PLOT_10:
    _skip("PLOT_10", "analytical result missing (run with RUN_ANALYTICAL=True)")

# =============================================================================
print("\n" + "="*60)
print("ALL PLOTS SAVED TO:", save_folder)
print("="*60)