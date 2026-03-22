import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# Polar data
# =============================================================================

data        = pd.read_excel('polar DU95W180 (3).xlsx', skiprows=3)
polar_alpha = data['Alfa'].to_numpy()
polar_cl    = data['Cl'].to_numpy()
polar_cd    = data['Cd'].to_numpy()

# =============================================================================
# Specifications  (identical to the final reference BEM class)
# =============================================================================

Radius         = 50          # Rotor radius [m]
NBlades        = 3           # Number of blades
U0             = 10          # Freestream velocity [m/s]
RootLocation_R = 0.2
TipLocation_R  = 1.0

TSR_design = 8.0
CT_target  = 0.75

CHORD_ROOT    = 3.4          # Fixed root chord [m]  (structural pin)
CHORD_MIN     = 0.3          # Minimum chord anywhere [m]  (manufacturing)
CHORD_MAX_REG = 6.0          # Soft upper limit for regularisation [m]

# Spanwise resolution: coarse during optimisation, fine for final evaluation
delta_r_R_opt   = 0.01
delta_r_R_final = 0.005

# =============================================================================
# Prandtl formula toggle
# =============================================================================

# Set to True  -> helical-wake spacing formula (original Prandtl 1919, more
#                 physically correct, used by QBlade / OpenFAST).
# Set to False -> simplified trigonometric form (Glauert 1935, matches the
#                 reference notebook exactly, standard in most BEM courses).
USE_HELICAL_PRANDTL = False

# =============================================================================
# BEM core functions  — verbatim from the final reference class
# =============================================================================

def ainduction(CT):
    """
    Axial induction factor from local thrust coefficient CT,
    including Glauert correction for heavily loaded rotors.
    """
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1

    if np.isscalar(CT):
        if CT >= CT2:
            return 1 + (CT - CT1) / (4 * (np.sqrt(CT1) - 1))
        return 0.5 - 0.5 * np.sqrt(max(0, 1 - CT))
    else:
        a = np.zeros(np.shape(CT))
        a[CT >= CT2] = 1 + (CT[CT >= CT2] - CT1) / (4 * (np.sqrt(CT1) - 1))
        a[CT < CT2]  = 0.5 - 0.5 * np.sqrt(np.maximum(0, 1 - CT[CT < CT2]))
        return a


def _PrandtlSimplified(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """Glauert (1935) simplified form — matches the reference notebook."""
    a = np.clip(axial_induction, -0.9, 0.99)
    t_tip  = (-NBlades / 2 * (tipradius_R - r_R) / r_R
              * np.sqrt(1 + ((TSR * r_R) ** 2) / ((1 - a) ** 2)))
    Ftip   = np.array(2 / np.pi * np.arccos(np.clip(np.exp(t_tip), 0, 1)))
    t_root = (NBlades / 2 * (rootradius_R - r_R) / r_R
              * np.sqrt(1 + ((TSR * r_R) ** 2) / ((1 - a) ** 2)))
    Froot  = np.array(2 / np.pi * np.arccos(np.clip(np.exp(t_root), 0, 1)))
    return Froot * Ftip


def _PrandtlHelical(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """Original Prandtl (1919) helical-wake spacing form."""
    a = np.clip(axial_induction, -0.9, 0.99)
    d = max(2 * np.pi / NBlades * (1 - a) / np.sqrt(TSR ** 2 + (1 - a) ** 2), 1e-8)
    Ftip  = float(2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (tipradius_R - r_R) / d, -500, 0))))
    Froot = float(2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (r_R - rootradius_R) / d, -500, 0))))
    Ftip  = 0.0 if np.isnan(Ftip)  else Ftip
    Froot = 0.0 if np.isnan(Froot) else Froot
    return Froot * Ftip


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """Dispatcher — controlled by USE_HELICAL_PRANDTL flag above."""
    if USE_HELICAL_PRANDTL:
        return _PrandtlHelical(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction)
    return _PrandtlSimplified(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction)


def LoadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Sectional forces and AoA.
    twist [deg], sign convention: alpha = twist + phi_deg.
    """
    vmag2 = vnorm ** 2 + vtan ** 2
    phi   = np.arctan2(vnorm, vtan)
    alpha = twist + np.degrees(phi)
    cl    = np.interp(alpha, polar_alpha, polar_cl)
    cd    = np.interp(alpha, polar_alpha, polar_cd)
    lift  = 0.5 * vmag2 * cl * chord
    drag  = 0.5 * vmag2 * cd * chord
    fnorm = lift * np.cos(phi) + drag * np.sin(phi)
    ftan  = lift * np.sin(phi) - drag * np.cos(phi)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord
    return fnorm, ftan, gamma, alpha, np.degrees(phi)


def SolveStreamtube(
    U0, r1_R, r2_R, rootradius_R, tipradius_R,
    Omega, Radius, NBlades, chord, twist,
    polar_alpha, polar_cl, polar_cd,
    max_iter=300, tol=1e-5
):
    """
    Solve momentum balance for one annular streamtube.
    Returns [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi], fnorm_history.
    """
    Area    = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid   = (r1_R + r2_R) / 2
    r_local = r_mid * Radius

    a     = 0.0
    aline = 0.0
    fnorm_history = []

    for _ in range(max_iter):
        Urotor = U0 * (1 - a)
        Utan   = (1 + aline) * Omega * r_local

        fnorm, ftan, gamma, alpha, phi = LoadBladeElement(
            Urotor, Utan, chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        fnorm_history.append(fnorm)

        CT_loc = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0 ** 2)
        anew   = ainduction(CT_loc)

        F = max(
            PrandtlTipRootCorrection(
                r_mid, rootradius_R, tipradius_R,
                Omega * Radius / U0, NBlades, anew
            ),
            0.0001
        )
        anew /= F
        a = 0.75 * a + 0.25 * anew

        aline = (ftan * NBlades) / (2 * np.pi * U0 * (1 - a) * Omega * 2 * r_local ** 2)
        aline /= F

        if abs(a - anew) < tol:
            break

    return [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi], np.array(fnorm_history)


# =============================================================================
# Rotor evaluator
# =============================================================================

def evaluate_rotor(r_nodes, c_nodes, twist_nodes_deg, tsr=TSR_design):
    """
    Evaluate CT and CP for an arbitrary blade geometry.

    Parameters
    ----------
    r_nodes         : radial node positions [m]
    c_nodes         : chord at nodes [m]
    twist_nodes_deg : twist [deg], convention: alpha = twist + phi_deg
    tsr             : tip-speed ratio

    Returns
    -------
    CT, CP, res_arr   (res_arr columns: [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi])
    """
    Omega   = U0 * tsr / Radius
    results = []

    for i in range(len(r_nodes) - 1):
        r1    = r_nodes[i]   / Radius
        r2    = r_nodes[i+1] / Radius
        chord = 0.5 * (c_nodes[i]          + c_nodes[i+1])
        twist = 0.5 * (twist_nodes_deg[i]  + twist_nodes_deg[i+1])

        res, _ = SolveStreamtube(
            U0, r1, r2, RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades, chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        results.append(res)

    res_arr = np.array(results)
    dr      = np.diff(r_nodes)

    CT = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0 ** 2 * np.pi * Radius ** 2))
    CP = np.sum(
        dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega
        / (0.5 * U0 ** 3 * np.pi * Radius ** 2)
    )
    return CT, CP, res_arr


# =============================================================================
# Polynomial design parameterisation
# =============================================================================
#
# 8 free parameters:
#   params = [pitch, t_root, t_tip, t_curve, c_tip, c2, c3, c4]
#
# Chord — quartic pinned at root:
#   x = (r/R - RootLocation_R) / (TipLocation_R - RootLocation_R)
#   c(x) = CHORD_ROOT + b1*x + c2*x^2 + c3*x^3 + c4*x^4
#   with b1 chosen so c(1) = c_tip  =>  b1 = c_tip - CHORD_ROOT - c2 - c3 - c4
#
# Twist — quadratic with global pitch offset:
#   twist(x) = pitch + t_root*(1-x) + t_tip*x + t_curve*x*(1-x)
#   sign convention: alpha = twist + phi_deg  (reference class convention)
# =============================================================================

def _span_x(r_R):
    """Map r/R in [RootLocation_R, TipLocation_R] to x in [0, 1]."""
    return (r_R - RootLocation_R) / (TipLocation_R - RootLocation_R)


def chord_poly(r_R, c_tip, c2, c3, c4):
    x  = _span_x(r_R)
    b1 = c_tip - CHORD_ROOT - c2 - c3 - c4
    return CHORD_ROOT + b1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4


def twist_poly(r_R, pitch, t_root, t_tip, t_curve):
    x = _span_x(r_R)
    return pitch + t_root * (1 - x) + t_tip * x + t_curve * x * (1 - x)


def build_geometry(params, delta_r_R=delta_r_R_opt):
    """
    Convert parameter vector to (r_nodes, c_nodes, twist_nodes_deg) arrays
    ready to pass into evaluate_rotor.
    """
    pitch, t_root, t_tip, t_curve, c_tip, c2, c3, c4 = params

    r_R_nodes = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)
    r_nodes   = r_R_nodes * Radius
    c_nodes   = np.array([chord_poly(r, c_tip, c2, c3, c4) for r in r_R_nodes])
    tw_nodes  = np.array([twist_poly(r, pitch, t_root, t_tip, t_curve) for r in r_R_nodes])

    return r_nodes, c_nodes, tw_nodes


def chord_min_max(params, n=300):
    """Evaluate min and max chord over a dense spanwise grid."""
    _, _, _, _, c_tip, c2, c3, c4 = params
    r_R_dense = np.linspace(RootLocation_R, TipLocation_R, n)
    c_dense   = np.array([chord_poly(r, c_tip, c2, c3, c4) for r in r_R_dense])
    return float(np.min(c_dense)), float(np.max(c_dense))


# =============================================================================
# Objective function
# =============================================================================

def objective(params):
    """
    Minimise  −CP  subject to  CT = CT_target,  chord ∈ [CHORD_MIN, CHORD_MAX_REG].

    Penalties:
      • Quadratic CT constraint:  5000 * (CT − CT_target)²
      • Hard chord lower bound:   1e4  * max(0, CHORD_MIN − c_min)²
      • Soft chord upper bound:   1e3  * max(0, c_max − CHORD_MAX_REG)²
      • Curvature regularisation: 0.05 * t_curve² + 0.01 * (c2² + c3²)
        (discourages wildly non-linear distributions without over-constraining)
    """
    pitch, t_root, t_tip, t_curve, c_tip, c2, c3, c4 = params

    c_min, c_max = chord_min_max(params)
    penalty = 0.0

    # Hard lower bound — infeasible chord causes BEM to diverge
    if c_min < CHORD_MIN:
        penalty += 1e4 * (CHORD_MIN - c_min) ** 2

    # Soft upper bound
    if c_max > CHORD_MAX_REG:
        penalty += 1e3 * (c_max - CHORD_MAX_REG) ** 2

    # If chord is already infeasible just return the penalty — skip BEM
    if c_min < CHORD_MIN * 0.5:
        return 1e9

    r_nodes, c_nodes, tw_nodes = build_geometry(params, delta_r_R=delta_r_R_opt)
    CT, CP, _ = evaluate_rotor(r_nodes, c_nodes, tw_nodes)

    penalty += 5e3 * (CT - CT_target) ** 2
    penalty += 0.05 * t_curve ** 2 + 0.01 * (c2 ** 2 + c3 ** 2 + c4 ** 2)

    return -CP + penalty


# =============================================================================
# Multi-start optimisation
# =============================================================================

def optimize(n_starts=12, random_seed=42):
    """
    Multi-start L-BFGS-B optimisation.

    Parameter order: [pitch, t_root, t_tip, t_curve, c_tip, c2, c3, c4]

    Bounds are generous — the CT penalty and chord penalties do the real
    constraining.  The first start is a nominal guess near the baseline;
    subsequent starts are random within the bounds.
    """
    rng = np.random.default_rng(random_seed)

    bounds = [
        (-8.0,  8.0),    # pitch     [deg]
        (-25.0, 5.0),    # t_root    [deg]
        (-10.0, 15.0),   # t_tip     [deg]
        (-20.0, 20.0),   # t_curve   [deg]
        (0.3,   2.0),    # c_tip     [m]
        (-5.0,  5.0),    # c2
        (-5.0,  5.0),    # c3
        (-5.0,  5.0),    # c4  (quartic term)
    ]

    # Nominal start close to the original baseline geometry
    x0_nominal = np.array([-2.0, -7.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    starts = [x0_nominal]
    for _ in range(n_starts - 1):
        starts.append(np.array([
            rng.uniform(-6.0,   6.0),
            rng.uniform(-20.0,  0.0),
            rng.uniform(-5.0,  10.0),
            rng.uniform(-10.0, 10.0),
            rng.uniform(0.3,    1.5),
            rng.uniform(-3.0,   3.0),
            rng.uniform(-3.0,   3.0),
            rng.uniform(-3.0,   3.0),
        ]))

    best_res = None
    best_obj = np.inf

    for k, x0 in enumerate(starts, start=1):
        res = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_res = res
        print(f"  Start {k:02d}/{n_starts}  obj = {res.fun:+.6f}  "
              f"CT = {_quick_ct(res.x):.4f}")

    return best_res


def _quick_ct(params):
    """Fast CT read-back for progress printing."""
    try:
        r, c, tw = build_geometry(params, delta_r_R_opt)
        CT, _, _ = evaluate_rotor(r, c, tw)
        return CT
    except Exception:
        return float("nan")


# =============================================================================
# Baseline geometry
# =============================================================================

def get_baseline(delta_r_R=delta_r_R_final):
    """Original linear chord/twist blade, pitch = −2 deg."""
    r_R_nodes = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)
    r_nodes   = r_R_nodes * Radius
    c_nodes   = 3.0 * (1 - r_R_nodes) + 1.0
    tw_nodes  = -(14.0 * (1 - r_R_nodes) + (-2.0))  # pitch = -2
    return r_nodes, c_nodes, tw_nodes


# =============================================================================
# Plotting
# =============================================================================

base_path   = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_poly_opt_quartic")
os.makedirs(save_folder, exist_ok=True)


def save_and_show(filename):
    path = os.path.join(save_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


def plot_all(params_opt,
             r_base, c_base, tw_base, res_base,
             r_opt,  c_opt,  tw_opt,  res_opt,
             CT_base, CP_base, CT_opt, CP_opt):

    pitch, t_root, t_tip, t_curve, c_tip, c2, c3, c4 = params_opt

    r_R_base = res_base[:, 2]
    r_R_opt  = res_opt[:, 2]
    norm_val = 0.5 * U0 ** 2 * Radius

    # Dense curves for geometry plots
    r_R_dense  = np.linspace(RootLocation_R, TipLocation_R, 400)
    c_base_den = 3.0 * (1 - r_R_dense) + 1.0
    tw_base_den = -(14.0 * (1 - r_R_dense) + (-2.0))
    c_opt_den  = np.array([chord_poly(r, c_tip, c2, c3, c4) for r in r_R_dense])
    tw_opt_den = np.array([twist_poly(r, pitch, t_root, t_tip, t_curve) for r in r_R_dense])

    # ── Plot 1: chord ─────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_dense, c_base_den, label=f"Baseline  (CT={CT_base:.3f})")
    plt.plot(r_R_dense, c_opt_den,  label=f"Polynomial opt  (CT={CT_opt:.3f})", linestyle="--")
    plt.axhline(CHORD_MIN,     color="grey", linestyle=":",  linewidth=1, label="Min chord")
    plt.axhline(CHORD_MAX_REG, color="grey", linestyle="--", linewidth=1, label="Max chord (soft)")
    plt.scatter([RootLocation_R], [CHORD_ROOT], zorder=5, color="k", label="Fixed root chord")
    plt.xlabel("r/R"); plt.ylabel("Chord [m]")
    plt.title("Chord distribution")
    plt.legend(); plt.grid(True)
    save_and_show("01_chord_distribution.png")

    # ── Plot 2: twist ─────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_dense, tw_base_den, label="Baseline")
    plt.plot(r_R_dense, tw_opt_den,  label="Polynomial opt", linestyle="--")
    plt.xlabel("r/R"); plt.ylabel("Twist [deg]")
    plt.title("Twist distribution  (convention: α = twist + φ)")
    plt.legend(); plt.grid(True)
    save_and_show("02_twist_distribution.png")

    # ── Plot 3: induction ──────────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, res_base[:, 0], label="Baseline  a")
    plt.plot(r_R_base, res_base[:, 1], label="Baseline  a'")
    plt.plot(r_R_opt,  res_opt[:,  0], label="Opt  a",  linestyle="--")
    plt.plot(r_R_opt,  res_opt[:,  1], label="Opt  a'", linestyle="--")
    plt.xlabel("r/R"); plt.ylabel("Induction factor [-]")
    plt.title("Induction distributions")
    plt.legend(); plt.grid(True)
    save_and_show("03_induction_distribution.png")

    # ── Plot 4: section loading ────────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, res_base[:, 3] / norm_val, label="Baseline  fnorm")
    plt.plot(r_R_base, res_base[:, 4] / norm_val, label="Baseline  ftan")
    plt.plot(r_R_opt,  res_opt[:,  3] / norm_val, label="Opt  fnorm", linestyle="--")
    plt.plot(r_R_opt,  res_opt[:,  4] / norm_val, label="Opt  ftan",  linestyle="--")
    plt.xlabel("r/R"); plt.ylabel(r"$F\,/\,(\frac{1}{2}\,U_\infty^2\,R)$")
    plt.title("Section loading")
    plt.legend(); plt.grid(True)
    save_and_show("04_loading_distribution.png")

    # ── Plot 5: AoA and inflow angle ──────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, res_base[:, 6], label=r"Baseline $\alpha$")
    plt.plot(r_R_base, res_base[:, 7], label=r"Baseline $\phi$")
    plt.plot(r_R_opt,  res_opt[:,  6], label=r"Opt $\alpha$",  linestyle="--")
    plt.plot(r_R_opt,  res_opt[:,  7], label=r"Opt $\phi$",    linestyle="--")
    plt.xlabel("r/R"); plt.ylabel("Angle [deg]")
    plt.title("Angle of attack and inflow angle")
    plt.legend(); plt.grid(True)
    save_and_show("05_angles_distribution.png")

    # ── Plot 6: performance bar ────────────────────────────────────────────────
    a_ad  = 0.5 * (1.0 - np.sqrt(1.0 - CT_target))
    CP_ad = 4.0 * a_ad * (1.0 - a_ad) ** 2

    fig, ax = plt.subplots(figsize=(6, 5))
    labels  = ["Baseline", "Poly opt", "Actuator disk"]
    cp_vals = [CP_base, CP_opt, CP_ad]
    ct_vals = [CT_base, CT_opt, CT_target]
    x = np.arange(len(labels)); w = 0.35
    b1 = ax.bar(x - w/2, cp_vals, w, label="CP")
    b2 = ax.bar(x + w/2, ct_vals, w, label="CT")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Coefficient [-]")
    ax.set_title("Performance comparison")
    ax.legend(); ax.bar_label(b1, fmt="%.4f", padding=3)
    ax.bar_label(b2, fmt="%.4f", padding=3)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_and_show("06_performance_summary.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("BASELINE  (linear chord/twist, pitch = −2 deg, TSR = 8)")
    print("=" * 65)

    r_base, c_base, tw_base = get_baseline(delta_r_R_final)
    CT_base, CP_base, res_base = evaluate_rotor(r_base, c_base, tw_base)
    print(f"Baseline  CT = {CT_base:.6f},  CP = {CP_base:.6f}")

    # ── Optimisation ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RUNNING MULTI-START OPTIMISATION  (12 starts, L-BFGS-B)")
    print("=" * 65)

    opt_res     = optimize(n_starts=12, random_seed=42)
    best_params = opt_res.x

    # Final high-resolution evaluation
    r_opt, c_opt, tw_opt = build_geometry(best_params, delta_r_R_final)
    CT_opt, CP_opt, res_opt = evaluate_rotor(r_opt, c_opt, tw_opt)

    pitch_opt, t_root_opt, t_tip_opt, t_curve_opt, c_tip_opt, c2_opt, c3_opt, c4_opt = best_params
    c_min_opt, c_max_opt = chord_min_max(best_params)

    print("\n" + "=" * 65)
    print("OPTIMISED DESIGN")
    print("=" * 65)
    print(f"  pitch       : {pitch_opt:+.4f} deg")
    print(f"  t_root      : {t_root_opt:+.4f} deg")
    print(f"  t_tip       : {t_tip_opt:+.4f} deg")
    print(f"  t_curve     : {t_curve_opt:+.4f} deg")
    print(f"  c_tip       : {c_tip_opt:.4f} m")
    print(f"  c2          : {c2_opt:+.4f}")
    print(f"  c3          : {c3_opt:+.4f}")
    print(f"  c4          : {c4_opt:+.4f}")
    print(f"  CT          : {CT_opt:.6f}  (target {CT_target})")
    print(f"  CP          : {CP_opt:.6f}")
    print(f"  chord range : {c_min_opt:.4f} – {c_max_opt:.4f} m")

    # ── Actuator disk reference ───────────────────────────────────────────────
    a_ad  = 0.5 * (1.0 - np.sqrt(1.0 - CT_target))
    CP_ad = 4.0 * a_ad * (1.0 - a_ad) ** 2

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Actuator disk  CT={CT_target}:  a = {a_ad:.4f},  CP = {CP_ad:.6f}")
    print(f"Baseline        CT = {CT_base:.6f},  CP = {CP_base:.6f}"
          f"  (CP/CP_AD = {CP_base/CP_ad:.4f})")
    print(f"Polynomial opt  CT = {CT_opt:.6f},  CP = {CP_opt:.6f}"
          f"  (CP/CP_AD = {CP_opt/CP_ad:.4f})")
    print(f"CP improvement  = {100*(CP_opt - CP_base)/abs(CP_base):.2f} %")

    # ── Polynomial form ───────────────────────────────────────────────────────
    b1 = c_tip_opt - CHORD_ROOT - c2_opt - c3_opt - c4_opt
    print("\n" + "=" * 65)
    print("POLYNOMIAL DISTRIBUTIONS  (x = (r/R − 0.2) / 0.8)")
    print("=" * 65)
    print(f"Chord:  c(x) = {CHORD_ROOT:.4f}"
          f" + ({b1:+.4f})·x"
          f" + ({c2_opt:+.4f})·x²"
          f" + ({c3_opt:+.4f})·x³"
          f" + ({c4_opt:+.4f})·x⁴   [m]")
    print(f"Twist:  t(x) = {pitch_opt:+.4f}"
          f" + ({t_root_opt:+.4f})·(1−x)"
          f" + ({t_tip_opt:+.4f})·x"
          f" + ({t_curve_opt:+.4f})·x(1−x)   [deg]")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(
        best_params,
        r_base, c_base, tw_base, res_base,
        r_opt,  c_opt,  tw_opt,  res_opt,
        CT_base, CP_base, CT_opt, CP_opt,
    )

    # ── Save geometry to CSV ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "r_m":       r_opt,
        "r_R":       r_opt / Radius,
        "chord_m":   c_opt,
        "twist_deg": tw_opt,
    })
    csv_path = os.path.join(save_folder, "poly_opt_quartic_geometry.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nGeometry saved to: {csv_path}")
    print("\nDone.")