import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

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

Radius         = 50        # Rotor radius [m]
NBlades        = 3         # Number of blades
U0             = 10        # Freestream velocity [m/s]
RootLocation_R = 0.2
TipLocation_R  = 1.0

TSR_design = 8.0
CT_target  = 0.75

MAX_CHORD     = 3.4        # Structural root limit [m]
MIN_CHORD_TIP = 0.2        # Manufacturing tip limit [m]

# =============================================================================
# Prandtl formula toggle
# =============================================================================

# Set to True  -> helical-wake spacing formula (original Prandtl 1919, more
#                 physically correct, used by QBlade / OpenFAST).
# Set to False -> simplified trigonometric form (Glauert 1935, matches the
#                 reference notebook exactly, standard in most BEM courses).
USE_HELICAL_PRANDTL = False

# =============================================================================
# BEM core functions  — copied verbatim from the final reference class
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
    """
    Glauert (1935) simplified form.
    Uses the local speed ratio TSR*r_R at each station to approximate
    1/sin(phi). Exact at r/R=1, increasingly approximate inboard.
    Matches the reference notebook exactly.
    """
    a = np.clip(axial_induction, -0.9, 0.99)

    temp_tip = (
        -NBlades / 2 * (tipradius_R - r_R) / r_R
        * np.sqrt(1 + ((TSR * r_R) ** 2) / ((1 - a) ** 2))
    )
    Ftip = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0, 1)))

    temp_root = (
        NBlades / 2 * (rootradius_R - r_R) / r_R
        * np.sqrt(1 + ((TSR * r_R) ** 2) / ((1 - a) ** 2))
    )
    Froot = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0, 1)))

    return Froot * Ftip


def _PrandtlHelical(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Original Prandtl (1919) helical-wake spacing form.
    Computes d = axial distance between successive wake sheets (normalised
    by R) from the global tip TSR, then uses -pi*(tipR-r_R)/d as the exponent.
    More physically correct: the wake sheet spacing is set by the global tip
    geometry, not the local speed ratio. tipradius_R is used directly in the
    tip exponent so the formula works for any tip location.
    """
    a = np.clip(axial_induction, -0.9, 0.99)
    d = max(2 * np.pi / NBlades * (1 - a) / np.sqrt(TSR ** 2 + (1 - a) ** 2), 1e-8)

    Ftip  = float(2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (tipradius_R - r_R) / d, -500, 0))))
    Froot = float(2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (r_R - rootradius_R) / d, -500, 0))))

    Ftip  = 0.0 if np.isnan(Ftip)  else Ftip
    Froot = 0.0 if np.isnan(Froot) else Froot

    return Froot * Ftip


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Dispatcher: routes to the simplified or helical-wake Prandtl formula
    depending on the USE_HELICAL_PRANDTL flag at the top of this file.
    Both implementations share the same argument signature so all call
    sites remain unchanged.
    """
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
# Rotor evaluator — runs SolveStreamtube on a full (r, chord, twist) geometry
# =============================================================================

def evaluate_rotor(r_nodes, c_nodes, twist_nodes_deg, tsr=TSR_design):
    """
    Evaluate CT and CP for an arbitrary blade geometry.

    Parameters
    ----------
    r_nodes        : radial node positions [m],  length N
    c_nodes        : chord at nodes [m],          length N
    twist_nodes_deg: twist at nodes [deg], sign convention: alpha = twist + phi_deg
    tsr            : tip-speed ratio

    Returns
    -------
    CT, CP, results_arr
        results_arr columns: [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi]
    """
    Omega = U0 * tsr / Radius

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

    CT = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP = np.sum(
        dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega
        / (0.5 * U0**3 * np.pi * Radius**2)
    )

    return CT, CP, res_arr


# =============================================================================
# Step 1 — find aerodynamic optimum (max Cl/Cd) from the polar
# =============================================================================

def find_optimal_alpha():
    """Return (alpha_opt_deg, cl_opt, cd_opt) at maximum Cl/Cd."""
    alphas  = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_vals = np.interp(alphas, polar_alpha, polar_cl)
    cd_vals = np.interp(alphas, polar_alpha, polar_cd)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cd_vals > 1e-6, cl_vals / cd_vals, 0.0)

    idx           = int(np.argmax(ratio))
    alpha_opt_deg = float(alphas[idx])
    cl_opt        = float(cl_vals[idx])
    cd_opt        = float(cd_vals[idx])

    print(f"Aerodynamic optimum: alpha = {alpha_opt_deg:.2f} deg, "
          f"Cl = {cl_opt:.4f}, Cd = {cd_opt:.5f}, "
          f"Cl/Cd = {cl_opt / cd_opt:.1f}")

    return alpha_opt_deg, cl_opt, cd_opt


# =============================================================================
# Step 2 — analytically generate chord and twist for a target induction
# =============================================================================

def generate_ideal_rotor(target_a, n_nodes=101,
                         alpha_opt_deg=None, cl_opt=None, cd_opt=None,
                         ap_in=None):
    """
    Compute the ideal chord and twist at each span station so the rotor
    operates at max Cl/Cd everywhere with uniform axial induction target_a.

    Chord is derived by inverting the BEM CT momentum equation:
        CT_loc = fnorm·R·Δr·B / (0.5·Area·U0²)
    Setting CT_loc = 4·a·F·(1 − a·F) and solving for chord gives:
        c = 8π·r · a·F·(1−a·F) · sin²φ / (B·(1−a)²·Cn)

    Twist uses the reference class sign convention:
        twist_deg = alpha_opt − phi_deg
    so that alpha = twist + phi_deg = alpha_opt.

    Parameters
    ----------
    target_a      : uniform design axial induction
    n_nodes       : spanwise node count
    alpha_opt_deg : override from find_optimal_alpha (computed if None)
    cl_opt, cd_opt: aero coefficients at alpha_opt (computed if None)
    ap_in         : tangential induction from previous BEM run (annulus midpoints);
                    refines phi on subsequent Brent iterations

    Returns
    -------
    r_nodes [m], c_nodes [m], twist_nodes_deg [deg]
    """
    if alpha_opt_deg is None:
        alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()

    Omega = U0 * TSR_design / Radius

    # Cosine-spaced nodes — denser near root and tip
    r_start  = RootLocation_R * Radius + 0.005 * Radius
    r_end    = TipLocation_R  * Radius - 0.005 * Radius
    r_nodes  = r_start + (r_end - r_start) * (
        1 - np.cos(np.linspace(0, np.pi, n_nodes))
    ) / 2

    c_nodes         = np.zeros(n_nodes)
    twist_nodes_deg = np.zeros(n_nodes)

    # Interpolate previous ap onto current node positions
    if ap_in is not None:
        r_mid_prev  = 0.5 * (r_nodes[:-1] + r_nodes[1:])
        ap_at_nodes = np.interp(r_nodes, r_mid_prev, ap_in,
                                left=ap_in[0], right=ap_in[-1])
    else:
        ap_at_nodes = np.zeros(n_nodes)

    for i, r in enumerate(r_nodes):
        r_R       = r / Radius
        local_tsr = TSR_design * r_R

        # Prandtl correction at design condition
        F = float(max(
            PrandtlTipRootCorrection(
                r_R, RootLocation_R, TipLocation_R,
                TSR_design, NBlades, target_a
            ),
            0.0001
        ))

        ap  = float(ap_at_nodes[i])
        phi = np.arctan2(1.0 - target_a, local_tsr * (1.0 + ap))   # [rad]

        # Twist so the section sits at alpha_opt
        # Convention: alpha = twist_deg + phi_deg  =>  twist_deg = alpha_opt − phi_deg
        twist_nodes_deg[i] = alpha_opt_deg - np.degrees(phi)

        # Normal force coefficient
        Cn = cl_opt * np.cos(phi) + cd_opt * np.sin(phi)

        # Ideal chord from momentum-BEM equality:
        #   4·a·F·(1−a·F) = fnorm·R·Δr·B / (0.5·Area·U0²)
        # Which reduces to:
        #   c = 8π·r · a·F·(1−a·F) · sin²φ / (B·(1−a)²·Cn)
        numerator   = 8 * np.pi * r * target_a * F * (1 - target_a * F) * (np.sin(phi) ** 2)
        denominator = NBlades * (1 - target_a) ** 2 * max(Cn, 1e-8)
        c_nodes[i]  = numerator / denominator

    # --- Physical constraints ---
    # 1. Cap at structural maximum (root)
    c_nodes = np.clip(c_nodes, 0.0, MAX_CHORD)
    # 2. Hold everything inboard of the chord maximum at that maximum
    max_idx = int(np.argmax(c_nodes))
    c_nodes[: max_idx + 1] = c_nodes[max_idx]
    # 3. Enforce minimum tip chord
    c_nodes = np.clip(c_nodes, MIN_CHORD_TIP, None)

    return r_nodes, c_nodes, twist_nodes_deg


# =============================================================================
# Step 3 — root-find the exact target_a that yields CT = CT_target
# =============================================================================

def design_for_exact_ct():
    """
    Use Brent's method to find the axial induction a such that the
    analytically designed rotor achieves exactly CT_target when
    evaluated by the reference BEM solver.

    The tangential induction profile from each BEM call is fed back
    into generate_ideal_rotor to refine phi on the next iteration.

    Returns
    -------
    r_opt, c_opt, twist_opt_deg, CT_final, CP_final, results_arr
    """
    alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()

    last_ap   = [None]
    last_geom = [None]

    def residual(target_a):
        r, c, tw = generate_ideal_rotor(
            target_a,
            alpha_opt_deg=alpha_opt_deg,
            cl_opt=cl_opt,
            cd_opt=cd_opt,
            ap_in=last_ap[0],
        )
        last_geom[0] = (r, c, tw)

        # Silence solver prints during root-finding
        old_out, sys.stdout = sys.stdout, io.StringIO()
        try:
            CT, CP, res_arr = evaluate_rotor(r, c, tw)
        finally:
            sys.stdout = old_out

        # Store aline column for next call
        last_ap[0] = res_arr[:, 1]

        print(f"  target_a = {target_a:.5f}  →  CT = {CT:.5f},  CP = {CP:.5f}")
        return CT - CT_target

    print("\n" + "=" * 65)
    print(f"Root-finding: searching for a such that CT = {CT_target}")
    print("=" * 65)

    sol = root_scalar(residual, bracket=[0.20, 0.35], method="brentq", xtol=1e-5)

    if not sol.converged:
        raise RuntimeError("root_scalar did not converge — try widening the bracket.")

    best_a = sol.root
    print(f"\nConverged:  design induction a = {best_a:.6f}")

    r_opt, c_opt, tw_opt = last_geom[0]
    CT_final, CP_final, results_arr = evaluate_rotor(r_opt, c_opt, tw_opt)

    print(f"Final BEM:  CT = {CT_final:.6f}  (target {CT_target})")
    print(f"            CP = {CP_final:.6f}")

    return r_opt, c_opt, tw_opt, CT_final, CP_final, results_arr


# =============================================================================
# Baseline geometry and evaluation (original assignment blade)
# =============================================================================

def get_baseline_geometry(n_nodes=201):
    """
    Original linear chord/twist blade.
    Uses the reference class sign convention: twist = -(14*(1-r/R) + Pitch).
    """
    r_nodes        = np.linspace(RootLocation_R * Radius, TipLocation_R * Radius, n_nodes)
    r_R            = r_nodes / Radius
    c_nodes        = 3.0 * (1 - r_R) + 1.0
    twist_nodes    = -(14.0 * (1 - r_R) + (-2.0))   # Pitch = -2 deg
    return r_nodes, c_nodes, twist_nodes


# =============================================================================
# Plotting helpers
# =============================================================================

base_path   = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_analytical")
os.makedirs(save_folder, exist_ok=True)


def save_and_show(filename):
    path = os.path.join(save_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


def plot_all(r_base, c_base, tw_base, res_base,
             r_opt,  c_opt,  tw_opt,  res_opt,
             CT_base, CP_base, CT_opt, CP_opt):

    r_R_base = np.array([r[2] for r in res_base])   # r_mid from results
    r_R_opt  = np.array([r[2] for r in res_opt])

    # Midpoint chord/twist for baseline (nodes → midpoints for plotting)
    c_base_mid  = 0.5 * (c_base[:-1]  + c_base[1:])
    tw_base_mid = 0.5 * (tw_base[:-1] + tw_base[1:])

    norm_val = 0.5 * U0**2 * Radius

    # ── Plot 1: chord distribution ──────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, c_base_mid, label=f"Baseline  (CT={CT_base:.3f})")
    plt.plot(r_opt / Radius, c_opt,  label=f"Analytical opt  (CT={CT_opt:.3f})")
    plt.axhline(MIN_CHORD_TIP, color="grey", linestyle=":", linewidth=1, label="Min chord")
    plt.axhline(MAX_CHORD,     color="grey", linestyle="--", linewidth=1, label="Max chord")
    plt.xlabel("r/R")
    plt.ylabel("Chord [m]")
    plt.title("Chord distribution")
    plt.legend()
    plt.grid(True)
    save_and_show("01_chord_distribution.png")

    # ── Plot 2: twist distribution ──────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, tw_base_mid, label="Baseline")
    plt.plot(r_opt / Radius, tw_opt,  label="Analytical opt", linestyle="--")
    plt.xlabel("r/R")
    plt.ylabel("Twist [deg]")
    plt.title("Twist distribution  (convention: alpha = twist + phi_deg)")
    plt.legend()
    plt.grid(True)
    save_and_show("02_twist_distribution.png")

    # ── Plot 3: induction factors ────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, [r[0] for r in res_base], label="Baseline axial a")
    plt.plot(r_R_base, [r[1] for r in res_base], label="Baseline tangential a'")
    plt.plot(r_R_opt,  [r[0] for r in res_opt],  label="Analytical axial a",       linestyle="--")
    plt.plot(r_R_opt,  [r[1] for r in res_opt],  label="Analytical tangential a'", linestyle="--")
    plt.xlabel("r/R")
    plt.ylabel("Induction factor [-]")
    plt.title("Induction distributions")
    plt.legend()
    plt.grid(True)
    save_and_show("03_induction_distribution.png")

    # ── Plot 4: section loading ──────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, [r[3] / norm_val for r in res_base], label="Baseline fnorm")
    plt.plot(r_R_base, [r[4] / norm_val for r in res_base], label="Baseline ftan")
    plt.plot(r_R_opt,  [r[3] / norm_val for r in res_opt],  label="Analytical fnorm", linestyle="--")
    plt.plot(r_R_opt,  [r[4] / norm_val for r in res_opt],  label="Analytical ftan",  linestyle="--")
    plt.xlabel("r/R")
    plt.ylabel(r"$F\,/\,({\frac{1}{2}}\,U_\infty^2\,R)$")
    plt.title("Section loading")
    plt.legend()
    plt.grid(True)
    save_and_show("04_loading_distribution.png")

    # ── Plot 5: angle of attack and inflow angle ─────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(r_R_base, [r[6] for r in res_base], label=r"Baseline $\alpha$")
    plt.plot(r_R_base, [r[7] for r in res_base], label=r"Baseline $\phi$")
    plt.plot(r_R_opt,  [r[6] for r in res_opt],  label=r"Analytical $\alpha$", linestyle="--")
    plt.plot(r_R_opt,  [r[7] for r in res_opt],  label=r"Analytical $\phi$",   linestyle="--")
    plt.xlabel("r/R")
    plt.ylabel("Angle [deg]")
    plt.title("Angle of attack and inflow angle")
    plt.legend()
    plt.grid(True)
    save_and_show("05_angles_distribution.png")

    # ── Plot 6: performance bar chart ────────────────────────────────────────
    a_ad  = 0.5 * (1.0 - np.sqrt(1.0 - CT_target))
    CP_ad = 4.0 * a_ad * (1.0 - a_ad) ** 2

    fig, ax = plt.subplots(figsize=(6, 5))
    labels  = ["Baseline", "Analytical opt", "Actuator disk"]
    cp_vals = [CP_base, CP_opt, CP_ad]
    ct_vals = [CT_base, CT_opt, CT_target]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, cp_vals, w, label="CP")
    b2 = ax.bar(x + w/2, ct_vals, w, label="CT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coefficient [-]")
    ax.set_title("Performance comparison")
    ax.legend()
    ax.bar_label(b1, fmt="%.4f", padding=3)
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
    print("BASELINE BLADE  (linear chord/twist, pitch = −2 deg, TSR = 8)")
    print("=" * 65)

    r_base, c_base, tw_base = get_baseline_geometry()
    CT_base, CP_base, res_base_arr = evaluate_rotor(r_base, c_base, tw_base)
    res_base = [list(row) for row in res_base_arr]

    print(f"Baseline  CT = {CT_base:.6f}")
    print(f"Baseline  CP = {CP_base:.6f}")

    # ── Analytical optimal design ─────────────────────────────────────────────
    r_opt, c_opt, tw_opt, CT_opt, CP_opt, res_opt_arr = design_for_exact_ct()
    res_opt = [list(row) for row in res_opt_arr]

    # ── Actuator disk reference ───────────────────────────────────────────────
    a_ad  = 0.5 * (1.0 - np.sqrt(1.0 - CT_target))
    CP_ad = 4.0 * a_ad * (1.0 - a_ad) ** 2

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Actuator disk  CT={CT_target}:  a = {a_ad:.4f},  CP = {CP_ad:.6f}")
    print(f"Baseline       CT = {CT_base:.6f},  CP = {CP_base:.6f}"
          f"  (CP/CP_AD = {CP_base/CP_ad:.4f})")
    print(f"Analytical opt CT = {CT_opt:.6f},  CP = {CP_opt:.6f}"
          f"  (CP/CP_AD = {CP_opt/CP_ad:.4f})")
    print(f"CP improvement = {100*(CP_opt - CP_base)/abs(CP_base):.2f} %")

    # ── Geometry table ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("OPTIMIZED GEOMETRY (selected stations)")
    print("=" * 65)
    idxs = [0, len(r_opt)//4, len(r_opt)//2, 3*len(r_opt)//4, -1]
    print(f"{'r/R':>7}  {'Chord [m]':>10}  {'Twist [deg]':>12}")
    for i in idxs:
        print(f"{r_opt[i]/Radius:>7.3f}  {c_opt[i]:>10.4f}  {tw_opt[i]:>12.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(r_base, c_base, tw_base, res_base,
             r_opt,  c_opt,  tw_opt,  res_opt,
             CT_base, CP_base, CT_opt, CP_opt)

    # ── Save geometry to CSV ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "r_m":       r_opt,
        "r_R":       r_opt / Radius,
        "chord_m":   c_opt,
        "twist_deg": tw_opt,
    })
    csv_path = os.path.join(save_folder, "analytical_optimal_geometry.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nGeometry saved to: {csv_path}")
    print("\nDone.")