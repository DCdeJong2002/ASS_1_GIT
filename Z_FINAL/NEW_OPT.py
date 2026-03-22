"""
optimal_a_distribution.py
==========================
Constrained spanwise a(r) optimisation using the existing BEM core.

Finds the spanwise axial induction distribution a(r) that maximises CP
subject to the integral thrust constraint CT = CT_TARGET.

Method: SLSQP (Sequential Least Squares Programming) via scipy.optimize.minimize.
The tangential induction a'(r) at each station is obtained from the converged
BEM solution evaluated at the current a(r) guess, so the blade element physics
are fully respected at every function evaluation.

Usage:
    python optimal_a_distribution.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# PASTE / IMPORT YOUR BASE BEM FUNCTIONS HERE
# (or place this file in the same directory and import them)
# =============================================================================

import pandas as pd
import sys, io

# ── Polar ─────────────────────────────────────────────────────────────────────
_df         = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl    = _df["Cl"].to_numpy()
polar_cd    = _df["Cd"].to_numpy()

# ── Configuration (must match assignment.py) ──────────────────────────────────
USE_HELICAL_PRANDTL = False
Radius         = 50.0
NBlades        = 3
U0             = 10.0
rho            = 1.225
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0
TSR_DESIGN     = 8.0
CT_TARGET      = 0.75
DELTA_R_R      = 0.005

Omega = U0 * TSR_DESIGN / Radius

# ── Copy the four core BEM functions verbatim from assignment.py ──────────────

def ainduction(CT):
    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1
    if np.isscalar(CT):
        if CT >= CT2:
            return 1.0 + (CT - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    a = np.zeros_like(np.asarray(CT, dtype=float))
    a[CT >= CT2] = 1.0 + (CT[CT >= CT2] - CT1) / (4.0*(np.sqrt(CT1)-1.0))
    a[CT < CT2]  = 0.5 - 0.5*np.sqrt(np.maximum(0.0, 1.0 - CT[CT < CT2]))
    return a


def _prandtl_simplified(r_R, rootR, tipR, TSR, NB, a_in):
    a  = float(np.clip(a_in, -0.9, 0.99))
    sq = np.sqrt(1.0 + (TSR * r_R)**2 / (1.0 - a)**2)
    t1 = -NB / 2.0 * (tipR  - r_R) / r_R * sq
    t2 =  NB / 2.0 * (rootR - r_R) / r_R * sq
    Ft = 2.0/np.pi * np.arccos(float(np.clip(np.exp(t1), 0.0, 1.0)))
    Fr = 2.0/np.pi * np.arccos(float(np.clip(np.exp(t2), 0.0, 1.0)))
    return float(Fr * Ft)


def prandtl(r_R, rootR, tipR, TSR, NB, a):
    return _prandtl_simplified(r_R, rootR, tipR, TSR, NB, a)


def load_blade_element(vnorm, vtan, chord, twist):
    vmag2 = vnorm**2 + vtan**2
    phi   = np.arctan2(vnorm, vtan)
    alpha = twist + np.degrees(phi)
    cl    = float(np.interp(alpha, polar_alpha, polar_cl))
    cd    = float(np.interp(alpha, polar_alpha, polar_cd))
    lift  = 0.5 * vmag2 * cl * chord
    drag  = 0.5 * vmag2 * cd * chord
    fnorm = lift * np.cos(phi) + drag * np.sin(phi)
    ftan  = lift * np.sin(phi) - drag * np.cos(phi)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord
    return fnorm, ftan, gamma, alpha, float(np.degrees(phi)), cl, cd


def solve_streamtube(r1_R, r2_R, Omega, chord, twist, max_iter=300, tol=1e-5):
    Area    = np.pi * ((r2_R*Radius)**2 - (r1_R*Radius)**2)
    r_mid   = 0.5*(r1_R + r2_R)
    r_local = r_mid * Radius
    TSR_now = Omega * Radius / U0
    a = 0.0; aline = 0.0; cl_f = cd_f = 0.0

    for _ in range(max_iter):
        Vax  = U0 * (1.0 - a)
        Vtan = (1.0 + aline) * Omega * r_local
        fnorm, ftan, gamma, alpha, phi, cl, cd = load_blade_element(
            Vax, Vtan, chord, twist)
        cl_f = cl; cd_f = cd
        CT_loc = fnorm*Radius*(r2_R-r1_R)*NBlades / (0.5*Area*U0**2)
        anew   = ainduction(CT_loc)
        F      = max(prandtl(r_mid, RootLocation_R, TipLocation_R,
                             TSR_now, NBlades, anew), 1e-4)
        anew  /= F; a_old = a
        a      = 0.75*a + 0.25*anew
        aline  = ftan*NBlades / (
            2.0*np.pi*U0*(1.0-a)*Omega*2.0*r_local**2) / F
        if abs(a - a_old) < tol:
            break

    return (np.array([a, aline, r_mid, fnorm, ftan, gamma,
                      alpha, phi, cl_f, cd_f], dtype=float))

# =============================================================================
# GRID
# =============================================================================

bins   = np.arange(RootLocation_R, TipLocation_R + DELTA_R_R/2, DELTA_R_R)
r_mid  = 0.5*(bins[:-1] + bins[1:])        # strip centres  [N]
dr     = np.diff(bins) * Radius             # strip widths   [N]  (metres)
N      = len(r_mid)

# Baseline chord and twist at each strip centre (same as assignment.py)
chord_base = 3.0*(1.0 - r_mid) + 1.0
twist_base = -(14.0*(1.0 - r_mid) + Pitch)

# =============================================================================
# FORWARD MODEL
# Evaluate the rotor given a prescribed spanwise a(r) vector.
#
# For each strip we fix the axial induction to the prescribed value,
# derive a' from the tangential momentum equation, then compute forces
# using the blade element equations. This keeps full BEM physics while
# allowing the optimiser to directly control a(r).
# =============================================================================

def evaluate_prescribed_a(a_vec, chord=chord_base, twist=twist_base):
    """
    Given a spanwise axial induction vector a_vec [N], compute forces
    at each strip using the blade element equations with Prandtl correction.

    The tangential induction a' is iterated to self-consistency at each
    strip with a fixed, so the blade element physics are fully respected.

    Returns
    -------
    fnorm : np.ndarray [N]   normal force per unit span [N/m]
    ftan  : np.ndarray [N]   tangential force per unit span [N/m]
    aline : np.ndarray [N]   converged tangential induction [-]
    """
    fnorm = np.zeros(N)
    ftan  = np.zeros(N)
    aline = np.zeros(N)
    TSR_now = Omega * Radius / U0

    for i in range(N):
        a       = float(np.clip(a_vec[i], 0.0, 0.5))
        r_local = r_mid[i] * Radius
        chord_i = chord[i]
        twist_i = twist[i]

        # Prandtl factor at this station using prescribed a
        F = max(prandtl(r_mid[i], RootLocation_R, TipLocation_R,
                        TSR_now, NBlades, a), 1e-4)

        # Iterate a' to self-consistency with fixed a
        ap = 0.0
        for _ in range(100):
            Vax  = U0 * (1.0 - a)
            Vtan = (1.0 + ap) * Omega * r_local
            fn, ft, *_ = load_blade_element(Vax, Vtan, chord_i, twist_i)
            ap_new = ft * NBlades / (
                2.0*np.pi * U0*(1.0-a) * Omega * 2.0*r_local**2) / F
            if abs(ap_new - ap) < 1e-7:
                ap = ap_new
                fn_f, ft_f = fn, ft
                break
            ap = 0.75*ap + 0.25*ap_new
        else:
            fn_f, ft_f = fn, ft

        fnorm[i] = fn_f
        ftan[i]  = ft_f
        aline[i] = ap

    return fnorm, ftan, aline


def compute_CT_CP(a_vec, chord=chord_base, twist=twist_base):
    """Integrate fnorm and ftan to get rotor CT and CP."""
    fnorm, ftan, _ = evaluate_prescribed_a(a_vec, chord, twist)
    CT = float(np.sum(dr * fnorm * NBlades / (0.5*U0**2*np.pi*Radius**2)))
    CP = float(np.sum(dr * ftan * r_mid * Radius * NBlades * Omega
                      / (0.5*U0**3*np.pi*Radius**2)))
    return CT, CP

# =============================================================================
# SLSQP OPTIMISATION
# Maximise CP subject to CT = CT_TARGET.
# Decision variable: a(r) — one value per strip.
# =============================================================================

def run_optimal_a():
    print("="*60)
    print("Constrained spanwise a(r) optimisation  (SLSQP)")
    print(f"Target CT = {CT_TARGET},  TSR = {TSR_DESIGN},  N strips = {N}")
    print("="*60)

    # ── Initial guess: uniform a from scalar Betz approximation ──────────────
    a0_scalar = 0.5*(1.0 - np.sqrt(1.0 - CT_TARGET))
    a0        = np.full(N, a0_scalar)
    CT0, CP0  = compute_CT_CP(a0)
    print(f"\nInitial guess:  a = {a0_scalar:.4f} (uniform)")
    print(f"  CT = {CT0:.4f},  CP = {CP0:.4f}")

    # ── Objective: negative CP (we minimise) ─────────────────────────────────
    call_count = [0]

    def objective(a_vec):
        call_count[0] += 1
        _, CP = compute_CT_CP(a_vec)
        if call_count[0] % 50 == 0:
            CT_cur, _ = compute_CT_CP(a_vec)
            print(f"  iter ~{call_count[0]:4d}  CP = {CP:.5f}  "
                  f"CT = {CT_cur:.5f}")
        return -CP

    # ── Equality constraint: CT = CT_TARGET ───────────────────────────────────
    def ct_constraint(a_vec):
        CT, _ = compute_CT_CP(a_vec)
        return CT - CT_TARGET          # = 0 at feasibility

    constraints = [{'type': 'eq', 'fun': ct_constraint}]

    # ── Bounds: 0 <= a(r) <= 0.5 at every strip ───────────────────────────────
    bounds = [(0.0, 0.5)] * N

    # ── Run SLSQP ─────────────────────────────────────────────────────────────
    print("\nRunning SLSQP ...")
    result = minimize(
        objective,
        a0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'ftol'   : 1e-8,
            'maxiter': 500,
            'disp'   : True,
        }
    )

    a_opt       = result.x
    CT_opt, CP_opt = compute_CT_CP(a_opt)

    print(f"\nOptimisation complete:  converged = {result.success}")
    print(f"  Message : {result.message}")
    print(f"  CT      = {CT_opt:.6f}  (target {CT_TARGET})")
    print(f"  CP      = {CP_opt:.6f}")
    print(f"  CP/CP0  = {CP_opt/CP0:.4f}  (improvement over uniform-a guess)")

    # ── Comparison: standard BEM at TSR=8 (baseline geometry) ────────────────
    print("\nRunning baseline BEM for comparison ...")
    rows = []
    for i in range(N):
        row = solve_streamtube(bins[i], bins[i+1], Omega,
                               chord_base[i], twist_base[i])
        rows.append(row)
    res_base = np.vstack(rows)
    CT_base  = float(np.sum(dr*res_base[:,3]*NBlades
                            / (0.5*U0**2*np.pi*Radius**2)))
    CP_base  = float(np.sum(dr*res_base[:,4]*res_base[:,2]*Radius*NBlades*Omega
                            / (0.5*U0**3*np.pi*Radius**2)))
    print(f"  Baseline BEM:  CT = {CT_base:.6f},  CP = {CP_base:.6f}")

    return a_opt, CT_opt, CP_opt, res_base, CT_base, CP_base

# =============================================================================
# PLOTS
# =============================================================================

def plot_results(a_opt, CT_opt, CP_opt, res_base, CT_base, CP_base):
    fnorm_opt, ftan_opt, aline_opt = evaluate_prescribed_a(a_opt)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Axial induction ───────────────────────────────────────────────────────
    axes[0].plot(r_mid, a_opt,        label="Optimal a(r)  [SLSQP]", lw=2)
    axes[0].plot(res_base[:,2], res_base[:,0], "--",
                 label="Baseline BEM", lw=1.5)
    axes[0].axhline(1/3, color="grey", ls=":", lw=1,
                    label="Betz optimum  a=1/3")
    axes[0].set_xlabel("r/R")
    axes[0].set_ylabel("a  [-]")
    axes[0].set_title("Axial induction distribution")
    axes[0].legend(); axes[0].grid(True)

    # ── Tangential induction ──────────────────────────────────────────────────
    axes[1].plot(r_mid, aline_opt,    label="Optimal a′(r)  [SLSQP]", lw=2)
    axes[1].plot(res_base[:,2], res_base[:,1], "--",
                 label="Baseline BEM", lw=1.5)
    axes[1].set_xlabel("r/R")
    axes[1].set_ylabel("a′  [-]")
    axes[1].set_title("Tangential induction distribution")
    axes[1].legend(); axes[1].grid(True)

    # ── Normal loading ────────────────────────────────────────────────────────
    norm_val = 0.5 * U0**2 * Radius
    axes[2].plot(r_mid, fnorm_opt/norm_val,
                 label=f"Optimal  CT={CT_opt:.3f}  CP={CP_opt:.4f}", lw=2)
    axes[2].plot(res_base[:,2], res_base[:,3]/norm_val, "--",
                 label=f"Baseline CT={CT_base:.3f}  CP={CP_base:.4f}", lw=1.5)
    axes[2].set_xlabel("r/R")
    axes[2].set_ylabel(r"$C_n = f_n\,/\,(\frac{1}{2}U_0^2 R)$  [-]")
    axes[2].set_title("Normal (thrust) loading")
    axes[2].legend(); axes[2].grid(True)

    fig.suptitle(
        f"Constrained a(r) optimisation  —  "
        f"TSR={TSR_DESIGN},  CT target={CT_TARGET}",
        fontsize=13)
    fig.tight_layout()
    plt.savefig("optimal_a_distribution.png", dpi=200, bbox_inches="tight")
    print("\nPlot saved: optimal_a_distribution.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    a_opt, CT_opt, CP_opt, res_base, CT_base, CP_base = run_optimal_a()
    plot_results(a_opt, CT_opt, CP_opt, res_base, CT_base, CP_base)