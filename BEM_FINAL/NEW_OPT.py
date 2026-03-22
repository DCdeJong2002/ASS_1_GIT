"""
optimal_a_distribution.py  —  corrected version
=================================================
Optimises the spanwise axial induction a(r) subject to CT = CT_TARGET.

At each function evaluation, chord and twist are rebuilt analytically
from the current a(r) using the Glauert optimality conditions, so the
blade element physics are always self-consistent.

CP is bounded above by the actuator disk limit for the given CT target.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# CONFIGURATION
# =============================================================================

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

Omega   = U0 * TSR_DESIGN / Radius
TSR_now = Omega * Radius / U0

# Actuator disk upper bound for this CT target
a_AD    = 0.5 * (1.0 - np.sqrt(1.0 - CT_TARGET))
CP_AD   = 4.0 * a_AD * (1.0 - a_AD)**2
print(f"Actuator disk limit:  a = {a_AD:.4f},  CP_AD = {CP_AD:.4f}")

# =============================================================================
# POLAR
# =============================================================================

_df         = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl    = _df["Cl"].to_numpy()
polar_cd    = _df["Cd"].to_numpy()

# Optimal angle of attack — maximises Cl/Cd
def find_optimal_alpha():
    alphas = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_v   = np.interp(alphas, polar_alpha, polar_cl)
    cd_v   = np.interp(alphas, polar_alpha, polar_cd)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cd_v > 1e-6, cl_v / cd_v, 0.0)
    idx = int(np.argmax(ratio))
    return float(alphas[idx]), float(cl_v[idx]), float(cd_v[idx])

alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
print(f"Optimal AoA: α = {alpha_opt:.2f}°,  Cl = {cl_opt:.4f},  "
      f"Cd = {cd_opt:.5f},  Cl/Cd = {cl_opt/cd_opt:.1f}")

# =============================================================================
# BEM CORE
# =============================================================================

def ainduction(CT):
    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1
    if np.isscalar(CT):
        if CT >= CT2:
            return 1.0 + (CT - CT1) / (4.0*(np.sqrt(CT1) - 1.0))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    a = np.zeros_like(np.asarray(CT, dtype=float))
    a[CT >= CT2] = 1.0 + (CT[CT >= CT2] - CT1) / (4.0*(np.sqrt(CT1) - 1.0))
    a[CT < CT2]  = 0.5 - 0.5*np.sqrt(np.maximum(0.0, 1.0 - CT[CT < CT2]))
    return a


def prandtl_F(r_R, a):
    a  = float(np.clip(a, -0.9, 0.99))
    sq = np.sqrt(1.0 + (TSR_now * r_R)**2 / (1.0 - a)**2)
    t1 = -NBlades/2.0 * (TipLocation_R  - r_R) / r_R * sq
    t2 =  NBlades/2.0 * (RootLocation_R - r_R) / r_R * sq
    Ft = 2.0/np.pi * np.arccos(float(np.clip(np.exp(t1), 0.0, 1.0)))
    Fr = 2.0/np.pi * np.arccos(float(np.clip(np.exp(t2), 0.0, 1.0)))
    return max(float(Fr * Ft), 1e-4)


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
    return fnorm, ftan, cl, cd


def solve_streamtube(r1_R, r2_R, chord, twist, max_iter=300, tol=1e-5):
    """Standard BEM iteration — used for baseline comparison."""
    Area    = np.pi * ((r2_R*Radius)**2 - (r1_R*Radius)**2)
    r_mid   = 0.5*(r1_R + r2_R)
    r_local = r_mid * Radius
    a = 0.0; ap = 0.0

    for _ in range(max_iter):
        Vax  = U0*(1.0 - a)
        Vtan = (1.0 + ap)*Omega*r_local
        fn, ft, cl, cd = load_blade_element(Vax, Vtan, chord, twist)
        CT_loc = fn*Radius*(r2_R-r1_R)*NBlades / (0.5*Area*U0**2)
        anew   = ainduction(CT_loc)
        F      = prandtl_F(r_mid, anew)
        anew  /= F; a_old = a
        a      = 0.75*a + 0.25*anew
        ap     = ft*NBlades / (2.0*np.pi*U0*(1.0-a)*Omega*2.0*r_local**2) / F
        if abs(a - a_old) < tol:
            break

    return np.array([a, ap, r_mid, fn, ft, cl, cd])

# =============================================================================
# GRID
# =============================================================================

bins  = np.arange(RootLocation_R, TipLocation_R + DELTA_R_R/2, DELTA_R_R)
r_mid = 0.5*(bins[:-1] + bins[1:])
dr    = np.diff(bins) * Radius
N     = len(r_mid)

# Baseline geometry for comparison
chord_base = 3.0*(1.0 - r_mid) + 1.0
twist_base = -(14.0*(1.0 - r_mid) + Pitch)

# =============================================================================
# PHYSICALLY CONSISTENT FORWARD MODEL
#
# Given a spanwise a(r), rebuild chord and twist analytically from the
# Glauert optimality conditions at each strip, then evaluate forces.
# This ensures every function evaluation is physically self-consistent.
#
# Chord from equating momentum and blade element thrust (Hansen 2008):
#
#   c(r) = 8π r a F (1 - aF) sin²φ / [B (1-a)² Cn]
#
# Twist set so every section operates at α_opt (max Cl/Cd):
#
#   β(r) = α_opt - φ(r) - Pitch
#
# The Pitch offset is included so the design is consistent with the
# baseline and polynomial methods (see pitch treatment discussion).
# =============================================================================

def geometry_from_a(a_vec):
    """
    Build chord and twist analytically from a prescribed a(r).
    Returns chord [N], twist [N] arrays at strip centres.
    """
    chord = np.zeros(N)
    twist = np.zeros(N)

    Cn_opt = cl_opt * np.cos(np.radians(alpha_opt)) \
           + cd_opt * np.sin(np.radians(alpha_opt))

    for i in range(N):
        a       = float(np.clip(a_vec[i], 1e-4, 0.499))
        r_local = r_mid[i] * Radius
        F       = prandtl_F(r_mid[i], a)

        # Inflow angle at optimum: a' is small so use a' ≈ 0 first,
        # then refine once with the Glauert relation a'=(1-3a)/(4a-1)
        # (valid when 1/4 < a < 1/3; clip outside this range)
        if 0.26 < a < 0.333:
            ap = (1.0 - 3.0*a) / (4.0*a - 1.0)
            ap = max(ap, 0.0)
        else:
            ap = 0.0          # fallback: ignore wake rotation correction

        phi_rad = np.arctan2(1.0 - a, TSR_now * r_mid[i] * (1.0 + ap))

        # Twist: β = α_opt − φ − Pitch  (Pitch absorbed so β is geometric twist)
        twist[i] = alpha_opt - np.degrees(phi_rad) - Pitch

        # Chord from momentum / blade element balance
        sin2_phi = np.sin(phi_rad)**2
        denom    = NBlades * (1.0 - a)**2 * max(Cn_opt, 1e-6)
        numer    = 8.0*np.pi*r_local * a * F * (1.0 - a*F) * sin2_phi
        c        = numer / denom

        # Physical chord limits
        chord[i] = float(np.clip(c, 0.3, 6.0))

    return chord, twist


def evaluate_consistent(a_vec):
    """
    Evaluate CT and CP for a given a(r), with geometry rebuilt each time.
    Returns CT, CP, fnorm [N], ftan [N], aline [N].
    """
    chord, twist = geometry_from_a(a_vec)
    fnorm = np.zeros(N)
    ftan  = np.zeros(N)
    aline = np.zeros(N)

    for i in range(N):
        a       = float(np.clip(a_vec[i], 1e-4, 0.499))
        r_local = r_mid[i] * Radius
        chord_i = chord[i]
        twist_i = twist[i]
        F       = prandtl_F(r_mid[i], a)

        # Iterate a' to self-consistency with fixed a
        ap = 0.0
        for _ in range(100):
            Vax  = U0*(1.0 - a)
            Vtan = (1.0 + ap)*Omega*r_local
            fn, ft, *_ = load_blade_element(Vax, Vtan, chord_i, twist_i)
            ap_new = ft*NBlades / (
                2.0*np.pi*U0*(1.0 - a)*Omega*2.0*r_local**2) / F
            if abs(ap_new - ap) < 1e-7:
                ap = ap_new; fn_f = fn; ft_f = ft
                break
            ap = 0.75*ap + 0.25*ap_new
        else:
            fn_f, ft_f = fn, ft

        fnorm[i] = fn_f
        ftan[i]  = ft_f
        aline[i] = ap

    CT = float(np.sum(dr*fnorm*NBlades / (0.5*U0**2*np.pi*Radius**2)))
    CP = float(np.sum(dr*ftan*r_mid*Radius*NBlades*Omega
                      / (0.5*U0**3*np.pi*Radius**2)))
    return CT, CP, fnorm, ftan, aline

# =============================================================================
# SLSQP OPTIMISATION
# =============================================================================

def run_optimal_a():
    print("\n" + "="*60)
    print("Constrained spanwise a(r) optimisation  (SLSQP)")
    print(f"Target CT = {CT_TARGET},  TSR = {TSR_DESIGN},  N = {N} strips")
    print(f"CP must stay below actuator disk limit CP_AD = {CP_AD:.4f}")
    print("="*60)

    # Initial guess: uniform a matching the actuator disk for CT_TARGET
    a0        = np.full(N, a_AD)
    CT0, CP0, *_ = evaluate_consistent(a0)
    print(f"\nInitial guess:  a = {a_AD:.4f} (uniform)")
    print(f"  CT = {CT0:.4f},  CP = {CP0:.4f},  CP/CP_AD = {CP0/CP_AD:.4f}")

    iteration = [0]

    def objective(a_vec):
        iteration[0] += 1
        CT, CP, *_ = evaluate_consistent(a_vec)
        if iteration[0] % 20 == 0:
            print(f"  iter {iteration[0]:4d}  CP = {CP:.5f}  "
                  f"CT = {CT:.5f}  CT_err = {CT-CT_TARGET:+.5f}")
        return -CP

    def ct_constraint(a_vec):
        CT, *_ = evaluate_consistent(a_vec)
        return CT - CT_TARGET

    # Upper bound on CP as an inequality constraint — physics guard
    def cp_upper_bound(a_vec):
        _, CP, *_ = evaluate_consistent(a_vec)
        return CP_AD - CP          # must be >= 0

    constraints = [
        {'type': 'eq',  'fun': ct_constraint},
        {'type': 'ineq','fun': cp_upper_bound},
    ]

    # Box bounds: 0.05 <= a <= 0.45 (stay away from singularities)
    bounds = [(0.05, 0.45)] * N

    print("\nRunning SLSQP ...")
    result = minimize(
        objective,
        a0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-7, 'maxiter': 300, 'disp': True}
    )

    a_opt = result.x
    CT_opt, CP_opt, fnorm_opt, ftan_opt, aline_opt = evaluate_consistent(a_opt)
    chord_opt, twist_opt = geometry_from_a(a_opt)

    print(f"\nResult:  converged = {result.success}")
    print(f"  {result.message}")
    print(f"  CT   = {CT_opt:.6f}  (target {CT_TARGET},"
          f"  error {CT_opt-CT_TARGET:+.2e})")
    print(f"  CP   = {CP_opt:.6f}")
    print(f"  CP_AD= {CP_AD:.6f}")
    print(f"  CP/CP_AD = {CP_opt/CP_AD:.4f}")

    if CP_opt > CP_AD + 1e-4:
        print("  WARNING: CP exceeds actuator disk limit — result unphysical")
    else:
        print("  CP is within actuator disk bound — result is physical")

    # Baseline BEM for comparison
    print("\nRunning baseline BEM ...")
    rows = [solve_streamtube(bins[i], bins[i+1],
                             chord_base[i], twist_base[i])
            for i in range(N)]
    res_base = np.vstack(rows)
    CT_base  = float(np.sum(dr*res_base[:,3]*NBlades
                            / (0.5*U0**2*np.pi*Radius**2)))
    CP_base  = float(np.sum(dr*res_base[:,4]*res_base[:,2]*Radius*NBlades*Omega
                            / (0.5*U0**3*np.pi*Radius**2)))
    print(f"  Baseline:  CT = {CT_base:.6f},  CP = {CP_base:.6f}")

    return (a_opt, CT_opt, CP_opt,
            fnorm_opt, ftan_opt, aline_opt,
            chord_opt, twist_opt,
            res_base, CT_base, CP_base)

# =============================================================================
# PLOTS
# =============================================================================

def plot_results(a_opt, CT_opt, CP_opt,
                 fnorm_opt, ftan_opt, aline_opt,
                 chord_opt, twist_opt,
                 res_base, CT_base, CP_base):

    norm_val = 0.5 * U0**2 * Radius
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ── Axial induction ───────────────────────────────────────────────────────
    axes[0,0].plot(r_mid, a_opt, lw=2, label="Optimal a(r)  [SLSQP]")
    axes[0,0].plot(res_base[:,2], res_base[:,0], "--", lw=1.5,
                   label="Baseline BEM")
    axes[0,0].axhline(1/3, color="grey", ls=":", lw=1, label="Betz  a=1/3")
    axes[0,0].axhline(a_AD, color="red", ls=":", lw=1,
                      label=f"CT={CT_TARGET} uniform  a={a_AD:.3f}")
    axes[0,0].set_xlabel("r/R"); axes[0,0].set_ylabel("a  [-]")
    axes[0,0].set_title("Axial induction")
    axes[0,0].legend(fontsize=8); axes[0,0].grid(True)

    # ── Tangential induction ──────────────────────────────────────────────────
    axes[0,1].plot(r_mid, aline_opt, lw=2, label="Optimal a′(r)")
    axes[0,1].plot(res_base[:,2], res_base[:,1], "--", lw=1.5,
                   label="Baseline BEM")
    axes[0,1].set_xlabel("r/R"); axes[0,1].set_ylabel("a′  [-]")
    axes[0,1].set_title("Tangential induction")
    axes[0,1].legend(fontsize=8); axes[0,1].grid(True)

    # ── Normal loading ────────────────────────────────────────────────────────
    axes[0,2].plot(r_mid, fnorm_opt/norm_val, lw=2,
                   label=f"Optimal  CT={CT_opt:.3f}  CP={CP_opt:.4f}")
    axes[0,2].plot(res_base[:,2], res_base[:,3]/norm_val, "--", lw=1.5,
                   label=f"Baseline CT={CT_base:.3f}  CP={CP_base:.4f}")
    axes[0,2].set_xlabel("r/R"); axes[0,2].set_ylabel(r"$C_n$  [-]")
    axes[0,2].set_title("Normal (thrust) loading")
    axes[0,2].legend(fontsize=8); axes[0,2].grid(True)

    # ── Chord ─────────────────────────────────────────────────────────────────
    axes[1,0].plot(r_mid, chord_opt, lw=2, label="Optimal chord")
    axes[1,0].plot(r_mid, chord_base, "--", lw=1.5, label="Baseline chord")
    axes[1,0].set_xlabel("r/R"); axes[1,0].set_ylabel("Chord  [m]")
    axes[1,0].set_title("Chord distribution")
    axes[1,0].legend(fontsize=8); axes[1,0].grid(True)

    # ── Twist ─────────────────────────────────────────────────────────────────
    axes[1,1].plot(r_mid, twist_opt, lw=2, label="Optimal twist")
    axes[1,1].plot(r_mid, twist_base, "--", lw=1.5, label="Baseline twist")
    axes[1,1].set_xlabel("r/R"); axes[1,1].set_ylabel("Twist  [deg]")
    axes[1,1].set_title("Twist distribution")
    axes[1,1].legend(fontsize=8); axes[1,1].grid(True)

    # ── Performance bar ───────────────────────────────────────────────────────
    labels = ["Baseline", "Optimal a(r)", "Actuator disk"]
    CP_vals = [CP_base, CP_opt, CP_AD]
    CT_vals = [CT_base, CT_opt, CT_TARGET]
    x = np.arange(3); w = 0.35
    b1 = axes[1,2].bar(x - w/2, CP_vals, w, label="CP", color="steelblue")
    b2 = axes[1,2].bar(x + w/2, CT_vals, w, label="CT", color="coral")
    axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(labels)
    axes[1,2].set_ylabel("Coefficient  [-]")
    axes[1,2].set_title("Performance summary")
    axes[1,2].bar_label(b1, fmt="%.4f", padding=3, fontsize=8)
    axes[1,2].bar_label(b2, fmt="%.4f", padding=3, fontsize=8)
    axes[1,2].legend(); axes[1,2].grid(True, axis="y")

    fig.suptitle(
        f"Constrained a(r) optimisation  —  TSR={TSR_DESIGN},  "
        f"CT target={CT_TARGET},  CP_AD={CP_AD:.4f}",
        fontsize=13)
    fig.tight_layout()
    plt.savefig("optimal_a_distribution.png", dpi=200, bbox_inches="tight")
    print("\nPlot saved: optimal_a_distribution.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_optimal_a()
    plot_results(*results)