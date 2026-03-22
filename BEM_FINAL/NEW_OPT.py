
"""
optimal_a_distribution.py  —  v3
==================================
Constrained spanwise a(r) optimisation subject to:
  - CT = CT_TARGET          (SLSQP equality)
  - chord(root) = 3.4 m    (enforced by construction — a0 is pinned)
  - chord(tip)  >= 0.3 m   (SLSQP inequality)

Root chord is enforced by construction:
  Before the optimisation we solve the chord equation for the unique
  axial induction a_root that produces c_root = 3.4 m at r/R = 0.2.
  The polynomial constant term a0 is then fixed to a_root and only
  the shape coefficients [a1, a2, a3, a4] are optimised.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, brentq

# =============================================================================
# CONFIGURATION
# =============================================================================

Radius         = 50.0
NBlades        = 3
U0             = 10.0
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0
TSR_DESIGN     = 8.0
CT_TARGET      = 0.75
DELTA_R_R      = 0.005

CHORD_ROOT_TARGET = 3.4
CHORD_TIP_MIN     = 0.3

Omega   = U0 * TSR_DESIGN / Radius
TSR_now = Omega * Radius / U0

a_AD  = 0.5 * (1.0 - np.sqrt(1.0 - CT_TARGET))
CP_AD = 4.0 * a_AD * (1.0 - a_AD)**2

# =============================================================================
# POLAR
# =============================================================================

_df         = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl    = _df["Cl"].to_numpy()
polar_cd    = _df["Cd"].to_numpy()


def find_optimal_alpha():
    alphas = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_v   = np.interp(alphas, polar_alpha, polar_cl)
    cd_v   = np.interp(alphas, polar_alpha, polar_cd)
    ratio  = np.where(cd_v > 1e-6, cl_v / cd_v, 0.0)
    idx    = int(np.argmax(ratio))
    return float(alphas[idx]), float(cl_v[idx]), float(cd_v[idx])


alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
Cn_opt = (cl_opt * np.cos(np.radians(alpha_opt))
          + cd_opt * np.sin(np.radians(alpha_opt)))

print(f"Actuator disk limit:  a = {a_AD:.4f},  CP_AD = {CP_AD:.4f}")
print(f"Optimal AoA: alpha = {alpha_opt:.2f} deg,  Cl/Cd = {cl_opt/cd_opt:.1f}")

# =============================================================================
# BEM CORE
# =============================================================================

def ainduction(CT):
    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1
    if CT >= CT2:
        return 1.0 + (CT - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
    return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))


def prandtl_F(r_R, a):
    a  = float(np.clip(a, -0.9, 0.99))
    sq = np.sqrt(1.0 + (TSR_now * r_R)**2 / (1.0 - a)**2)
    t1 = -NBlades / 2.0 * (TipLocation_R  - r_R) / r_R * sq
    t2 =  NBlades / 2.0 * (RootLocation_R - r_R) / r_R * sq
    Ft = 2.0 / np.pi * np.arccos(float(np.clip(np.exp(t1), 0.0, 1.0)))
    Fr = 2.0 / np.pi * np.arccos(float(np.clip(np.exp(t2), 0.0, 1.0)))
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
    return fnorm, ftan


def solve_streamtube_baseline(r1_R, r2_R, chord, twist,
                               max_iter=300, tol=1e-5):
    Area    = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_s     = 0.5 * (r1_R + r2_R)
    r_local = r_s * Radius
    a = 0.0; ap = 0.0
    for _ in range(max_iter):
        fn, ft = load_blade_element(
            U0 * (1 - a), (1 + ap) * Omega * r_local, chord, twist)
        CT_loc = fn * Radius * (r2_R - r1_R) * NBlades / (0.5 * Area * U0**2)
        anew   = ainduction(CT_loc)
        F      = prandtl_F(r_s, anew)
        anew  /= F; a_old = a
        a      = 0.75 * a + 0.25 * anew
        ap     = ft * NBlades / (
            2.0 * np.pi * U0 * (1 - a) * Omega * 2.0 * r_local**2) / F
        if abs(a - a_old) < tol:
            break
    return a, ap, r_s, fn, ft

# =============================================================================
# GRID
# =============================================================================

bins  = np.arange(RootLocation_R, TipLocation_R + DELTA_R_R / 2, DELTA_R_R)
r_mid = 0.5 * (bins[:-1] + bins[1:])
dr    = np.diff(bins) * Radius
N     = len(r_mid)
x_mid = (r_mid - RootLocation_R) / (TipLocation_R - RootLocation_R)

chord_base = 3.0 * (1.0 - r_mid) + 1.0
twist_base = -(14.0 * (1.0 - r_mid) + Pitch)

# =============================================================================
# CHORD EQUATION — single station
# Iterates a' to self-consistency and returns the analytical chord.
# No clipping so Brent can find exact zero crossings.
# =============================================================================

def chord_from_a(a_val, r_R):
    """Analytical chord at r_R for a given axial induction a_val."""
    a       = float(np.clip(a_val, 0.01, 0.49))
    r_local = r_R * Radius
    F       = prandtl_F(r_R, a)
    ap      = 0.0

    for _ in range(80):
        phi_rad = np.arctan2(U0 * (1.0 - a),
                             Omega * r_local * (1.0 + ap))
        sin2phi = np.sin(phi_rad)**2
        numer   = 8.0 * np.pi * r_local * a * F * (1.0 - a * F) * sin2phi
        denom   = NBlades * (1.0 - a)**2 * max(Cn_opt, 1e-6)
        c_i     = numer / denom                   # no clipping
        twist_i = alpha_opt - np.degrees(phi_rad) - Pitch
        fn, ft  = load_blade_element(
            U0 * (1.0 - a), Omega * r_local * (1.0 + ap), c_i, twist_i)
        ap_new  = float(np.clip(
            ft * NBlades / (
                2.0 * np.pi * U0 * (1.0 - a)
                * Omega * 2.0 * r_local**2) / F,
            0.0, 0.5))
        if abs(ap_new - ap) < 1e-8:
            break
        ap = 0.6 * ap + 0.4 * ap_new

    return float(c_i)

# =============================================================================
# FIND a_root BY INVERTING THE CHORD EQUATION
#
# Solve  chord_from_a(a, RootLocation_R) - CHORD_ROOT_TARGET = 0
# using Brent's method over a physically sensible bracket.
# This gives the exact a_root that produces c_root = 3.4 m.
# a0 in the polynomial is then pinned to this value.
# =============================================================================

def _root_chord_residual(a_val):
    return chord_from_a(a_val, RootLocation_R) - CHORD_ROOT_TARGET


print(f"\nSolving for a_root that gives root chord = {CHORD_ROOT_TARGET} m ...")
a_scan = np.linspace(0.05, 0.45, 200)
c_scan = np.array([chord_from_a(a, RootLocation_R) for a in a_scan])

print(f"  Chord range at root over a=[0.05,0.45]: "
      f"[{c_scan.min():.3f}, {c_scan.max():.3f}] m")

sign_changes = np.where(np.diff(np.sign(c_scan - CHORD_ROOT_TARGET)))[0]
if len(sign_changes) == 0:
    idx_nearest = int(np.argmin(np.abs(c_scan - CHORD_ROOT_TARGET)))
    a_root_pin  = float(a_scan[idx_nearest])
    c_achieved  = float(c_scan[idx_nearest])
    print(f"  WARNING: target chord {CHORD_ROOT_TARGET} m not reachable.")
    print(f"  Nearest achievable: c = {c_achieved:.4f} m at a = {a_root_pin:.4f}")
else:
    idx = sign_changes[0]
    a_lo, a_hi = float(a_scan[idx]), float(a_scan[idx + 1])
    a_root_pin = brentq(_root_chord_residual, a_lo, a_hi, xtol=1e-8)
    c_check    = chord_from_a(a_root_pin, RootLocation_R)
    print(f"  Solved: a_root = {a_root_pin:.6f},  "
          f"chord = {c_check:.6f} m  (target {CHORD_ROOT_TARGET} m)")

# =============================================================================
# POLYNOMIAL PARAMETERISATION — a0 PINNED, [a1..a4] FREE
#
# Full polynomial:  a(r) = a_root_pin + a1*x + a2*x^2 + a3*x^3 + a4*x^4
#
# The optimiser only sees [a1, a2, a3, a4].
# a0 = a_root_pin is fixed, so chord(root) = 3.4 m by construction.
# =============================================================================

def a_from_params(shape_params, x=None):
    """Reconstruct full a(r) from the 4 shape coefficients."""
    if x is None:
        x = x_mid
    a1, a2, a3, a4 = shape_params
    a_vec = a_root_pin + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    return np.clip(a_vec, 0.05, 0.45)


def chord_at_tip(shape_params):
    a_tip = float(a_from_params(shape_params, x=np.array([1.0]))[0])
    return chord_from_a(a_tip, TipLocation_R)

# =============================================================================
# FORWARD MODEL — rebuilds geometry at every call
# =============================================================================

def evaluate(shape_params):
    a_vec = a_from_params(shape_params)
    fnorm = np.zeros(N)
    ftan  = np.zeros(N)
    aline = np.zeros(N)
    chord = np.zeros(N)
    twist = np.zeros(N)

    for i in range(N):
        a       = float(a_vec[i])
        r_local = r_mid[i] * Radius
        F       = prandtl_F(r_mid[i], a)
        ap      = 0.0

        for _ in range(60):
            phi_rad = np.arctan2(U0 * (1.0 - a),
                                 Omega * r_local * (1.0 + ap))
            sin2phi = np.sin(phi_rad)**2
            numer   = (8.0 * np.pi * r_local * a * F
                       * (1.0 - a * F) * sin2phi)
            denom   = NBlades * (1.0 - a)**2 * max(Cn_opt, 1e-6)
            chord_i = float(np.clip(numer / denom, CHORD_TIP_MIN, 6.0))
            twist_i = alpha_opt - np.degrees(phi_rad) - Pitch

            fn, ft  = load_blade_element(
                U0 * (1.0 - a), Omega * r_local * (1.0 + ap),
                chord_i, twist_i)
            ap_new  = float(np.clip(
                ft * NBlades / (
                    2.0 * np.pi * U0 * (1.0 - a)
                    * Omega * 2.0 * r_local**2) / F,
                0.0, 0.5))
            if abs(ap_new - ap) < 1e-7:
                ap = ap_new
                break
            ap = 0.6 * ap + 0.4 * ap_new

        fn, ft  = load_blade_element(
            U0 * (1.0 - a), Omega * r_local * (1.0 + ap),
            chord_i, twist_i)

        fnorm[i] = fn
        ftan[i]  = ft
        aline[i] = ap
        chord[i] = chord_i
        twist[i] = twist_i

    CT = float(np.sum(dr * fnorm * NBlades
                      / (0.5 * U0**2 * np.pi * Radius**2)))
    CP = float(np.sum(dr * ftan * r_mid * Radius * NBlades * Omega
                      / (0.5 * U0**3 * np.pi * Radius**2)))
    return CT, CP, fnorm, ftan, aline, chord, twist

# =============================================================================
# OPTIMISATION — 4 free shape parameters
# =============================================================================

def run():
    print("\n" + "=" * 60)
    print("Polynomial a(r) optimisation  (SLSQP, 4 shape parameters)")
    print(f"CT target        = {CT_TARGET}")
    print(f"a0 pinned        = {a_root_pin:.6f}  "
          f"-> root chord = {CHORD_ROOT_TARGET} m  (by construction)")
    print(f"Tip chord min    = {CHORD_TIP_MIN} m  (inequality)")
    print(f"CP_AD            = {CP_AD:.4f}")
    print("=" * 60)

    p0 = np.array([0.0, 0.0, 0.0, 0.0])
    CT0, CP0, *_ = evaluate(p0)
    c_tip0 = chord_at_tip(p0)
    print(f"\nInitial guess (flat a = {a_root_pin:.4f}):")
    print(f"  CT = {CT0:.4f},  CP = {CP0:.4f}")
    print(f"  Root chord = {chord_from_a(a_root_pin, RootLocation_R):.4f} m"
          f"  (should be {CHORD_ROOT_TARGET} m)")
    print(f"  Tip chord  = {c_tip0:.4f} m")

    call_count = [0]

    def objective(p):
        call_count[0] += 1
        CT, CP, *_ = evaluate(p)
        if call_count[0] % 10 == 0:
            c_t = chord_at_tip(p)
            print(f"  eval {call_count[0]:4d}  CP={CP:.5f}  CT={CT:.5f}"
                  f"  CT_err={CT - CT_TARGET:+.4f}"
                  f"  c_tip={c_t:.3f}")
        return -CP

    def ct_constraint(p):
        CT, *_ = evaluate(p)
        return CT - CT_TARGET

    def tip_chord_constraint(p):
        return chord_at_tip(p) - CHORD_TIP_MIN

    constraints = [
        {'type': 'eq',   'fun': ct_constraint},
        {'type': 'ineq', 'fun': tip_chord_constraint},
    ]

    bounds = [
        (-0.20, 0.20),
        (-0.20, 0.20),
        (-0.10, 0.10),
        (-0.10, 0.10),
    ]

    print("\nRunning SLSQP ...")
    result = minimize(
        objective,
        p0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-8, 'maxiter': 300, 'disp': True}
    )

    print(f"\nConverged: {result.success}  -  {result.message}")
    print(f"Total evaluations: {call_count[0]}")

    p_opt = result.x
    CT_opt, CP_opt, fnorm_opt, ftan_opt, aline_opt, \
        chord_opt, twist_opt = evaluate(p_opt)
    a_opt  = a_from_params(p_opt)
    c_root = chord_from_a(float(a_opt[0]), RootLocation_R)
    c_tip  = chord_at_tip(p_opt)

    print(f"\nOptimal result:")
    print(f"  CT        = {CT_opt:.6f}  "
          f"(target {CT_TARGET},  error {CT_opt - CT_TARGET:+.2e})")
    print(f"  CP        = {CP_opt:.6f}")
    print(f"  CP_AD     = {CP_AD:.6f}")
    print(f"  CP/CP_AD  = {CP_opt / CP_AD:.4f}")
    print(f"  Root chord= {c_root:.6f} m  (target {CHORD_ROOT_TARGET} m)")
    print(f"  Tip chord = {c_tip:.6f} m  (min {CHORD_TIP_MIN} m)")
    print(f"  a(r) range: [{a_opt.min():.3f}, {a_opt.max():.3f}]")

    if CP_opt > CP_AD + 1e-4:
        print("  WARNING: CP exceeds actuator disk limit")
    else:
        print("  CP within actuator disk bound - result physical")

    print("\nBaseline BEM ...")
    rows  = [solve_streamtube_baseline(bins[i], bins[i + 1],
                                       chord_base[i], twist_base[i])
             for i in range(N)]
    res_b = np.array(rows)
    CT_b  = float(np.sum(dr * res_b[:, 3] * NBlades
                         / (0.5 * U0**2 * np.pi * Radius**2)))
    CP_b  = float(np.sum(dr * res_b[:, 4] * res_b[:, 2] * Radius
                         * NBlades * Omega
                         / (0.5 * U0**3 * np.pi * Radius**2)))
    print(f"  CT = {CT_b:.6f},  CP = {CP_b:.6f}")

    return (p_opt, a_opt, CT_opt, CP_opt,
            fnorm_opt, ftan_opt, aline_opt,
            chord_opt, twist_opt,
            res_b, CT_b, CP_b)

# =============================================================================
# PLOTS
# =============================================================================

def plot_results(p_opt, a_opt, CT_opt, CP_opt,
                 fnorm_opt, ftan_opt, aline_opt,
                 chord_opt, twist_opt,
                 res_b, CT_b, CP_b):

    norm_val = 0.5 * U0**2 * Radius
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    axes[0, 0].plot(r_mid, a_opt, lw=2, label="Optimal a(r)  [SLSQP]")
    axes[0, 0].plot(res_b[:, 2], res_b[:, 0], "--", lw=1.5,
                    label="Baseline BEM")
    axes[0, 0].axhline(1 / 3, color="grey", ls=":", lw=1,
                       label="Betz  a=1/3")
    axes[0, 0].axhline(a_root_pin, color="green", ls=":", lw=1,
                       label=f"a_root = {a_root_pin:.3f}")
    axes[0, 0].set_xlabel("r/R"); axes[0, 0].set_ylabel("a  [-]")
    axes[0, 0].set_title("Axial induction distribution")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True)

    axes[0, 1].plot(r_mid, aline_opt, lw=2, label="Optimal a'(r)")
    axes[0, 1].plot(res_b[:, 2], res_b[:, 1], "--", lw=1.5,
                    label="Baseline BEM")
    axes[0, 1].set_xlabel("r/R"); axes[0, 1].set_ylabel("a'  [-]")
    axes[0, 1].set_title("Tangential induction distribution")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True)

    axes[0, 2].plot(r_mid, fnorm_opt / norm_val, lw=2,
                    label=f"Optimal  CT={CT_opt:.3f}  CP={CP_opt:.4f}")
    axes[0, 2].plot(res_b[:, 2], res_b[:, 3] / norm_val, "--", lw=1.5,
                    label=f"Baseline CT={CT_b:.3f}  CP={CP_b:.4f}")
    axes[0, 2].set_xlabel("r/R"); axes[0, 2].set_ylabel(r"$C_n$  [-]")
    axes[0, 2].set_title("Normal (thrust) loading")
    axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True)

    axes[1, 0].plot(r_mid, chord_opt, lw=2, label="Optimal chord")
    axes[1, 0].plot(r_mid, chord_base, "--", lw=1.5, label="Baseline chord")
    axes[1, 0].axhline(CHORD_ROOT_TARGET, color="green", ls=":", lw=1,
                       label=f"Root target {CHORD_ROOT_TARGET} m")
    axes[1, 0].axhline(CHORD_TIP_MIN, color="orange", ls=":", lw=1,
                       label=f"Tip min {CHORD_TIP_MIN} m")
    axes[1, 0].set_xlabel("r/R"); axes[1, 0].set_ylabel("Chord  [m]")
    axes[1, 0].set_title("Chord distribution")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True)

    axes[1, 1].plot(r_mid, twist_opt, lw=2, label="Optimal twist")
    axes[1, 1].plot(r_mid, twist_base, "--", lw=1.5, label="Baseline twist")
    axes[1, 1].set_xlabel("r/R"); axes[1, 1].set_ylabel("Twist  [deg]")
    axes[1, 1].set_title("Twist distribution")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True)

    labels  = ["Baseline", "Optimal a(r)", "Actuator disk"]
    cp_vals = [CP_b,   CP_opt,  CP_AD]
    ct_vals = [CT_b,   CT_opt,  CT_TARGET]
    x = np.arange(3); w = 0.35
    b1 = axes[1, 2].bar(x - w / 2, cp_vals, w, label="CP", color="steelblue")
    b2 = axes[1, 2].bar(x + w / 2, ct_vals, w, label="CT", color="coral")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].set_ylabel("Coefficient  [-]")
    axes[1, 2].set_title("Performance summary")
    axes[1, 2].bar_label(b1, fmt="%.4f", padding=3, fontsize=8)
    axes[1, 2].bar_label(b2, fmt="%.4f", padding=3, fontsize=8)
    axes[1, 2].legend(); axes[1, 2].grid(True, axis="y")

    fig.suptitle(
        f"Polynomial a(r) optimisation  -  TSR={TSR_DESIGN},  "
        f"CT={CT_TARGET},  c_root={CHORD_ROOT_TARGET} m (exact),  "
        f"c_tip>={CHORD_TIP_MIN} m,  CP_AD={CP_AD:.4f}",
        fontsize=11)
    fig.tight_layout()
    plt.savefig("optimal_a_distribution.png", dpi=200, bbox_inches="tight")
    print("\nPlot saved: optimal_a_distribution.png")
    plt.show()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run()
    plot_results(*results)











































    