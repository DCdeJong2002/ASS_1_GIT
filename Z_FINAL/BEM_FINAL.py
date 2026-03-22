"""
assignment.py  —  AE4135 Rotor/Wake Aerodynamics, Assignment 1
================================================================
Self-contained script producing all required plots.

Sections
--------
 4.1  Alpha and inflow angle vs r/R              (TSR sweep)
 4.2  Axial and azimuthal induction vs r/R       (TSR sweep)
 4.3  Thrust and azimuthal loading vs r/R        (TSR sweep)
 4.4  Total CT and CQ vs tip-speed ratio
  5   Tip-loss correction influence
  6   Number of annuli, spacing method, convergence history
  7   Stagnation pressure at four streamwise stations
  8   All designs — chord, twist, induction, loading, performance
  9   Lift coefficient and chord relation  (analytical optimum)
 10   Cl/Cd polar with operating points

Run:  python assignment.py
All plots saved to ./plots_assignment/
"""

import os, sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

USE_HELICAL_PRANDTL = False   # True -> Prandtl 1919; False -> Glauert 1935

N_STARTS   = 12
TSR_SWEEP  = [6, 7, 8, 9, 10]
TSR_DESIGN = 8.0
CT_TARGET  = 0.75

Radius         = 50.0
NBlades        = 3
U0             = 10.0
rho            = 1.225
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0          # baseline pitch [deg]

# Unified chord constraints — identical for ALL optimisation methods
CHORD_ROOT    = 3.4            # root chord [m]
CHORD_MIN     = 0.3            # minimum chord [m]
CHORD_MAX_REG = 6.0            # soft upper limit for polynomial penalty [m]

# Single unified grid for all BEM evaluations
# Constant spacing: 160 annuli, same quadrature for every method,
# CT constraint satisfied at same resolution that is reported.
DELTA_R_R = 0.005

# =============================================================================
# MENU — control what runs and what plots are produced
# =============================================================================
#
# RUN_* flags control which BEM computations execute.
# Dependency notes are listed next to each flag.
#
#   If you already have bem_results.npz from a previous run, set all
#   RUN_* flags to False and use plot_results.py instead.
#   If you only want to re-run specific sections, set others to False
#   but note that downstream sections may require upstream results.
#
# PLOT_* flags control which figures are saved.
# You can disable any plot independently of the computation flags.
# =============================================================================

# ── Computations ─────────────────────────────────────────────────────────────
RUN_TSR_SWEEP       = True   # baseline geometry sweep over TSR_SWEEP
                              #   required by: PLOT_4_1, PLOT_4_2, PLOT_4_3,
                              #                PLOT_4_4, PLOT_5, PLOT_6, PLOT_7
RUN_NO_CORRECTION   = True   # F=1 run at TSR=8
                              #   required by: PLOT_5
                              #   requires:    RUN_TSR_SWEEP (uses Omega8)
RUN_ANALYTICAL      = True   # analytical optimum via Brent root-finder
                              #   required by: PLOT_8, PLOT_9, PLOT_10
RUN_CUBIC           = True   # cubic polynomial optimiser  (slowest step)
                              #   required by: PLOT_8
RUN_QUARTIC         = True   # quartic polynomial optimiser (slowest step)
                              #   required by: PLOT_8

# ── Plots ─────────────────────────────────────────────────────────────────────
PLOT_4_1 = True   # alpha and inflow angle vs r/R        (requires RUN_TSR_SWEEP)
PLOT_4_2 = True   # axial and tangential induction vs r/R (requires RUN_TSR_SWEEP)
PLOT_4_3 = True   # thrust and azimuthal loading vs r/R   (requires RUN_TSR_SWEEP)
PLOT_4_4 = True   # CT and CQ vs TSR                      (requires RUN_TSR_SWEEP)
PLOT_5   = True   # tip correction influence               (requires RUN_TSR_SWEEP
                  #                                         and RUN_NO_CORRECTION)
PLOT_6   = True   # annuli count, spacing, convergence     (requires RUN_TSR_SWEEP)
PLOT_7   = True   # stagnation pressure                    (requires RUN_TSR_SWEEP)
PLOT_8   = True   # all designs — chord/twist/loading/perf (requires all RUN_* flags
                  #                                         that are relevant)
PLOT_9   = True   # Cl and chord relation                  (requires RUN_ANALYTICAL)
PLOT_10  = True   # Cl/Cd polar with operating points      (requires RUN_ANALYTICAL)

# ── Save results ──────────────────────────────────────────────────────────────
SAVE_RESULTS = True   # write bem_results.npz after all computations
                       #   only saves results for sections that were run


# =============================================================================
# 1.  POLAR DATA
# =============================================================================

_df         = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = _df["Alfa"].to_numpy()
polar_cl    = _df["Cl"].to_numpy()
polar_cd    = _df["Cd"].to_numpy()

# =============================================================================
# 2.  BEM CORE FUNCTIONS
# =============================================================================

def ainduction(CT):
    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1
    if np.isscalar(CT):
        if CT >= CT2:
            return 1.0 + (CT - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    a = np.zeros_like(np.asarray(CT, dtype=float))
    a[CT >= CT2] = 1.0 + (CT[CT >= CT2] - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
    a[CT < CT2]  = 0.5 - 0.5 * np.sqrt(np.maximum(0.0, 1.0 - CT[CT < CT2]))
    return a


def _prandtl_simplified(r_R, rootR, tipR, TSR, NB, a_in):
    a  = float(np.clip(a_in, -0.9, 0.99))
    sq = np.sqrt(1.0 + (TSR * r_R) ** 2 / (1.0 - a) ** 2)
    t1 = -NB / 2.0 * (tipR  - r_R) / r_R * sq
    t2 =  NB / 2.0 * (rootR - r_R) / r_R * sq
    Ft = 2.0 / np.pi * np.arccos(float(np.clip(np.exp(t1), 0.0, 1.0)))
    Fr = 2.0 / np.pi * np.arccos(float(np.clip(np.exp(t2), 0.0, 1.0)))
    return float(Fr * Ft)


def _prandtl_helical(r_R, rootR, tipR, TSR, NB, a_in):
    a  = float(np.clip(a_in, -0.9, 0.99))
    d  = max(2.0 * np.pi / NB * (1.0 - a) / np.sqrt(TSR ** 2 + (1.0 - a) ** 2), 1e-8)
    Ft = 2.0 / np.pi * np.arccos(np.exp(max(-np.pi * (tipR  - r_R) / d, -500.0)))
    Fr = 2.0 / np.pi * np.arccos(np.exp(max(-np.pi * (r_R - rootR)  / d, -500.0)))
    return float((0.0 if np.isnan(Fr) else Fr) * (0.0 if np.isnan(Ft) else Ft))


def prandtl(r_R, rootR, tipR, TSR, NB, a):
    return (_prandtl_helical(r_R, rootR, tipR, TSR, NB, a)
            if USE_HELICAL_PRANDTL
            else _prandtl_simplified(r_R, rootR, tipR, TSR, NB, a))


def load_blade_element(vnorm, vtan, chord, twist):
    """twist [deg].  Convention: alpha = twist + phi_deg."""
    vmag2 = vnorm ** 2 + vtan ** 2
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
    """
    Returns row=np.array([a,aline,r_mid,fnorm,ftan,gamma,alpha,phi,cl,cd])
    and fnorm history.
    """
    Area    = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid   = 0.5 * (r1_R + r2_R)
    r_local = r_mid * Radius
    TSR_now = Omega * Radius / U0
    a = 0.0; aline = 0.0; hist = []; cl_f = cd_f = 0.0

    for _ in range(max_iter):
        Vax  = U0 * (1.0 - a)
        Vtan = (1.0 + aline) * Omega * r_local
        fnorm, ftan, gamma, alpha, phi, cl, cd = load_blade_element(
            Vax, Vtan, chord, twist)
        cl_f = cl; cd_f = cd
        hist.append(fnorm)
        CT_loc = fnorm * Radius * (r2_R - r1_R) * NBlades / (0.5 * Area * U0 ** 2)
        anew   = ainduction(CT_loc)
        F      = max(prandtl(r_mid, RootLocation_R, TipLocation_R,
                             TSR_now, NBlades, anew), 1e-4)
        anew  /= F; a_old = a
        a      = 0.75 * a + 0.25 * anew
        aline  = ftan * NBlades / (
            2.0 * np.pi * U0 * (1.0 - a) * Omega * 2.0 * r_local ** 2) / F
        if abs(a - a_old) < tol:
            break

    return (np.array([a, aline, r_mid, fnorm, ftan, gamma,
                      alpha, phi, cl_f, cd_f], dtype=float),
            np.array(hist))


def solve_streamtube_nocorr(r1_R, r2_R, Omega, chord, twist,
                             max_iter=300, tol=1e-5):
    Area    = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid   = 0.5 * (r1_R + r2_R)
    r_local = r_mid * Radius
    a = 0.0; aline = 0.0

    for _ in range(max_iter):
        Vax  = U0 * (1.0 - a); Vtan = (1.0 + aline) * Omega * r_local
        fnorm, ftan, *_ = load_blade_element(Vax, Vtan, chord, twist)
        CT_loc = fnorm * Radius * (r2_R - r1_R) * NBlades / (0.5 * Area * U0 ** 2)
        anew   = ainduction(CT_loc)
        a      = 0.75 * a + 0.25 * anew
        aline  = ftan * NBlades / (
            2.0 * np.pi * U0 * max(1.0 - a, 1e-4) * Omega * 2.0 * r_local ** 2)
        if abs(a - anew) < tol:
            break
    return np.array([a, aline, r_mid, fnorm, ftan], dtype=float)

# =============================================================================
# 3.  ROTOR EVALUATOR
# =============================================================================

def make_bins(delta=DELTA_R_R):
    return np.arange(RootLocation_R, TipLocation_R + delta / 2.0, delta)


def evaluate_rotor(r_nodes, c_nodes, tw_nodes, tsr=TSR_DESIGN):
    """Returns CT, CP, res (shape [N,10])."""
    Omega = U0 * tsr / Radius
    rows  = []
    for i in range(len(r_nodes) - 1):
        r1    = r_nodes[i]     / Radius
        r2    = r_nodes[i + 1] / Radius
        chord = 0.5 * (c_nodes[i]  + c_nodes[i + 1])
        twist = 0.5 * (tw_nodes[i] + tw_nodes[i + 1])
        row, _ = solve_streamtube(r1, r2, Omega, chord, twist)
        rows.append(row)
    res = np.vstack(rows)
    dr  = np.diff(r_nodes)
    CT  = float(np.sum(dr * res[:, 3] * NBlades
                        / (0.5 * U0 ** 2 * np.pi * Radius ** 2)))
    CP  = float(np.sum(dr * res[:, 4] * res[:, 2] * NBlades * Radius * Omega
                        / (0.5 * U0 ** 3 * np.pi * Radius ** 2)))
    return CT, CP, res

# =============================================================================
# 4.  BASELINE GEOMETRY
# =============================================================================

def baseline_geometry():
    bins = make_bins()
    return (bins * Radius,
            3.0 * (1.0 - bins) + 1.0,
            -(14.0 * (1.0 - bins) + Pitch))

# =============================================================================
# 5.  TSR SWEEP
# =============================================================================

# Initialise to None so downstream guards can check safely
bins_main    = make_bins()
rmid_R       = 0.5 * (bins_main[:-1] + bins_main[1:])
tsr_perf     = {}; sweep_data = {}
results_tsr8 = None; ct_hist_tsr8 = None; F_tsr8 = None
Omega8       = U0 * 8.0 / Radius   # needed by no-correction run and section 6

if RUN_TSR_SWEEP:
    print("Running TSR sweep ...")
    for TSR in TSR_SWEEP:
        Omega = U0 * TSR / Radius
        rows_t = []; hists_t = []
        for i in range(len(bins_main) - 1):
            rm    = rmid_R[i]
            row, hist = solve_streamtube(bins_main[i], bins_main[i + 1], Omega,
                                          3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows_t.append(row); hists_t.append(hist)

        res  = np.vstack(rows_t)
        dr_m = np.diff(bins_main) * Radius
        CT   = float(np.sum(dr_m * res[:, 3] * NBlades
                             / (0.5 * U0 ** 2 * np.pi * Radius ** 2)))
        CP   = float(np.sum(dr_m * res[:, 4] * res[:, 2] * NBlades * Radius * Omega
                             / (0.5 * U0 ** 3 * np.pi * Radius ** 2)))
        tsr_perf[TSR] = {"CT": CT, "CP": CP}
        sweep_data[TSR] = res
        print(f"  TSR={TSR}  CT={CT:.4f}  CP={CP:.4f}")

        if TSR == 8:
            results_tsr8 = res
            max_len = max(len(h) for h in hists_t)
            padded  = np.array([np.concatenate([h, np.full(max_len - len(h), h[-1])])
                                 for h in hists_t])
            ct_hist_tsr8 = np.sum(
                padded * dr_m[:, None] * NBlades
                / (0.5 * U0 ** 2 * np.pi * Radius ** 2), axis=0)
            F_tsr8 = np.array([
                max(prandtl(rmid_R[i], RootLocation_R, TipLocation_R,
                            Omega * Radius / U0, NBlades, res[i, 0]), 1e-4)
                for i in range(len(rmid_R))])

# =============================================================================
# 6.  NO-CORRECTION RUN
# =============================================================================

res_nc = None

if RUN_NO_CORRECTION:
    print("Running no-correction BEM (TSR=8) ...")
    rows_nc = []
    for i in range(len(bins_main) - 1):
        rm = rmid_R[i]
        rows_nc.append(solve_streamtube_nocorr(
            bins_main[i], bins_main[i + 1], Omega8,
            3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch)))
    res_nc = np.vstack(rows_nc)

# =============================================================================
# 7.  ANALYTICAL OPTIMUM
# =============================================================================

def find_optimal_alpha():
    alphas = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_v   = np.interp(alphas, polar_alpha, polar_cl)
    cd_v   = np.interp(alphas, polar_alpha, polar_cd)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cd_v > 1e-6, cl_v / cd_v, 0.0)
    idx = int(np.argmax(ratio))
    return float(alphas[idx]), float(cl_v[idx]), float(cd_v[idx])


def generate_ideal_rotor(target_a, n_nodes=101,
                          alpha_opt_deg=None, cl_opt=None, cd_opt=None,
                          ap_in=None):
    if alpha_opt_deg is None:
        alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()
    r_start = RootLocation_R * Radius + 0.005 * Radius
    r_end   = TipLocation_R  * Radius - 0.005 * Radius
    r_nodes = r_start + (r_end - r_start) * (
        1.0 - np.cos(np.linspace(0.0, np.pi, n_nodes))) / 2.0
    c_nodes = np.zeros(n_nodes); tw_nodes = np.zeros(n_nodes)
    ap_nodes = (np.zeros(n_nodes) if ap_in is None
                else np.interp(r_nodes, 0.5*(r_nodes[:-1]+r_nodes[1:]),
                               ap_in, left=ap_in[0], right=ap_in[-1]))
    for i, r in enumerate(r_nodes):
        r_R  = r / Radius
        F    = float(max(prandtl(r_R, RootLocation_R, TipLocation_R,
                                  TSR_DESIGN, NBlades, target_a), 1e-4))
        phi  = np.arctan2(1.0 - target_a,
                          TSR_DESIGN * r_R * (1.0 + float(ap_nodes[i])))
        tw_nodes[i] = alpha_opt_deg - np.degrees(phi)
        Cn   = cl_opt * np.cos(phi) + cd_opt * np.sin(phi)
        num  = 8.0*np.pi*r*target_a*F*(1.0-target_a*F)*np.sin(phi)**2
        den  = NBlades * (1.0 - target_a) ** 2 * max(Cn, 1e-8)
        c_nodes[i] = num / den
    c_nodes = np.clip(c_nodes, 0.0, CHORD_ROOT)
    mx = int(np.argmax(c_nodes)); c_nodes[:mx+1] = c_nodes[mx]
    c_nodes = np.clip(c_nodes, CHORD_MIN, None)
    return r_nodes, c_nodes, tw_nodes


def design_for_exact_ct():
    alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
    last_ap = [None]; last_geom = [None]

    def residual(ta):
        r, c, tw = generate_ideal_rotor(ta, alpha_opt_deg=alpha_opt,
                                         cl_opt=cl_opt, cd_opt=cd_opt,
                                         ap_in=last_ap[0])
        last_geom[0] = (r, c, tw)
        _out, sys.stdout = sys.stdout, io.StringIO()
        try:
            CT, CP, res_arr = evaluate_rotor(r, c, tw)
        finally:
            sys.stdout = _out
        last_ap[0] = res_arr[:, 1]
        print(f"  target_a={ta:.5f}  CT={CT:.5f}  CP={CP:.5f}")
        return CT - CT_TARGET

    print("\nAnalytical: root-finding for CT =", CT_TARGET)
    sol = root_scalar(residual, bracket=[0.20, 0.35], method="brentq", xtol=1e-5)
    if not sol.converged:
        raise RuntimeError("Brent did not converge.")
    print(f"  Converged: a={sol.root:.6f}")
    r, c, tw = last_geom[0]
    CT, CP, res_arr = evaluate_rotor(r, c, tw)
    print(f"  CT={CT:.6f}  CP={CP:.6f}")
    return r, c, tw, CT, CP, res_arr

# =============================================================================
# 8.  POLYNOMIAL OPTIMISERS
# =============================================================================

def _span_x(r_R):
    return (r_R - RootLocation_R) / (TipLocation_R - RootLocation_R)

def chord_poly(r_R, c_tip, c2, c3, c4=0.0):
    x = _span_x(r_R)
    b1 = c_tip - CHORD_ROOT - c2 - c3 - c4
    return CHORD_ROOT + b1*x + c2*x**2 + c3*x**3 + c4*x**4

def twist_poly(r_R, pitch, t_root, t_tip, t_curve):
    x = _span_x(r_R)
    return pitch + t_root*(1-x) + t_tip*x + t_curve*x*(1-x)

def build_poly_geometry(params):
    n = len(params)
    if n == 7:
        pitch, t_root, t_tip, t_curve, c_tip, c2, c3 = params; c4 = 0.0
    else:
        pitch, t_root, t_tip, t_curve, c_tip, c2, c3, c4 = params
    bins = make_bins()
    r  = bins * Radius
    c  = np.array([chord_poly(rr, c_tip, c2, c3, c4) for rr in bins])
    tw = np.array([twist_poly(rr, pitch, t_root, t_tip, t_curve) for rr in bins])
    return r, c, tw

def _chord_minmax(params, n=300):
    n_p = len(params)
    if n_p == 7:
        _, _, _, _, c_tip, c2, c3 = params; c4 = 0.0
    else:
        _, _, _, _, c_tip, c2, c3, c4 = params
    rr = np.linspace(RootLocation_R, TipLocation_R, n)
    cv = np.array([chord_poly(r, c_tip, c2, c3, c4) for r in rr])
    return float(cv.min()), float(cv.max())

def _make_objective(quartic):
    def obj(params):
        c_min, c_max = _chord_minmax(params)
        pen = 0.0
        if c_min < CHORD_MIN:
            pen += 1e4 * (CHORD_MIN - c_min) ** 2
        if c_max > CHORD_MAX_REG:
            pen += 1e3 * (c_max - CHORD_MAX_REG) ** 2
        if c_min < CHORD_MIN * 0.5:
            return 1e9
        r, c, tw   = build_poly_geometry(params)
        CT, CP, _  = evaluate_rotor(r, c, tw)
        pen       += 5e3 * (CT - CT_TARGET) ** 2
        if quartic:
            _, _, _, t_curve, _, c2, c3, c4 = params
            pen += 0.05*t_curve**2 + 0.01*(c2**2+c3**2+c4**2)
        else:
            _, _, _, t_curve, _, c2, c3 = params
            pen += 0.05*t_curve**2 + 0.01*(c2**2+c3**2)
        return -CP + pen
    return obj

def run_poly_optimizer(quartic=False, n_starts=N_STARTS, seed=42):
    rng   = np.random.default_rng(seed)
    obj   = _make_objective(quartic)
    label = "quartic" if quartic else "cubic"
    if quartic:
        bounds = [(-8,8),(-25,5),(-10,15),(-20,20),(0.3,2),(-5,5),(-5,5),(-5,5)]
        x0_nom = np.array([-2,-7,2,0,1,0,0,0], dtype=float)
        def _rand():
            return np.array([rng.uniform(-6,6), rng.uniform(-20,0),
                             rng.uniform(-5,10), rng.uniform(-10,10),
                             rng.uniform(0.3,1.5), rng.uniform(-3,3),
                             rng.uniform(-3,3), rng.uniform(-3,3)])
    else:
        bounds = [(-8,8),(-25,5),(-10,15),(-20,20),(0.3,2),(-5,5),(-5,5)]
        x0_nom = np.array([-2,-7,2,0,1,0,0], dtype=float)
        def _rand():
            return np.array([rng.uniform(-6,6), rng.uniform(-20,0),
                             rng.uniform(-5,10), rng.uniform(-10,10),
                             rng.uniform(0.3,1.5), rng.uniform(-3,3),
                             rng.uniform(-3,3)])
    starts = [x0_nom] + [_rand() for _ in range(n_starts-1)]
    best_obj = np.inf; best_p = x0_nom.copy()
    for k, x0 in enumerate(starts, 1):
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
        try:
            r_, c_, tw_ = build_poly_geometry(res.x)
            CT_, _, _   = evaluate_rotor(r_, c_, tw_)
        except Exception:
            CT_ = float("nan")
        print(f"  [{label}] {k:02d}/{n_starts}  obj={res.fun:+.6f}  CT={CT_:.4f}")
        if res.fun < best_obj:
            best_obj = res.fun; best_p = res.x.copy()
    r, c, tw      = build_poly_geometry(best_p)
    CT, CP, res_a = evaluate_rotor(r, c, tw)
    return best_p, r, c, tw, CT, CP, res_a

# =============================================================================
# 9.  RUN ALL METHODS
# =============================================================================

# Initialise design results to None
r_base = c_base = tw_base = res_base = None
CT_base = CP_base = None
r_anal = c_anal = tw_anal = res_anal = None
CT_anal = CP_anal = None
p_cubic = r_cubic = c_cubic = tw_cubic = res_cubic = None
CT_cubic = CP_cubic = None
p_qrt = r_qrt = c_qrt = tw_qrt = res_qrt = None
CT_qrt = CP_qrt = None
a_ad  = 0.5 * (1.0 - np.sqrt(1.0 - CT_TARGET))
CP_ad = 4.0 * a_ad * (1.0 - a_ad) ** 2

print("\n" + "="*60); print("BASELINE")
r_base, c_base, tw_base = baseline_geometry()
CT_base, CP_base, res_base = evaluate_rotor(r_base, c_base, tw_base)
print(f"  CT={CT_base:.6f}  CP={CP_base:.6f}")

if RUN_ANALYTICAL:
    print("\n" + "="*60); print("ANALYTICAL OPTIMUM")
    r_anal, c_anal, tw_anal, CT_anal, CP_anal, res_anal = design_for_exact_ct()

if RUN_CUBIC:
    print("\n" + "="*60); print("CUBIC POLYNOMIAL OPTIMISER")
    p_cubic, r_cubic, c_cubic, tw_cubic, CT_cubic, CP_cubic, res_cubic = \
        run_poly_optimizer(quartic=False)
    print(f"  Best: CT={CT_cubic:.6f}  CP={CP_cubic:.6f}")

if RUN_QUARTIC:
    print("\n" + "="*60); print("QUARTIC POLYNOMIAL OPTIMISER")
    p_qrt, r_qrt, c_qrt, tw_qrt, CT_qrt, CP_qrt, res_qrt = \
        run_poly_optimizer(quartic=True)
    print(f"  Best: CT={CT_qrt:.6f}  CP={CP_qrt:.6f}")

print("\n" + "="*60); print("PERFORMANCE SUMMARY")
print(f"  Actuator disk  CT={CT_TARGET}  a={a_ad:.4f}  CP={CP_ad:.6f}")
print(f"  Baseline       CT={CT_base:.6f}  CP={CP_base:.6f}  CP/CP_AD={CP_base/CP_ad:.4f}")
if RUN_ANALYTICAL:
    print(f"  Analytical     CT={CT_anal:.6f}  CP={CP_anal:.6f}  CP/CP_AD={CP_anal/CP_ad:.4f}")
if RUN_CUBIC:
    print(f"  Cubic poly     CT={CT_cubic:.6f}  CP={CP_cubic:.6f}  CP/CP_AD={CP_cubic/CP_ad:.4f}")
if RUN_QUARTIC:
    print(f"  Quartic poly   CT={CT_qrt:.6f}  CP={CP_qrt:.6f}  CP/CP_AD={CP_qrt/CP_ad:.4f}")

# =============================================================================
# 10.  PLOTS
# =============================================================================

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "plots_assignment")
os.makedirs(save_folder, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(save_folder, name), dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}")
    plt.show()

norm_val = 0.5 * U0 ** 2 * Radius

if PLOT_4_1:
    # ── 4.1  Alpha and inflow angle ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for TSR in TSR_SWEEP:
        res = sweep_data[TSR]
        axes[0].plot(res[:,2], res[:,6], label=rf"$\lambda$={TSR}")
        axes[1].plot(res[:,2], res[:,7], label=rf"$\lambda$={TSR}")
    for ax, yl, tl in zip(axes, [r"$\alpha$ [deg]",r"$\phi$ [deg]"],
                           ["Angle of attack","Inflow angle"]):
        ax.set_xlabel("r/R"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.grid(True); ax.legend()
    fig.suptitle("Baseline geometry — angle distributions")
    fig.tight_layout(); save_fig("4_1_alpha_phi.png")


if PLOT_4_2:
    # ── 4.2  Induction ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for TSR in TSR_SWEEP:
        res = sweep_data[TSR]
        axes[0].plot(res[:,2], res[:,0], label=rf"$\lambda$={TSR}")
        axes[1].plot(res[:,2], res[:,1], label=rf"$\lambda$={TSR}")
    for ax, yl, tl in zip(axes, [r"$a$ [-]",r"$a'$ [-]"],
                           ["Axial induction","Tangential induction"]):
        ax.set_xlabel("r/R"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.grid(True); ax.legend()
    fig.suptitle("Baseline geometry — induction factors")
    fig.tight_layout(); save_fig("4_2_induction.png")


if PLOT_4_3:
    # ── 4.3  Loading ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for TSR in TSR_SWEEP:
        res = sweep_data[TSR]
        axes[0].plot(res[:,2], res[:,3]/norm_val, label=rf"$\lambda$={TSR}")
        axes[1].plot(res[:,2], res[:,4]/norm_val, label=rf"$\lambda$={TSR}")
    for ax, yl, tl in zip(axes,
            [r"$C_n = F_n/(½U_\infty^2 R)$",r"$C_t = F_t/(½U_\infty^2 R)$"],
            ["Normal (thrust) loading","Tangential (torque) loading"]):
        ax.set_xlabel("r/R"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.grid(True); ax.legend()
    fig.suptitle("Baseline geometry — spanwise loading")
    fig.tight_layout(); save_fig("4_3_loading.png")


if PLOT_4_4:
    # ── 4.4  CT and CQ vs TSR ────────────────────────────────────────────────────
    tsr_list = sorted(tsr_perf)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(tsr_list, [tsr_perf[t]["CT"] for t in tsr_list], "bo-")
    axes[0].set_xlabel(r"$\lambda$"); axes[0].set_ylabel(r"$C_T$")
    axes[0].set_title(r"$C_T$ vs TSR"); axes[0].grid(True)
    axes[1].plot(tsr_list, [tsr_perf[t]["CP"]/t for t in tsr_list], "ro-")
    axes[1].set_xlabel(r"$\lambda$"); axes[1].set_ylabel(r"$C_Q$")
    axes[1].set_title(r"$C_Q$ vs TSR"); axes[1].grid(True)
    fig.tight_layout(); save_fig("4_4_CT_CQ_TSR.png")


if PLOT_5:
    # ── 5  Tip correction ─────────────────────────────────────────────────────────
    r_R8 = results_tsr8[:,2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(r_R8, results_tsr8[:,0], label="With Prandtl correction")
    axes[0].plot(res_nc[:,2], res_nc[:,0], "--", label="No correction (F=1)")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$a$ [-]")
    axes[0].set_title("Axial induction (TSR=8)"); axes[0].grid(True); axes[0].legend()
    axes[1].plot(r_R8, results_tsr8[:,3]/norm_val, label="With Prandtl correction")
    axes[1].plot(res_nc[:,2], res_nc[:,3]/norm_val, "--", label="No correction (F=1)")
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_n$ [-]")
    axes[1].set_title("Normal loading (TSR=8)"); axes[1].grid(True); axes[1].legend()
    fig.suptitle("Influence of Prandtl tip/root correction")
    fig.tight_layout(); save_fig("5_tip_correction.png")


if PLOT_6:
    # ── 6a  Number of annuli ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for N in [8, 20, 100]:
        b = np.linspace(RootLocation_R, TipLocation_R, N + 1)
        rows_N = []
        for i in range(N):
            rm = 0.5 * (b[i] + b[i+1])
            row, _ = solve_streamtube(b[i], b[i+1], Omega8,
                                       3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows_N.append(row)
        res_N = np.vstack(rows_N)
        ax.plot(res_N[:,2], res_N[:,3]/norm_val, "-o", markersize=4, label=f"N={N}")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.set_title("Normal loading — influence of N annuli (TSR=8)")
    ax.grid(True); ax.legend(); save_fig("6a_annuli.png")

    # ── 6b  Spacing method ────────────────────────────────────────────────────────
    N_sp = 40
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, b in {"Constant": np.linspace(RootLocation_R, TipLocation_R, N_sp+1),
                   "Cosine":   RootLocation_R + (TipLocation_R-RootLocation_R)
                               * 0.5*(1-np.cos(np.linspace(0,np.pi,N_sp+1)))}.items():
        rows_sp = []
        for i in range(N_sp):
            rm = 0.5*(b[i]+b[i+1])
            row, _ = solve_streamtube(b[i], b[i+1], Omega8,
                                       3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows_sp.append(row)
        res_sp = np.vstack(rows_sp)
        ax.plot(res_sp[:,2], res_sp[:,3]/norm_val, "-o", markersize=5, label=lbl)
    ax.set_xlim(0.85, 1.01); ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
    ax.set_title("Spacing method near tip (N=40, TSR=8)")
    ax.grid(True); ax.legend(); save_fig("6b_spacing.png")

    # ── 6c  Convergence ───────────────────────────────────────────────────────────
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


if PLOT_7:
    # ── 7  Stagnation pressure ────────────────────────────────────────────────────
    P0_up   = 0.5 * rho * U0**2 * np.ones(len(r_R8))
    dP0     = 2.0*rho*U0**2 * results_tsr8[:,0] * F_tsr8 * (1.0 - results_tsr8[:,0]*F_tsr8)
    P0_down = P0_up - dP0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_R8, P0_up,   "b-",  lw=2, label=r"$P_0$ far upstream  (stat. 1)")
    ax.plot(r_R8, P0_up,   "b--", lw=1.5, alpha=0.6, label=r"$P_0$ rotor upwind  (stat. 2)")
    ax.plot(r_R8, P0_down, "r--", lw=1.5, alpha=0.6, label=r"$P_0$ rotor downwind (stat. 3)")
    ax.plot(r_R8, P0_down, "r-",  lw=2, label=r"$P_0$ far downstream (stat. 4)")
    ax.set_xlabel("r/R"); ax.set_ylabel(r"$P_0$ [Pa]")
    ax.set_title("Stagnation pressure at four streamwise stations (TSR=8)")
    ax.legend(); ax.grid(True); save_fig("7_stagnation_pressure.png")


if PLOT_8:
    # ── 8  All designs ────────────────────────────────────────────────────────────
    r_R_d = np.linspace(RootLocation_R, TipLocation_R, 400)

    n_c = len(p_cubic); n_q = len(p_qrt)
    _, _, _, _, c_tip_c, c2_c, c3_c = p_cubic[:7]; c4_c = p_cubic[7] if n_c==8 else 0.0
    _, _, _, _, c_tip_q, c2_q, c3_q = p_qrt[:7];   c4_q = p_qrt[7]   if n_q==8 else 0.0
    pitch_c, tr_c, tt_c, tc_c = p_cubic[:4]
    pitch_q, tr_q, tt_q, tc_q = p_qrt[:4]

    designs = [
        ("Baseline",
         3.0*(1-r_R_d)+1.0, -(14.0*(1-r_R_d)+Pitch),
         res_base, CT_base, CP_base),
        ("Analytical",
         np.interp(r_R_d, r_anal/Radius, c_anal),
         np.interp(r_R_d, r_anal/Radius, tw_anal),
         res_anal, CT_anal, CP_anal),
        ("Cubic poly",
         np.array([chord_poly(r, c_tip_c,c2_c,c3_c,c4_c) for r in r_R_d]),
         np.array([twist_poly(r, pitch_c,tr_c,tt_c,tc_c) for r in r_R_d]),
         res_cubic, CT_cubic, CP_cubic),
        ("Quartic poly",
         np.array([chord_poly(r, c_tip_q,c2_q,c3_q,c4_q) for r in r_R_d]),
         np.array([twist_poly(r, pitch_q,tr_q,tt_q,tc_q) for r in r_R_d]),
         res_qrt, CT_qrt, CP_qrt),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, c_d, tw_d, *_ in designs:
        axes[0].plot(r_R_d, c_d, label=lbl)
        axes[1].plot(r_R_d, tw_d, label=lbl)
    axes[0].axhline(CHORD_MIN, color="grey", ls=":", lw=1, label=f"Min {CHORD_MIN} m")
    axes[0].scatter([RootLocation_R],[CHORD_ROOT], zorder=5, color="k",
                    label=f"Root pin {CHORD_ROOT} m")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
    axes[0].set_title("Chord distribution"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
    axes[1].set_title(r"Twist  ($\alpha=\mathrm{twist}+\phi$)")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); save_fig("8a_chord_twist.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lbl, _, _, res, *_ in designs:
        axes[0].plot(res[:,2], res[:,0], label=lbl)
        axes[1].plot(res[:,2], res[:,3]/norm_val, label=lbl)
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$a$ [-]")
    axes[0].set_title("Axial induction"); axes[0].legend(); axes[0].grid(True)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_n$ [-]")
    axes[1].set_title("Normal loading"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); save_fig("8b_induction_loading.png")

    x = np.arange(len(designs)+1); w = 0.35
    labels_b = [d[0] for d in designs]+["Actuator disk"]
    cp_b     = [d[5] for d in designs]+[CP_ad]
    ct_b     = [d[4] for d in designs]+[CT_TARGET]
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x-w/2, cp_b, w, label="CP", color="steelblue")
    b2 = ax.bar(x+w/2, ct_b, w, label="CT", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(labels_b, rotation=15, ha="right")
    ax.set_ylabel("Coefficient [-]"); ax.set_title("Performance — all designs")
    ax.legend()
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(True, axis="y"); fig.tight_layout(); save_fig("8c_performance.png")


if PLOT_9:
    # ── 9  Cl and chord ──────────────────────────────────────────────────────────
    r_R_am = res_anal[:,2]; cl_am = res_anal[:,8]
    c_am   = np.interp(r_R_am, r_anal/Radius, c_anal)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(r_R_am, cl_am, "b-", lw=2, label=r"$C_l$")
    axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$C_l$ [-]"); axes[0].grid(True)
    ax2 = axes[0].twinx()
    ax2.plot(r_R_am, c_am, "r--", lw=2, label="Chord [m]")
    ax2.set_ylabel("Chord [m]", color="red"); ax2.tick_params(axis="y", labelcolor="red")
    axes[0].set_zorder(ax2.get_zorder()+1); axes[0].patch.set_visible(False)
    h1,l1 = axes[0].get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1+h2, l1+l2)
    axes[0].set_title(r"$C_l$ and chord (analytical optimum)")
    axes[1].plot(r_R_am, cl_am*c_am, "g-", lw=2)
    axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_l \cdot c$  [m]")
    axes[1].set_title(r"Circulation proxy $\Gamma \propto C_l \cdot c$")
    axes[1].grid(True)
    fig.tight_layout(); save_fig("9_cl_chord.png")


if PLOT_10:
    # ── 10  Polar ─────────────────────────────────────────────────────────────────
    alpha_opt, cl_opt, cd_opt = find_optimal_alpha()
    alphas_d = np.linspace(polar_alpha[0], polar_alpha[-1], 500)
    cl_d = np.interp(alphas_d, polar_alpha, polar_cl)
    cd_d = np.interp(alphas_d, polar_alpha, polar_cd)
    ld_d = cl_d / np.maximum(cd_d, 1e-8)

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
    sc2 = axes[1].scatter(res_anal[:,6], ld_ops, c=r_R_am, cmap="viridis", s=25, zorder=5)
    plt.colorbar(sc2, ax=axes[1], label="r/R")
    axes[1].set_xlabel(r"$\alpha$ [deg]"); axes[1].set_ylabel(r"$C_l/C_d$")
    axes[1].set_title(r"$C_l/C_d$ vs $\alpha$"); axes[1].grid(True); axes[1].legend()
    fig.tight_layout(); save_fig("10_polar.png")

print("\n" + "="*60)
print("ALL PLOTS SAVED TO:", save_folder)
print("="*60)

# =============================================================================
# 11.  SAVE ALL RESULTS  (run once; reload with plot_results.py)
# =============================================================================

def save_results(path="bem_results.npz"):
    """
    Save every array needed by plot_results.py into a single .npz file.

    Arrays stored
    -------------
    # Polar
    polar_alpha, polar_cl, polar_cd

    # Configuration scalars (stored as 0-d arrays)
    cfg_*   : Radius, NBlades, U0, rho, RootLocation_R, TipLocation_R,
              Pitch, CHORD_ROOT, CHORD_MIN, CT_TARGET, TSR_DESIGN, DELTA_R_R

    # TSR sweep  (one stacked block per TSR in TSR_SWEEP)
    sweep_tsrs         : 1-D array of TSR values  [5]
    sweep_res_<TSR>    : BEM result array          [N_annuli, 10]
    tsr_CT, tsr_CP     : scalar CT/CP per TSR      [5]

    # TSR=8 specific
    results_tsr8       : BEM result at TSR=8       [N_annuli, 10]
    res_nc             : no-correction BEM TSR=8   [N_annuli, 5]
    ct_hist_tsr8       : convergence history       [max_iter]
    F_tsr8             : Prandtl F spanwise        [N_annuli]

    # Section-6 annuli/spacing  (pre-computed so plot_results.py is fast)
    annuli_N8, annuli_N20, annuli_N100   : BEM results for N=8,20,100
    spacing_constant, spacing_cosine     : BEM results for N=40 spacing study

    # Designs — geometry nodes
    r_base, c_base, tw_base
    r_anal, c_anal, tw_anal
    r_cubic, c_cubic, tw_cubic
    r_qrt,  c_qrt,  tw_qrt

    # Designs — BEM results  [N_annuli, 10]
    res_base, res_anal, res_cubic, res_qrt

    # Designs — scalar performance
    CT_base, CP_base, CT_anal, CP_anal,
    CT_cubic, CP_cubic, CT_qrt, CP_qrt, CP_ad

    # Polynomial parameters
    p_cubic  : cubic parameter vector   [7]
    p_qrt    : quartic parameter vector [8]
    """
    # ── Section-6 pre-computed results ───────────────────────────────────────
    annuli_results = {}
    for N in [8, 20, 100]:
        b = np.linspace(RootLocation_R, TipLocation_R, N + 1)
        rows = []
        for i in range(N):
            rm = 0.5*(b[i]+b[i+1])
            row, _ = solve_streamtube(b[i], b[i+1], Omega8,
                                       3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows.append(row)
        annuli_results[N] = np.vstack(rows)

    spacing_results = {}
    N_sp = 40
    for lbl, b in {
        "constant": np.linspace(RootLocation_R, TipLocation_R, N_sp+1),
        "cosine":   RootLocation_R + (TipLocation_R-RootLocation_R)
                    * 0.5*(1-np.cos(np.linspace(0, np.pi, N_sp+1)))
    }.items():
        rows = []
        for i in range(N_sp):
            rm = 0.5*(b[i]+b[i+1])
            row, _ = solve_streamtube(b[i], b[i+1], Omega8,
                                       3.0*(1-rm)+1.0, -(14.0*(1-rm)+Pitch))
            rows.append(row)
        spacing_results[lbl] = np.vstack(rows)

    # ── Build the keyword dict ────────────────────────────────────────────────
    kw = dict(
        # Polar
        polar_alpha=polar_alpha, polar_cl=polar_cl, polar_cd=polar_cd,

        # Config scalars
        cfg_Radius=Radius, cfg_NBlades=NBlades, cfg_U0=U0, cfg_rho=rho,
        cfg_RootLocation_R=RootLocation_R, cfg_TipLocation_R=TipLocation_R,
        cfg_Pitch=Pitch, cfg_CHORD_ROOT=CHORD_ROOT, cfg_CHORD_MIN=CHORD_MIN,
        cfg_CT_TARGET=CT_TARGET, cfg_TSR_DESIGN=TSR_DESIGN,
        cfg_DELTA_R_R=DELTA_R_R,

        # TSR sweep
        sweep_tsrs=np.array(TSR_SWEEP, dtype=float),
        tsr_CT=np.array([tsr_perf[t]["CT"] for t in TSR_SWEEP]),
        tsr_CP=np.array([tsr_perf[t]["CP"] for t in TSR_SWEEP]),

        # TSR=8
        results_tsr8=results_tsr8,
        res_nc=res_nc,
        ct_hist_tsr8=ct_hist_tsr8,
        F_tsr8=F_tsr8,

        # Section-6
        annuli_N8=annuli_results[8],
        annuli_N20=annuli_results[20],
        annuli_N100=annuli_results[100],
        spacing_constant=spacing_results["constant"],
        spacing_cosine=spacing_results["cosine"],

        # Geometry nodes
        r_base=r_base, c_base=c_base, tw_base=tw_base,
        r_anal=r_anal, c_anal=c_anal, tw_anal=tw_anal,
        r_cubic=r_cubic, c_cubic=c_cubic, tw_cubic=tw_cubic,
        r_qrt=r_qrt, c_qrt=c_qrt, tw_qrt=tw_qrt,

        # BEM results
        res_base=res_base, res_anal=res_anal,
        res_cubic=res_cubic, res_qrt=res_qrt,

        # Scalar performance
        CT_base=CT_base, CP_base=CP_base,
        CT_anal=CT_anal, CP_anal=CP_anal,
        CT_cubic=CT_cubic, CP_cubic=CP_cubic,
        CT_qrt=CT_qrt, CP_qrt=CP_qrt,
        CP_ad=CP_ad,

        # Polynomial parameters
        p_cubic=np.array(p_cubic, dtype=float),
        p_qrt=np.array(p_qrt,   dtype=float),
    )

    # One entry per TSR sweep result
    for TSR in TSR_SWEEP:
        kw[f"sweep_res_{int(TSR)}"] = sweep_data[TSR]

    np.savez(path, **kw)
    print(f"\nResults saved to: {path}")
    print(f"  ({len(kw)} arrays,  "
          f"{sum(v.nbytes for v in kw.values() if hasattr(v,'nbytes'))//1024} KB)")


if SAVE_RESULTS:
    save_results("full_bem_results.npz")
else:
    print("\nSAVE_RESULTS=False — bem_results.npz not written.")