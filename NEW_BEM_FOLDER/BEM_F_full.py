"""
assignment.py  —  AE4135 Rotor Wake Aerodynamics, Assignment 1
================================================================
Self-contained script that produces all required plots.

Sections produced
-----------------
 4.1  Alpha and inflow angle vs r/R                  (TSR sweep)
 4.2  Axial and azimuthal induction vs r/R            (TSR sweep)
 4.3  Thrust and azimuthal loading vs r/R             (TSR sweep)
 4.4  Total CT and CQ vs tip-speed ratio
  5   Tip-loss correction influence on a and loading
  6   Number of annuli, spacing method, convergence history
  7   Stagnation pressure at four streamwise stations
  8   All four designs — chord, twist, CP/CT comparison
  9   Lift coefficient and chord relation  (analytical optimum)
 10   Cl/Cd polar with operating point highlighted

Run:   python assignment.py
All plots are saved to  ./plots_assignment/
"""

import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

# Set to True  -> helical-wake Prandtl (Prandtl 1919, more physical)
# Set to False -> simplified Glauert form (matches reference notebook)
USE_HELICAL_PRANDTL = False

# Number of multi-start runs for each polynomial optimiser
N_STARTS = 12

# Tip-speed ratios for the baseline sweep
TSR_SWEEP = list(np.arange(6, 11, 1))

# Design point
TSR_DESIGN = 8.0
CT_TARGET  = 0.75

# Geometry constants
Radius         = 50.0
NBlades        = 3
U0             = 10.0
rho            = 1.225
RootLocation_R = 0.2
TipLocation_R  = 1.0
Pitch          = -2.0       # baseline blade pitch [deg]
A_disk         = np.pi * Radius ** 2

# Chord constraints for polynomial optimiser
CHORD_ROOT    = 3.4         # fixed root chord [m]
CHORD_MIN     = 0.3         # manufacturing minimum [m]
CHORD_MAX_REG = 6.0         # soft upper limit [m]
MAX_CHORD     = 3.4         # analytical method structural cap [m]
MIN_CHORD_TIP = 0.3         # analytical method tip minimum [m]

# Spanwise resolutions
delta_r_R_final = 0.005
delta_r_R_opt   = 0.01

# =============================================================================
# 1.  POLAR DATA
# =============================================================================

data        = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = data["Alfa"].to_numpy()
polar_cl    = data["Cl"].to_numpy()
polar_cd    = data["Cd"].to_numpy()

# =============================================================================
# 2.  BEM CORE FUNCTIONS  (reference class — verbatim)
# =============================================================================

def ainduction(CT):
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1
    if np.isscalar(CT):
        if CT >= CT2:
            return 1 + (CT - CT1) / (4 * (np.sqrt(CT1) - 1))
        return 0.5 - 0.5 * np.sqrt(max(0, 1 - CT))
    a = np.zeros(np.shape(CT))
    a[CT >= CT2] = 1 + (CT[CT >= CT2] - CT1) / (4 * (np.sqrt(CT1) - 1))
    a[CT < CT2]  = 0.5 - 0.5 * np.sqrt(np.maximum(0, 1 - CT[CT < CT2]))
    return a


def _PrandtlSimplified(r_R, rootR, tipR, TSR, NB, a_in):
    a = np.clip(a_in, -0.9, 0.99)
    t1 = -NB/2*(tipR - r_R)/r_R * np.sqrt(1 + ((TSR*r_R)**2)/((1-a)**2))
    t2 =  NB/2*(rootR - r_R)/r_R * np.sqrt(1 + ((TSR*r_R)**2)/((1-a)**2))
    Ftip  = np.array(2/np.pi * np.arccos(np.clip(np.exp(t1), 0, 1)))
    Froot = np.array(2/np.pi * np.arccos(np.clip(np.exp(t2), 0, 1)))
    return Froot * Ftip


def _PrandtlHelical(r_R, rootR, tipR, TSR, NB, a_in):
    a = np.clip(a_in, -0.9, 0.99)
    d = max(2*np.pi/NB * (1-a) / np.sqrt(TSR**2 + (1-a)**2), 1e-8)
    Ftip  = float(2/np.pi * np.arccos(np.exp(np.clip(-np.pi*(tipR  - r_R)/d, -500, 0))))
    Froot = float(2/np.pi * np.arccos(np.exp(np.clip(-np.pi*(r_R - rootR)/d, -500, 0))))
    Ftip  = 0.0 if np.isnan(Ftip)  else Ftip
    Froot = 0.0 if np.isnan(Froot) else Froot
    return Froot * Ftip


def PrandtlTipRootCorrection(r_R, rootR, tipR, TSR, NB, a):
    if USE_HELICAL_PRANDTL:
        return _PrandtlHelical(r_R, rootR, tipR, TSR, NB, a)
    return _PrandtlSimplified(r_R, rootR, tipR, TSR, NB, a)


def LoadBladeElement(vnorm, vtan, chord, twist, pa, pcl, pcd):
    """twist in deg; convention alpha = twist + phi_deg."""
    vmag2 = vnorm**2 + vtan**2
    phi   = np.arctan2(vnorm, vtan)
    alpha = twist + np.degrees(phi)
    cl    = np.interp(alpha, pa, pcl)
    cd    = np.interp(alpha, pa, pcd)
    lift  = 0.5 * vmag2 * cl * chord
    drag  = 0.5 * vmag2 * cd * chord
    fnorm = lift * np.cos(phi) + drag * np.sin(phi)
    ftan  = lift * np.sin(phi) - drag * np.cos(phi)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord
    return fnorm, ftan, gamma, alpha, np.degrees(phi), cl, cd


def SolveStreamtube(U0, r1_R, r2_R, rootR, tipR,
                    Omega, Radius, NB, chord, twist,
                    max_iter=300, tol=1e-5):
    """
    Returns [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi, cl, cd],
    fnorm_history.
    cl and cd are the final-iteration aerodynamic coefficients — needed
    for sections 9 and 10 of the report.
    """
    Area    = np.pi * ((r2_R*Radius)**2 - (r1_R*Radius)**2)
    r_mid   = (r1_R + r2_R) / 2
    r_local = r_mid * Radius
    a = 0.0; aline = 0.0
    hist = []
    cl_final = cd_final = 0.0

    for _ in range(max_iter):
        Urotor = U0 * (1 - a)
        Utan   = (1 + aline) * Omega * r_local

        fnorm, ftan, gamma, alpha, phi, cl, cd = LoadBladeElement(
            Urotor, Utan, chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        cl_final = cl; cd_final = cd
        hist.append(fnorm)

        CT_loc = (fnorm * Radius * (r2_R - r1_R) * NB) / (0.5 * Area * U0**2)
        anew   = ainduction(CT_loc)
        F = max(PrandtlTipRootCorrection(r_mid, rootR, tipR,
                                          Omega*Radius/U0, NB, anew), 0.0001)
        anew /= F
        a_old = a
        a     = 0.75*a + 0.25*anew
        aline = (ftan*NB) / (2*np.pi*U0*(1-a)*Omega*2*r_local**2)
        aline /= F

        if abs(a - a_old) < tol:
            break

    return ([a, aline, r_mid, fnorm, ftan, gamma, alpha, phi, cl_final, cd_final],
            np.array(hist))


def SolveStreamtube_NoCorrection(U0, r1_R, r2_R, rootR, tipR,
                                  Omega, Radius, NB, chord, twist,
                                  max_iter=300, tol=1e-5):
    """F forced to 1 — no Prandtl correction."""
    Area    = np.pi * ((r2_R*Radius)**2 - (r1_R*Radius)**2)
    r_mid   = (r1_R + r2_R) / 2
    r_local = r_mid * Radius
    a = 0.0; aline = 0.0

    for _ in range(max_iter):
        Urotor = U0*(1-a); Utan = (1+aline)*Omega*r_local
        fnorm, ftan, gamma, alpha, phi, cl, cd = LoadBladeElement(
            Urotor, Utan, chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        CT_loc = (fnorm*Radius*(r2_R-r1_R)*NB) / (0.5*Area*U0**2)
        anew   = ainduction(CT_loc)
        a      = 0.75*a + 0.25*anew
        aline  = (ftan*NB) / (2*np.pi*U0*max(1-a,1e-4)*Omega*2*r_local**2)
        if abs(a - anew) < tol:
            break

    return [a, aline, r_mid, fnorm, ftan]


# =============================================================================
# 3.  ROTOR EVALUATOR  (used by both optimisers)
# =============================================================================

def evaluate_rotor(r_nodes, c_nodes, tw_nodes, tsr=TSR_DESIGN):
    """
    Run BEM on an arbitrary geometry.
    Returns CT, CP, res_arr  (columns: a,aline,r_mid,fnorm,ftan,gamma,alpha,phi,cl,cd)
    """
    Omega   = U0 * tsr / Radius
    results = []
    for i in range(len(r_nodes) - 1):
        r1    = r_nodes[i]   / Radius
        r2    = r_nodes[i+1] / Radius
        chord = 0.5*(c_nodes[i]  + c_nodes[i+1])
        twist = 0.5*(tw_nodes[i] + tw_nodes[i+1])
        res, _ = SolveStreamtube(U0, r1, r2, RootLocation_R, TipLocation_R,
                                  Omega, Radius, NBlades, chord, twist)
        results.append(res)

    res_arr = np.array(results)
    dr      = np.diff(r_nodes)
    CT = np.sum(dr * res_arr[:,3] * NBlades / (0.5*U0**2*np.pi*Radius**2))
    CP = np.sum(dr * res_arr[:,4] * res_arr[:,2] * NBlades * Radius * Omega
                / (0.5*U0**3*np.pi*Radius**2))
    return CT, CP, res_arr


# =============================================================================
# 4.  BASELINE GEOMETRY
# =============================================================================

def get_baseline(delta_r_R=delta_r_R_final):
    r_R  = np.arange(RootLocation_R, TipLocation_R + delta_r_R/2, delta_r_R)
    r    = r_R * Radius
    c    = 3.0*(1 - r_R) + 1.0
    tw   = -(14.0*(1 - r_R) + Pitch)
    return r, c, tw


# =============================================================================
# 5.  TSR SWEEP  (sections 4.1 – 4.4)
# =============================================================================

print("Running TSR sweep (baseline geometry) ...")
r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R_final/2, delta_r_R_final)

tsr_perf    = {}
results_tsr8 = None
ct_hist_tsr8 = None
F_tsr8       = None

for TSR in TSR_SWEEP:
    Omega = U0 * TSR / Radius
    temp  = []; hists = []

    for i in range(len(r_R_bins) - 1):
        rm    = (r_R_bins[i] + r_R_bins[i+1]) / 2
        chord = 3*(1 - rm) + 1
        twist = -(14*(1 - rm) + Pitch)
        res, hist = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1],
                                     RootLocation_R, TipLocation_R,
                                     Omega, Radius, NBlades, chord, twist)
        temp.append(res); hists.append(hist)

    res_arr = np.array(temp)
    dr      = (r_R_bins[1:] - r_R_bins[:-1]) * Radius
    CT = np.sum(dr * res_arr[:,3] * NBlades / (0.5*U0**2*np.pi*Radius**2))
    CP = np.sum(dr * res_arr[:,4] * res_arr[:,2] * NBlades * Radius * Omega
                / (0.5*U0**3*np.pi*Radius**2))
    tsr_perf[TSR] = {"CT": CT, "CP": CP}
    print(f"  TSR={TSR:.0f}  CT={CT:.4f}  CP={CP:.4f}")

    if TSR == 8:
        results_tsr8 = res_arr
        max_len      = max(len(h) for h in hists)
        padded       = np.array([np.concatenate([h, np.full(max_len-len(h), h[-1])])
                                  for h in hists])
        ct_hist_tsr8 = np.sum(padded * dr[:,None] * NBlades
                               / (0.5*U0**2*np.pi*Radius**2), axis=0)
        # Compute F spanwise at TSR=8 for stagnation pressure plot
        F_tsr8 = np.array([
            max(PrandtlTipRootCorrection(
                (r_R_bins[i]+r_R_bins[i+1])/2,
                RootLocation_R, TipLocation_R,
                Omega*Radius/U0, NBlades, res_arr[i,0]), 0.0001)
            for i in range(len(r_R_bins)-1)
        ])

# =============================================================================
# 6.  NO-CORRECTION RUN  (section 5)
# =============================================================================

print("Running no-correction BEM (TSR=8) ...")
Omega8  = U0 * 8 / Radius
res_nc_list = []
for i in range(len(r_R_bins) - 1):
    rm    = (r_R_bins[i] + r_R_bins[i+1]) / 2
    chord = 3*(1 - rm) + 1
    twist = -(14*(1 - rm) + Pitch)
    res_nc_list.append(SolveStreamtube_NoCorrection(
        U0, r_R_bins[i], r_R_bins[i+1],
        RootLocation_R, TipLocation_R,
        Omega8, Radius, NBlades, chord, twist))
res_nc = np.array(res_nc_list)

# =============================================================================
# 7.  ANALYTICAL OPTIMUM  (design_for_exact_ct)
# =============================================================================

def find_optimal_alpha():
    alphas = np.linspace(polar_alpha[0], polar_alpha[-1], 2000)
    cl_v   = np.interp(alphas, polar_alpha, polar_cl)
    cd_v   = np.interp(alphas, polar_alpha, polar_cd)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cd_v > 1e-6, cl_v/cd_v, 0.0)
    idx = int(np.argmax(ratio))
    return float(alphas[idx]), float(cl_v[idx]), float(cd_v[idx])


def generate_ideal_rotor(target_a, n_nodes=101,
                         alpha_opt_deg=None, cl_opt=None, cd_opt=None,
                         ap_in=None):
    if alpha_opt_deg is None:
        alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()

    r_start = RootLocation_R*Radius + 0.005*Radius
    r_end   = TipLocation_R*Radius  - 0.005*Radius
    r_nodes = r_start + (r_end-r_start)*(1-np.cos(np.linspace(0,np.pi,n_nodes)))/2

    c_nodes = np.zeros(n_nodes); tw_nodes = np.zeros(n_nodes)

    if ap_in is not None:
        r_mid_prev  = 0.5*(r_nodes[:-1]+r_nodes[1:])
        ap_at_nodes = np.interp(r_nodes, r_mid_prev, ap_in,
                                left=ap_in[0], right=ap_in[-1])
    else:
        ap_at_nodes = np.zeros(n_nodes)

    alpha_opt_rad = np.radians(alpha_opt_deg)
    for i, r in enumerate(r_nodes):
        r_R   = r/Radius; local_tsr = TSR_DESIGN*r_R
        F = float(max(PrandtlTipRootCorrection(r_R, RootLocation_R, TipLocation_R,
                                                TSR_DESIGN, NBlades, target_a), 0.0001))
        ap  = float(ap_at_nodes[i])
        phi = np.arctan2(1-target_a, local_tsr*(1+ap))
        tw_nodes[i] = alpha_opt_deg - np.degrees(phi)
        Cn  = cl_opt*np.cos(phi) + cd_opt*np.sin(phi)
        c_nodes[i] = (8*np.pi*r*target_a*F*(1-target_a*F)*np.sin(phi)**2
                      / (NBlades*(1-target_a)**2*max(Cn,1e-8)))

    c_nodes = np.clip(c_nodes, 0.0, MAX_CHORD)
    max_idx = int(np.argmax(c_nodes)); c_nodes[:max_idx+1] = c_nodes[max_idx]
    c_nodes = np.clip(c_nodes, MIN_CHORD_TIP, None)
    return r_nodes, c_nodes, tw_nodes


def design_for_exact_ct():
    alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()
    last_ap = [None]; last_geom = [None]

    def residual(ta):
        r,c,tw = generate_ideal_rotor(ta, alpha_opt_deg=alpha_opt_deg,
                                      cl_opt=cl_opt, cd_opt=cd_opt, ap_in=last_ap[0])
        last_geom[0] = (r,c,tw)
        old_out, sys.stdout = sys.stdout, io.StringIO()
        try:
            CT,CP,res_arr = evaluate_rotor(r,c,tw)
        finally:
            sys.stdout = old_out
        last_ap[0] = res_arr[:,1]
        print(f"  target_a={ta:.5f}  CT={CT:.5f}  CP={CP:.5f}")
        return CT - CT_TARGET

    sol = root_scalar(residual, bracket=[0.20,0.35], method="brentq", xtol=1e-5)
    if not sol.converged:
        raise RuntimeError("Brent root-finder did not converge.")
    print(f"  Converged: a = {sol.root:.6f}")
    r,c,tw = last_geom[0]
    CT,CP,res_arr = evaluate_rotor(r,c,tw)
    return r, c, tw, CT, CP, res_arr


# =============================================================================
# 8.  POLYNOMIAL OPTIMISERS  (cubic and quartic chord)
# =============================================================================

def _span_x(r_R):
    return (r_R - RootLocation_R) / (TipLocation_R - RootLocation_R)

def chord_poly(r_R, c_tip, c2, c3, c4=0.0):
    x  = _span_x(r_R)
    b1 = c_tip - CHORD_ROOT - c2 - c3 - c4
    return CHORD_ROOT + b1*x + c2*x**2 + c3*x**3 + c4*x**4

def twist_poly(r_R, pitch, t_root, t_tip, t_curve):
    x = _span_x(r_R)
    return pitch + t_root*(1-x) + t_tip*x + t_curve*x*(1-x)

def build_poly_geometry(params, delta_r_R=delta_r_R_opt):
    n_params = len(params)
    if n_params == 7:
        pitch,t_root,t_tip,t_curve,c_tip,c2,c3 = params; c4 = 0.0
    else:
        pitch,t_root,t_tip,t_curve,c_tip,c2,c3,c4 = params
    r_R = np.arange(RootLocation_R, TipLocation_R+delta_r_R/2, delta_r_R)
    r   = r_R * Radius
    c   = np.array([chord_poly(rr, c_tip,c2,c3,c4) for rr in r_R])
    tw  = np.array([twist_poly(rr, pitch,t_root,t_tip,t_curve) for rr in r_R])
    return r, c, tw

def _chord_min_max(params, n=300):
    n_params = len(params)
    if n_params == 7:
        _,_,_,_,c_tip,c2,c3 = params; c4 = 0.0
    else:
        _,_,_,_,c_tip,c2,c3,c4 = params
    r_R = np.linspace(RootLocation_R, TipLocation_R, n)
    c   = np.array([chord_poly(rr,c_tip,c2,c3,c4) for rr in r_R])
    return float(np.min(c)), float(np.max(c))

def _make_objective(quartic=False):
    def objective(params):
        c_min, c_max = _chord_min_max(params)
        pen = 0.0
        if c_min < CHORD_MIN:
            pen += 1e4*(CHORD_MIN-c_min)**2
        if c_max > CHORD_MAX_REG:
            pen += 1e3*(c_max-CHORD_MAX_REG)**2
        if c_min < CHORD_MIN*0.5:
            return 1e9
        r,c,tw = build_poly_geometry(params, delta_r_R_opt)
        CT,CP,_ = evaluate_rotor(r,c,tw)
        pen += 5e3*(CT-CT_TARGET)**2
        if quartic:
            _,_,_,t_curve,_,c2,c3,c4 = params
            pen += 0.05*t_curve**2 + 0.01*(c2**2+c3**2+c4**2)
        else:
            _,_,_,t_curve,_,c2,c3 = params
            pen += 0.05*t_curve**2 + 0.01*(c2**2+c3**2)
        return -CP + pen
    return objective

def run_poly_optimizer(quartic=False, n_starts=N_STARTS, seed=42):
    rng  = np.random.default_rng(seed)
    obj  = _make_objective(quartic)
    label = "quartic" if quartic else "cubic"

    if quartic:
        bounds  = [(-8,8),(-25,5),(-10,15),(-20,20),(0.3,2),(-5,5),(-5,5),(-5,5)]
        x0_nom  = np.array([-2,-7,2,0,1,0,0,0])
        def rand_start():
            return np.array([rng.uniform(-6,6), rng.uniform(-20,0),
                             rng.uniform(-5,10), rng.uniform(-10,10),
                             rng.uniform(0.3,1.5), rng.uniform(-3,3),
                             rng.uniform(-3,3), rng.uniform(-3,3)])
    else:
        bounds  = [(-8,8),(-25,5),(-10,15),(-20,20),(0.3,2),(-5,5),(-5,5)]
        x0_nom  = np.array([-2,-7,2,0,1,0,0])
        def rand_start():
            return np.array([rng.uniform(-6,6), rng.uniform(-20,0),
                             rng.uniform(-5,10), rng.uniform(-10,10),
                             rng.uniform(0.3,1.5), rng.uniform(-3,3),
                             rng.uniform(-3,3)])

    starts = [x0_nom] + [rand_start() for _ in range(n_starts-1)]
    best_res = None; best_obj = np.inf

    for k, x0 in enumerate(starts, 1):
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter":300,"ftol":1e-10,"gtol":1e-7})
        if res.fun < best_obj:
            best_obj = res.fun; best_res = res
        r,c,tw = build_poly_geometry(res.x, delta_r_R_opt)
        CT,_,_ = evaluate_rotor(r,c,tw)
        print(f"  [{label}] Start {k:02d}/{n_starts}  obj={res.fun:+.6f}  CT={CT:.4f}")

    # Final high-resolution evaluation
    r,c,tw = build_poly_geometry(best_res.x, delta_r_R_final)
    CT,CP,res_arr = evaluate_rotor(r,c,tw)
    return best_res.x, r, c, tw, CT, CP, res_arr


# =============================================================================
# 9.  RUN ALL METHODS
# =============================================================================

print("\n" + "="*65)
print("BASELINE BEM EVALUATION")
print("="*65)
r_base, c_base, tw_base = get_baseline(delta_r_R_final)
CT_base, CP_base, res_base = evaluate_rotor(r_base, c_base, tw_base)
print(f"  Baseline  CT={CT_base:.6f}  CP={CP_base:.6f}")

print("\n" + "="*65)
print("ANALYTICAL OPTIMUM")
print("="*65)
r_anal, c_anal, tw_anal, CT_anal, CP_anal, res_anal = design_for_exact_ct()

print("\n" + "="*65)
print("CUBIC POLYNOMIAL OPTIMISER")
print("="*65)
p_cubic, r_cubic, c_cubic, tw_cubic, CT_cubic, CP_cubic, res_cubic = \
    run_poly_optimizer(quartic=False)

print("\n" + "="*65)
print("QUARTIC POLYNOMIAL OPTIMISER")
print("="*65)
p_qrt, r_qrt, c_qrt, tw_qrt, CT_qrt, CP_qrt, res_qrt = \
    run_poly_optimizer(quartic=True)

# Actuator disk reference
a_ad  = 0.5*(1 - np.sqrt(1-CT_TARGET))
CP_ad = 4*a_ad*(1-a_ad)**2

print("\n" + "="*65)
print("PERFORMANCE SUMMARY")
print("="*65)
print(f"Actuator disk (CT={CT_TARGET}):  a={a_ad:.4f}  CP={CP_ad:.6f}")
print(f"Baseline:       CT={CT_base:.6f}  CP={CP_base:.6f}  "
      f"CP/CP_AD={CP_base/CP_ad:.4f}")
print(f"Analytical:     CT={CT_anal:.6f}  CP={CP_anal:.6f}  "
      f"CP/CP_AD={CP_anal/CP_ad:.4f}")
print(f"Cubic poly:     CT={CT_cubic:.6f}  CP={CP_cubic:.6f}  "
      f"CP/CP_AD={CP_cubic/CP_ad:.4f}")
print(f"Quartic poly:   CT={CT_qrt:.6f}  CP={CP_qrt:.6f}  "
      f"CP/CP_AD={CP_qrt/CP_ad:.4f}")

# =============================================================================
# 10.  PLOTTING
# =============================================================================

base_path   = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_assignment")
os.makedirs(save_folder, exist_ok=True)

def save_and_show(filename):
    path = os.path.join(save_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()

norm_val = 0.5 * U0**2 * Radius   # normalisation for loading plots
r_mid8   = results_tsr8[:, 2]     # r_mid at TSR=8

# ── 4.1  Alpha and inflow angle ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5))
for TSR in TSR_SWEEP:
    r_R_arr = np.arange(RootLocation_R, TipLocation_R+delta_r_R_final/2, delta_r_R_final)
    r_R_mid = 0.5*(r_R_arr[:-1]+r_R_arr[1:])
    # collect results for each TSR from scratch (they were collected earlier)
ax.set_prop_cycle(None)
# Re-run sweep to get per-TSR spanwise arrays for alpha/phi
alpha_all = {}; phi_all = {}
for TSR in TSR_SWEEP:
    Omega = U0*TSR/Radius
    tmp   = []
    for i in range(len(r_R_bins)-1):
        rm    = (r_R_bins[i]+r_R_bins[i+1])/2
        chord = 3*(1-rm)+1; twist = -(14*(1-rm)+Pitch)
        res,_ = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1],
                                  RootLocation_R, TipLocation_R,
                                  Omega, Radius, NBlades, chord, twist)
        tmp.append(res)
    arr = np.array(tmp)
    alpha_all[TSR] = arr[:,6]; phi_all[TSR] = arr[:,7]

fig, axes = plt.subplots(1,2,figsize=(12,5))
for TSR in TSR_SWEEP:
    axes[0].plot(r_mid8, alpha_all[TSR], label=rf"$\lambda={TSR:.0f}$")
    axes[1].plot(r_mid8, phi_all[TSR],   label=rf"$\lambda={TSR:.0f}$")
for ax, ylabel, title in zip(axes,
        [r"$\alpha$ [deg]", r"$\phi$ [deg]"],
        ["Angle of attack", "Inflow angle"]):
    ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True); ax.legend()
fig.suptitle("Spanwise distribution of angle of attack and inflow angle")
fig.tight_layout()
save_and_show("4_1_alpha_phi_vs_rR.png")

# ── 4.2  Axial and azimuthal induction ───────────────────────────────────────
a_all = {}; ap_all = {}
for TSR in TSR_SWEEP:
    Omega = U0*TSR/Radius; tmp = []
    for i in range(len(r_R_bins)-1):
        rm    = (r_R_bins[i]+r_R_bins[i+1])/2
        chord = 3*(1-rm)+1; twist = -(14*(1-rm)+Pitch)
        res,_ = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1],
                                  RootLocation_R, TipLocation_R,
                                  Omega, Radius, NBlades, chord, twist)
        tmp.append(res)
    arr = np.array(tmp)
    a_all[TSR] = arr[:,0]; ap_all[TSR] = arr[:,1]

fig, axes = plt.subplots(1,2,figsize=(12,5))
for TSR in TSR_SWEEP:
    axes[0].plot(r_mid8, a_all[TSR],  label=rf"$\lambda={TSR:.0f}$")
    axes[1].plot(r_mid8, ap_all[TSR], label=rf"$\lambda={TSR:.0f}$")
for ax, ylabel, title in zip(axes,
        [r"$a$ [-]", r"$a'$ [-]"],
        ["Axial induction", "Tangential induction"]):
    ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True); ax.legend()
fig.suptitle("Spanwise distribution of induction factors")
fig.tight_layout()
save_and_show("4_2_induction_vs_rR.png")

# ── 4.3  Thrust and azimuthal loading ────────────────────────────────────────
fn_all = {}; ft_all = {}
for TSR in TSR_SWEEP:
    Omega = U0*TSR/Radius; tmp = []
    for i in range(len(r_R_bins)-1):
        rm    = (r_R_bins[i]+r_R_bins[i+1])/2
        chord = 3*(1-rm)+1; twist = -(14*(1-rm)+Pitch)
        res,_ = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1],
                                  RootLocation_R, TipLocation_R,
                                  Omega, Radius, NBlades, chord, twist)
        tmp.append(res)
    arr = np.array(tmp)
    fn_all[TSR] = arr[:,3]/norm_val; ft_all[TSR] = arr[:,4]/norm_val

fig, axes = plt.subplots(1,2,figsize=(12,5))
for TSR in TSR_SWEEP:
    axes[0].plot(r_mid8, fn_all[TSR], label=rf"$\lambda={TSR:.0f}$")
    axes[1].plot(r_mid8, ft_all[TSR], label=rf"$\lambda={TSR:.0f}$")
for ax, ylabel, title in zip(axes,
        [r"$C_n = F_{norm}/(½U_\infty^2 R)$", r"$C_t = F_{tan}/(½U_\infty^2 R)$"],
        ["Normal (thrust) loading", "Tangential (azimuthal) loading"]):
    ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True); ax.legend()
fig.suptitle("Spanwise loading distribution")
fig.tight_layout()
save_and_show("4_3_loading_vs_rR.png")

# ── 4.4  CT and CQ vs TSR ────────────────────────────────────────────────────
tsr_list = sorted(tsr_perf.keys())
CT_list  = [tsr_perf[t]["CT"] for t in tsr_list]
CP_list  = [tsr_perf[t]["CP"] for t in tsr_list]
CQ_list  = [tsr_perf[t]["CP"]/t for t in tsr_list]

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(tsr_list, CT_list, "bo-", label=r"$C_T$")
axes[0].set_xlabel(r"Tip-speed ratio $\lambda$"); axes[0].set_ylabel(r"$C_T$")
axes[0].set_title("Thrust coefficient vs TSR"); axes[0].grid(True); axes[0].legend()
axes[1].plot(tsr_list, CQ_list, "ro-", label=r"$C_Q$")
axes[1].set_xlabel(r"Tip-speed ratio $\lambda$"); axes[1].set_ylabel(r"$C_Q$")
axes[1].set_title("Torque coefficient vs TSR"); axes[1].grid(True); axes[1].legend()
fig.tight_layout()
save_and_show("4_4_CT_CQ_vs_TSR.png")

# ── 5  Tip correction influence ───────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(r_mid8, results_tsr8[:,0], "b-", label="With correction")
axes[0].plot(res_nc[:,2], res_nc[:,0],  "r-", label="No correction")
axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$a$ [-]")
axes[0].set_title("Axial induction"); axes[0].grid(True); axes[0].legend()

axes[1].plot(r_mid8, results_tsr8[:,3]/norm_val, "b-", label="With correction")
axes[1].plot(res_nc[:,2], res_nc[:,3]/norm_val,   "r-", label="No correction")
axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_n$ [-]")
axes[1].set_title("Normal loading"); axes[1].grid(True); axes[1].legend()
fig.suptitle("Influence of Prandtl tip/root correction (TSR=8)")
fig.tight_layout()
save_and_show("5_tip_correction_influence.png")

# ── 6a  Number of annuli ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5))
for N in [8, 20, 100]:
    bins_N = np.linspace(RootLocation_R, TipLocation_R, N+1); tmp = []
    for i in range(N):
        rm    = (bins_N[i]+bins_N[i+1])/2
        chord = 3*(1-rm)+1; twist = -(14*(1-rm)+Pitch)
        res,_ = SolveStreamtube(U0, bins_N[i], bins_N[i+1],
                                  RootLocation_R, TipLocation_R,
                                  Omega8, Radius, NBlades, chord, twist)
        tmp.append(res)
    arr = np.array(tmp); rm_arr = arr[:,2]
    ax.plot(rm_arr, arr[:,3]/norm_val, "-o", markersize=4, label=f"N={N}")
ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
ax.set_title("Normal loading — influence of number of annuli (TSR=8)")
ax.grid(True); ax.legend()
save_and_show("6a_annuli_influence.png")

# ── 6b  Spacing method ────────────────────────────────────────────────────────
N_sp  = 40
beta_cos = np.linspace(0,np.pi,N_sp+1)
spacings = {
    "Constant": np.linspace(RootLocation_R, TipLocation_R, N_sp+1),
    "Cosine":   RootLocation_R+(TipLocation_R-RootLocation_R)*0.5*(1-np.cos(beta_cos))
}
fig, ax = plt.subplots(figsize=(9,5))
for label, bins in spacings.items():
    tmp = []
    for i in range(N_sp):
        rm    = (bins[i]+bins[i+1])/2
        chord = 3*(1-rm)+1; twist = -(14*(1-rm)+Pitch)
        res,_ = SolveStreamtube(U0, bins[i], bins[i+1],
                                  RootLocation_R, TipLocation_R,
                                  Omega8, Radius, NBlades, chord, twist)
        tmp.append(res)
    arr = np.array(tmp)
    ax.plot(arr[:,2], arr[:,3]/norm_val, "-o", markersize=5, label=f"{label} spacing")
ax.set_xlim(0.85,1.01); ax.set_ylim(0.5,1.5)
ax.set_xlabel("r/R"); ax.set_ylabel(r"$C_n$ [-]")
ax.set_title("Spacing method comparison near tip (N=40, TSR=8)")
ax.grid(True); ax.legend()
save_and_show("6b_spacing_method.png")

# ── 6c  Convergence history ───────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
iters = range(1, len(ct_hist_tsr8)+1)
axes[0].plot(iters, ct_hist_tsr8, "b-", linewidth=2)
axes[0].set_xlim(1,min(60,len(ct_hist_tsr8)))
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel(r"$C_T$ [-]")
axes[0].set_title(r"CT convergence history (TSR=8)"); axes[0].grid(True)

residuals = np.abs(np.diff(ct_hist_tsr8))
axes[1].semilogy(range(2,len(ct_hist_tsr8)+1), residuals, "r-", linewidth=2,
                 label=r"$|C_{T,i}-C_{T,i-1}|$")
axes[1].axhline(1e-5, color="k", linestyle="--", linewidth=0.8, label="tol=1e-5")
axes[1].set_xlim(1,min(60,len(ct_hist_tsr8)))
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel(r"$|\Delta C_T|$")
axes[1].set_title("Convergence residuals (log scale)"); axes[1].grid(True,which="both"); axes[1].legend()
fig.tight_layout()
save_and_show("6c_convergence_history.png")

# ── 7  Stagnation pressure ────────────────────────────────────────────────────
# Four stations: (1) far upstream, (2) rotor upwind, (3) rotor downwind, (4) far downstream
# P0_1 = P0_2 (stagnation pressure preserved upstream of rotor)
# P0_3 = P0_4 (stagnation pressure preserved downstream of rotor)
# Pressure drop: ΔP0 = 2·ρ·U0²·a·F·(1 − a·F)

P_static = 0.0
P01 = P_static + 0.5*rho*U0**2   # far upstream stagnation pressure [Pa]
P01_arr = P01 * np.ones(len(r_mid8))

dP0_arr = (2*rho*U0**2
           * results_tsr8[:,0]      # a
           * F_tsr8                  # F (Prandtl)
           * (1 - results_tsr8[:,0]*F_tsr8))

P03_arr = P01_arr - dP0_arr        # downwind stagnation pressure

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(r_mid8, P01_arr, "b-",  linewidth=2, label=r"$P_0^\infty$ (far upstream)")
ax.plot(r_mid8, P01_arr, "b--", linewidth=2, label=r"$P_0^+$ (rotor upwind face)",  alpha=0.5)
ax.plot(r_mid8, P03_arr, "r--", linewidth=2, label=r"$P_0^-$ (rotor downwind face)", alpha=0.5)
ax.plot(r_mid8, P03_arr, "r-",  linewidth=2, label=r"$P_0^\infty$ (far downstream)")
ax.set_xlabel("r/R")
ax.set_ylabel(r"Stagnation pressure $P_0$ [Pa]")
ax.set_title("Stagnation pressure at four streamwise stations (TSR=8)")
ax.legend(); ax.grid(True)
save_and_show("7_stagnation_pressure.png")

# ── 8  All designs — chord, twist, CP/CT comparison ──────────────────────────
r_R_dense = np.linspace(RootLocation_R, TipLocation_R, 400)

# Chord curves
c_base_d  = 3*(1-r_R_dense)+1
p_c = p_cubic;  p_q = p_qrt
if len(p_c)==7: _,_,_,_, c_tip_c,c2_c,c3_c = p_c; c4_c=0.0
else:            _,_,_,_, c_tip_c,c2_c,c3_c,c4_c = p_c
if len(p_q)==7: _,_,_,_, c_tip_q,c2_q,c3_q = p_q; c4_q=0.0
else:            _,_,_,_, c_tip_q,c2_q,c3_q,c4_q = p_q

c_cubic_d  = np.array([chord_poly(r, c_tip_c,c2_c,c3_c,c4_c) for r in r_R_dense])
c_qrt_d    = np.array([chord_poly(r, c_tip_q,c2_q,c3_q,c4_q) for r in r_R_dense])
c_anal_d   = np.interp(r_R_dense, r_anal/Radius, c_anal)

# Twist curves
if len(p_c)==7: pitch_c,tr_c,tt_c,tc_c,_,_,_ = p_c
else:            pitch_c,tr_c,tt_c,tc_c,_,_,_,_ = p_c
if len(p_q)==7: pitch_q,tr_q,tt_q,tc_q,_,_,_ = p_q
else:            pitch_q,tr_q,tt_q,tc_q,_,_,_,_ = p_q

tw_base_d  = -(14*(1-r_R_dense)+Pitch)
tw_cubic_d = np.array([twist_poly(r,pitch_c,tr_c,tt_c,tc_c) for r in r_R_dense])
tw_qrt_d   = np.array([twist_poly(r,pitch_q,tr_q,tt_q,tc_q) for r in r_R_dense])
tw_anal_d  = np.interp(r_R_dense, r_anal/Radius, tw_anal)

fig, axes = plt.subplots(1,2,figsize=(14,5))
for c_d, lbl in [(c_base_d,"Baseline"), (c_anal_d,"Analytical"),
                  (c_cubic_d,"Cubic poly"), (c_qrt_d,"Quartic poly")]:
    axes[0].plot(r_R_dense, c_d, label=lbl)
axes[0].axhline(CHORD_MIN, color="grey", linestyle=":", label="Min chord")
axes[0].scatter([RootLocation_R],[CHORD_ROOT], zorder=5, color="k", label="Root pin")
axes[0].set_xlabel("r/R"); axes[0].set_ylabel("Chord [m]")
axes[0].set_title("Chord distribution — all designs"); axes[0].legend(); axes[0].grid(True)

for tw_d, lbl in [(tw_base_d,"Baseline"), (tw_anal_d,"Analytical"),
                   (tw_cubic_d,"Cubic poly"), (tw_qrt_d,"Quartic poly")]:
    axes[1].plot(r_R_dense, tw_d, label=lbl)
axes[1].set_xlabel("r/R"); axes[1].set_ylabel("Twist [deg]")
axes[1].set_title("Twist distribution — all designs"); axes[1].legend(); axes[1].grid(True)
fig.tight_layout()
save_and_show("8a_chord_twist_all_designs.png")

# Performance bar chart
fig, ax = plt.subplots(figsize=(8,5))
labels   = ["Baseline", "Analytical", "Cubic poly", "Quartic poly", "Actuator disk"]
cp_vals  = [CP_base, CP_anal, CP_cubic, CP_qrt, CP_ad]
ct_vals  = [CT_base, CT_anal, CT_cubic, CT_qrt, CT_TARGET]
x = np.arange(len(labels)); w = 0.35
b1 = ax.bar(x-w/2, cp_vals, w, label="CP", color="steelblue")
b2 = ax.bar(x+w/2, ct_vals, w, label="CT", color="coral")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Coefficient [-]"); ax.set_title("Performance comparison — all designs")
ax.legend(); ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=8)
ax.grid(True, axis="y"); fig.tight_layout()
save_and_show("8b_performance_comparison.png")

# Spanwise a and loading for all optimised designs
fig, axes = plt.subplots(1,2,figsize=(14,5))
for res, lbl in [(res_base,"Baseline"), (res_anal,"Analytical"),
                  (res_cubic,"Cubic poly"), (res_qrt,"Quartic poly")]:
    axes[0].plot(res[:,2], res[:,0], label=lbl)
    axes[1].plot(res[:,2], res[:,3]/norm_val, label=lbl)
axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$a$ [-]")
axes[0].set_title("Axial induction — all designs"); axes[0].legend(); axes[0].grid(True)
axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_n$ [-]")
axes[1].set_title("Normal loading — all designs"); axes[1].legend(); axes[1].grid(True)
fig.tight_layout()
save_and_show("8c_induction_loading_all_designs.png")

# ── 9  Cl distribution and chord relation (analytical optimum) ────────────────
# cl stored in column 8 of res_arr, chord recomputed from geometry
r_R_anal_mid = res_anal[:,2]   # r_mid values
cl_anal      = res_anal[:,8]   # lift coefficient at each annulus
# Chord at each midpoint (interp from nodes)
c_anal_mid   = np.interp(r_R_anal_mid, r_anal/Radius, c_anal)

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(r_R_anal_mid, cl_anal, "b-", linewidth=2)
axes[0].set_xlabel("r/R"); axes[0].set_ylabel(r"$C_l$ [-]")
axes[0].set_title("Lift coefficient distribution (analytical optimum)")
axes[0].grid(True)

ax2 = axes[0].twinx()
ax2.plot(r_R_anal_mid, c_anal_mid, "r--", linewidth=2, label="Chord")
ax2.set_ylabel("Chord [m]", color="red"); ax2.tick_params(axis="y", labelcolor="red")
axes[0].set_zorder(ax2.get_zorder()+1); axes[0].patch.set_visible(False)

# Cl·c product — proportional to circulation Γ
circ = cl_anal * c_anal_mid
axes[1].plot(r_R_anal_mid, circ, "g-", linewidth=2)
axes[1].set_xlabel("r/R"); axes[1].set_ylabel(r"$C_l \cdot c$ [m]")
axes[1].set_title(r"Circulation proxy $C_l \cdot c$ (analytical optimum)")
axes[1].grid(True)
fig.tight_layout()
save_and_show("9_cl_chord_relation.png")

# ── 10  Cl/Cd polar with operating points ─────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5))

# Polar
axes[0].plot(polar_cd, polar_cl, "k-", linewidth=1.5, label="Polar")
# Operating points for analytical optimum
cd_anal = res_anal[:,9]
axes[0].scatter(cd_anal, cl_anal, c=r_R_anal_mid, cmap="viridis",
                s=30, zorder=5, label="Analytical opt (coloured by r/R)")
sm = plt.cm.ScalarMappable(cmap="viridis",
     norm=plt.Normalize(r_R_anal_mid.min(), r_R_anal_mid.max()))
plt.colorbar(sm, ax=axes[0], label="r/R")
axes[0].set_xlabel(r"$C_d$"); axes[0].set_ylabel(r"$C_l$")
axes[0].set_title("Cl-Cd polar with operating points"); axes[0].grid(True); axes[0].legend()

# Cl/Cd ratio vs alpha
alphas_dense = np.linspace(polar_alpha[0], polar_alpha[-1], 500)
cl_dense     = np.interp(alphas_dense, polar_alpha, polar_cl)
cd_dense     = np.interp(alphas_dense, polar_alpha, polar_cd)
ld_dense     = cl_dense / np.maximum(cd_dense, 1e-8)
alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()

axes[1].plot(alphas_dense, ld_dense, "k-", linewidth=1.5, label=r"$C_l/C_d$")
axes[1].axvline(alpha_opt_deg, color="r", linestyle="--",
                label=rf"$\alpha_{{opt}}={alpha_opt_deg:.1f}°$  "
                      rf"$(C_l/C_d)_{{max}}={cl_opt/cd_opt:.0f}$")
# Scatter operating alphas of analytical optimum
alpha_anal = res_anal[:,6]
axes[1].scatter(alpha_anal, np.interp(alpha_anal, polar_alpha, polar_cl)
                             / np.maximum(np.interp(alpha_anal, polar_alpha, polar_cd),1e-8),
                c=r_R_anal_mid, cmap="viridis", s=30, zorder=5)
axes[1].set_xlabel(r"$\alpha$ [deg]"); axes[1].set_ylabel(r"$C_l/C_d$")
axes[1].set_title(r"$C_l/C_d$ vs $\alpha$ — operating points"); axes[1].grid(True); axes[1].legend()
fig.tight_layout()
save_and_show("10_polar_operating_points.png")

# =============================================================================
# 11.  DONE
# =============================================================================

print("\n" + "="*65)
print("ALL PLOTS SAVED TO:", save_folder)
print("="*65)