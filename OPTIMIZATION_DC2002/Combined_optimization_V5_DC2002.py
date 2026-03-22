import os
import io
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize

# =============================================================================
# LOAD POLAR DATA
# =============================================================================

data = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = data["Alfa"].to_numpy()
polar_cl = data["Cl"].to_numpy()
polar_cd = data["Cd"].to_numpy()

# =============================================================================
# SPECIFICATIONS
# =============================================================================

Pitch = -2.0                  # baseline pitch angle [deg]
delta_r_R = 0.005             # final spanwise resolution

Radius = 50.0                 # [m]
NBlades = 3
U0 = 10.0                     # freestream [m/s]

RootLocation_R = 0.2
TipLocation_R = 1.0

r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)

CT_TARGET = 0.75
TSR_DESIGN = 8.0

ROOT_CHORD = 3.4              # fixed root chord at r/R = 0.2 [m]
MIN_CHORD = 0.3               # minimum chord anywhere / at tip [m]

# optimization resolution (coarser for speed)
delta_r_R_opt = 0.01

# =============================================================================
# INDUCTION AND CORRECTION FUNCTIONS
# =============================================================================

def ainduction(CT):
    """
    Calculate the induction factor a as a function of the thrust coefficient CT.
    Glauert's correction is applied for heavily loaded rotors.
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
        a[CT < CT2] = 0.5 - 0.5 * np.sqrt(np.maximum(0, 1 - CT[CT < CT2]))
        return a


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Calculate the combined tip and root Prandtl correction factor F.
    """
    eps = 1e-10
    r_R = max(r_R, eps)
    denom = max((1 - axial_induction), 1e-8)

    temp_tip = -NBlades / 2 * (tipradius_R - r_R) / r_R * np.sqrt(
        1 + ((TSR * r_R) ** 2) / (denom ** 2)
    )
    Ftip = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0, 1)))

    temp_root = NBlades / 2 * (rootradius_R - r_R) / r_R * np.sqrt(
        1 + ((TSR * r_R) ** 2) / (denom ** 2)
    )
    Froot = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0, 1)))

    return Froot * Ftip


def LoadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Calculate the local forces (normal and tangential) and circulation on a blade element.
    twist in degrees.
    """
    vmag2 = vnorm**2 + vtan**2
    phi = np.arctan2(vnorm, vtan)

    alpha = twist + np.degrees(phi)
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord

    fnorm = lift * np.cos(phi) + drag * np.sin(phi)
    ftan = lift * np.sin(phi) - drag * np.cos(phi)

    gamma = 0.5 * np.sqrt(vmag2) * cl * chord

    return fnorm, ftan, gamma, alpha, np.degrees(phi)


def SolveStreamtube(U0, r1_R, r2_R, rootradius_R, tipradius_R, Omega, Radius,
                    NBlades, chord, twist, polar_alpha, polar_cl, polar_cd,
                    n_iter=200):
    """
    Solve the momentum balance for a single streamtube using BEM method.
    """
    Area = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid = (r1_R + r2_R) / 2
    r_local = r_mid * Radius

    a, aline = 0.1, 0.0
    fnorm_history = np.zeros(n_iter)

    for i in range(n_iter):
        Urotor = U0 * (1 - a)
        Utan = (1 + aline) * Omega * r_local

        fnorm, ftan, gamma, alpha, phi = LoadBladeElement(
            Urotor, Utan, chord, twist, polar_alpha, polar_cl, polar_cd
        )
        fnorm_history[i] = fnorm

        CT = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0 ** 2)

        anew = ainduction(CT)
        F = max(
            PrandtlTipRootCorrection(
                r_mid, rootradius_R, tipradius_R, Omega * Radius / U0, NBlades, anew
            ),
            0.0001,
        )
        anew /= F

        a = 0.75 * a + 0.25 * anew

        denom = 2 * np.pi * U0 * max((1 - a), 1e-8) * Omega * 2 * r_local ** 2
        aline = (ftan * NBlades) / denom
        aline /= F

    return [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi], fnorm_history


# =============================================================================
# GENERAL ROTOR EVALUATION USING THE BASE BEM SOLVER
# =============================================================================

def evaluate_rotor_from_functions(chord_fun, twist_fun, TSR=TSR_DESIGN,
                                  delta_r=delta_r_R, n_iter=200,
                                  return_histories=False):
    """
    Evaluate a rotor defined by chord_fun(r/R) and twist_fun(r/R) [deg]
    using the base BEM solver.
    """
    bins = np.arange(RootLocation_R, TipLocation_R + delta_r / 2, delta_r)
    Omega = U0 * TSR / Radius

    results = []
    histories = []

    for i in range(len(bins) - 1):
        r1 = bins[i]
        r2 = bins[i + 1]
        rm = 0.5 * (r1 + r2)

        chord = chord_fun(rm)
        twist = twist_fun(rm)

        if chord < MIN_CHORD:
            return {
                "CT": -1e9,
                "CP": -1e9,
                "results": None,
                "histories": None,
                "feasible": False,
                "bins": bins,
            }

        res, f_hist = SolveStreamtube(
            U0, r1, r2, RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades,
            chord, twist, polar_alpha, polar_cl, polar_cd,
            n_iter=n_iter
        )
        results.append(res)
        histories.append(f_hist)

    res_arr = np.array(results)
    dr = (bins[1:] - bins[:-1]) * Radius

    CT = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0 ** 2 * np.pi * Radius ** 2))
    CP = np.sum(dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega /
                (0.5 * U0 ** 3 * np.pi * Radius ** 2))

    out = {
        "CT": CT,
        "CP": CP,
        "results": res_arr,
        "histories": histories if return_histories else None,
        "feasible": True,
        "bins": bins,
    }
    return out


def evaluate_rotor_from_nodes(r_nodes, c_nodes, beta_nodes_deg, TSR=TSR_DESIGN,
                              n_iter=200, return_histories=False):
    """
    Evaluate rotor from nodal geometry by midpoint interpolation onto annuli.
    beta in degrees.
    """
    r_mid_nodes = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    c_mid_nodes = 0.5 * (c_nodes[:-1] + c_nodes[1:])
    beta_mid_nodes = 0.5 * (beta_nodes_deg[:-1] + beta_nodes_deg[1:])

    def chord_fun(r_R):
        r_abs = r_R * Radius
        return np.interp(r_abs, r_mid_nodes, c_mid_nodes)

    def twist_fun(r_R):
        r_abs = r_R * Radius
        return np.interp(r_abs, r_mid_nodes, beta_mid_nodes)

    return evaluate_rotor_from_functions(
        chord_fun, twist_fun, TSR=TSR, delta_r=delta_r_R,
        n_iter=n_iter, return_histories=return_histories
    )


# =============================================================================
# BASELINE GEOMETRY
# =============================================================================

def baseline_chord(r_R):
    return 3.0 * (1.0 - r_R) + 1.0


def baseline_twist(r_R, pitch_deg=Pitch):
    return -(14.0 * (1.0 - r_R) + pitch_deg)


def evaluate_baseline(TSR=TSR_DESIGN, pitch_deg=Pitch, delta_r=delta_r_R,
                      n_iter=200, return_histories=False):
    return evaluate_rotor_from_functions(
        lambda r: baseline_chord(r),
        lambda r: baseline_twist(r, pitch_deg=pitch_deg),
        TSR=TSR,
        delta_r=delta_r,
        n_iter=n_iter,
        return_histories=return_histories
    )


# =============================================================================
# METHOD 1: ANALYTICAL / INVERSE-DESIGN METHOD
# =============================================================================

def find_optimal_alpha(alpha_min_deg=-5.0, alpha_max_deg=20.0, npts=800):
    """
    Find alpha with maximum Cl/Cd.
    """
    alphas = np.linspace(alpha_min_deg, alpha_max_deg, npts)
    cl_vals = np.interp(alphas, polar_alpha, polar_cl)
    cd_vals = np.interp(alphas, polar_alpha, polar_cd)
    ld_vals = cl_vals / np.maximum(cd_vals, 1e-8)

    idx = int(np.argmax(ld_vals))
    alpha_opt = float(alphas[idx])
    cl_opt = float(cl_vals[idx])
    cd_opt = float(cd_vals[idx])

    return alpha_opt, cl_opt, cd_opt


def generate_analytical_rotor(target_a=0.25, n_nodes=101):
    """
    Generate rotor geometry from an analytical inverse-design idea:
    - choose alpha_opt from max Cl/Cd
    - prescribe design induction a
    - derive phi from local TSR
    - derive chord from inverse design expression
    - enforce practical chord constraints
    """
    alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha()

    r_start = RootLocation_R * Radius + 0.005 * Radius
    r_end = TipLocation_R * Radius - 0.005 * Radius
    r_nodes = np.linspace(r_start, r_end, n_nodes)

    c_nodes = np.zeros_like(r_nodes)
    beta_nodes_deg = np.zeros_like(r_nodes)

    a = target_a
    TSR = TSR_DESIGN

    for i, r in enumerate(r_nodes):
        r_R = r / Radius
        local_tsr = TSR * r_R

        F = max(
            PrandtlTipRootCorrection(r_R, RootLocation_R, TipLocation_R, TSR, NBlades, a),
            1e-4
        )

        phi = np.arctan2((1 - a), local_tsr)
        phi_deg = np.degrees(phi)

        beta_nodes_deg[i] = alpha_opt_deg - phi_deg

        Cn = cl_opt * np.cos(phi) + cd_opt * np.sin(phi)

        c_nodes[i] = (
            8 * np.pi * r * a * F * (1 - a * F) * np.sin(phi) ** 2
        ) / (
            NBlades * (1 - a) ** 2 * max(Cn, 1e-8)
        )

    # enforce constraints
    c_nodes = np.minimum(c_nodes, ROOT_CHORD)
    c_nodes = np.maximum(c_nodes, MIN_CHORD)
    c_nodes[0] = ROOT_CHORD
    c_nodes[-1] = max(c_nodes[-1], MIN_CHORD)

    return r_nodes, c_nodes, beta_nodes_deg


def design_analytical_for_exact_ct():
    """
    Solve for the design induction a such that the final BEM CT = 0.75.
    """
    print("\n--- Solving analytical method for CT = 0.75 ---")

    def residual(a):
        r_nodes, c_nodes, beta_nodes_deg = generate_analytical_rotor(target_a=a, n_nodes=101)
        perf = evaluate_rotor_from_nodes(
            r_nodes, c_nodes, beta_nodes_deg,
            TSR=TSR_DESIGN, n_iter=200, return_histories=False
        )
        ct_val = perf["CT"]
        cp_val = perf["CP"]
        print(f"  tested a = {a:.5f} -> CT = {ct_val:.5f}, CP = {cp_val:.5f}")
        return ct_val - CT_TARGET

    res = root_scalar(residual, bracket=[0.20, 0.35], method="brentq")

    if not res.converged:
        raise RuntimeError("Could not find analytical design induction for CT = 0.75.")

    best_a = float(res.root)
    r_nodes, c_nodes, beta_nodes_deg = generate_analytical_rotor(target_a=best_a, n_nodes=101)
    perf = evaluate_rotor_from_nodes(
        r_nodes, c_nodes, beta_nodes_deg,
        TSR=TSR_DESIGN, n_iter=200, return_histories=True
    )

    print(f"Analytical method converged to design a = {best_a:.6f}")
    print(f"Final analytical CT = {perf['CT']:.6f}")
    print(f"Final analytical CP = {perf['CP']:.6f}")

    return r_nodes, c_nodes, beta_nodes_deg, best_a, perf


# =============================================================================
# METHOD 2: POLYNOMIAL OPTIMIZATION METHOD
# =============================================================================

def x_from_rR(r_R):
    return (r_R - RootLocation_R) / (TipLocation_R - RootLocation_R)


def chord_poly(r_R, c_tip, c2, c3):
    """
    Cubic chord distribution with fixed root chord:
        c(x) = ROOT_CHORD + b1*x + c2*x^2 + c3*x^3
    and c(1) = c_tip.
    """
    x = x_from_rR(np.asarray(r_R))
    b1 = c_tip - ROOT_CHORD - c2 - c3
    return ROOT_CHORD + b1 * x + c2 * x**2 + c3 * x**3


def twist_poly(r_R, pitch_deg, t_root_deg, t_tip_deg, t_curve_deg):
    """
    Smooth twist / blade angle polynomial in degrees:
        beta(x) = pitch + t_root*(1-x) + t_tip*x + t_curve*x*(1-x)
    """
    x = x_from_rR(np.asarray(r_R))
    return pitch_deg + t_root_deg * (1 - x) + t_tip_deg * x + t_curve_deg * x * (1 - x)


def build_poly_functions(params):
    """
    params = [pitch_deg, t_root_deg, t_tip_deg, t_curve_deg, c_tip, c2, c3]
    """
    pitch_deg, t_root_deg, t_tip_deg, t_curve_deg, c_tip, c2, c3 = params

    def cfun(r_R):
        return chord_poly(r_R, c_tip, c2, c3)

    def tfun(r_R):
        return twist_poly(r_R, pitch_deg, t_root_deg, t_tip_deg, t_curve_deg)

    return cfun, tfun


def polynomial_penalty(params):
    """
    Geometry penalty.
    """
    _, _, _, _, c_tip, c2, c3 = params

    r_dense = np.linspace(RootLocation_R, TipLocation_R, 300)
    c_dense = chord_poly(r_dense, c_tip, c2, c3)

    penalty = 0.0

    min_c = np.min(c_dense)
    max_c = np.max(c_dense)

    if min_c < MIN_CHORD:
        penalty += 1e6 * (MIN_CHORD - min_c) ** 2

    if max_c > ROOT_CHORD:
        penalty += 1e6 * (max_c - ROOT_CHORD) ** 2

    penalty += 0.03 * params[3] ** 2
    penalty += 0.01 * params[5] ** 2
    penalty += 0.01 * params[6] ** 2

    return penalty


def objective_polynomial(params):
    """
    Maximize CP at TSR=8 while enforcing CT=0.75 via penalty.
    """
    penalty = polynomial_penalty(params)

    if penalty > 1e5:
        return penalty

    cfun, tfun = build_poly_functions(params)

    perf = evaluate_rotor_from_functions(
        cfun, tfun,
        TSR=TSR_DESIGN,
        delta_r=delta_r_R_opt,
        n_iter=120,
        return_histories=False
    )

    if not perf["feasible"]:
        return 1e9

    CT = perf["CT"]
    CP = perf["CP"]

    penalty += 5e4 * (CT - CT_TARGET) ** 2

    return -CP + penalty


def optimize_polynomial_rotor():
    """
    Optimize:
    params = [pitch_deg, t_root_deg, t_tip_deg, t_curve_deg, c_tip, c2, c3]
    """
    print("\n--- Running polynomial optimization method ---")

    bounds = [
        (-8.0, 8.0),      # pitch_deg
        (-20.0, 15.0),    # t_root_deg
        (-15.0, 15.0),    # t_tip_deg
        (-20.0, 20.0),    # t_curve_deg
        (MIN_CHORD, ROOT_CHORD),  # c_tip
        (-5.0, 5.0),      # c2
        (-5.0, 5.0),      # c3
    ]

    # use one robust nominal start; can add more later if needed
    starts = [
        np.array([
            -2.0,   # pitch
            11.2,   # root shaping
            -2.0,   # tip shaping
            0.0,    # curvature
            1.0,    # c_tip
            0.0,    # c2
            0.0,    # c3
        ])
    ]

    best_res = None
    best_fun = np.inf

    for k, x0 in enumerate(starts, start=1):
        print(f"  start {k}/{len(starts)}")
        res = minimize(
            objective_polynomial,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 220}
        )
        print(f"    final objective = {res.fun:.6f}")

        if res.fun < best_fun:
            best_fun = res.fun
            best_res = res

    if best_res is None:
        raise RuntimeError("Polynomial optimization failed.")

    best_params = best_res.x
    cfun, tfun = build_poly_functions(best_params)

    perf = evaluate_rotor_from_functions(
        cfun, tfun,
        TSR=TSR_DESIGN,
        delta_r=delta_r_R,
        n_iter=200,
        return_histories=True
    )

    print("\nBest polynomial parameters:")
    print(f"  pitch_deg = {best_params[0]:.6f}")
    print(f"  t_root_deg = {best_params[1]:.6f}")
    print(f"  t_tip_deg = {best_params[2]:.6f}")
    print(f"  t_curve_deg = {best_params[3]:.6f}")
    print(f"  c_tip = {best_params[4]:.6f}")
    print(f"  c2 = {best_params[5]:.6f}")
    print(f"  c3 = {best_params[6]:.6f}")
    print(f"Final polynomial CT = {perf['CT']:.6f}")
    print(f"Final polynomial CP = {perf['CP']:.6f}")

    return best_params, perf


# =============================================================================
# ACTUATOR DISK THEORY
# =============================================================================

def actuator_disk_cp_from_ct(CT):
    """
    Ideal actuator disk in axial flow:
        CT = 4 a (1-a)
        CP = 4 a (1-a)^2
    """
    if CT < 0.0 or CT > 1.0:
        return np.nan, np.nan

    a = 0.5 * (1.0 - np.sqrt(1.0 - CT))
    CP = 4.0 * a * (1.0 - a) ** 2
    return a, CP


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

base_path = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_BEM")
os.makedirs(save_folder, exist_ok=True)


def save_and_show(filename):
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


def extract_perf_arrays(perf):
    arr = perf["results"]
    return {
        "a": arr[:, 0],
        "aline": arr[:, 1],
        "r_mid": arr[:, 2],
        "fnorm": arr[:, 3],
        "ftan": arr[:, 4],
        "gamma": arr[:, 5],
        "alpha": arr[:, 6],
        "phi": arr[:, 7],
    }


def plot_geometry_comparison(rR_dense, c_baseline, beta_baseline,
                             c_analytical, beta_analytical,
                             c_poly, beta_poly):
    plt.figure(figsize=(8, 5))
    plt.plot(rR_dense, c_baseline, label="Baseline")
    plt.plot(rR_dense, c_analytical, label="Analytical method")
    plt.plot(rR_dense, c_poly, label="Polynomial method")
    plt.axhline(MIN_CHORD, linestyle="--", label="Minimum chord")
    plt.xlabel("r/R")
    plt.ylabel("Chord [m]")
    plt.title("Chord distribution comparison")
    plt.grid(True)
    plt.legend()
    save_and_show("comparison_chord_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.plot(rR_dense, beta_baseline, label="Baseline")
    plt.plot(rR_dense, beta_analytical, label="Analytical method")
    plt.plot(rR_dense, beta_poly, label="Polynomial method")
    plt.xlabel("r/R")
    plt.ylabel("Blade angle / twist [deg]")
    plt.title("Blade angle comparison")
    plt.grid(True)
    plt.legend()
    save_and_show("comparison_twist_distribution.png")


def plot_loading_comparison(baseline_perf, analytical_perf, poly_perf):
    b = extract_perf_arrays(baseline_perf)
    a = extract_perf_arrays(analytical_perf)
    p = extract_perf_arrays(poly_perf)

    plt.figure(figsize=(8, 5))
    plt.plot(b["r_mid"], b["fnorm"], label="Baseline fnorm")
    plt.plot(a["r_mid"], a["fnorm"], label="Analytical fnorm")
    plt.plot(p["r_mid"], p["fnorm"], label="Polynomial fnorm")
    plt.xlabel("r/R")
    plt.ylabel("Normal load [N/m]")
    plt.title("Normal loading comparison")
    plt.grid(True)
    plt.legend()
    save_and_show("comparison_normal_loading.png")

    plt.figure(figsize=(8, 5))
    plt.plot(b["r_mid"], b["ftan"], label="Baseline ftan")
    plt.plot(a["r_mid"], a["ftan"], label="Analytical ftan")
    plt.plot(p["r_mid"], p["ftan"], label="Polynomial ftan")
    plt.xlabel("r/R")
    plt.ylabel("Tangential load [N/m]")
    plt.title("Tangential loading comparison")
    plt.grid(True)
    plt.legend()
    save_and_show("comparison_tangential_loading.png")


def plot_induction_comparison(baseline_perf, analytical_perf, poly_perf):
    b = extract_perf_arrays(baseline_perf)
    a = extract_perf_arrays(analytical_perf)
    p = extract_perf_arrays(poly_perf)

    plt.figure(figsize=(8, 5))
    plt.plot(b["r_mid"], b["a"], label="Baseline a")
    plt.plot(a["r_mid"], a["a"], label="Analytical a")
    plt.plot(p["r_mid"], p["a"], label="Polynomial a")
    plt.xlabel("r/R")
    plt.ylabel("Axial induction [-]")
    plt.title("Axial induction comparison")
    plt.grid(True)
    plt.legend()
    save_and_show("comparison_axial_induction.png")

    plt.figure(figsize=(8, 5))
    plt.plot(b["r_mid"], b["aline"], label="Baseline a'")
    plt.plot(a["r_mid"], a["aline"], label="Analytical a'")
    plt.plot(p["r_mid"], p["aline"], label="Polynomial a'")
    plt.xlabel("r/R")
    plt.ylabel("Tangential induction [-]")
    plt.title("Tangential induction comparison")
    plt.grid(True)
    plt.legend()
    save_and_show("comparison_tangential_induction.png")


def plot_ct_convergence(histories, bins, filename, title):
    dr = (bins[1:] - bins[:-1]) * Radius
    hist_arr = np.array(histories)
    dr_col = dr[:, np.newaxis]

    ct_history = np.sum(
        hist_arr * dr_col * NBlades / (0.5 * U0 ** 2 * np.pi * Radius ** 2),
        axis=0
    )

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(ct_history) + 1), ct_history)
    plt.axhline(CT_TARGET, linestyle="--", label="Target CT")
    plt.xlabel("Iteration")
    plt.ylabel("Total CT [-]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    save_and_show(filename)


# =============================================================================
# RUN BASELINE ACROSS TSR
# =============================================================================

def run_baseline_tsr_sweep():
    tsr_performance = {}
    results_tsr8 = []
    ct_history_tsr8 = []

    print(f"{'TSR':<10} | {'CT':<10} | {'CP':<10}")
    print("-" * 35)

    for TSR in np.arange(6, 11, 1):
        perf = evaluate_baseline(TSR=TSR, pitch_deg=Pitch, delta_r=delta_r_R,
                                 n_iter=200, return_histories=True)
        CT = perf["CT"]
        CP = perf["CP"]

        print(f"{TSR:<10.1f} | {CT:<10.4f} | {CP:<10.4f}")

        tsr_performance[TSR] = {"CT": CT, "CP": CP}

        if TSR == 8:
            results_tsr8 = perf["results"]
            histories = perf["histories"]
            bins = perf["bins"]

            dr = (bins[1:] - bins[:-1]) * Radius
            dr_col = dr[:, np.newaxis]
            hist_arr = np.array(histories)
            ct_history_tsr8 = np.sum(
                hist_arr * dr_col * NBlades / (0.5 * U0 ** 2 * np.pi * Radius ** 2),
                axis=0
            )

    return tsr_performance, results_tsr8, ct_history_tsr8


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # BASELINE TSR SWEEP
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("BASELINE TSR SWEEP")
    print("============================================================")
    tsr_performance, results_tsr8, ct_history_tsr8 = run_baseline_tsr_sweep()

    # -------------------------------------------------------------------------
    # BASELINE AT TSR = 8
    # -------------------------------------------------------------------------
    baseline_perf = evaluate_baseline(
        TSR=TSR_DESIGN, pitch_deg=Pitch, delta_r=delta_r_R,
        n_iter=200, return_histories=True
    )
    print("\nBaseline at TSR=8")
    print(f"CT = {baseline_perf['CT']:.6f}")
    print(f"CP = {baseline_perf['CP']:.6f}")

    # -------------------------------------------------------------------------
    # ANALYTICAL METHOD
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("ANALYTICAL METHOD")
    print("============================================================")
    r_anal, c_anal, beta_anal_deg, a_design, analytical_perf = design_analytical_for_exact_ct()

    # -------------------------------------------------------------------------
    # POLYNOMIAL METHOD
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("POLYNOMIAL OPTIMIZATION METHOD")
    print("============================================================")
    poly_params, poly_perf = optimize_polynomial_rotor()

    cfun_poly, tfun_poly = build_poly_functions(poly_params)

    # -------------------------------------------------------------------------
    # ACTUATOR DISK REFERENCE
    # -------------------------------------------------------------------------
    a_ad, cp_ad = actuator_disk_cp_from_ct(CT_TARGET)

    # -------------------------------------------------------------------------
    # COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n============================================================")
    print("COMPARISON AT TSR = 8")
    print("============================================================")

    comparison_df = pd.DataFrame({
        "Method": [
            "Baseline",
            "Analytical method",
            "Polynomial method",
            "Actuator disk theory"
        ],
        "CT": [
            baseline_perf["CT"],
            analytical_perf["CT"],
            poly_perf["CT"],
            CT_TARGET
        ],
        "CP": [
            baseline_perf["CP"],
            analytical_perf["CP"],
            poly_perf["CP"],
            cp_ad
        ]
    })

    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    csv_path = os.path.join(save_folder, "comparison_table_methods.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print("\nRelative to actuator disk CP:")
    print(f"Baseline          : {baseline_perf['CP'] / cp_ad:.4f}")
    print(f"Analytical method : {analytical_perf['CP'] / cp_ad:.4f}")
    print(f"Polynomial method : {poly_perf['CP'] / cp_ad:.4f}")

    # -------------------------------------------------------------------------
    # EXPLICIT POLYNOMIALS
    # -------------------------------------------------------------------------
    pitch_deg, t_root_deg, t_tip_deg, t_curve_deg, c_tip, c2, c3 = poly_params
    b1 = c_tip - ROOT_CHORD - c2 - c3

    print("\nPolynomial method distributions:")
    print("x = (r/R - 0.2) / 0.8")
    print(f"c(x) = {ROOT_CHORD:.6f} + ({b1:.6f}) x + ({c2:.6f}) x^2 + ({c3:.6f}) x^3")
    print(f"beta(x) = {pitch_deg:.6f} + {t_root_deg:.6f}(1-x) + {t_tip_deg:.6f}x + {t_curve_deg:.6f}x(1-x)")

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------
    rR_dense = np.linspace(RootLocation_R, TipLocation_R, 400)
    c_baseline = baseline_chord(rR_dense)
    beta_baseline = baseline_twist(rR_dense, pitch_deg=Pitch)

    c_anal_dense = np.interp(rR_dense * Radius, r_anal, c_anal)
    beta_anal_dense = np.interp(rR_dense * Radius, r_anal, beta_anal_deg)

    c_poly_dense = cfun_poly(rR_dense)
    beta_poly_dense = tfun_poly(rR_dense)

    plot_geometry_comparison(
        rR_dense,
        c_baseline, beta_baseline,
        c_anal_dense, beta_anal_dense,
        c_poly_dense, beta_poly_dense
    )

    plot_loading_comparison(baseline_perf, analytical_perf, poly_perf)
    plot_induction_comparison(baseline_perf, analytical_perf, poly_perf)

    # convergence histories
    plot_ct_convergence(
        baseline_perf["histories"], baseline_perf["bins"],
        "convergence_baseline_ct.png",
        "Convergence history of total CT - Baseline"
    )
    plot_ct_convergence(
        analytical_perf["histories"], analytical_perf["bins"],
        "convergence_analytical_ct.png",
        "Convergence history of total CT - Analytical method"
    )
    plot_ct_convergence(
        poly_perf["histories"], poly_perf["bins"],
        "convergence_polynomial_ct.png",
        "Convergence history of total CT - Polynomial method"
    )

    print("\nDone.")