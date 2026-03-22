import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

# ============================================================
# Load polar data
# ============================================================

data = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = data["Alfa"].to_numpy()
polar_cl = data["Cl"].to_numpy()
polar_cd = data["Cd"].to_numpy()

# ============================================================
# Global specifications
# ============================================================

Radius = 50.0                 # rotor radius [m]
NBlades = 3                   # number of blades
U0 = 10.0                     # freestream [m/s]
TSR_DESIGN = 8.0              # design tip speed ratio
CT_TARGET = 0.75              # target thrust coefficient

RootLocation_R = 0.2
TipLocation_R = 1.0

# final evaluation resolution
delta_r_R_final = 0.005

# coarser resolution for optimization to keep runtime reasonable
delta_r_R_opt = 0.01

# fixed root chord requirement
CHORD_ROOT = 3.4              # [m]

# minimum allowable chord anywhere
CHORD_MIN = 0.3               # [m]

# ============================================================
# Induction and correction functions
# ============================================================

def ainduction(CT):
    """
    Compute axial induction factor a from local thrust coefficient CT
    using Glauert correction for heavily loaded conditions.
    """
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1

    if np.isscalar(CT):
        if CT >= CT2:
            return 1 + (CT - CT1) / (4 * (np.sqrt(CT1) - 1))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    else:
        a = np.zeros(np.shape(CT))
        mask_high = CT >= CT2
        mask_low = ~mask_high
        a[mask_high] = 1 + (CT[mask_high] - CT1) / (4 * (np.sqrt(CT1) - 1))
        a[mask_low] = 0.5 - 0.5 * np.sqrt(np.maximum(0.0, 1.0 - CT[mask_low]))
        return a


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Combined Prandtl tip and root loss correction factor.
    """
    eps = 1e-8
    r_R = np.maximum(r_R, eps)
    denom = np.maximum((1 - axial_induction), 1e-6)

    temp_tip = -NBlades / 2 * (tipradius_R - r_R) / r_R * np.sqrt(
        1 + ((TSR * r_R) ** 2) / (denom ** 2)
    )
    Ftip = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0, 1)))

    temp_root = NBlades / 2 * (rootradius_R - r_R) / r_R * np.sqrt(
        1 + ((TSR * r_R) ** 2) / (denom ** 2)
    )
    Froot = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0, 1)))

    F = Froot * Ftip
    return np.maximum(F, 1e-4)


def LoadBladeElement(vnorm, vtan, chord, twist_deg, polar_alpha, polar_cl, polar_cd):
    """
    Compute sectional normal force, tangential force, circulation,
    angle of attack, inflow angle and aerodynamic coefficients.
    """
    vmag2 = vnorm**2 + vtan**2
    phi = np.arctan2(vnorm, vtan)              # [rad]
    alpha = twist_deg + np.degrees(phi)        # [deg]

    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord

    fnorm = lift * np.cos(phi) + drag * np.sin(phi)
    ftan = lift * np.sin(phi) - drag * np.cos(phi)

    gamma = 0.5 * np.sqrt(vmag2) * cl * chord

    return fnorm, ftan, gamma, alpha, np.degrees(phi), cl, cd


def SolveStreamtube(
    U0, r1_R, r2_R, rootradius_R, tipradius_R,
    Omega, Radius, NBlades,
    chord, twist_deg,
    polar_alpha, polar_cl, polar_cd,
    max_iter=300, tol=1e-5
):
    """
    Solve one annular streamtube using BEM.
    """
    Area = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_mid = 0.5 * (r1_R + r2_R)
    r_local = r_mid * Radius

    a = 0.1
    aline = 0.0

    fnorm_history = np.zeros(max_iter)

    for i in range(max_iter):
        Urotor = U0 * (1 - a)
        Utan = (1 + aline) * Omega * r_local

        fnorm, ftan, gamma, alpha, phi_deg, cl, cd = LoadBladeElement(
            Urotor, Utan, chord, twist_deg, polar_alpha, polar_cl, polar_cd
        )

        fnorm_history[i] = fnorm

        CT_loc = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0**2)

        anew = ainduction(CT_loc)
        F = PrandtlTipRootCorrection(
            r_mid, rootradius_R, tipradius_R, Omega * Radius / U0, NBlades, anew
        )
        anew = anew / F

        a_old = a
        aline_old = aline

        a = 0.75 * a + 0.25 * anew

        denom = 2 * np.pi * U0 * max((1 - a), 1e-6) * Omega * 2 * r_local**2
        aline_new = (ftan * NBlades) / denom
        aline_new = aline_new / F
        aline = 0.75 * aline + 0.25 * aline_new

        if abs(a - a_old) < tol and abs(aline - aline_old) < tol:
            fnorm_history = fnorm_history[:i+1]
            break

    result = {
        "a": a,
        "aline": aline,
        "r_mid": r_mid,
        "fnorm": fnorm,
        "ftan": ftan,
        "gamma": gamma,
        "alpha": alpha,
        "phi": phi_deg,
        "cl": cl,
        "cd": cd,
        "chord": chord,
        "twist": twist_deg,
    }

    return result, fnorm_history


# ============================================================
# Design parameterization
# ============================================================

def span_coordinate_x(r_R, root_r=RootLocation_R, tip_r=TipLocation_R):
    """
    Map r/R from [root, tip] to x in [0, 1].
    """
    return (r_R - root_r) / (tip_r - root_r)


def chord_distribution_poly(r_R, c_tip, c2, c3):
    """
    Cubic chord distribution:
        c(x) = c_root + b1*x + c2*x^2 + c3*x^3
    with:
        c(0) = c_root = 3.4 m
        c(1) = c_tip
    so:
        b1 = c_tip - c_root - c2 - c3

    Free design variables:
        c_tip, c2, c3
    """
    x = span_coordinate_x(r_R)
    c_root = CHORD_ROOT
    b1 = c_tip - c_root - c2 - c3
    return c_root + b1 * x + c2 * x**2 + c3 * x**3


def twist_distribution_poly(r_R, pitch, t_root, t_tip, t_curve):
    """
    Quadratic twist distribution with a separate global pitch offset:
        twist(x) = pitch + linear endpoint interpolation + curvature*x*(1-x)

    Free design variables:
        pitch, t_root, t_tip, t_curve
    """
    x = span_coordinate_x(r_R)
    base = t_root * (1 - x) + t_tip * x + t_curve * x * (1 - x)
    return pitch + base


def build_design_functions(params):
    """
    Convert optimization parameter vector into callable chord and twist functions.

    params = [pitch, t_root, t_tip, t_curve, c_tip, c2, c3]
    """
    pitch, t_root, t_tip, t_curve, c_tip, c2, c3 = params

    def chord_fun(r_R):
        return chord_distribution_poly(r_R, c_tip, c2, c3)

    def twist_fun(r_R):
        return twist_distribution_poly(r_R, pitch, t_root, t_tip, t_curve)

    return chord_fun, twist_fun


# ============================================================
# Rotor evaluation
# ============================================================

def evaluate_rotor(params, tsr=TSR_DESIGN, delta_r_R=delta_r_R_opt, return_histories=False):
    """
    Evaluate rotor performance for a given design parameter vector.
    """
    chord_fun, twist_fun = build_design_functions(params)

    Omega = U0 * tsr / Radius
    r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)

    results = []
    histories = []

    for i in range(len(r_R_bins) - 1):
        r1 = r_R_bins[i]
        r2 = r_R_bins[i + 1]
        r_mid = 0.5 * (r1 + r2)

        chord = chord_fun(r_mid)
        twist = twist_fun(r_mid)

        # hard feasibility screening
        if chord < CHORD_MIN:
            return {
                "CT": -1e9,
                "CP": -1e9,
                "results": None,
                "histories": None,
                "feasible": False,
            }

        res, hist = SolveStreamtube(
            U0, r1, r2, RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades,
            chord, twist,
            polar_alpha, polar_cl, polar_cd
        )

        results.append(res)
        histories.append(hist)

    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

    fnorm = np.array([r["fnorm"] for r in results])
    ftan = np.array([r["ftan"] for r in results])
    r_mid_arr = np.array([r["r_mid"] for r in results])

    CT = np.sum(dr * fnorm * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP = np.sum(dr * ftan * r_mid_arr * NBlades * Radius * (U0 * tsr / Radius)
                / (0.5 * U0**3 * np.pi * Radius**2))

    out = {
        "CT": CT,
        "CP": CP,
        "results": results,
        "histories": histories if return_histories else None,
        "feasible": True,
    }
    return out


# ============================================================
# Baseline design
# ============================================================

def baseline_chord(r_R):
    """
    Original baseline chord:
        c = 3*(1-r/R) + 1
    Gives c(0.2)=3.4 m and c(1)=1.0 m
    """
    return 3.0 * (1.0 - r_R) + 1.0


def baseline_twist(r_R, pitch=-2.0):
    """
    Original baseline twist from your script:
        twist = -(14*(1-r/R) + pitch)
    """
    return -(14.0 * (1.0 - r_R) + pitch)


def evaluate_baseline(pitch=-2.0, tsr=TSR_DESIGN, delta_r_R=delta_r_R_final):
    Omega = U0 * tsr / Radius
    r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)

    results = []
    histories = []

    for i in range(len(r_R_bins) - 1):
        r1 = r_R_bins[i]
        r2 = r_R_bins[i + 1]
        rm = 0.5 * (r1 + r2)

        chord = baseline_chord(rm)
        twist = baseline_twist(rm, pitch=pitch)

        res, hist = SolveStreamtube(
            U0, r1, r2, RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades,
            chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        results.append(res)
        histories.append(hist)

    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius
    fnorm = np.array([r["fnorm"] for r in results])
    ftan = np.array([r["ftan"] for r in results])
    r_mid_arr = np.array([r["r_mid"] for r in results])

    CT = np.sum(dr * fnorm * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP = np.sum(dr * ftan * r_mid_arr * NBlades * Radius * Omega
                / (0.5 * U0**3 * np.pi * Radius**2))

    return {
        "CT": CT,
        "CP": CP,
        "results": results,
        "histories": histories,
    }


def fit_baseline_pitch_to_target():
    """
    Adjust only the baseline pitch so that the baseline rotor hits CT_target.
    This gives a fair baseline comparison against the optimized design.
    """
    def objective_pitch(p):
        perf = evaluate_baseline(pitch=p[0], delta_r_R=delta_r_R_opt)
        return (perf["CT"] - CT_TARGET) ** 2

    res = minimize(
        objective_pitch,
        x0=np.array([-2.0]),
        bounds=[(-10.0, 10.0)],
        method="L-BFGS-B"
    )

    best_pitch = res.x[0]
    perf = evaluate_baseline(pitch=best_pitch, delta_r_R=delta_r_R_final)
    return best_pitch, perf


# ============================================================
# Optimization setup
# ============================================================

def dense_chord_check(params, n=200):
    """
    Evaluate minimum chord over a dense spanwise grid.
    """
    chord_fun, _ = build_design_functions(params)
    r_dense = np.linspace(RootLocation_R, TipLocation_R, n)
    c_dense = chord_fun(r_dense)
    return np.min(c_dense), np.max(c_dense)


def objective_with_penalty(params):
    """
    Objective to maximize CP while enforcing CT=0.75 and chord feasibility.
    Implemented as minimization of:
        -CP + penalty
    """
    min_c, max_c = dense_chord_check(params)

    penalty = 0.0

    if min_c < CHORD_MIN:
        penalty += 1e4 * (CHORD_MIN - min_c) ** 2

    # optional upper chord regularization to avoid unrealistic very fat blades
    if max_c > 6.0:
        penalty += 1e3 * (max_c - 6.0) ** 2

    perf = evaluate_rotor(params, tsr=TSR_DESIGN, delta_r_R=delta_r_R_opt)

    if not perf["feasible"]:
        return 1e9

    CT_err = perf["CT"] - CT_TARGET
    penalty += 5e3 * CT_err ** 2

    # mild smoothness/regularization so optimizer does not create extreme polynomials
    _, _, _, t_curve, _, c2, c3 = params
    penalty += 0.05 * (t_curve**2) + 0.01 * (c2**2 + c3**2)

    return -perf["CP"] + penalty


def optimize_design(n_starts=12, random_seed=42):
    """
    Multi-start local optimization.
    Parameters:
        params = [pitch, t_root, t_tip, t_curve, c_tip, c2, c3]
    """
    rng = np.random.default_rng(random_seed)

    bounds = [
        (-8.0, 8.0),    # pitch
        (-25.0, 5.0),   # t_root
        (-10.0, 15.0),  # t_tip
        (-20.0, 20.0),  # t_curve
        (0.3, 2.0),     # c_tip
        (-5.0, 5.0),    # c2
        (-5.0, 5.0),    # c3
    ]

    # sensible initial guess near the original design
    # original root twist around -9.2 deg and tip around +2 deg for pitch=-2
    x0_nominal = np.array([
        -2.0,   # pitch
        -7.0,   # t_root
        2.0,    # t_tip
        0.0,    # t_curve
        1.0,    # c_tip
        0.0,    # c2
        0.0,    # c3
    ])

    starts = [x0_nominal]

    for _ in range(n_starts - 1):
        trial = np.array([
            rng.uniform(-6.0, 6.0),     # pitch
            rng.uniform(-20.0, 0.0),    # t_root
            rng.uniform(-5.0, 10.0),    # t_tip
            rng.uniform(-10.0, 10.0),   # t_curve
            rng.uniform(0.3, 1.5),      # c_tip
            rng.uniform(-3.0, 3.0),     # c2
            rng.uniform(-3.0, 3.0),     # c3
        ])
        starts.append(trial)

    best_res = None
    best_obj = np.inf

    for k, x0 in enumerate(starts, start=1):
        res = minimize(
            objective_with_penalty,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250}
        )

        if res.fun < best_obj:
            best_obj = res.fun
            best_res = res

        print(f"Start {k:02d}/{len(starts)} | objective = {res.fun:.6f}")

    return best_res


# ============================================================
# Actuator disk theory
# ============================================================

def actuator_disk_cp_from_ct(CT):
    """
    For ideal actuator disk in axial flow:
        CT = 4a(1-a)
        CP = 4a(1-a)^2
    Physical branch is a <= 0.5:
        a = 0.5 * (1 - sqrt(1 - CT))
    """
    if CT < 0.0 or CT > 1.0:
        return np.nan, np.nan

    a = 0.5 * (1.0 - np.sqrt(1.0 - CT))
    CP = 4.0 * a * (1.0 - a)**2
    return a, CP


# ============================================================
# Plotting helpers
# ============================================================

def ensure_output_folder():
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(base_path, "plots_BEM_optimization")
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def save_and_show(fig, save_folder, filename):
    save_path = os.path.join(save_folder, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


def extract_arrays(result_dict):
    """
    Turn list of section dictionaries into arrays.
    """
    results = result_dict["results"]
    arr = {}
    keys = results[0].keys()
    for key in keys:
        arr[key] = np.array([r[key] for r in results])
    return arr


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":

    save_folder = ensure_output_folder()

    # --------------------------------------------------------
    # Baseline with pitch retuned to CT = 0.75
    # --------------------------------------------------------
    baseline_pitch, baseline_perf = fit_baseline_pitch_to_target()
    baseline_arr = extract_arrays(baseline_perf)

    print("\n" + "=" * 70)
    print("BASELINE DESIGN (original chord/twist shape, pitch retuned for CT=0.75)")
    print("=" * 70)
    print(f"Baseline pitch         : {baseline_pitch:.4f} deg")
    print(f"Baseline CT            : {baseline_perf['CT']:.6f}")
    print(f"Baseline CP            : {baseline_perf['CP']:.6f}")

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)

    opt_res = optimize_design(n_starts=12, random_seed=42)
    best_params = opt_res.x

    # final high-resolution evaluation
    opt_perf = evaluate_rotor(best_params, tsr=TSR_DESIGN, delta_r_R=delta_r_R_final, return_histories=True)
    opt_arr = extract_arrays(opt_perf)

    pitch_opt, t_root_opt, t_tip_opt, t_curve_opt, c_tip_opt, c2_opt, c3_opt = best_params

    print("\n" + "=" * 70)
    print("OPTIMIZED DESIGN")
    print("=" * 70)
    print(f"Optimized pitch        : {pitch_opt:.4f} deg")
    print(f"Optimized t_root       : {t_root_opt:.4f} deg")
    print(f"Optimized t_tip        : {t_tip_opt:.4f} deg")
    print(f"Optimized t_curve      : {t_curve_opt:.4f} deg")
    print(f"Optimized c_tip        : {c_tip_opt:.4f} m")
    print(f"Optimized c2           : {c2_opt:.4f}")
    print(f"Optimized c3           : {c3_opt:.4f}")
    print(f"Optimized CT           : {opt_perf['CT']:.6f}")
    print(f"Optimized CP           : {opt_perf['CP']:.6f}")

    min_c, max_c = dense_chord_check(best_params)
    print(f"Minimum chord on span  : {min_c:.4f} m")
    print(f"Maximum chord on span  : {max_c:.4f} m")

    # --------------------------------------------------------
    # Actuator disk comparison
    # --------------------------------------------------------
    a_ad, cp_ad = actuator_disk_cp_from_ct(CT_TARGET)

    print("\n" + "=" * 70)
    print("ACTUATOR DISK THEORY")
    print("=" * 70)
    print(f"For CT = {CT_TARGET:.4f}:")
    print(f"Ideal axial induction a = {a_ad:.6f}")
    print(f"Ideal CP               = {cp_ad:.6f}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Baseline CP / CP_AD    = {baseline_perf['CP'] / cp_ad:.4f}")
    print(f"Optimized CP / CP_AD   = {opt_perf['CP'] / cp_ad:.4f}")
    print(f"CP improvement         = {100 * (opt_perf['CP'] - baseline_perf['CP']) / abs(baseline_perf['CP']):.2f} %")

    # --------------------------------------------------------
    # Build dense distributions for plotting
    # --------------------------------------------------------
    r_dense = np.linspace(RootLocation_R, TipLocation_R, 400)

    chord_baseline_dense = baseline_chord(r_dense)
    twist_baseline_dense = baseline_twist(r_dense, pitch=baseline_pitch)

    chord_fun_opt, twist_fun_opt = build_design_functions(best_params)
    chord_opt_dense = chord_fun_opt(r_dense)
    twist_opt_dense = twist_fun_opt(r_dense)

    # --------------------------------------------------------
    # Plot 1: chord distribution
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 5))
    plt.plot(r_dense, chord_baseline_dense, label="Baseline")
    plt.plot(r_dense, chord_opt_dense, label="Optimized")
    plt.axhline(CHORD_MIN, linestyle="--", label="Minimum allowed chord")
    plt.scatter([RootLocation_R], [CHORD_ROOT], zorder=5, label="Fixed root chord")
    plt.xlabel("r/R")
    plt.ylabel("Chord [m]")
    plt.title("Chord distribution")
    plt.grid(True)
    plt.legend()
    save_and_show(fig, save_folder, "01_chord_distribution.png")

    # --------------------------------------------------------
    # Plot 2: twist distribution
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 5))
    plt.plot(r_dense, twist_baseline_dense, label="Baseline")
    plt.plot(r_dense, twist_opt_dense, label="Optimized")
    plt.xlabel("r/R")
    plt.ylabel("Twist [deg]")
    plt.title("Twist distribution")
    plt.grid(True)
    plt.legend()
    save_and_show(fig, save_folder, "02_twist_distribution.png")

    # --------------------------------------------------------
    # Plot 3: induction distribution
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 5))
    plt.plot(baseline_arr["r_mid"], baseline_arr["a"], label="Baseline axial induction")
    plt.plot(opt_arr["r_mid"], opt_arr["a"], label="Optimized axial induction")
    plt.plot(baseline_arr["r_mid"], baseline_arr["aline"], label="Baseline tangential induction")
    plt.plot(opt_arr["r_mid"], opt_arr["aline"], label="Optimized tangential induction")
    plt.xlabel("r/R")
    plt.ylabel("Induction factor [-]")
    plt.title("Induction distributions")
    plt.grid(True)
    plt.legend()
    save_and_show(fig, save_folder, "03_induction_distribution.png")

    # --------------------------------------------------------
    # Plot 4: loading
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 5))
    plt.plot(baseline_arr["r_mid"], baseline_arr["fnorm"], label="Baseline normal load")
    plt.plot(opt_arr["r_mid"], opt_arr["fnorm"], label="Optimized normal load")
    plt.plot(baseline_arr["r_mid"], baseline_arr["ftan"], label="Baseline tangential load")
    plt.plot(opt_arr["r_mid"], opt_arr["ftan"], label="Optimized tangential load")
    plt.xlabel("r/R")
    plt.ylabel("Section load [N/m]")
    plt.title("Section loading")
    plt.grid(True)
    plt.legend()
    save_and_show(fig, save_folder, "04_loading_distribution.png")

    # --------------------------------------------------------
    # Plot 5: angle distributions
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 5))
    plt.plot(baseline_arr["r_mid"], baseline_arr["alpha"], label="Baseline alpha")
    plt.plot(opt_arr["r_mid"], opt_arr["alpha"], label="Optimized alpha")
    plt.plot(baseline_arr["r_mid"], baseline_arr["phi"], label="Baseline phi")
    plt.plot(opt_arr["r_mid"], opt_arr["phi"], label="Optimized phi")
    plt.xlabel("r/R")
    plt.ylabel("Angle [deg]")
    plt.title("Angle of attack and inflow angle")
    plt.grid(True)
    plt.legend()
    save_and_show(fig, save_folder, "05_angle_distribution.png")

    # --------------------------------------------------------
    # Plot 6: convergence history of total CT for optimized design
    # --------------------------------------------------------
    histories = opt_perf["histories"]
    max_len = max(len(h) for h in histories)

    # pad histories with their last value so all have equal length
    hist_padded = []
    for h in histories:
        if len(h) < max_len:
            pad = np.full(max_len - len(h), h[-1])
            h_full = np.concatenate([h, pad])
        else:
            h_full = h
        hist_padded.append(h_full)

    hist_arr = np.array(hist_padded)
    dr = (np.arange(RootLocation_R + delta_r_R_final, TipLocation_R + delta_r_R_final / 2, delta_r_R_final)
          - np.arange(RootLocation_R, TipLocation_R, delta_r_R_final)) * Radius

    ct_history = np.sum(hist_arr * dr[:, np.newaxis] * NBlades / (0.5 * U0**2 * np.pi * Radius**2), axis=0)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(ct_history) + 1), ct_history)
    plt.axhline(CT_TARGET, linestyle="--", label="Target CT")
    plt.xlabel("Iteration")
    plt.ylabel("Total CT [-]")
    plt.title("Convergence history of total thrust coefficient")
    plt.grid(True)
    plt.legend()
    save_and_show(fig, save_folder, "06_ct_convergence_history_optimized.png")

    # --------------------------------------------------------
    # Save distributions to CSV
    # --------------------------------------------------------
    df_sections = pd.DataFrame({
        "r_R": opt_arr["r_mid"],
        "chord_m": opt_arr["chord"],
        "twist_deg": opt_arr["twist"],
        "a": opt_arr["a"],
        "aline": opt_arr["aline"],
        "fnorm": opt_arr["fnorm"],
        "ftan": opt_arr["ftan"],
        "gamma": opt_arr["gamma"],
        "alpha_deg": opt_arr["alpha"],
        "phi_deg": opt_arr["phi"],
        "cl": opt_arr["cl"],
        "cd": opt_arr["cd"],
    })
    csv_path = os.path.join(save_folder, "optimized_blade_sections.csv")
    df_sections.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # --------------------------------------------------------
    # Print polynomial form explicitly
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("OPTIMIZED POLYNOMIAL DISTRIBUTIONS")
    print("=" * 70)

    b1 = c_tip_opt - CHORD_ROOT - c2_opt - c3_opt

    print("Chord distribution:")
    print("x = (r/R - 0.2) / 0.8")
    print(f"c(x) = {CHORD_ROOT:.6f} + ({b1:.6f}) x + ({c2_opt:.6f}) x^2 + ({c3_opt:.6f}) x^3   [m]")

    print("\nTwist distribution:")
    print("x = (r/R - 0.2) / 0.8")
    print(f"twist(x) = {pitch_opt:.6f} + {t_root_opt:.6f}(1-x) + {t_tip_opt:.6f}x + {t_curve_opt:.6f}x(1-x)   [deg]")

    print("\nDone.")