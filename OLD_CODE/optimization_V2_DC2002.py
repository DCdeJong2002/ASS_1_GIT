import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

# ============================================================
# Load airfoil polar
# ============================================================
data = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
polar_alpha = data["Alfa"].to_numpy()
polar_cl = data["Cl"].to_numpy()
polar_cd = data["Cd"].to_numpy()

# ============================================================
# Specifications
# ============================================================
Radius = 50.0
NBlades = 3
U0 = 10.0
TSR_target = 8.0
CT_target = 0.75

RootLocation_R = 0.2
TipLocation_R = 1.0

# Coarse grid for optimization, fine grid for final evaluation
N_annuli_opt = 40
N_annuli_final = 160

r_R_bins_opt = np.linspace(RootLocation_R, TipLocation_R, N_annuli_opt + 1)
r_R_bins_final = np.linspace(RootLocation_R, TipLocation_R, N_annuli_final + 1)

Omega_target = U0 * TSR_target / Radius

# ============================================================
# Induction and correction functions
# ============================================================
def ainduction(CT):
    """
    Calculate the axial induction factor a from the local thrust coefficient CT.
    Glauert correction is included for heavily loaded conditions.
    """
    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1

    if np.isscalar(CT):
        if CT >= CT2:
            return 1.0 + (CT - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    else:
        a = np.zeros(np.shape(CT))
        heavy = CT >= CT2
        light = ~heavy
        a[heavy] = 1.0 + (CT[heavy] - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
        a[light] = 0.5 - 0.5 * np.sqrt(np.maximum(0.0, 1.0 - CT[light]))
        return a


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Combined Prandtl tip and root loss correction factor.
    """
    denom = max(1e-8, 1.0 - axial_induction)

    temp_tip = -NBlades / 2.0 * (tipradius_R - r_R) / r_R * np.sqrt(
        1.0 + ((TSR * r_R) ** 2) / (denom ** 2)
    )
    Ftip = 2.0 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0.0, 1.0))

    temp_root = NBlades / 2.0 * (rootradius_R - r_R) / r_R * np.sqrt(
        1.0 + ((TSR * r_R) ** 2) / (denom ** 2)
    )
    Froot = 2.0 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0.0, 1.0))

    return max(Froot * Ftip, 1e-4)


def LoadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Compute local blade element loads and aerodynamic state.
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

    return fnorm, ftan, gamma, alpha, np.degrees(phi), cl, cd


def SolveStreamtube(
    U0,
    r1_R,
    r2_R,
    rootradius_R,
    tipradius_R,
    Omega,
    Radius,
    NBlades,
    chord,
    twist,
    polar_alpha,
    polar_cl,
    polar_cd,
    max_iter=200,
    tol=1e-5,
):
    """
    Solve a single annular streamtube using the BEM method.
    Includes under-relaxation and early stopping.
    """
    Area = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid = 0.5 * (r1_R + r2_R)
    r_local = r_mid * Radius

    a = 0.1
    aline = 0.0
    fnorm_history = np.zeros(max_iter)

    TSR = Omega * Radius / U0

    for i in range(max_iter):
        a_old = a
        aline_old = aline

        Urotor = U0 * (1.0 - a)
        Utan = (1.0 + aline) * Omega * r_local

        fnorm, ftan, gamma, alpha, phi, cl, cd = LoadBladeElement(
            Urotor, Utan, chord, twist, polar_alpha, polar_cl, polar_cd
        )
        fnorm_history[i] = fnorm

        CT = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0**2)

        anew = ainduction(CT)
        F = PrandtlTipRootCorrection(
            r_mid, rootradius_R, tipradius_R, TSR, NBlades, anew
        )
        anew /= F

        a = 0.75 * a + 0.25 * anew

        denom_aline = 2.0 * np.pi * U0 * (1.0 - a) * Omega * 2.0 * r_local**2
        denom_aline = max(denom_aline, 1e-8)
        aline = (ftan * NBlades) / denom_aline
        aline /= F

        if abs(a - a_old) < tol and abs(aline - aline_old) < tol:
            fnorm_history[i + 1:] = fnorm
            break

    return [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi, cl, cd], fnorm_history


# ============================================================
# Full-rotor BEM analysis for arbitrary chord/twist distributions
# ============================================================
def run_bem_from_distributions(chord_func, twist_func, r_R_bins, return_spanwise=False):
    """
    Run the BEM solver for an arbitrary chord and twist distribution.
    """
    results = []
    histories = []

    for i in range(len(r_R_bins) - 1):
        r1 = r_R_bins[i]
        r2 = r_R_bins[i + 1]
        mu = 0.5 * (r1 + r2)

        chord = chord_func(mu)
        twist = twist_func(mu)

        res, f_hist = SolveStreamtube(
            U0, r1, r2, RootLocation_R, TipLocation_R,
            Omega_target, Radius, NBlades,
            chord, twist, polar_alpha, polar_cl, polar_cd
        )
        results.append(res)
        histories.append(f_hist)

    results = np.array(results)
    histories = np.array(histories)

    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

    CT = np.sum(dr * results[:, 3] * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP = np.sum(
        dr * results[:, 4] * results[:, 2] * NBlades * Radius * Omega_target
        / (0.5 * U0**3 * np.pi * Radius**2)
    )

    if return_spanwise:
        mu_mid = results[:, 2]
        chord_dist = np.array([chord_func(mu) for mu in mu_mid])
        twist_dist = np.array([twist_func(mu) for mu in mu_mid])

        return {
            "CT": CT,
            "CP": CP,
            "results": results,
            "histories": histories,
            "mu_mid": mu_mid,
            "chord_dist": chord_dist,
            "twist_dist": twist_dist,
        }

    return CT, CP


# ============================================================
# Classical design ingredients
# ============================================================
def find_alpha_opt(polar_alpha, polar_cl, polar_cd):
    """
    Find the angle of attack that maximizes Cl/Cd.
    """
    ld = polar_cl / np.maximum(polar_cd, 1e-8)
    idx = np.argmax(ld)
    return polar_alpha[idx], polar_cl[idx], polar_cd[idx]


alpha_opt, cl_opt, cd_opt = find_alpha_opt(polar_alpha, polar_cl, polar_cd)


def make_interp_function(x, y):
    """
    Return a callable interpolation function.
    """
    def f(xq):
        return np.interp(xq, x, y)
    return f


def phi_from_inductions(mu, a, aprime, TSR):
    """
    Inflow angle from the velocity triangle.
    """
    return np.arctan2((1.0 - a), (TSR * mu * (1.0 + aprime)))


def cn_from_phi(phi_rad, cl, cd):
    """
    Normal coefficient from aerodynamic coefficients and inflow angle.
    """
    return cl * np.cos(phi_rad) + cd * np.sin(phi_rad)


def make_classical_blade_fixed_a(
    a_scalar,
    aprime_func,
    mu_grid,
    tip_chord_min=0.3,
    root_flat_until=0.28,
):
    """
    Build the classical blade for a fixed scalar axial induction a.

    Classical method:
    - theta(mu) = phi(mu) - alpha_opt
    - chord(mu) from analytical formula

    Boundary treatment:
    - Root: chord flattened to value at mu = root_flat_until
    - Tip : minimum chord tip_chord_min
    """
    chord_vals = []
    twist_vals = []

    for mu in mu_grid:
        aprime_here = aprime_func(mu)
        phi = phi_from_inductions(mu, a_scalar, aprime_here, TSR_target)

        F = PrandtlTipRootCorrection(
            mu, RootLocation_R, TipLocation_R, TSR_target, NBlades, a_scalar
        )

        cn = cn_from_phi(phi, cl_opt, cd_opt)

        numerator = (
            8.0
            * np.pi
            * (mu * Radius)
            * a_scalar
            * F
            * (1.0 - a_scalar * F)
            * (np.sin(phi) ** 2)
        )
        denominator = NBlades * ((1.0 - a_scalar) ** 2) * max(cn, 1e-8)

        chord_here = numerator / denominator
        twist_here = np.degrees(phi) - alpha_opt

        chord_vals.append(chord_here)
        twist_vals.append(twist_here)

    chord_vals = np.array(chord_vals)
    twist_vals = np.array(twist_vals)

    # Root boundary treatment:
    c_root_ref = np.interp(root_flat_until, mu_grid, chord_vals)
    chord_vals[mu_grid <= root_flat_until] = c_root_ref

    # Tip floor:
    chord_vals = np.maximum(chord_vals, tip_chord_min)

    return chord_vals, twist_vals


def find_sign_change_bracket_scalar(func, x_min, x_max, n_scan=30):
    """
    Scan an interval and return the first sub-interval where the function changes sign.
    """
    x_vals = np.linspace(x_min, x_max, n_scan)
    f_vals = []

    for xv in x_vals:
        try:
            fv = func(xv)
        except Exception:
            fv = np.nan
        f_vals.append(fv)

    f_vals = np.array(f_vals)

    print("\nScanning for sign-changing bracket:")
    for xv, fv in zip(x_vals, f_vals):
        print(f"x = {xv:.6f}, residual = {fv:.6f}")

    for i in range(len(x_vals) - 1):
        f1 = f_vals[i]
        f2 = f_vals[i + 1]

        if np.isfinite(f1) and np.isfinite(f2):
            if f1 == 0.0:
                return x_vals[i], x_vals[i]
            if f1 * f2 < 0.0:
                return x_vals[i], x_vals[i + 1]

    return None


def build_classical_design(
    a_scalar=0.25,
    tip_chord_min=0.3,
    root_flat_until=0.28,
):
    """
    Build a robust classical baseline design.

    Strategy:
    1. Fix a = 0.25 from actuator-disk theory for CT=0.75
    2. Compute classical twist and raw chord shape
    3. Solve for a global chord scale factor kc such that
       final BEM CT = CT_target
    """
    mu_design = np.linspace(RootLocation_R, TipLocation_R, 300)

    # First assume a' = 0
    aprime_func = lambda mu: 0.0

    # Build raw classical shape
    chord_raw, twist_vals = make_classical_blade_fixed_a(
        a_scalar=a_scalar,
        aprime_func=aprime_func,
        mu_grid=mu_design,
        tip_chord_min=tip_chord_min,
        root_flat_until=root_flat_until,
    )

    twist_func = make_interp_function(mu_design, twist_vals)

    def ct_residual_chord_scale(kc):
        chord_scaled = kc * chord_raw
        chord_scaled = np.maximum(chord_scaled, tip_chord_min)

        c_root_ref = np.interp(root_flat_until, mu_design, chord_scaled)
        chord_scaled[mu_design <= root_flat_until] = c_root_ref

        chord_func = make_interp_function(mu_design, chord_scaled)

        CT, _ = run_bem_from_distributions(chord_func, twist_func, r_R_bins_opt)
        return CT - CT_target

    bracket = find_sign_change_bracket_scalar(
        ct_residual_chord_scale,
        x_min=0.4,
        x_max=2.5,
        n_scan=40,
    )

    if bracket is None:
        raise RuntimeError(
            "Could not find a valid sign-changing bracket for the chord scale factor kc. "
            "Check the classical chord construction or widen the kc scan range."
        )

    kc1, kc2 = bracket

    if kc1 == kc2:
        kc_star = kc1
    else:
        sol = root_scalar(
            ct_residual_chord_scale,
            bracket=[kc1, kc2],
            method="brentq",
            xtol=1e-5,
        )
        kc_star = sol.root

    chord_vals = kc_star * chord_raw
    chord_vals = np.maximum(chord_vals, tip_chord_min)

    c_root_ref = np.interp(root_flat_until, mu_design, chord_vals)
    chord_vals[mu_design <= root_flat_until] = c_root_ref

    chord_func = make_interp_function(mu_design, chord_vals)
    twist_func = make_interp_function(mu_design, twist_vals)

    coarse_analysis = run_bem_from_distributions(
        chord_func, twist_func, r_R_bins_opt, return_spanwise=True
    )
    fine_analysis = run_bem_from_distributions(
        chord_func, twist_func, r_R_bins_final, return_spanwise=True
    )

    print("\n========================================================")
    print("CLASSICAL DESIGN BUILT SUCCESSFULLY")
    print("========================================================")
    print(f"alpha_opt                  = {alpha_opt:.6f} deg")
    print(f"fixed a_scalar             = {a_scalar:.6f}")
    print(f"optimized chord scale kc   = {kc_star:.6f}")
    print(f"coarse-grid CT             = {coarse_analysis['CT']:.6f}")
    print(f"coarse-grid CP             = {coarse_analysis['CP']:.6f}")
    print(f"final-grid CT              = {fine_analysis['CT']:.6f}")
    print(f"final-grid CP              = {fine_analysis['CP']:.6f}")
    print("========================================================\n")

    return {
        "alpha_opt": alpha_opt,
        "cl_opt": cl_opt,
        "cd_opt": cd_opt,
        "a_star": a_scalar,
        "kc_star": kc_star,
        "mu_design": mu_design,
        "chord_raw": chord_raw,
        "chord_vals": chord_vals,
        "twist_vals": twist_vals,
        "chord_func": chord_func,
        "twist_func": twist_func,
        "coarse_analysis": coarse_analysis,
        "fine_analysis": fine_analysis,
        "tip_chord_min": tip_chord_min,
        "root_flat_until": root_flat_until,
    }


# ============================================================
# Build classical baseline
# ============================================================
classical = build_classical_design(
    a_scalar=0.25,
    tip_chord_min=0.3,
    root_flat_until=0.28,
)

classical_chord_func = classical["chord_func"]
classical_twist_func = classical["twist_func"]

CT_classical_coarse = classical["coarse_analysis"]["CT"]
CP_classical_coarse = classical["coarse_analysis"]["CP"]
CT_classical_final = classical["fine_analysis"]["CT"]
CP_classical_final = classical["fine_analysis"]["CP"]

print("========================================================")
print("CLASSICAL BASELINE")
print("========================================================")
print(f"alpha_opt                      = {classical['alpha_opt']:.6f} deg")
print(f"classical scalar a             = {classical['a_star']:.6f}")
print(f"classical chord scale kc       = {classical['kc_star']:.6f}")
print(f"classical coarse-grid CT       = {CT_classical_coarse:.6f}")
print(f"classical coarse-grid CP       = {CP_classical_coarse:.6f}")
print(f"classical final-grid  CT       = {CT_classical_final:.6f}")
print(f"classical final-grid  CP       = {CP_classical_final:.6f}")
print("========================================================")


# ============================================================
# Hybrid correction model around classical blade
# ============================================================
# chord(mu) = c_classical(mu) * (1 + a0 + a1*mu)
# twist(mu) = theta_classical(mu) + (b0 + b1*mu + b2*mu^2)
#
# x = [a0, a1, b0, b1, b2]
# ============================================================
def hybrid_distributions_from_x(x):
    a0, a1, b0, b1, b2 = x

    def chord_func(mu):
        corr = 1.0 + a0 + a1 * mu
        cval = classical_chord_func(mu) * corr
        return max(cval, classical["tip_chord_min"])

    def twist_func(mu):
        return classical_twist_func(mu) + b0 + b1 * mu + b2 * mu**2

    return chord_func, twist_func


def design_penalty(x):
    chord_func, twist_func = hybrid_distributions_from_x(x)

    mu_eval = np.linspace(RootLocation_R, TipLocation_R, 100)
    chord_vals = np.array([chord_func(mu) for mu in mu_eval])
    twist_vals = np.array([twist_func(mu) for mu in mu_eval])

    penalty = 0.0

    # Keep chord physical
    penalty += 1e6 * np.sum(np.clip(0.2 - chord_vals, 0.0, None) ** 2)
    penalty += 1e4 * np.sum(np.clip(chord_vals - 8.0, 0.0, None) ** 2)

    # Tip floor
    penalty += 1e6 * np.sum(np.clip(classical["tip_chord_min"] - chord_vals[-5:], 0.0, None) ** 2)

    # Encourage taper
    dchord = np.diff(chord_vals)
    penalty += 2e2 * np.sum(np.clip(dchord, 0.0, None) ** 2)

    # Twist reasonable range
    penalty += 1e3 * np.sum(np.clip(twist_vals - 25.0, 0.0, None) ** 2)
    penalty += 1e3 * np.sum(np.clip(-35.0 - twist_vals, 0.0, None) ** 2)

    return penalty


def objective(x):
    try:
        chord_func, twist_func = hybrid_distributions_from_x(x)
        CT, CP = run_bem_from_distributions(chord_func, twist_func, r_R_bins_opt)
    except Exception:
        return 1e9

    if not np.isfinite(CT) or not np.isfinite(CP):
        return 1e9

    return -CP + design_penalty(x)


def ct_constraint(x):
    try:
        chord_func, twist_func = hybrid_distributions_from_x(x)
        CT, _ = run_bem_from_distributions(chord_func, twist_func, r_R_bins_opt)
        return CT - CT_target
    except Exception:
        return 1e6


# ============================================================
# Optimize small correction around classical blade
# ============================================================
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

bounds = [
    (-0.30, 0.30),   # a0
    (-0.50, 0.50),   # a1
    (-5.0, 5.0),     # b0
    (-10.0, 10.0),   # b1
    (-10.0, 10.0),   # b2
]

constraints = [{"type": "eq", "fun": ct_constraint}]

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 120, "ftol": 1e-8, "disp": True},
)

x_opt = result.x
chord_opt_func, twist_opt_func = hybrid_distributions_from_x(x_opt)

CT_opt_coarse, CP_opt_coarse = run_bem_from_distributions(
    chord_opt_func, twist_opt_func, r_R_bins_opt
)
CT_opt_final, CP_opt_final = run_bem_from_distributions(
    chord_opt_func, twist_opt_func, r_R_bins_final
)

print("\n========================================================")
print("HYBRID OPTIMIZED DESIGN")
print("========================================================")
print(result.message)
print(f"Target CT                      = {CT_target:.6f}")
print(f"Hybrid coarse-grid CT          = {CT_opt_coarse:.6f}")
print(f"Hybrid coarse-grid CP          = {CP_opt_coarse:.6f}")
print(f"Hybrid final-grid  CT          = {CT_opt_final:.6f}")
print(f"Hybrid final-grid  CP          = {CP_opt_final:.6f}")
print()
print("Correction parameters")
print(f"a0 = {x_opt[0]: .6f}")
print(f"a1 = {x_opt[1]: .6f}")
print(f"b0 = {x_opt[2]: .6f}")
print(f"b1 = {x_opt[3]: .6f}")
print(f"b2 = {x_opt[4]: .6f}")
print()
print("Hybrid correction formulas")
print(f"c(mu)     = c_classical(mu) * (1 + ({x_opt[0]:.6f}) + ({x_opt[1]:.6f})*mu)")
print(f"theta(mu) = theta_classical(mu) + ({x_opt[2]:.6f}) + ({x_opt[3]:.6f})*mu + ({x_opt[4]:.6f})*mu^2")
print("========================================================")


# ============================================================
# Actuator disk comparison
# ============================================================
a_AD = 0.25
CP_AD = 4.0 * a_AD * (1.0 - a_AD) ** 2

print("\n========================================================")
print("ACTUATOR DISK COMPARISON")
print("========================================================")
print(f"CP_actuator_disk              = {CP_AD:.6f}")
print(f"CP_classical_final            = {CP_classical_final:.6f}")
print(f"CP_hybrid_final               = {CP_opt_final:.6f}")
print(f"classical / actuator disk     = {CP_classical_final / CP_AD:.6f}")
print(f"hybrid / actuator disk        = {CP_opt_final / CP_AD:.6f}")
print("========================================================")


# ============================================================
# Collect final detailed data
# ============================================================
classical_data = run_bem_from_distributions(
    classical_chord_func, classical_twist_func, r_R_bins_final, return_spanwise=True
)
hybrid_data = run_bem_from_distributions(
    chord_opt_func, twist_opt_func, r_R_bins_final, return_spanwise=True
)


# ============================================================
# Output folder
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_BEM_hybrid_classical")
os.makedirs(save_folder, exist_ok=True)

def save_and_show(filename):
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


# ============================================================
# Plots
# ============================================================
plt.figure(figsize=(9, 5))
plt.plot(classical_data["mu_mid"], classical_data["chord_dist"], label="Classical")
plt.plot(hybrid_data["mu_mid"], hybrid_data["chord_dist"], label="Hybrid optimized")
plt.title("Chord distribution")
plt.xlabel("r/R")
plt.ylabel("Chord [m]")
plt.grid(True)
plt.legend()
save_and_show("1_chord_distribution.png")

plt.figure(figsize=(9, 5))
plt.plot(classical_data["mu_mid"], classical_data["twist_dist"], label="Classical")
plt.plot(hybrid_data["mu_mid"], hybrid_data["twist_dist"], label="Hybrid optimized")
plt.title("Twist distribution")
plt.xlabel("r/R")
plt.ylabel("Twist [deg]")
plt.grid(True)
plt.legend()
save_and_show("2_twist_distribution.png")

plt.figure(figsize=(9, 5))
plt.plot(classical_data["results"][:, 2], classical_data["results"][:, 6], label="Classical")
plt.plot(hybrid_data["results"][:, 2], hybrid_data["results"][:, 6], label="Hybrid optimized")
plt.title("Angle of attack distribution")
plt.xlabel("r/R")
plt.ylabel("Alpha [deg]")
plt.grid(True)
plt.legend()
save_and_show("3_alpha_distribution.png")

plt.figure(figsize=(9, 5))
plt.plot(classical_data["results"][:, 2], classical_data["results"][:, 0], label="Classical")
plt.plot(hybrid_data["results"][:, 2], hybrid_data["results"][:, 0], label="Hybrid optimized")
plt.title("Axial induction")
plt.xlabel("r/R")
plt.ylabel("a")
plt.grid(True)
plt.legend()
save_and_show("4_axial_induction.png")

plt.figure(figsize=(8, 5))
labels = ["Actuator disk", "Classical", "Hybrid"]
values = [CP_AD, CP_classical_final, CP_opt_final]
plt.bar(labels, values)
plt.title("CP comparison")
plt.ylabel("CP")
plt.grid(True, axis="y")
save_and_show("5_cp_comparison.png")

plt.figure(figsize=(8, 5))
labels = ["Target", "Classical", "Hybrid"]
values = [CT_target, CT_classical_final, CT_opt_final]
plt.bar(labels, values)
plt.title("CT comparison")
plt.ylabel("CT")
plt.grid(True, axis="y")
save_and_show("6_ct_comparison.png")


# ============================================================
# Save data
# ============================================================
summary_df = pd.DataFrame({
    "quantity": [
        "alpha_opt_deg",
        "classical_scalar_a",
        "classical_kc",
        "CT_target",
        "CP_actuator_disk",
        "CT_classical_coarse",
        "CP_classical_coarse",
        "CT_classical_final",
        "CP_classical_final",
        "CT_hybrid_coarse",
        "CP_hybrid_coarse",
        "CT_hybrid_final",
        "CP_hybrid_final",
    ],
    "value": [
        alpha_opt,
        classical["a_star"],
        classical["kc_star"],
        CT_target,
        CP_AD,
        CT_classical_coarse,
        CP_classical_coarse,
        CT_classical_final,
        CP_classical_final,
        CT_opt_coarse,
        CP_opt_coarse,
        CT_opt_final,
        CP_opt_final,
    ],
})
summary_df.to_csv(os.path.join(save_folder, "hybrid_summary.csv"), index=False)

x_df = pd.DataFrame({
    "parameter": ["a0", "a1", "b0", "b1", "b2"],
    "value": x_opt,
})
x_df.to_csv(os.path.join(save_folder, "hybrid_correction_parameters.csv"), index=False)

spanwise_df = pd.DataFrame({
    "r_R": hybrid_data["mu_mid"],
    "chord_classical": classical_data["chord_dist"],
    "twist_classical_deg": classical_data["twist_dist"],
    "a_classical": classical_data["results"][:, 0],
    "aprime_classical": classical_data["results"][:, 1],
    "alpha_classical_deg": classical_data["results"][:, 6],
    "phi_classical_deg": classical_data["results"][:, 7],
    "fnorm_classical": classical_data["results"][:, 3],
    "ftan_classical": classical_data["results"][:, 4],
    "chord_hybrid": hybrid_data["chord_dist"],
    "twist_hybrid_deg": hybrid_data["twist_dist"],
    "a_hybrid": hybrid_data["results"][:, 0],
    "aprime_hybrid": hybrid_data["results"][:, 1],
    "alpha_hybrid_deg": hybrid_data["results"][:, 6],
    "phi_hybrid_deg": hybrid_data["results"][:, 7],
    "fnorm_hybrid": hybrid_data["results"][:, 3],
    "ftan_hybrid": hybrid_data["results"][:, 4],
})
spanwise_df.to_csv(os.path.join(save_folder, "hybrid_spanwise_comparison.csv"), index=False)

print("\nSaved CSV outputs:")
print("- hybrid_summary.csv")
print("- hybrid_correction_parameters.csv")
print("- hybrid_spanwise_comparison.csv")
print(f"\nAll outputs saved in:\n{save_folder}")