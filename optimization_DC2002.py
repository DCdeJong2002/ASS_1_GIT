import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize


# ============================================================
# 1. BEM helper functions
# ============================================================

def ainduction(CT):
    """
    Compute axial induction factor a from thrust coefficient CT
    using the same Glauert-type correction logic as in your code.
    """
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1

    if np.isscalar(CT):
        if CT >= CT2:
            return 1 + (CT - CT1) / (4 * (np.sqrt(CT1) - 1))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    else:
        a = np.zeros(np.shape(CT))
        mask = CT >= CT2
        a[mask] = 1 + (CT[mask] - CT1) / (4 * (np.sqrt(CT1) - 1))
        a[~mask] = 0.5 - 0.5 * np.sqrt(np.maximum(0.0, 1.0 - CT[~mask]))
        return a


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Compute combined Prandtl tip and root loss factor.
    """
    ai = np.clip(axial_induction, -1.0, 0.999)

    temp_tip = (
        -NBlades / 2
        * (tipradius_R - r_R) / r_R
        * np.sqrt(1 + ((TSR * r_R) ** 2) / ((1 - ai) ** 2))
    )
    Ftip = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0.0, 1.0)))

    temp_root = (
        NBlades / 2
        * (rootradius_R - r_R) / r_R
        * np.sqrt(1 + ((TSR * r_R) ** 2) / ((1 - ai) ** 2))
    )
    Froot = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0.0, 1.0)))

    return Froot * Ftip


def loadBladeElement(vnorm, vtan, chord, twist_input, polar_alpha, polar_cl, polar_cd):
    """
    Compute sectional normal force, tangential force, circulation,
    angle of attack, and inflow angle.

    Note:
    alpha = twist_input + phi_deg
    This is kept exactly in line with your original implementation.
    """
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm, vtan)
    phi_deg = np.degrees(inflowangle)

    alpha = twist_input + phi_deg
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord

    fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
    ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord

    return fnorm, ftan, gamma, alpha, phi_deg


def solveStreamtube(
    Uinf,
    r1_R,
    r2_R,
    rootradius_R,
    tipradius_R,
    Omega,
    Radius,
    NBlades,
    chord,
    twist_input,
    polar_alpha,
    polar_cl,
    polar_cd,
):
    """
    Solve one annular streamtube using iterative BEM.
    """
    area = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_R = 0.5 * (r1_R + r2_R)

    a = 0.1
    aline = 0.01

    for _ in range(200):
        Urotor = Uinf * (1 - a)
        Utan = (1 + aline) * Omega * r_R * Radius

        fnorm, ftan, gamma, alpha, phi = loadBladeElement(
            Urotor, Utan, chord, twist_input,
            polar_alpha, polar_cl, polar_cd
        )

        load3Daxial = fnorm * Radius * (r2_R - r1_R) * NBlades
        CT_local = load3Daxial / (0.5 * area * Uinf**2)

        anew = ainduction(CT_local)

        F = PrandtlTipRootCorrection(
            r_R, rootradius_R, tipradius_R,
            Omega * Radius / Uinf, NBlades, anew
        )
        F = max(F, 1e-4)

        anew /= F

        if np.abs(a - anew) < 1e-5:
            a = anew
            break

        a = 0.75 * a + 0.25 * anew

        denom = 2 * np.pi * Uinf * (1 - a) * Omega * 2 * (r_R * Radius) ** 2
        if np.abs(denom) < 1e-12:
            aline = 0.0
        else:
            aline = ftan * NBlades / denom
            aline /= F

    return [a, aline, r_R, fnorm, ftan, gamma, alpha, phi]


# ============================================================
# 2. Blade parameterization
# ============================================================

def normalized_span_coordinate(r_R, root_R, tip_R):
    """
    Map r/R in [root_R, tip_R] to s in [0, 1].
    """
    return (r_R - root_R) / (tip_R - root_R)


def twist_distribution(r_R, root_R, tip_R, t0, t1, t2):
    """
    Quadratic twist-input distribution:
        twist_input(s) = t0 + t1*s + t2*s^2

    This is the value directly passed into the BEM alpha relation:
        alpha = twist_input + phi
    """
    s = normalized_span_coordinate(r_R, root_R, tip_R)
    return t0 + t1 * s + t2 * s**2


def chord_distribution(r_R, root_R, tip_R, c_root, c_tip):
    """
    Linear chord distribution between root and tip:
        c(s) = c_root + (c_tip - c_root) * s
    """
    s = normalized_span_coordinate(r_R, root_R, tip_R)
    return c_root + (c_tip - c_root) * s


# ============================================================
# 3. Rotor performance evaluation
# ============================================================

def evaluate_rotor(
    x,
    polar_alpha,
    polar_cl,
    polar_cd,
    Uinf=10.0,
    TSR=8.0,
    Radius=50.0,
    NBlades=3,
    RootLocation_R=0.2,
    TipLocation_R=1.0,
    delta_r_R=0.01,
):
    """
    Evaluate a blade defined by:
        x = [t0, t1, t2, c_root, c_tip]

    Returns:
        CT, CP, results_array, r_bins

    If something goes numerically wrong, returns large penalty-style values.
    """
    t0, t1, t2, c_root, c_tip = x

    # Basic geometric sanity checks
    if c_root <= 0 or c_tip <= 0:
        return np.nan, -np.inf, None, None

    r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)
    Omega = Uinf * TSR / Radius

    results = []

    try:
        for i in range(len(r_R_bins) - 1):
            r_mid = 0.5 * (r_R_bins[i] + r_R_bins[i + 1])

            chord = chord_distribution(r_mid, RootLocation_R, TipLocation_R, c_root, c_tip)
            twist_input = twist_distribution(r_mid, RootLocation_R, TipLocation_R, t0, t1, t2)

            res = solveStreamtube(
                Uinf,
                r_R_bins[i],
                r_R_bins[i + 1],
                RootLocation_R,
                TipLocation_R,
                Omega,
                Radius,
                NBlades,
                chord,
                twist_input,
                polar_alpha,
                polar_cl,
                polar_cd,
            )
            results.append(res)

        res_arr = np.array(results, dtype=float)
        dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

        CT = np.sum(
            dr * res_arr[:, 3] * NBlades
            / (0.5 * Uinf**2 * np.pi * Radius**2)
        )

        CP = np.sum(
            dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega
            / (0.5 * Uinf**3 * np.pi * Radius**2)
        )

        if not np.isfinite(CT) or not np.isfinite(CP):
            return np.nan, -np.inf, None, None

        return CT, CP, res_arr, r_R_bins

    except Exception:
        return np.nan, -np.inf, None, None


# ============================================================
# 4. Objective and constraint definitions
# ============================================================

TARGET_CT = 0.75


def penalized_objective(
    x,
    polar_alpha,
    polar_cl,
    polar_cd,
    penalty_weight=500.0,
):
    """
    Penalized objective for global search:
        minimize[ -CP + w*(CT - TARGET_CT)^2 ]
    """
    CT, CP, _, _ = evaluate_rotor(x, polar_alpha, polar_cl, polar_cd)

    if not np.isfinite(CT) or not np.isfinite(CP):
        return 1e6

    penalty = penalty_weight * (CT - TARGET_CT) ** 2
    return -CP + penalty


def objective_slsqp(x, polar_alpha, polar_cl, polar_cd):
    """
    Objective for local constrained refinement:
        minimize -CP
    """
    CT, CP, _, _ = evaluate_rotor(x, polar_alpha, polar_cl, polar_cd)

    if not np.isfinite(CT) or not np.isfinite(CP):
        return 1e6

    return -CP


def ct_constraint(x, polar_alpha, polar_cl, polar_cd):
    """
    Equality constraint for SLSQP:
        CT - TARGET_CT = 0
    """
    CT, _, _, _ = evaluate_rotor(x, polar_alpha, polar_cl, polar_cd)

    if not np.isfinite(CT):
        return 1e6

    return CT - TARGET_CT


# ============================================================
# 5. Baseline blade definition from your original script
# ============================================================

def baseline_geometry(r_R, Pitch=-2.0):
    """
    Reproduce your original blade setup.

    Original chord:
        chord = 3*(1-r/R) + 1

    Original twist input:
        twist_input = -(14*(1-r/R) + Pitch)
    """
    chord = 3.0 * (1.0 - r_R) + 1.0
    twist_input = -(14.0 * (1.0 - r_R) + Pitch)
    return chord, twist_input


def evaluate_baseline(
    polar_alpha,
    polar_cl,
    polar_cd,
    Uinf=10.0,
    TSR=8.0,
    Radius=50.0,
    NBlades=3,
    RootLocation_R=0.2,
    TipLocation_R=1.0,
    delta_r_R=0.01,
    Pitch=-2.0,
):
    """
    Evaluate the original baseline blade from your current script.
    """
    r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)
    Omega = Uinf * TSR / Radius

    results = []

    for i in range(len(r_R_bins) - 1):
        r_mid = 0.5 * (r_R_bins[i] + r_R_bins[i + 1])

        chord, twist_input = baseline_geometry(r_mid, Pitch=Pitch)

        res = solveStreamtube(
            Uinf,
            r_R_bins[i],
            r_R_bins[i + 1],
            RootLocation_R,
            TipLocation_R,
            Omega,
            Radius,
            NBlades,
            chord,
            twist_input,
            polar_alpha,
            polar_cl,
            polar_cd,
        )
        results.append(res)

    res_arr = np.array(results, dtype=float)
    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

    CT = np.sum(
        dr * res_arr[:, 3] * NBlades
        / (0.5 * Uinf**2 * np.pi * Radius**2)
    )

    CP = np.sum(
        dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega
        / (0.5 * Uinf**3 * np.pi * Radius**2)
    )

    return CT, CP, res_arr, r_R_bins


# ============================================================
# 6. Actuator disk comparison
# ============================================================

def actuator_disk_cp_from_ct(CT):
    """
    For classical actuator disk theory:
        CT = 4a(1-a)
        CP = 4a(1-a)^2

    For CT < 1, the physically relevant solution is:
        a = (1 - sqrt(1 - CT)) / 2
    """
    if CT < 0 or CT > 1:
        return np.nan, np.nan

    a = 0.5 * (1.0 - np.sqrt(1.0 - CT))
    CP = 4.0 * a * (1.0 - a) ** 2
    return a, CP


# ============================================================
# 7. Main script
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load polar data
    # --------------------------------------------------------
    data = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
    polar_alpha = data["Alfa"].to_numpy(dtype=float)
    polar_cl = data["Cl"].to_numpy(dtype=float)
    polar_cd = data["Cd"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Fixed operating conditions
    # --------------------------------------------------------
    Uinf = 10.0
    TSR = 8.0
    Radius = 50.0
    NBlades = 3
    RootLocation_R = 0.2
    TipLocation_R = 1.0
    delta_r_R = 0.01

    # --------------------------------------------------------
    # Baseline evaluation
    # --------------------------------------------------------
    CT_base, CP_base, res_base, r_bins_base = evaluate_baseline(
        polar_alpha, polar_cl, polar_cd,
        Uinf=Uinf,
        TSR=TSR,
        Radius=Radius,
        NBlades=NBlades,
        RootLocation_R=RootLocation_R,
        TipLocation_R=TipLocation_R,
        delta_r_R=delta_r_R,
        Pitch=-2.0
    )

    print("\nBaseline blade at TSR = 8")
    print(f"CT_baseline = {CT_base:.5f}")
    print(f"CP_baseline = {CP_base:.5f}")

    # --------------------------------------------------------
    # Optimization bounds
    # x = [t0, t1, t2, c_root, c_tip]
    #
    # twist_input(s) = t0 + t1*s + t2*s^2
    # chord(s)      = c_root + (c_tip-c_root)*s
    # --------------------------------------------------------
    bounds = [
        (-25.0, 10.0),   # t0
        (-30.0, 30.0),   # t1
        (-30.0, 30.0),   # t2
        (1.0, 8.0),      # c_root [m]
        (0.3, 4.0),      # c_tip  [m]
    ]

    # --------------------------------------------------------
    # Stage 1: global search with penalty objective
    # --------------------------------------------------------
    print("\nStarting global optimization (differential evolution)...")

    result_de = differential_evolution(
        penalized_objective,
        bounds=bounds,
        args=(polar_alpha, polar_cl, polar_cd),
        strategy="best1bin",
        maxiter=40,
        popsize=12,
        tol=1e-3,
        polish=False,
        seed=42,
        updating="deferred",
        workers=1,
    )

    x0 = result_de.x

    CT_de, CP_de, _, _ = evaluate_rotor(x0, polar_alpha, polar_cl, polar_cd)

    print("\nGlobal search result")
    print(f"x_DE = {x0}")
    print(f"CT_DE = {CT_de:.5f}")
    print(f"CP_DE = {CP_de:.5f}")

    # --------------------------------------------------------
    # Stage 2: local constrained refinement with SLSQP
    # --------------------------------------------------------
    print("\nStarting local constrained refinement (SLSQP)...")

    cons = {
        "type": "eq",
        "fun": ct_constraint,
        "args": (polar_alpha, polar_cl, polar_cd),
    }

    result_slsqp = minimize(
        objective_slsqp,
        x0=x0,
        args=(polar_alpha, polar_cl, polar_cd),
        method="SLSQP",
        bounds=bounds,
        constraints=[cons],
        options={"maxiter": 200, "ftol": 1e-8, "disp": True},
    )

    x_opt = result_slsqp.x
    CT_opt, CP_opt, res_opt, r_bins_opt = evaluate_rotor(
        x_opt, polar_alpha, polar_cl, polar_cd,
        Uinf=Uinf,
        TSR=TSR,
        Radius=Radius,
        NBlades=NBlades,
        RootLocation_R=RootLocation_R,
        TipLocation_R=TipLocation_R,
        delta_r_R=delta_r_R,
    )

    print("\nOptimized blade at TSR = 8, CT = 0.75 target")
    print(f"x_opt = {x_opt}")
    print(f"CT_opt = {CT_opt:.8f}")
    print(f"CP_opt = {CP_opt:.8f}")

    # --------------------------------------------------------
    # Actuator disk comparison
    # --------------------------------------------------------
    a_ad, CP_ad = actuator_disk_cp_from_ct(TARGET_CT)

    print("\nActuator disk theory comparison")
    print(f"Target CT = {TARGET_CT:.5f}")
    print(f"a_actuator_disk = {a_ad:.5f}")
    print(f"CP_actuator_disk = {CP_ad:.5f}")

    if np.isfinite(CP_opt):
        print(f"CP_opt / CP_actuator_disk = {CP_opt / CP_ad:.5f}")

    # --------------------------------------------------------
    # Reconstruct optimized chord and twist distributions
    # --------------------------------------------------------
    r_mid_opt = res_opt[:, 2]
    t0, t1, t2, c_root, c_tip = x_opt

    chord_opt = chord_distribution(
        r_mid_opt, RootLocation_R, TipLocation_R, c_root, c_tip
    )
    twist_opt = twist_distribution(
        r_mid_opt, RootLocation_R, TipLocation_R, t0, t1, t2
    )

    # Baseline distributions on same radial grid
    chord_base = np.array([baseline_geometry(r, Pitch=-2.0)[0] for r in r_mid_opt])
    twist_base = np.array([baseline_geometry(r, Pitch=-2.0)[1] for r in r_mid_opt])

    # --------------------------------------------------------
    # Plot 1: chord distribution
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(r_mid_opt, chord_base, label="Baseline chord")
    plt.plot(r_mid_opt, chord_opt, label="Optimized chord")
    plt.xlabel("r/R")
    plt.ylabel("Chord [m]")
    plt.title("Baseline vs optimized chord distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 2: twist-input distribution
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(r_mid_opt, twist_base, label="Baseline twist input")
    plt.plot(r_mid_opt, twist_opt, label="Optimized twist input")
    plt.xlabel("r/R")
    plt.ylabel("Twist input [deg]")
    plt.title("Baseline vs optimized twist distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 3: induction factors
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(res_base[:, 2], res_base[:, 0], label="Baseline a")
    plt.plot(res_base[:, 2], res_base[:, 1], "--", label="Baseline a'")
    plt.plot(res_opt[:, 2], res_opt[:, 0], label="Optimized a")
    plt.plot(res_opt[:, 2], res_opt[:, 1], "--", label="Optimized a'")
    plt.xlabel("r/R")
    plt.ylabel("Induction factor [-]")
    plt.title("Spanwise induction factors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 4: angle of attack and inflow angle for optimized blade
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(res_opt[:, 2], res_opt[:, 6], label=r"Optimized $\alpha$")
    plt.plot(res_opt[:, 2], res_opt[:, 7], "--", label=r"Optimized $\phi$")
    plt.xlabel("r/R")
    plt.ylabel("Angle [deg]")
    plt.title("Optimized spanwise angle of attack and inflow angle")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot 5: sectional loads
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(
        res_base[:, 2],
        res_base[:, 3] / (0.5 * Uinf**2 * Radius),
        label=r"Baseline $F_{norm}$"
    )
    plt.plot(
        res_base[:, 2],
        res_base[:, 4] / (0.5 * Uinf**2 * Radius),
        "--",
        label=r"Baseline $F_{tan}$"
    )
    plt.plot(
        res_opt[:, 2],
        res_opt[:, 3] / (0.5 * Uinf**2 * Radius),
        label=r"Optimized $F_{norm}$"
    )
    plt.plot(
        res_opt[:, 2],
        res_opt[:, 4] / (0.5 * Uinf**2 * Radius),
        "--",
        label=r"Optimized $F_{tan}$"
    )
    plt.xlabel("r/R")
    plt.ylabel(r"$F / \left(\frac{1}{2}U_\infty^2 R\right)$")
    plt.title("Spanwise sectional loads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Final summary table
    # --------------------------------------------------------
    summary = pd.DataFrame({
        "Case": ["Baseline", "Optimized", "Actuator disk"],
        "CT": [CT_base, CT_opt, TARGET_CT],
        "CP": [CP_base, CP_opt, CP_ad],
    })

    print("\nSummary:")
    print(summary.to_string(index=False))

    # Optional: save summary
    summary.to_csv("bem_optimization_summary.csv", index=False)