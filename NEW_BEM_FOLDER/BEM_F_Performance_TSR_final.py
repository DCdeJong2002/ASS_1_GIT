import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel('polar DU95W180 (3).xlsx', skiprows=3)
polar_alpha = data['Alfa'].to_numpy()
polar_cl    = data['Cl'].to_numpy()
polar_cd    = data['Cd'].to_numpy()

# =============================================================================
# Specifications
# =============================================================================

Pitch     = -2       # Blade pitch angle [deg]
delta_r_R = 0.005    # Spanwise resolution

Radius   = 50        # Rotor radius [m]
NBlades  = 3         # Number of blades
U0       = 10        # Freestream velocity [m/s]

RootLocation_R = 0.2
TipLocation_R  = 1.0

r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R / 2, delta_r_R)

# =============================================================================
# Prandtl formula toggle
# =============================================================================

# Set to True  → helical-wake spacing formula (original Prandtl 1919, more
#                 physically correct, used by QBlade / OpenFAST).
# Set to False → simplified trigonometric form (Glauert 1935, matches the
#                 reference notebook exactly, standard in most BEM courses).
USE_HELICAL_PRANDTL = False

# =============================================================================
# BEM core functions
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
    sites in SolveStreamtube remain unchanged.
    """
    if USE_HELICAL_PRANDTL:
        return _PrandtlHelical(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction)
    return _PrandtlSimplified(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction)


def LoadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Sectional normal force, tangential force, circulation, AoA and inflow angle.
    twist [deg] uses the sign convention: alpha = twist + phi_deg.
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

    Convergence: early exit when |a - anew| < tol.
    Safety ceiling: max_iter (300 is safe; this geometry converges in < 60 iters).

    Returns
    -------
    [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi] , fnorm_history
    """
    Area    = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid   = (r1_R + r2_R) / 2
    r_local = r_mid * Radius

    a     = 0.0   # matches reference notebook initialisation
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

        anew = ainduction(CT_loc)

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

        # Early exit once axial induction has converged
        if abs(a - anew) < tol:
            break

    return [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi], np.array(fnorm_history)


# =============================================================================
# TSR sweep
# =============================================================================

tsr_performance = {}
results_tsr8    = None
ct_history_tsr8 = None

print(f"{'TSR':<10} | {'CT':<10} | {'CP':<10}")
print("-" * 35)

for TSR in np.arange(6, 11, 1):
    Omega = U0 * TSR / Radius

    temp_results = []
    histories    = []

    for i in range(len(r_R_bins) - 1):
        r_mid = (r_R_bins[i] + r_R_bins[i + 1]) / 2

        chord = 3 * (1 - r_mid) + 1
        twist = -(14 * (1 - r_mid) + Pitch)

        res, f_hist = SolveStreamtube(
            U0, r_R_bins[i], r_R_bins[i + 1],
            RootLocation_R, TipLocation_R,
            Omega, Radius, NBlades,
            chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        temp_results.append(res)
        histories.append(f_hist)

    res_arr = np.array(temp_results)
    dr      = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

    CT = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0 ** 2 * np.pi * Radius ** 2))
    CP = np.sum(
        dr * res_arr[:, 4] * res_arr[:, 2]
        * NBlades * Radius * Omega
        / (0.5 * U0 ** 3 * np.pi * Radius ** 2)
    )

    print(f"{TSR:<10.1f} | {CT:<10.4f} | {CP:<10.4f}")
    tsr_performance[TSR] = {"CT": CT, "CP": CP}

    if TSR == 8:
        results_tsr8 = res_arr

        # Build CT history across iterations by integrating fnorm histories.
        # Each history may be shorter than the others (early exit),
        # so pad each one with its final converged value before stacking.
        max_len = max(len(h) for h in histories)
        hist_padded = np.array([
            np.concatenate([h, np.full(max_len - len(h), h[-1])])
            for h in histories
        ])
        dr_col = dr[:, np.newaxis]
        ct_history_tsr8 = np.sum(
            hist_padded * dr_col * NBlades / (0.5 * U0 ** 2 * np.pi * Radius ** 2),
            axis=0
        )

# =============================================================================
# No-correction run (F = 1 everywhere, TSR=8)
# Uses the same SolveStreamtube with a wrapper that forces F=1.
# =============================================================================

TSR_comp  = 8
Omega_comp = U0 * TSR_comp / Radius


def SolveStreamtube_NoCorrection(
    U0, r1_R, r2_R, rootradius_R, tipradius_R,
    Omega, Radius, NBlades, chord, twist,
    polar_alpha, polar_cl, polar_cd,
    max_iter=300, tol=1e-5
):
    """Identical to SolveStreamtube but Prandtl F is held at 1.0."""
    Area    = np.pi * ((r2_R * Radius) ** 2 - (r1_R * Radius) ** 2)
    r_mid   = (r1_R + r2_R) / 2
    r_local = r_mid * Radius

    a     = 0.0
    aline = 0.0

    for _ in range(max_iter):
        Urotor = U0 * (1 - a)
        Utan   = (1 + aline) * Omega * r_local

        fnorm, ftan, gamma, alpha, phi = LoadBladeElement(
            Urotor, Utan, chord, twist,
            polar_alpha, polar_cl, polar_cd
        )

        CT_loc = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0 ** 2)
        anew   = ainduction(CT_loc)
        # F = 1.0 — no Prandtl correction applied

        a     = 0.75 * a + 0.25 * anew
        aline = (ftan * NBlades) / (
            2 * np.pi * U0 * max(1 - a, 1e-4) * Omega * 2 * r_local ** 2
        )

        if abs(a - anew) < tol:
            break

    return [a, aline, r_mid, fnorm, ftan]


results_tsr8_no_corr = []
for i in range(len(r_R_bins) - 1):
    rm    = (r_R_bins[i] + r_R_bins[i + 1]) / 2
    chord = 3 * (1 - rm) + 1
    twist = -(14 * (1 - rm) + Pitch)
    res_nc = SolveStreamtube_NoCorrection(
        U0, r_R_bins[i], r_R_bins[i + 1],
        RootLocation_R, TipLocation_R,
        Omega_comp, Radius, NBlades,
        chord, twist,
        polar_alpha, polar_cl, polar_cd
    )
    results_tsr8_no_corr.append(res_nc)

res_nc = np.array(results_tsr8_no_corr)

# =============================================================================
# Save / show helper
# =============================================================================

base_path   = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_BEM")
os.makedirs(save_folder, exist_ok=True)


def save_and_show(filename):
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


# =============================================================================
# Plot 1 — Angle of attack and inflow angle (TSR=8)
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 6], "b-", label=r"Angle of attack ($\alpha$)")
plt.plot(results_tsr8[:, 2], results_tsr8[:, 7], "r-", label=r"Inflow angle ($\phi$)")
plt.title("Spanwise distribution of angle of attack and inflow angle (TSR=8)")
plt.xlabel("r/R")
plt.ylabel("Angle [deg]")
plt.grid(True)
plt.legend()
save_and_show("1_Alpha_Phi_Distribution.png")

# =============================================================================
# Plot 2 — Axial and azimuthal induction (TSR=8)
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 0], "b-", label=r"Axial induction ($a$)")
plt.plot(results_tsr8[:, 2], results_tsr8[:, 1], "r-", label=r"Azimuthal induction ($a'$)")
plt.title("Spanwise distribution of axial and azimuthal inductions (TSR=8)")
plt.xlabel("r/R")
plt.ylabel("Induction factor")
plt.grid(True)
plt.legend()
save_and_show("2_Induction_Factors.png")

# =============================================================================
# Plot 3 — Thrust and azimuthal loading (TSR=8)
# =============================================================================

norm_val = 0.5 * U0 ** 2 * Radius
cn = results_tsr8[:, 3] / norm_val
ct = results_tsr8[:, 4] / norm_val

plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], cn, "b-", label=r"Thrust loading ($C_n$)")
plt.plot(results_tsr8[:, 2], ct, "r-", label=r"Azimuthal loading ($C_t$)")
plt.title("Spanwise distribution of thrust and azimuthal loading (TSR=8)")
plt.xlabel("r/R")
plt.ylabel(r"Sectional load coefficient $F\,/\,(\frac{1}{2} U_\infty^2 R)$")
plt.grid(True)
plt.legend()
save_and_show("3_Spanwise_Loading.png")

# =============================================================================
# Plot 4a — Total thrust coefficient (CT) vs TSR
# =============================================================================

tsr_plot_list = sorted(tsr_performance.keys())
ct_plot_list  = [tsr_performance[t]["CT"] for t in tsr_plot_list]
cq_plot_list  = [tsr_performance[t]["CP"] / t for t in tsr_plot_list]

plt.figure(figsize=(9, 5))
plt.plot(tsr_plot_list, ct_plot_list, "bo-", label=r"Total thrust coefficient ($C_T$)")
plt.title("Total thrust coefficient vs. Tip-Speed Ratio")
plt.xlabel("Tip-Speed Ratio (TSR)")
plt.ylabel(r"Thrust coefficient $C_T$")
plt.grid(True)
plt.legend()
save_and_show("4a_Thrust_vs_TSR.png")

# =============================================================================
# Plot 4b — Total torque coefficient (CQ) vs TSR
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(tsr_plot_list, cq_plot_list, "ro-", label=r"Total torque coefficient ($C_Q$)")
plt.title("Total torque coefficient vs. Tip-Speed Ratio")
plt.xlabel("Tip-Speed Ratio (TSR)")
plt.ylabel(r"Torque coefficient $C_Q$")
plt.grid(True)
plt.legend()
save_and_show("4b_Torque_vs_TSR.png")

# =============================================================================
# Plot 5a — BEM convergence history (total rotor CT, TSR=8)
# =============================================================================

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(ct_history_tsr8) + 1), ct_history_tsr8, "b-", linewidth=2)
plt.xlim(1, min(100, len(ct_history_tsr8)))
plt.title(r"Convergence history of total thrust coefficient ($C_T$) (TSR=8)")
plt.xlabel("Iteration step")
plt.ylabel(r"Total $C_T$")
plt.grid(True)
save_and_show("5a_Convergence_History.png")

# =============================================================================
# Plot 5b — BEM convergence residuals (log scale, TSR=8)
# =============================================================================

residuals = np.abs(np.diff(ct_history_tsr8))

plt.figure(figsize=(7, 5))
plt.semilogy(
    range(2, len(ct_history_tsr8) + 1), residuals,
    "r-", linewidth=2,
    label=r"Residual $|C_{T,i} - C_{T,i-1}|$"
)
plt.axhline(1e-5, color="k", linestyle="--", linewidth=0.8, label="Tolerance = 1e-5")
plt.xlim(1, min(100, len(ct_history_tsr8)))
plt.title("Convergence residuals of total thrust coefficient (TSR=8)")
plt.xlabel("Iteration step")
plt.ylabel(r"Absolute difference in $C_T$")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
save_and_show("5b_Convergence_Residuals_Log.png")

# =============================================================================
# Plot 6a — Influence of Prandtl correction on axial induction (TSR=8)
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 0], "b-", label="With correction")
plt.plot(res_nc[:, 2],       res_nc[:, 0],        "r-", label="No correction")
plt.title("Influence of Prandtl tip/root correction on axial induction (TSR=8)")
plt.xlabel("r/R")
plt.ylabel(r"Axial induction factor $a$")
plt.grid(True)
plt.legend()
save_and_show("6a_Induction_Correction_Influence.png")

# =============================================================================
# Plot 6b — Influence of Prandtl correction on normal loading (TSR=8)
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 3] / norm_val, "b-", label="With correction")
plt.plot(res_nc[:, 2],       res_nc[:, 3]        / norm_val, "r-", label="No correction")
plt.title("Influence of Prandtl tip/root correction on normal loading (TSR=8)")
plt.xlabel("r/R")
plt.ylabel(r"$F_{norm}\,/\,(0.5\,\rho\,U_{\infty}^2\,R)$")
plt.grid(True)
plt.legend()
save_and_show("6b_Loading_Correction_Influence.png")

# =============================================================================
# Plot 7 — Influence of number of annuli (TSR=8)
# =============================================================================

N_values = [8, 20, 100]

for N in N_values:
    r_R_bins_N = np.linspace(RootLocation_R, TipLocation_R, N + 1)
    results_N  = []

    for i in range(N):
        rm    = (r_R_bins_N[i] + r_R_bins_N[i + 1]) / 2
        chord = 3 * (1 - rm) + 1
        twist = -(14 * (1 - rm) + Pitch)
        res, _ = SolveStreamtube(
            U0, r_R_bins_N[i], r_R_bins_N[i + 1],
            RootLocation_R, TipLocation_R,
            Omega_comp, Radius, NBlades,
            chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        results_N.append(res)

    res_N_arr = np.array(results_N)

    plt.figure(figsize=(7, 5))
    plt.plot(
        res_N_arr[:, 2],
        res_N_arr[:, 3] / norm_val,
        "-o", markersize=4,
        label=f"Annuli = {N}"
    )
    plt.title(f"Normal loading distribution ($C_n$) with {N} annuli")
    plt.xlabel("r/R")
    plt.ylabel(r"$C_n = F_{norm}\,/\,(0.5\,\rho\,U_{\infty}^2\,R)$")
    plt.grid(True)
    plt.legend()
    save_and_show(f"7_Annuli_Influence_N{N}.png")

# =============================================================================
# Plot 8 — Influence of spacing method (TSR=8, N=40)
# =============================================================================

N_fixed = 40
beta_cos = np.linspace(0, np.pi, N_fixed + 1)

spacing_methods = {
    "Constant": np.linspace(RootLocation_R, TipLocation_R, N_fixed + 1),
    "Cosine":   RootLocation_R + (TipLocation_R - RootLocation_R) * 0.5 * (1 - np.cos(beta_cos)),
}

line_styles = {"Constant": "-",    "Cosine": "--"}
colors      = {"Constant": "blue", "Cosine": "red"}

plt.figure(figsize=(9, 5))

for label, bins in spacing_methods.items():
    results_spacing = []
    for i in range(N_fixed):
        rm    = (bins[i] + bins[i + 1]) / 2
        chord = 3 * (1 - rm) + 1
        twist = -(14 * (1 - rm) + Pitch)
        res, _ = SolveStreamtube(
            U0, bins[i], bins[i + 1],
            RootLocation_R, TipLocation_R,
            Omega_comp, Radius, NBlades,
            chord, twist,
            polar_alpha, polar_cl, polar_cd
        )
        results_spacing.append(res)

    res_S_arr = np.array(results_spacing)

    plt.plot(
        res_S_arr[:, 2],
        res_S_arr[:, 3] / norm_val,
        marker="o",
        linestyle=line_styles[label],
        color=colors[label],
        markersize=5,
        linewidth=2,
        label=f"{label} Spacing",
    )

plt.title("Comparison of spacing methods (N=40)")
plt.xlabel("r/R")
plt.ylabel(r"$C_n = F_{norm}\,/\,(0.5\,\rho\,U_{\infty}^2\,R)$")
plt.grid(True)
plt.legend()
plt.xlim(0.85, 1.01)
plt.ylim(0.5, 1.5)
save_and_show("8_Spacing_Method_Comparison.png")