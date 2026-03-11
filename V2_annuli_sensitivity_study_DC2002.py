import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Induction and correction functions
# ============================================================

def ainduction(CT):
    """
    Compute axial induction factor a from thrust coefficient CT,
    including Glauert correction.
    """
    CT1 = 1.816
    CT2 = 2.0 * np.sqrt(CT1) - CT1

    if np.isscalar(CT):
        if CT >= CT2:
            return 1.0 + (CT - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
        return 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT))
    else:
        a = np.zeros(np.shape(CT))
        mask_high = CT >= CT2
        mask_low = ~mask_high
        a[mask_high] = 1.0 + (CT[mask_high] - CT1) / (4.0 * (np.sqrt(CT1) - 1.0))
        a[mask_low] = 0.5 - 0.5 * np.sqrt(np.maximum(0.0, 1.0 - CT[mask_low]))
        return a


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    """
    Compute combined Prandtl tip and root loss factor.
    """
    # Avoid singularities
    r_R = np.clip(r_R, rootradius_R + 1e-6, tipradius_R - 1e-6)
    ai = np.clip(axial_induction, -0.2, 0.95)

    sqrt_term = np.sqrt(1.0 + ((TSR * r_R) ** 2) / ((1.0 - ai) ** 2))

    temp_tip = -NBlades / 2.0 * (tipradius_R - r_R) / r_R * sqrt_term
    temp_root =  NBlades / 2.0 * (rootradius_R - r_R) / r_R * sqrt_term

    exp_tip = np.clip(np.exp(temp_tip), 0.0, 1.0)
    exp_root = np.clip(np.exp(temp_root), 0.0, 1.0)

    Ftip = 2.0 / np.pi * np.arccos(exp_tip)
    Froot = 2.0 / np.pi * np.arccos(exp_root)

    F = Froot * Ftip

    return float(F)


def loadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Compute 2D blade-element forces from local inflow.
    """
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm, vtan)

    alpha = twist + np.degrees(inflowangle)

    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)

    lift = 0.5 * vmag2 * cl * chord
    drag = 0.5 * vmag2 * cd * chord

    fnorm = lift * np.cos(inflowangle) + drag * np.sin(inflowangle)
    ftan = lift * np.sin(inflowangle) - drag * np.cos(inflowangle)
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord

    return fnorm, ftan, gamma, alpha, np.degrees(inflowangle)


# ============================================================
# 2. Stable BEM annulus solver
# ============================================================

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
    twist,
    polar_alpha,
    polar_cl,
    polar_cd
):
    """
    Solve one annulus using a more numerically robust BEM iteration.
    """
    Area = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_R = 0.5 * (r1_R + r2_R)

    # Initial guesses
    a = 0.2
    aline = 0.01

    TSR = Omega * Radius / Uinf

    for _ in range(300):
        # Clip induction factors for stability
        a = np.clip(a, -0.2, 0.95)
        aline = np.clip(aline, -0.5, 1.0)

        Urotor = Uinf * (1.0 - a)
        Utan = (1.0 + aline) * Omega * r_R * Radius

        fnorm, ftan, gamma, alpha, phi = loadBladeElement(
            Urotor, Utan, chord, twist, polar_alpha, polar_cl, polar_cd
        )

        # Annulus axial load
        load3Daxial = fnorm * Radius * (r2_R - r1_R) * NBlades
        CT = load3Daxial / (0.5 * Area * Uinf**2)

        # Prevent unphysical extremes before inversion
        CT = np.clip(CT, -2.0, 3.0)

        anew = ainduction(CT)

        # Prandtl correction
        F = PrandtlTipRootCorrection(
            r_R, rootradius_R, tipradius_R, TSR, NBlades, anew
        )

        # IMPORTANT: use a less aggressive lower bound
        F = max(F, 0.05)

        anew = anew / F
        anew = np.clip(anew, -0.2, 0.95)

        # Tangential induction update
        denom = 2.0 * np.pi * Uinf * (1.0 - a) * Omega * 2.0 * (r_R * Radius)**2
        denom = max(abs(denom), 1e-8) * np.sign(denom if denom != 0 else 1.0)

        aline_new = ftan * NBlades / denom
        aline_new = aline_new / F
        aline_new = np.clip(aline_new, -0.5, 1.0)

        # Under-relax both a and a'
        a_next = 0.8 * a + 0.2 * anew
        aline_next = 0.8 * aline + 0.2 * aline_new

        if abs(a_next - a) < 1e-5 and abs(aline_next - aline) < 1e-5:
            a = a_next
            aline = aline_next
            break

        a = a_next
        aline = aline_next

    return [a, aline, r_R, fnorm, ftan, gamma, alpha, phi]


# ============================================================
# 3. Annulus distributions
# ============================================================

def constant_annuli(root_R, tip_R, n_annuli):
    """
    Uniform annulus spacing.
    """
    return np.linspace(root_R, tip_R, n_annuli + 1)


def cosine_annuli(root_R, tip_R, n_annuli, soften=True):
    """
    Cosine-clustered annulus spacing.

    If soften=True, the spacing is slightly regularized to avoid
    extremely tiny first/last annuli that can destabilize the solver.
    """
    theta = np.linspace(0.0, np.pi, n_annuli + 1)
    s = 0.5 * (1.0 - np.cos(theta))

    if soften:
        # Blend a little with uniform spacing for stability
        s_uniform = np.linspace(0.0, 1.0, n_annuli + 1)
        s = 0.85 * s + 0.15 * s_uniform

    edges = root_R + (tip_R - root_R) * s
    return edges


# ============================================================
# 4. Blade geometry
# ============================================================

def chord_distribution(r_mid):
    """
    Baseline chord distribution.
    """
    return 3.0 * (1.0 - r_mid) + 1.0


def twist_input_distribution(r_mid, Pitch):
    """
    Baseline twist-input distribution matching your current sign convention.
    """
    return -(14.0 * (1.0 - r_mid) + Pitch)


# ============================================================
# 5. One BEM run
# ============================================================

def run_bem_case(
    TSR,
    annulus_edges,
    polar_alpha,
    polar_cl,
    polar_cd,
    Radius=50.0,
    NBlades=3,
    Uinf=10.0,
    RootLocation_R=0.2,
    TipLocation_R=1.0,
    Pitch=-2.0
):
    """
    Run one BEM calculation for a given annulus discretization.
    """
    Omega = Uinf * TSR / Radius
    results = []

    for i in range(len(annulus_edges) - 1):
        r1 = annulus_edges[i]
        r2 = annulus_edges[i + 1]
        r_mid = 0.5 * (r1 + r2)

        chord = chord_distribution(r_mid)
        twist_input = twist_input_distribution(r_mid, Pitch)

        res = solveStreamtube(
            Uinf=Uinf,
            r1_R=r1,
            r2_R=r2,
            rootradius_R=RootLocation_R,
            tipradius_R=TipLocation_R,
            Omega=Omega,
            Radius=Radius,
            NBlades=NBlades,
            chord=chord,
            twist=twist_input,
            polar_alpha=polar_alpha,
            polar_cl=polar_cl,
            polar_cd=polar_cd
        )
        results.append(res)

    results = np.array(results)

    dr = (annulus_edges[1:] - annulus_edges[:-1]) * Radius

    CT = np.sum(
        dr * results[:, 3] * NBlades / (0.5 * Uinf**2 * np.pi * Radius**2)
    )

    CP = np.sum(
        dr
        * results[:, 4]
        * results[:, 2]
        * NBlades
        * Radius
        * Omega
        / (0.5 * Uinf**3 * np.pi * Radius**2)
    )

    CQ = CP / TSR

    return {
        "TSR": TSR,
        "Omega": Omega,
        "annulus_edges": annulus_edges,
        "results": results,
        "CT": CT,
        "CP": CP,
        "CQ": CQ,
        "dr": dr,
    }


# ============================================================
# 6. Sensitivity study
# ============================================================

def perform_sensitivity_study(
    annuli_list,
    distributions,
    TSR_values,
    polar_alpha,
    polar_cl,
    polar_cd,
    Radius=50.0,
    NBlades=3,
    Uinf=10.0,
    RootLocation_R=0.2,
    TipLocation_R=1.0,
    Pitch=-2.0
):
    """
    Perform sensitivity study for annulus count and spacing distribution.
    """
    rows = []
    case_results = {}

    for dist_name, dist_fun in distributions.items():
        for n_annuli in annuli_list:
            annulus_edges = dist_fun(RootLocation_R, TipLocation_R, n_annuli)

            for TSR in TSR_values:
                case_key = (dist_name, n_annuli, TSR)

                out = run_bem_case(
                    TSR=TSR,
                    annulus_edges=annulus_edges,
                    polar_alpha=polar_alpha,
                    polar_cl=polar_cl,
                    polar_cd=polar_cd,
                    Radius=Radius,
                    NBlades=NBlades,
                    Uinf=Uinf,
                    RootLocation_R=RootLocation_R,
                    TipLocation_R=TipLocation_R,
                    Pitch=Pitch
                )

                case_results[case_key] = out

                rows.append({
                    "distribution": dist_name,
                    "n_annuli": n_annuli,
                    "TSR": TSR,
                    "CT": out["CT"],
                    "CP": out["CP"],
                    "CQ": out["CQ"],
                })

    summary_df = pd.DataFrame(rows)
    return summary_df, case_results


# ============================================================
# 7. Plotting
# ============================================================

def plot_annulus_boundaries(distributions, annuli_to_show, RootLocation_R=0.2, TipLocation_R=1.0):
    plt.figure(figsize=(11, 4))
    y_levels = np.arange(len(distributions))[::-1] + 1

    for idx, (dist_name, dist_fun) in enumerate(distributions.items()):
        edges = dist_fun(RootLocation_R, TipLocation_R, annuli_to_show)
        y = np.full_like(edges, y_levels[idx], dtype=float)
        plt.plot(edges, y, "o-", label=f"{dist_name} ({annuli_to_show} annuli)")

    plt.yticks(y_levels, list(distributions.keys()))
    plt.xlabel("r/R")
    plt.title("Comparison of annulus boundary distributions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cp_ct_vs_annuli(summary_df, TSR_values):
    for TSR in TSR_values:
        df_tsr = summary_df[summary_df["TSR"] == TSR].copy()

        plt.figure(figsize=(10, 5))
        for dist_name in df_tsr["distribution"].unique():
            sub = df_tsr[df_tsr["distribution"] == dist_name].sort_values("n_annuli")
            plt.plot(sub["n_annuli"], sub["CP"], "o-", label=dist_name)
        plt.xlabel("Number of annuli")
        plt.ylabel("CP")
        plt.title(f"CP vs number of annuli (TSR={TSR})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        for dist_name in df_tsr["distribution"].unique():
            sub = df_tsr[df_tsr["distribution"] == dist_name].sort_values("n_annuli")
            plt.plot(sub["n_annuli"], sub["CT"], "o-", label=dist_name)
        plt.xlabel("Number of annuli")
        plt.ylabel("CT")
        plt.title(f"CT vs number of annuli (TSR={TSR})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_spanwise_comparison(case_results, TSR_target, n_annuli_target, distributions, Uinf, Radius, NBlades):
    plt.figure(figsize=(10, 5))
    for dist_name in distributions.keys():
        res = case_results[(dist_name, n_annuli_target, TSR_target)]["results"]
        plt.plot(res[:, 2], res[:, 0], label=f"{dist_name}: a")
        plt.plot(res[:, 2], res[:, 1], "--", label=f"{dist_name}: a'")
    plt.xlabel("r/R")
    plt.ylabel("Induction factor")
    plt.title(f"Induction comparison (TSR={TSR_target}, N={n_annuli_target})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for dist_name in distributions.keys():
        res = case_results[(dist_name, n_annuli_target, TSR_target)]["results"]
        plt.plot(res[:, 2], res[:, 3] / (0.5 * Uinf**2 * Radius), label=f"{dist_name}: Fnorm")
        plt.plot(res[:, 2], res[:, 4] / (0.5 * Uinf**2 * Radius), "--", label=f"{dist_name}: Ftan")
    plt.xlabel("r/R")
    plt.ylabel(r"$F / (\frac{1}{2} U_\infty^2 R)$")
    plt.title(f"Force comparison (TSR={TSR_target}, N={n_annuli_target})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for dist_name in distributions.keys():
        out = case_results[(dist_name, n_annuli_target, TSR_target)]
        res = out["results"]
        Omega = out["Omega"]
        gamma_nd = res[:, 5] / (np.pi * Uinf**2 / (NBlades * Omega))
        plt.plot(res[:, 2], gamma_nd, label=dist_name)
    plt.xlabel("r/R")
    plt.ylabel(r"$\Gamma / \left(\pi U_\infty^2/(N_{blades}\Omega)\right)$")
    plt.title(f"Circulation comparison (TSR={TSR_target}, N={n_annuli_target})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for dist_name in distributions.keys():
        res = case_results[(dist_name, n_annuli_target, TSR_target)]["results"]
        plt.plot(res[:, 2], res[:, 6], label=f"{dist_name}: alpha")
        plt.plot(res[:, 2], res[:, 7], "--", label=f"{dist_name}: phi")
    plt.xlabel("r/R")
    plt.ylabel("Angle [deg]")
    plt.title(f"Alpha / phi comparison (TSR={TSR_target}, N={n_annuli_target})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":
    # Load polar data
    data = pd.read_excel("polar DU95W180 (3).xlsx", skiprows=3)
    polar_alpha = data["Alfa"].to_numpy()
    polar_cl = data["Cl"].to_numpy()
    polar_cd = data["Cd"].to_numpy()

    # Specs
    Radius = 50.0
    NBlades = 3
    Uinf = 10.0
    RootLocation_R = 0.2
    TipLocation_R = 1.0
    Pitch = -2.0

    # Sensitivity settings
    TSR_values = [6, 8, 10]
    annuli_list = [10, 15, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 280, 320]

    distributions = {
        "constant": constant_annuli,
        "cosine": cosine_annuli,
    }

    # Run study
    summary_df, case_results = perform_sensitivity_study(
        annuli_list=annuli_list,
        distributions=distributions,
        TSR_values=TSR_values,
        polar_alpha=polar_alpha,
        polar_cl=polar_cl,
        polar_cd=polar_cd,
        Radius=Radius,
        NBlades=NBlades,
        Uinf=Uinf,
        RootLocation_R=RootLocation_R,
        TipLocation_R=TipLocation_R,
        Pitch=Pitch
    )

    print("\nSensitivity study summary:")
    print(summary_df.sort_values(["TSR", "distribution", "n_annuli"]).to_string(index=False))

    plot_annulus_boundaries(
        distributions=distributions,
        annuli_to_show=40,
        RootLocation_R=RootLocation_R,
        TipLocation_R=TipLocation_R
    )

    plot_cp_ct_vs_annuli(summary_df, TSR_values)

    plot_spanwise_comparison(
        case_results=case_results,
        TSR_target=8,
        n_annuli_target=80,
        distributions=distributions,
        Uinf=Uinf,
        Radius=Radius,
        NBlades=NBlades
    )