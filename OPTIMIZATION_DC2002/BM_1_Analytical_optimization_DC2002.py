import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import os
import sys
import io

# =============================================================================
# 1. AERODYNAMIC DATA
# =============================================================================

def load_polar(filename):
    """Load polar data and return Cl/Cd interpolators."""
    df = pd.read_excel(filename, skiprows=3)
    df.columns = [str(c).strip() for c in df.columns]
    cl_func = interp1d(df["Alfa"], df["Cl"], kind="linear", fill_value="extrapolate")
    cd_func = interp1d(df["Alfa"], df["Cd"], kind="linear", fill_value="extrapolate")
    return cl_func, cd_func


def get_coefficients(alpha_rad, cl_func, cd_func):
    """Return Cl, Cd for alpha in radians."""
    alpha_deg = np.degrees(alpha_rad)
    return float(cl_func(alpha_deg)), float(cd_func(alpha_deg))


def find_optimal_alpha(cl_func, cd_func, alpha_min_deg=-5.0, alpha_max_deg=20.0, npts=500):
    """Find alpha at maximum Cl/Cd from the polar."""
    alphas_deg = np.linspace(alpha_min_deg, alpha_max_deg, npts)
    cl_vals = cl_func(alphas_deg)
    cd_vals = cd_func(alphas_deg)

    ld = cl_vals / np.maximum(cd_vals, 1e-8)
    idx = int(np.argmax(ld))

    alpha_opt_deg = float(alphas_deg[idx])
    cl_opt        = float(cl_vals[idx])
    cd_opt        = float(cd_vals[idx])

    return alpha_opt_deg, cl_opt, cd_opt


# =============================================================================
# 2. GLOBAL SETTINGS
# =============================================================================

R        = 50.0
R_root   = 10.0
B        = 3

V0       = 10.0
rho      = 1.225
A_disk   = np.pi * R**2

TSR_design = 8.0
CT_target  = 0.75

MAX_CHORD    = 3.4
MIN_CHORD_TIP = 0.3

polar_path = r"polar DU95W180 (3).xlsx"


# =============================================================================
# 3. PRANDTL CORRECTION  (file 4 signature and formula — unchanged)
# =============================================================================

def PrandtlTipRootCorrection(r_R, rootradius_R, TSR, B, a):
    """
    Combined Prandtl tip and root correction factor.
    Returns (F, Ftip, Froot).
    """
    a = np.clip(a, -0.9, 0.99)

    d = 2 * np.pi / B * (1 - a) / np.sqrt(TSR**2 + (1 - a)**2)
    d = max(d, 1e-8)

    Ftip  = 2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (1 - r_R) / d, -500, 0)))
    Froot = 2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (r_R - rootradius_R) / d, -500, 0)))

    Ftip  = float(np.where(np.isnan(Ftip),  0.0, Ftip))
    Froot = float(np.where(np.isnan(Froot), 0.0, Froot))

    return Froot * Ftip, Ftip, Froot


# =============================================================================
# 4. BEM SOLVER  (file 4 — unchanged)
# =============================================================================

def run_bem_solver(plotting, r_arr, c_arr, twist_arr, tsr_range=[8.0], plotname=""):
    """
    Run the BEM solver on a given blade geometry.

    Parameters
    ----------
    r_arr      : radial node locations [m]
    c_arr      : chord at nodes [m]
    twist_arr  : blade angle beta at nodes [rad]
    """
    cl_func, cd_func = load_polar(polar_path)

    dr_arr    = np.diff(r_arr)
    r_mid_arr = r_arr[:-1] + dr_arr / 2.0

    c_mid_arr     = 0.5 * (c_arr[:-1]     + c_arr[1:])
    twist_mid_arr = 0.5 * (twist_arr[:-1] + twist_arr[1:])

    TSR_results = []
    Ct_results  = []
    Cp_results  = []

    alpha_span_all = {}
    phi_span_all   = {}
    a_span_all     = {}
    ap_span_all    = {}
    dTdr_span_all  = {}
    dQdr_span_all  = {}
    F_span_all     = {}
    residual_records = []

    relaxation_factor = 0.25
    max_iter = 300
    tol      = 1e-5

    print("Starting Wind Turbine BEM Analysis...")

    ap_out_for_first_tsr = None

    for TSR in tsr_range:
        omega = (TSR * V0) / R

        a_arr  = np.zeros(len(r_mid_arr))
        ap_arr = np.zeros(len(r_mid_arr))

        alpha_span = np.zeros(len(r_mid_arr))
        phi_span   = np.zeros(len(r_mid_arr))
        dTdr_arr   = np.zeros(len(r_mid_arr))
        dQdr_arr   = np.zeros(len(r_mid_arr))
        F_arr      = np.zeros(len(r_mid_arr))

        TSR_key  = round(TSR, 3)
        Ct_prev  = 0.0
        converged = False

        for iter_count in range(max_iter):
            anew_arr  = a_arr.copy()
            apnew_arr = ap_arr.copy()
            Fnew_arr  = F_arr.copy()

            for i in range(len(r_mid_arr)):
                r    = r_mid_arr[i]
                c    = c_mid_arr[i]
                beta = twist_mid_arr[i]
                a    = a_arr[i]
                ap   = ap_arr[i]

                sigma = (B * c) / (2 * np.pi * r)

                V_axial      = V0 * (1 - a)
                V_tangential = omega * r * (1 + ap)
                V_rel        = np.sqrt(V_axial**2 + V_tangential**2)

                phi = np.arctan2(V_axial, V_tangential)
                phi = max(phi, 1e-6)

                alpha = phi - beta
                cl, cd = get_coefficients(alpha, cl_func, cd_func)

                Cn     = cl * np.cos(phi) + cd * np.sin(phi)
                Ct_aero = cl * np.sin(phi) - cd * np.cos(phi)

                F, _, _ = PrandtlTipRootCorrection(r / R, R_root / R, TSR, B, a)
                F = max(F, 1e-4)
                Fnew_arr[i] = F

                CT_loc = (sigma * (1 - a)**2 * Cn) / (np.sin(phi)**2)

                CT_switch = 2 * np.sqrt(1.816) - 1.816
                if CT_loc < CT_switch:
                    a_new = 0.5 - 0.5 * np.sqrt(max(0.0, 1.0 - CT_loc))
                else:
                    a_new = 1 + (CT_loc - 1.816) / (4 * np.sqrt(1.816) - 4)

                denom = ((4 * np.sin(phi) * np.cos(phi)) / max(sigma * Ct_aero, 1e-8)) - 1
                if abs(denom) < 1e-8:
                    ap_new = ap
                else:
                    ap_new = 1.0 / denom

                a_new  = np.clip(a_new  / F, -0.5, 0.95)
                ap_new = np.clip(ap_new / F, -0.5, 0.95)

                anew_arr[i]  = (1 - relaxation_factor) * a  + relaxation_factor * a_new
                apnew_arr[i] = (1 - relaxation_factor) * ap + relaxation_factor * ap_new

                alpha_span[i] = alpha
                phi_span[i]   = phi

                dTdr_arr[i] = 0.5 * rho * V_rel**2 * B * c * Cn
                dQdr_arr[i] = 0.5 * rho * V_rel**2 * B * c * Ct_aero * r

            Ct_iter    = np.sum(dTdr_arr * dr_arr) / (0.5 * rho * A_disk * V0**2)
            Ct_residual = abs(Ct_iter - Ct_prev)

            residual_records.append({
                "TSR":         TSR_key,
                "iteration":   iter_count,
                "Ct_residual": Ct_residual,
            })

            Ct_prev = Ct_iter
            a_arr   = anew_arr
            ap_arr  = apnew_arr
            F_arr   = Fnew_arr

            if iter_count > 0 and Ct_residual < tol:
                print(f"TSR={TSR:.1f}: converged at iteration {iter_count:3d}, Ct residual = {Ct_residual:.3e}")
                converged = True
                break

        if not converged:
            print(
                f"Warning: TSR={TSR:.1f} did not converge in {max_iter} iterations, "
                f"final Ct residual = {Ct_residual:.3e}"
            )

        current_thrust = np.sum(dTdr_arr * dr_arr)
        current_torque = np.sum(dQdr_arr * dr_arr)
        current_power  = current_torque * omega

        Ct = current_thrust / (0.5 * rho * A_disk * V0**2)
        Cp = current_power  / (0.5 * rho * A_disk * V0**3)

        TSR_results.append(TSR)
        Ct_results.append(Ct)
        Cp_results.append(Cp)

        alpha_span_all[TSR] = np.degrees(alpha_span)
        phi_span_all[TSR]   = np.degrees(phi_span)
        a_span_all[TSR]     = a_arr.copy()
        ap_span_all[TSR]    = ap_arr.copy()
        F_span_all[TSR]     = F_arr.copy()
        dTdr_span_all[TSR]  = dTdr_arr.copy()
        dQdr_span_all[TSR]  = dQdr_arr.copy()

        if TSR == tsr_range[0]:
            ap_out_for_first_tsr = ap_arr.copy()

        print(f"TSR={TSR:.1f} | Cp = {Cp:.4f} | Ct = {Ct:.4f}")

    if plotting:
        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        os.makedirs(img_dir, exist_ok=True)

        r_R = r_mid_arr / R

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tsr in alpha_span_all:
            axes[0].plot(r_R, alpha_span_all[tsr], label=rf"$\lambda={tsr}$")
            axes[1].plot(r_R, phi_span_all[tsr],   label=rf"$\lambda={tsr}$")
        axes[0].set_title("Angle of Attack")
        axes[0].set_xlabel(r"$r/R$")
        axes[0].set_ylabel(r"$\alpha$ [deg]")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].set_title("Inflow Angle")
        axes[1].set_xlabel(r"$r/R$")
        axes[1].set_ylabel(r"$\phi$ [deg]")
        axes[1].grid(True)
        axes[1].legend()
        fig.suptitle(f"Spanwise Alpha and Phi - {plotname}")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"plot_alpha_phi_{plotname}.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tsr in a_span_all:
            axes[0].plot(r_R, a_span_all[tsr],  label=rf"$\lambda={tsr}$")
            axes[1].plot(r_R, ap_span_all[tsr], label=rf"$\lambda={tsr}$")
        axes[0].set_title("Axial induction")
        axes[0].set_xlabel(r"$r/R$")
        axes[0].set_ylabel(r"$a$")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].set_title("Tangential induction")
        axes[1].set_xlabel(r"$r/R$")
        axes[1].set_ylabel(r"$a'$")
        axes[1].grid(True)
        axes[1].legend()
        fig.suptitle(f"Spanwise inductions - {plotname}")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"plot_inductions_{plotname}.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tsr in dTdr_span_all:
            axes[0].plot(r_R, dTdr_span_all[tsr], label=rf"$\lambda={tsr}$")
            axes[1].plot(r_R, dQdr_span_all[tsr], label=rf"$\lambda={tsr}$")
        axes[0].set_title(r"$dT/dr$")
        axes[0].set_xlabel(r"$r/R$")
        axes[0].set_ylabel(r"$dT/dr$ [N/m]")
        axes[0].grid(True)
        axes[0].legend()
        axes[1].set_title(r"$dQ/dr$")
        axes[1].set_xlabel(r"$r/R$")
        axes[1].set_ylabel(r"$dQ/dr$ [N]")
        axes[1].grid(True)
        axes[1].legend()
        fig.suptitle(f"Spanwise loading - {plotname}")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"plot_loading_{plotname}.png"), dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(TSR_results, Cp_results, "-o")
        axes[0].set_title(r"$C_P$ vs $\lambda$")
        axes[0].set_xlabel(r"$\lambda$")
        axes[0].set_ylabel(r"$C_P$")
        axes[0].grid(True)
        axes[1].plot(TSR_results, Ct_results, "-s")
        axes[1].set_title(r"$C_T$ vs $\lambda$")
        axes[1].set_xlabel(r"$\lambda$")
        axes[1].set_ylabel(r"$C_T$")
        axes[1].grid(True)
        fig.suptitle(f"Performance - {plotname}")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"plot_performance_{plotname}.png"), dpi=150)
        plt.close(fig)

        df_residuals = pd.DataFrame(residual_records)
        fig, ax = plt.subplots(figsize=(8, 5))
        for TSR_k, grp in df_residuals.groupby("TSR"):
            ax.semilogy(grp["iteration"], grp["Ct_residual"], "-o", markersize=3, label=rf"$\lambda={TSR_k}$")
        ax.axhline(tol, color="k", linestyle="--", linewidth=0.8, label=f"tol = {tol:.0e}")
        ax.set_title(r"Convergence history of $C_T$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|\Delta C_T|$")
        ax.grid(True, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"plot_convergence_{plotname}.png"), dpi=150)
        plt.close(fig)

        P_static = 0.0
        P01_arr  = (P_static + 0.5 * rho * V0**2) * np.ones_like(r_mid_arr)
        fig, ax  = plt.subplots(figsize=(8, 5))
        ax.plot(r_R, P01_arr, label=r"$P_0^1, P_0^2$")
        for tsr in tsr_range:
            dP0_arr = (2 * rho * V0**2 * a_span_all[tsr] * F_span_all[tsr]
                       * (1 - a_span_all[tsr] * F_span_all[tsr]))
            P03_arr = P01_arr - dP0_arr
            ax.plot(r_R, P03_arr, label=rf"$P_0^3, P_0^4$ for $\lambda={tsr}$")
        ax.set_title(f"Stagnation pressure - {plotname}")
        ax.set_xlabel(r"$r/R$")
        ax.set_ylabel(r"$P_0$")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"plot_stagnation_pressure_{plotname}.png"), dpi=150)
        plt.close(fig)

        print(f"All plots saved to: {img_dir}")

    return Ct_results, Cp_results, ap_out_for_first_tsr


# =============================================================================
# 5. ANALYTICAL ROTOR DESIGN
# =============================================================================

def generate_ideal_rotor(target_a=0.25, n_annuli=100, ap_in=None):
    """
    Analytically generate chord and twist distributions for a given uniform
    axial induction `target_a`, operating at maximum Cl/Cd everywhere.

    The chord is derived by inverting the BEM CT momentum equation:
        CT_loc = sigma*(1-a)^2*Cn / sin^2(phi)  =  4*a*F*(1-a*F)
    Solving for chord:
        c = 8*pi*r * a*F*(1-a*F) * sin^2(phi) / (B * (1-a)^2 * Cn)

    The blade angle beta = phi - alpha_opt so the section always sits
    at maximum Cl/Cd.

    Parameters
    ----------
    target_a  : design axial induction factor
    n_annuli  : number of spanwise nodes
    ap_in     : tangential induction array from a previous BEM run (on annulus
                midpoints of the current node array). If given, refines phi.

    Returns
    -------
    r_nodes, c_nodes, beta_nodes   (all length n_annuli)
        beta_nodes in radians, consistent with twist_arr convention in run_bem_solver.
    """
    print("\n--- Analytically Designing Ideal Rotor ---")

    cl_func, cd_func = load_polar(polar_path)
    alpha_opt_deg, cl_opt, cd_opt = find_optimal_alpha(cl_func, cd_func)
    alpha_opt_rad = np.radians(alpha_opt_deg)

    print(
        f"Optimal airfoil point: alpha = {alpha_opt_deg:.2f} deg, "
        f"Cl = {cl_opt:.3f}, Cd = {cd_opt:.4f}, "
        f"Cl/Cd = {cl_opt / cd_opt:.1f}"
    )

    TSR = TSR_design
    a   = target_a

    # Cosine-spaced nodes — denser near root and tip where geometry varies most
    r_start = R_root + 0.005 * R
    r_end   = R      - 0.005 * R
    r_nodes = r_start + (r_end - r_start) * (
        1 - np.cos(np.linspace(0, np.pi, n_annuli))
    ) / 2

    c_nodes    = np.zeros_like(r_nodes)
    beta_nodes = np.zeros_like(r_nodes)

    # Interpolate previous ap onto the current node positions.
    # ap_in lives on annulus midpoints of the same node array from the last call.
    if ap_in is not None:
        r_centers   = r_nodes[:-1] + np.diff(r_nodes) / 2
        ap_at_nodes = np.interp(r_nodes, r_centers, ap_in,
                                left=ap_in[0], right=ap_in[-1])
    else:
        ap_at_nodes = np.zeros_like(r_nodes)

    for i, r in enumerate(r_nodes):
        r_R       = r / R
        local_tsr = TSR * r_R

        # Prandtl correction at design condition using file 4's formula
        F, _, _ = PrandtlTipRootCorrection(r_R, R_root / R, TSR, B, a)
        F = max(F, 1e-4)

        ap = float(ap_at_nodes[i])

        # Ideal inflow angle, accounting for tangential induction if available
        phi = np.arctan2((1 - a), local_tsr * (1 + ap))

        # Blade angle so the section operates at alpha_opt
        # alpha = phi - beta  =>  beta = phi - alpha_opt
        beta_nodes[i] = phi - alpha_opt_rad

        # Normal force coefficient at this operating point
        Cn = cl_opt * np.cos(phi) + cd_opt * np.sin(phi)

        # Ideal chord from inverting the BEM sigma-based CT formula:
        #   sigma*(1-a)^2*Cn / sin^2(phi) = 4*a*F*(1-a*F)
        #   sigma = B*c / (2*pi*r)
        #   => c = 8*pi*r * a*F*(1-a*F) * sin^2(phi) / (B*(1-a)^2*Cn)
        numerator   = 8 * np.pi * r * a * F * (1 - a * F) * (np.sin(phi) ** 2)
        denominator = B * (1 - a) ** 2 * max(Cn, 1e-8)
        c_nodes[i]  = numerator / denominator

    # --- Physical constraints (same logic as file 3 / 4) ---

    # 1. Cap at structural maximum chord
    c_nodes = np.clip(c_nodes, 0.0, MAX_CHORD)

    # 2. Root structural fix: everything inboard of the chord maximum is held
    #    constant at that maximum (hub attachment requires a flat root section)
    max_idx = int(np.argmax(c_nodes))
    c_nodes[: max_idx + 1] = c_nodes[max_idx]

    # 3. Minimum manufacturing chord at the tip
    c_nodes = np.clip(c_nodes, MIN_CHORD_TIP, None)

    mid_idx = len(r_nodes) // 2
    print("\nGenerated geometry:")
    print(f"Root: Chord = {c_nodes[0]:.3f} m | Beta = {np.degrees(beta_nodes[0]):.2f} deg")
    print(f"Mid : Chord = {c_nodes[mid_idx]:.3f} m | Beta = {np.degrees(beta_nodes[mid_idx]):.2f} deg")
    print(f"Tip : Chord = {c_nodes[-1]:.3f} m | Beta = {np.degrees(beta_nodes[-1]):.2f} deg")

    return r_nodes, c_nodes, beta_nodes


def design_for_exact_ct():
    """
    Use Brent's root-finding method to find the uniform design axial induction
    `a` such that the analytically designed rotor achieves exactly CT_target
    when evaluated by the full BEM solver.

    The tangential induction profile `ap` from each BEM run is fed back into
    the geometry generator to refine the inflow angle phi, so the design
    converges to a self-consistent solution.

    Returns
    -------
    r_final, c_final, beta_final : final blade geometry arrays
    best_a                       : the found design induction
    """
    print("\n" + "=" * 65)
    print(f"ROOT-FINDING: searching for design 'a' such that BEM CT = {CT_target}")
    print("=" * 65)

    last_ap = [None]   # list so the closure can write back to it

    def residual(target_a):
        r_opt, c_opt, beta_opt = generate_ideal_rotor(
            target_a=target_a, n_annuli=100, ap_in=last_ap[0]
        )

        # Silence the BEM prints during root-finding
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            Ct_res, Cp_res, ap_res = run_bem_solver(
                plotting=False,
                tsr_range=[TSR_design],
                r_arr=r_opt,
                c_arr=c_opt,
                twist_arr=beta_opt,
                plotname="",
            )
            ct_val = Ct_res[0]
            cp_val = Cp_res[0]
            last_ap[0] = ap_res
        except Exception:
            ct_val = 0.0
            cp_val = 0.0
        sys.stdout = old_stdout

        print(f"  design a = {target_a:.5f}  ->  BEM CT = {ct_val:.5f},  CP = {cp_val:.5f}")
        return ct_val - CT_target

    # Bracket: lightly loaded (a~0.25) gives CT near Betz; CT=0.75 sits
    # in the range [0.25, 0.33] for typical designs.
    sol = root_scalar(residual, bracket=[0.25, 0.33], method="brentq", xtol=1e-5)

    if not sol.converged:
        raise RuntimeError(
            "root_scalar did not converge — try widening the bracket [0.25, 0.33]."
        )

    best_a = sol.root
    print(f"\nConverged: design induction a = {best_a:.6f}")

    # Re-generate final geometry with the converged ap profile
    r_final, c_final, beta_final = generate_ideal_rotor(
        target_a=best_a, n_annuli=100, ap_in=last_ap[0]
    )

    # Final BEM verification (printed, not silenced)
    print("\n--- Final BEM verification ---")
    Ct_final, Cp_final, _ = run_bem_solver(
        plotting=False,
        tsr_range=[TSR_design],
        r_arr=r_final,
        c_arr=c_final,
        twist_arr=beta_final,
        plotname="",
    )

    print(f"Final CT = {Ct_final[0]:.6f}  (target {CT_target})")
    print(f"Final CP = {Cp_final[0]:.6f}")

    return r_final, c_final, beta_final, best_a


# =============================================================================
# 6. BASELINE GEOMETRY
# =============================================================================

def get_geometry_blade(spacing_method="constant", num_annuli=100):
    """Original assignment blade geometry."""
    r_start = R_root
    r_end   = R

    if spacing_method == "constant":
        r_nodes = np.linspace(r_start, r_end, num_annuli)
    elif spacing_method == "cosine":
        r_nodes = r_start + (r_end - r_start) * (
            1 - np.cos(np.linspace(0, np.pi, num_annuli))
        ) / 2
    else:
        raise ValueError("spacing_method must be 'constant' or 'cosine'")

    c_nodes   = 3.0 * (1 - (r_nodes / R)) + 1.0
    twist_deg = 14.0 * (1 - (r_nodes / R))
    pitch_deg = -2.0
    beta_rad  = np.radians(twist_deg + pitch_deg)

    return r_nodes, c_nodes, beta_rad


# =============================================================================
# 7. PLOTTING HELPERS
# =============================================================================

def plot_geometry(r, c, beta, title_suffix=""):
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(img_dir, exist_ok=True)

    r_R = r / R

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.set_xlabel(r"Normalized radius $r/R$")
    ax1.set_ylabel(r"Chord $c$ [m]", color=color1)
    line1 = ax1.plot(r_R, c, "-", color=color1, label=r"Chord $c$")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel(r"Blade angle $\beta$ [deg]", color=color2)
    line2 = ax2.plot(r_R, np.degrees(beta), "-", color=color2, label=r"Blade angle $\beta$")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines  = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(f"Optimized Blade Geometry{' - ' + title_suffix if title_suffix else ''}")
    fig.tight_layout()

    fname = f"optimized_blade_geometry{'_' + title_suffix if title_suffix else ''}.png"
    save_path = os.path.join(img_dir, fname)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Geometry plot saved to: {save_path}")


def plot_comparison(r_base, c_base, beta_base, CT_base, CP_base,
                    r_opt,  c_opt,  beta_opt,  CT_opt,  CP_opt):
    """Side-by-side comparison of baseline and analytical optimal geometry."""
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(img_dir, exist_ok=True)

    # Midpoints for baseline (BEM works on midpoints)
    r_base_mid = 0.5 * (r_base[:-1] + r_base[1:])
    c_base_mid = 0.5 * (c_base[:-1] + c_base[1:])
    b_base_mid = 0.5 * (beta_base[:-1] + beta_base[1:])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(r_base_mid / R, c_base_mid,         label=f"Baseline  (CT={CT_base:.3f})")
    axes[0].plot(r_opt / R,      c_opt,               label=f"Analytical (CT={CT_opt:.3f})", linestyle="--")
    axes[0].axhline(MIN_CHORD_TIP, color="grey", linestyle=":", label="Min chord")
    axes[0].axhline(MAX_CHORD,     color="grey", linestyle="--", label="Max chord")
    axes[0].set_xlabel(r"$r/R$")
    axes[0].set_ylabel("Chord [m]")
    axes[0].set_title("Chord Distribution")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(r_base_mid / R, np.degrees(b_base_mid), label="Baseline")
    axes[1].plot(r_opt / R,      np.degrees(beta_opt),    label="Analytical", linestyle="--")
    axes[1].set_xlabel(r"$r/R$")
    axes[1].set_ylabel("Blade angle [deg]")
    axes[1].set_title("Blade Angle Distribution")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(
        f"Baseline CP={CP_base:.4f}  vs  Analytical CP={CP_opt:.4f}  "
        f"(+{100*(CP_opt-CP_base)/abs(CP_base):.1f} %)"
    )
    fig.tight_layout()
    save_path = os.path.join(img_dir, "comparison_baseline_vs_analytical.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to: {save_path}")


# =============================================================================
# 8. RUNNER
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Baseline blade
    # ------------------------------------------------------------------
    print("=" * 65)
    print("BASELINE BLADE  (linear chord/twist, pitch = -2 deg, TSR = 8)")
    print("=" * 65)

    r0, c0, beta0 = get_geometry_blade("constant", 100)

    CT_base_list, CP_base_list, _ = run_bem_solver(
        plotting=True,
        tsr_range=[6.0, 8.0, 10.0],
        r_arr=r0,
        c_arr=c0,
        twist_arr=beta0,
        plotname="initial",
    )

    # Pick TSR=8 result for the comparison (index 1 in [6,8,10])
    CT_base = CT_base_list[1]
    CP_base = CP_base_list[1]
    print(f"\nBaseline at TSR=8:  CT = {CT_base:.5f},  CP = {CP_base:.5f}")

    # ------------------------------------------------------------------
    # Analytical optimal design
    # ------------------------------------------------------------------
    r_opt, c_opt, beta_opt, a_design = design_for_exact_ct()

    print(f"\nDesign induction used: a = {a_design:.6f}")

    # ------------------------------------------------------------------
    # Final BEM run with full plotting
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RUNNING FINAL BEM ON OPTIMIZED BLADE (with plots)")
    print("=" * 65)

    CT_opt_list, CP_opt_list, _ = run_bem_solver(
        plotting=True,
        tsr_range=[8.0],
        r_arr=r_opt,
        c_arr=c_opt,
        twist_arr=beta_opt,
        plotname="analytical_optimal",
    )

    CT_opt = CT_opt_list[0]
    CP_opt = CP_opt_list[0]

    # ------------------------------------------------------------------
    # Actuator disk reference
    # ------------------------------------------------------------------
    a_ad  = 0.5 * (1.0 - np.sqrt(1.0 - CT_target))
    CP_ad = 4.0 * a_ad * (1.0 - a_ad) ** 2

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Actuator disk  CT={CT_target}:  a = {a_ad:.4f},  CP = {CP_ad:.5f}")
    print(f"Baseline       CT = {CT_base:.5f},  CP = {CP_base:.5f}  "
          f"(CP/CP_AD = {CP_base/CP_ad:.4f})")
    print(f"Analytical opt CT = {CT_opt:.5f},  CP = {CP_opt:.5f}  "
          f"(CP/CP_AD = {CP_opt/CP_ad:.4f})")
    print(f"CP improvement = {100*(CP_opt - CP_base)/abs(CP_base):.2f} %")

    # ------------------------------------------------------------------
    # Geometry printout at selected stations
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("OPTIMIZED GEOMETRY (selected stations)")
    print("=" * 65)
    idxs = [0, len(r_opt) // 4, len(r_opt) // 2, 3 * len(r_opt) // 4, -1]
    print(f"{'r/R':>7}  {'Chord [m]':>10}  {'Beta [deg]':>11}")
    for idx in idxs:
        print(f"{r_opt[idx]/R:>7.3f}  {c_opt[idx]:>10.4f}  {np.degrees(beta_opt[idx]):>11.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_geometry(r_opt, c_opt, beta_opt, title_suffix="analytical_optimal")
    plot_comparison(r0, c0, beta0, CT_base, CP_base,
                    r_opt, c_opt, beta_opt, CT_opt, CP_opt)

    # ------------------------------------------------------------------
    # Save geometry to CSV
    # ------------------------------------------------------------------
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    df = pd.DataFrame({
        "r_m":       r_opt,
        "r_R":       r_opt / R,
        "chord_m":   c_opt,
        "beta_deg":  np.degrees(beta_opt),
        "beta_rad":  beta_opt,
    })
    csv_path = os.path.join(img_dir, "analytical_optimal_geometry.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nGeometry saved to: {csv_path}")

    print("\nDone.")