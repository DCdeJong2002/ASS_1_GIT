import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import os
from scipy.optimize import root_scalar
import sys, io

# =============================================================================
# 1. AERODYNAMIC DATA
# =============================================================================

def load_polar(filename):
    """Load a polar Excel/CSV file and return (cl_func, cd_func) interpolators."""
    df = pd.read_excel(filename, skiprows=3)
    df.columns = [str(c).strip() for c in df.columns]        
    cl_func = interp1d(df['Alfa'], df['Cl'], kind='linear', fill_value='extrapolate')
    cd_func = interp1d(df['Alfa'], df['Cd'], kind='linear', fill_value='extrapolate')
    return cl_func, cd_func

def get_coefficients(alpha_rad, cl_func, cd_func):
    """Return Cl, Cd for the given angle of attack in radians."""
    alpha_deg = np.degrees(alpha_rad)
    return float(cl_func(alpha_deg)), float(cd_func(alpha_deg))

# =============================================================================
# 2. GEOMETRY DEFINITION (Based on Assignment)
# =============================================================================
R = 50.0       # Rotor radius [m]
R_root = 10.0  # Root radius [m] (0.2 * R)
B = 3          # Number of blades

def PrandtlTipRootCorrection(r_R, rootradius_R, TSR, B, a):
    """Combined Prandtl tip and root correction factor."""
    a = np.clip(a, -0.9, 0.99)  # prevent division by zero
    d = 2 * np.pi / B * (1 - a) / np.sqrt(TSR**2 + (1-a)**2)
    Ftip  = 2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (1 - r_R) / d,  -500, 0)))
    Ftip  = float(np.where(np.isnan(Ftip),  0, Ftip))

    Froot = 2 / np.pi * np.arccos(np.exp(np.clip(-np.pi * (r_R - rootradius_R) / d, -500, 0)))
    Froot = float(np.where(np.isnan(Froot), 0, Froot))
    return Froot * Ftip, Ftip, Froot

def get_geometry_blade(spacing_method, num_annuli):
    """Generates the geometry for the 50m wind turbine blade."""
    # Radial spacing
    r_start = R_root
    r_end   = R
    if spacing_method == "constant":
        r_meters = np.linspace(r_start, r_end, num_annuli)
    elif spacing_method == "cosine":
        r_meters = r_start + (r_end - r_start) * (1 - np.cos(np.linspace(0, np.pi, num_annuli))) / 2

    # Chord and Twist distributions from the assignment
    c_meters = 3.0 * (1 - (r_meters / R)) + 1.0       # Chord [m]
    twist_deg = 14.0 * (1 - (r_meters / R))           # Twist [deg]
    
    pitch_deg = -2.0  # Constant pitch [deg]
    
    # Total blade angle (beta)
    beta_deg = twist_deg + pitch_deg
    twist_rad = np.radians(beta_deg)

    return r_meters, c_meters, twist_rad

# =============================================================================
# 3. WIND TURBINE BEM SOLVER
# =============================================================================

def run_bem_solver(plotting, r_arr, c_arr, twist_arr, tsr_range=[8.0], plotname=""):
    polar_path = r"ASS_1_GIT\polar DU95W180 (3).xlsx"
        
    cl_func, cd_func = load_polar(polar_path)

    # Wind Turbine Operating Conditions
    V0 = 10.0        # Free-stream wind speed [m/s]
    rho = 1.225      # Standard air density [kg/m^3]
    A_disk = np.pi * R**2

    dr_arr = np.diff(r_arr)
    r_arr = r_arr[:-1]  # Remove the last element to match dr length
    r_arr = r_arr + dr_arr / 2  # Midpoint of each annulus for better accuracy
    
    c_arr = (c_arr[:-1] + c_arr[1:]) / 2
    twist_arr = (twist_arr[:-1] + twist_arr[1:]) / 2

    TSR_results = []
    Ct_results = []
    Cp_results = []
    
    # Dictionaries to store spanwise data for plotting
    alpha_span_all = {}
    phi_span_all   = {}
    a_span_all     = {}
    ap_span_all    = {}
    dTdr_span_all  = {}
    dQdr_span_all  = {}
    F_span_all    = {}

    # Residual tracking
    residual_records = []

    relaxation_factor = 0.25
    max_iter = 300
    tol = 1e-5  # Convergence threshold for inductions

    print(f"Starting Wind Turbine BEM Analysis...")

    for TSR in tsr_range:
        omega = (TSR * V0) / R  # Rotational speed [rad/s]

        # --- Initialise arrays for this TSR ---
        a_arr = np.zeros(len(r_arr))
        ap_arr = np.zeros(len(r_arr)) # a' (tangential induction)

        alpha_span = np.zeros(len(r_arr))
        phi_span   = np.zeros(len(r_arr))
        dTdr_arr   = np.zeros(len(r_arr))
        dQdr_arr   = np.zeros(len(r_arr))
        F_arr     = np.zeros(len(r_arr))

        TSR_key = round(TSR, 1)
        Ct_prev = 0.0
        converged = False

        # --- Global iteration: sweep ALL annuli each step ---
        for iter_count in range(max_iter):
            anew_arr  = a_arr.copy()
            apnew_arr = ap_arr.copy()
            Fnew_arr  = F_arr.copy()

            for i in range(len(r_arr)):
                r    = r_arr[i]
                c    = c_arr[i]
                beta = twist_arr[i]
                a    = a_arr[i]
                ap   = ap_arr[i]

                sigma = (B * c) / (2 * np.pi * r)

                # 1. Velocities (Wind Turbine retards wind, so 1-a. Wake rotates, so 1+a')
                V_axial      = V0 * (1 - a)
                V_tangential = omega * r * (1 + ap)
                V_rel = np.sqrt(V_axial**2 + V_tangential**2)

                phi = np.arctan2(V_axial, V_tangential)

                # Prevent division by zero errors at root
                if phi < 1e-5:
                    phi = 1e-5

                # 2. Aerodynamics
                alpha = phi - beta
                cl, cd = get_coefficients(alpha, cl_func, cd_func)

                # Wind Turbine Force Coefficients
                Cn = cl * np.cos(phi) + cd * np.sin(phi)
                Ct_aero = cl * np.sin(phi) - cd * np.cos(phi) # Named Ct_aero to avoid confusion with Thrust Coeff

                # 3. Prandtl Tip and Hub Loss Factors
                F, _, _ = PrandtlTipRootCorrection(r/R, R_root/R, TSR, B, a)
                F = max(F, 1e-4)
                Fnew_arr[i] = F

                # 4. Momentum update & Glauert Correction
                # Local thrust coefficient, rewritten from blade element thrust formula: dT = 0.5 * rho * V_rel^2 * B * c * Cn * dr
                CT_loc = (sigma * (1 - a)**2 * Cn) / (np.sin(phi)**2)

                # Glauert correction for highly loaded rotors
                if CT_loc <  2 * np.sqrt(1.816) - 1.816: 
                    a_new = 0.5 - np.sqrt(1 - CT_loc) / 2
                else:
                    a_new = 1 + (CT_loc - 1.816) / (4 * np.sqrt(1.816) - 4)

                # From equating momentum and blade element torque
                ap_new = 1.0 / (((4 * np.sin(phi) * np.cos(phi)) / (sigma * Ct_aero)) - 1)

                # Clip inductions to physical bounds to prevent divergence
                a_new = np.clip(a_new/F, -0.5, 0.95)
                ap_new = np.clip(ap_new/F, -0.5, 0.95)

                # Apply relaxation
                anew_arr[i]  = (1 - relaxation_factor) * a  + relaxation_factor * a_new
                apnew_arr[i] = (1 - relaxation_factor) * ap + relaxation_factor * ap_new

                alpha_span[i] = alpha
                phi_span[i]   = phi

                # 5. Calculate Local Loads
                dTdr_arr[i] = 0.5 * rho * V_rel**2 * B * c * Cn
                dQdr_arr[i] = 0.5 * rho * V_rel**2 * B * c * Ct_aero * r

            # --- Convergence check on global Ct residual ---
            Ct_iter     = np.sum(dTdr_arr * dr_arr) / (0.5 * rho * A_disk * V0**2)
            Ct_residual = abs(Ct_iter - Ct_prev)

            residual_records.append({
                'TSR':         TSR_key,
                'iteration':   iter_count,
                'Ct_residual': Ct_residual,
            })

            Ct_prev = Ct_iter
            a_arr  = anew_arr
            ap_arr = apnew_arr
            F_arr  = Fnew_arr

            if iter_count > 0 and Ct_residual < tol:
                print(f"TSR={TSR:.1f}: converged at iteration {iter_count:3d},  Ct residual = {Ct_residual:.3e}")
                converged = True
                break

        if not converged:
            print(f"Warning: TSR={TSR:.1f} did not converge in {max_iter} iterations, "
                  f"final Ct residual = {Ct_residual:.3e}")

        # --- Integrate global thrust and torque ---
        current_thrust = np.sum(dTdr_arr * dr_arr)
        current_torque = np.sum(dQdr_arr * dr_arr)
        current_power  = current_torque * omega

        # Wind Turbine Coefficients
        Ct = current_thrust / (0.5 * rho * A_disk * V0**2)
        Cp = current_power / (0.5 * rho * A_disk * V0**3)
            
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

        print(f"TSR={TSR:.1f} | Cp = {Cp:.4f} | Ct = {Ct:.4f}")

    if plotting:
        # --- Save figures to images folder ---
        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
        os.makedirs(img_dir, exist_ok=True)

        r_R = r_arr / R  # normalised span axis

        # --- Plot a: spanwise angle of attack and inflow angle ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tsr in alpha_span_all:
            axes[0].plot(r_R, alpha_span_all[tsr], '-', markersize=3, label=rf'$\lambda={tsr}$')
            axes[1].plot(r_R, phi_span_all[tsr],   '-', markersize=3, label=rf'$\lambda={tsr}$')
        axes[0].set_title(r'Angle of Attack')
        axes[0].set_xlabel(r'$r/R$'); axes[0].set_ylabel(r'$\alpha$ (deg)')
        axes[0].legend(); axes[0].grid(True)
        axes[1].set_title(r'Inflow Angle')
        axes[1].set_xlabel(r'$r/R$'); axes[1].set_ylabel(r'$\phi$ (deg)')
        axes[1].legend(); axes[1].grid(True)
        fig.suptitle(rf'Spanwise Angle of Attack and Inflow Angle {plotname}')
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f'plot_a_alpha_phi_span_{plotname}.png'), dpi=150)
        plt.close(fig)

        # --- Plot b: spanwise axial and tangential inductions ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tsr in a_span_all:
            axes[0].plot(r_R, a_span_all[tsr], '-', markersize=3, label=rf'$\lambda={tsr}$')
            axes[1].plot(r_R, ap_span_all[tsr], '-', markersize=3, label=rf'$\lambda={tsr}$')
        axes[0].set_title(r'Axial Induction Factor $a$')
        axes[0].set_xlabel(r'$r/R$'); axes[0].set_ylabel(r'$a$')
        axes[0].legend(); axes[0].grid(True)
        axes[1].set_title(r"Tangential Induction Factor $a'$")
        axes[1].set_xlabel(r'$r/R$'); axes[1].set_ylabel(r"$a'$")
        axes[1].legend(); axes[1].grid(True)
        fig.suptitle(rf"Spanwise Inductions {plotname}")
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f'plot_b_inductions_span_{plotname}.png'), dpi=150)
        plt.close(fig)

        # --- Plot c: spanwise thrust and torque loading ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tsr in dTdr_span_all:
            axes[0].plot(r_R, dTdr_span_all[tsr], '-', markersize=3, label=rf'$\lambda={tsr}$')
            axes[1].plot(r_R, dQdr_span_all[tsr], '-', markersize=3, label=rf'$\lambda={tsr}$')
        axes[0].set_title(r'Thrust Loading $dT/dr$')
        axes[0].set_xlabel(r'$r/R$'); axes[0].set_ylabel(r'$dT/dr$ (N/m)')
        axes[0].legend(); axes[0].grid(True)
        axes[1].set_title(r'Torque Loading $dQ/dr$')
        axes[1].set_xlabel(r'$r/R$'); axes[1].set_ylabel(r'$dQ/dr$ (N)')
        axes[1].legend(); axes[1].grid(True)
        fig.suptitle(rf'Spanwise Thrust and Torque Loading {plotname}')
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f'plot_c_loading_span_{plotname}.png'), dpi=150)
        plt.close(fig)

        # --- Plot d: total performance vs TSR ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(TSR_results, Cp_results, '-o', color='blue')
        axes[0].set_title(r'Power Coefficient $C_P$ vs $\lambda$')
        axes[0].set_xlabel(r'Tip Speed Ratio $\lambda$'); axes[0].set_ylabel(r'$C_P$')
        axes[0].grid(True)
        
        axes[1].plot(TSR_results, Ct_results, '-s', color='red')
        axes[1].set_title(r'Thrust Coefficient $C_T$ vs $\lambda$')
        axes[1].set_xlabel(r'Tip Speed Ratio $\lambda$'); axes[1].set_ylabel(r'$C_T$')
        axes[1].grid(True)
        
        fig.suptitle(rf'Wind Turbine Performance {plotname}')
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f'plot_d_total_performance_{plotname}.png'), dpi=150)
        plt.close(fig)

        print(f"All plots saved to: {img_dir}")

        # Build and return the residual DataFrame
        df_residuals = pd.DataFrame(residual_records)

        # --- Plot e: convergence history of thrust RMS ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for TSR_k, grp in df_residuals.groupby('TSR'):
            ax.semilogy(grp['iteration'], grp['Ct_residual'], '-o', markersize=3, label=rf'$\lambda={TSR_k}$')
        ax.axhline(tol, color='k', linestyle='--', linewidth=0.8, label=f'Tolerance = {tol:.0e}')
        ax.set_title('Convergence History: $C_T$ Residual')
        ax.set_xlabel('Global Iteration')
        ax.set_ylabel(r'$|\Delta C_T|$')
        ax.legend()
        ax.grid(True, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f'plot_e_convergence_{plotname}.png'), dpi=150)
        plt.close(fig)
        print(f"Convergence plot saved to: {os.path.join(img_dir, f'plot_e_convergence_{plotname}.png')}")
    
        # --- Plot f: Stagnation pressure distribution
        P_static = 0
        P01_arr = (P_static + 0.5 * rho * (V0**2)) * np.ones_like(r_arr)
        dP0_arr_all = {}
        P03_arr_all = {}

        for tsr in tsr_range:
            dP0_arr = 2 * rho * V0**2 * a_span_all[tsr] * F_span_all[tsr] * (1 - a_span_all[tsr] * F_span_all[tsr])  # Pressure drop due to power extraction
            P03_arr = P01_arr - dP0_arr

            dP0_arr_all[tsr] = dP0_arr
            P03_arr_all[tsr] = P03_arr

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r_R, P01_arr, '-', label=rf'$P_0^1,\,P_0^2$ for $\lambda={tsr_range}$')

        for tsr in tsr_range:
            ax.plot(r_R, P03_arr_all[tsr], '-', label=rf'$P_0^3,\,P_0^4$ for $\lambda={tsr}$')
        ax.set_title(rf'Stagnation Pressure Distribution {plotname}')
        ax.set_xlabel(r'$r/R$')
        ax.set_ylabel(r'Stagnation Pressure $P_0$')
        ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')
        ax.grid(True, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f'plot_f_stagnation_pressure_{plotname}.png'), dpi=150)
        plt.close(fig)

    return Ct_results, Cp_results, ap_span_all[tsr_range[0]]

def discretization_study():
    n_annuli_values = [5, 10, 20, 50, 100, 200, 500]
    Ct_constant_values = []
    Ct_cosine_values = []
    for n_annuli in n_annuli_values:
        r, c, twist = get_geometry_blade("constant", n_annuli)
        Ct_const = run_bem_solver(plotting=False, r_arr=r, c_arr=c, twist_arr=twist)[0][-1]
        r, c, twist = get_geometry_blade("cosine", n_annuli)
        Ct_cosine = run_bem_solver(plotting=False, r_arr=r, c_arr=c, twist_arr=twist)[0][-1]
        Ct_constant_values.append(Ct_const)
        Ct_cosine_values.append(Ct_cosine)
    
    # Plot convergence of Ct with number of annuli
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(img_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(n_annuli_values, Ct_constant_values, '-o', label='Constant Spacing')
    plt.plot(n_annuli_values, Ct_cosine_values, '-s', label='Cosine Spacing')
    plt.xlabel('Number of Annuli')
    plt.ylabel('Thrust Coefficient $C_T$')
    plt.title('Convergence of Thrust Coefficient')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'plot_g_convergence.png'), dpi=150)
    plt.close()

def generate_ideal_rotor(target_a=0.25, n_annuli=100, ap_in=None):
    print("\n--- Analytically Designing Ideal Rotor ---")
    
    # 1. Load the polar and find optimal alpha (max Cl/Cd)
    polar_path = r"ASS_1_GIT\polar DU95W180 (3).xlsx"
        
    cl_func, cd_func = load_polar(polar_path)

    # Calculate Cl/Cd ratio to find the aerodynamic optimum
    alphas = np.linspace(0, 20, 200)
    cl_vals = cl_func(alphas)
    cd_vals = cd_func(alphas)
    cl_cd_ratio = cl_vals / cd_vals
    opt_idx = int(np.argmax(cl_cd_ratio))

    alpha_opt_deg = float(alphas[opt_idx])
    cl_opt        = float(cl_vals[opt_idx])
    cd_opt        = float(cd_vals[opt_idx])
    alpha_opt_rad = np.radians(alpha_opt_deg)
    
    print(f"Optimal Airfoil Point: Alpha = {alpha_opt_deg:.2f} deg, Cl = {cl_opt:.3f}, Cd = {cd_opt:.4f}")

    # 2. Setup Rotor Parameters
    TSR = 8.0
    a = target_a # Target induction from Actuator Disk Theory for Ct=0.75

    r_start = R_root + 0.005 * R
    r_end   = R - 0.005 * R
    r_meters = r_start + (r_end - r_start) * (1 - np.cos(np.linspace(0, np.pi, n_annuli))) / 2
    
    c_meters = np.zeros_like(r_meters)
    twist_rad = np.zeros_like(r_meters)
    
    if ap_in is not None:
        # Interpolate the ap from BEM (which is on annulus centers) to the node locations
        r_centers = r_meters[:-1] + np.diff(r_meters)/2
        ap_interp = np.interp(r_meters, r_centers, ap_in)
    
    # 3. Calculate Ideal Geometry Spanwise
    for i, r in enumerate(r_meters):
        local_tsr = TSR * (r / R)
        
        # Calculate exact F first (independent of phi)
        F, _, _ = PrandtlTipRootCorrection(r/R, R_root/R, TSR, B, a)
        F = max(F, 1e-4)
        
        # Iteratively solve for wake rotation ap exactly aligned with BEM formulation
        # Derived from setting (Ct/Cn) momentum = (Ct/Cn) blade element and accounting for F
        if ap_in is not None:
            ap = ap_interp[i]
        else:
            ap = 0.0  # Start with no rotation, will be refined by BEM

        
        # Calculate ideal inflow angle phi
        phi = np.arctan2((1 - a), (local_tsr * (1 + ap)))
        
        # Calculate required twist to maintain alpha_opt
        twist_rad[i] = phi - alpha_opt_rad
        
        # Normal force coefficient
        Cn = cl_opt * np.cos(phi) + cd_opt * np.sin(phi)
        
        # Calculate exact required chord length matching the BEM CT formula: CT_loc = 4(a*F)(1-a*F)
        c_meters[i] = (8 * np.pi * r * a * F * (1 - a * F) * (np.sin(phi)**2)) / (B * ((1 - a)**2) * Cn)

    # 4. Apply physical and structural constraints
    MAX_CHORD = 4.0      # Maximum allowed chord [m]
    MIN_CHORD_TIP = 0.3  # Minimum allowed tip chord [m]
    
    # First, cap the maximum size to avoid unphysical aerodynamic bulges
    c_meters = np.clip(c_meters, 0, MAX_CHORD)
    
    # STRUCTURAL ROOT FIX: 
    # The aerodynamic math drops the chord to 0 at the root due to hub losses.
    # We find where the chord reaches its maximum, and force everything 
    # inboard of that point to stay at that maximum thickness for the hub attachment.
    max_idx = np.argmax(c_meters)
    c_meters[:max_idx] = c_meters[max_idx]
    
    # Finally, cap the tip so it doesn't drop below the minimum manufacturing limit
    c_meters = np.clip(c_meters, MIN_CHORD_TIP, None)

    # 5. Print and return the design
    twist_deg = np.degrees(twist_rad)
    
    print("\nIdeal Geometry Generated:")
    print(f"Root: Chord = {c_meters[0]:.2f} m | Twist = {twist_deg[0]:.2f} deg")
    print(f"Mid : Chord = {c_meters[25]:.2f} m | Twist = {twist_deg[25]:.2f} deg")
    print(f"Tip : Chord = {c_meters[-1]:.2f} m | Twist = {twist_deg[-1]:.2f} deg")
    
    return r_meters, c_meters, twist_rad

def design_for_exact_ct():
    print("\n--- Finding Exact Induction for Ct = 0.75 ---")
    
    last_ap = None

    def evaluate_target_a(target_a):
        nonlocal last_ap
        # 1. Generate the geometry for this 'a'
        r_opt, c_opt, tw_opt = generate_ideal_rotor(target_a, n_annuli=100, ap_in=last_ap)
        
        # 3. Run BEM silently
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            Ct_res, Cp_res, ap_res = run_bem_solver(plotting=False, tsr_range=[8.0], r_arr=r_opt, c_arr=c_opt, twist_arr=tw_opt)
            ct_val = Ct_res[0]
            cp_val = Cp_res[0]
            last_ap = ap_res
        except Exception as e:
            ct_val, cp_val = 0.0, 0.0
            
        sys.stdout = old_stdout
        
        print(f"Tested target a={target_a:.4f} -> Resulting BEM Ct={ct_val:.4f}, Cp={cp_val:.4f}")
        
        # We want the difference between actual Ct and 0.75 to be zero
        return ct_val - 0.75

    # Use a root finder. We know 'a' must be between 0.25 and 0.33
    # root_scalar will quickly zero in on the exact 'a' needed!
    res = root_scalar(evaluate_target_a, bracket=[0.25, 0.33], method='brentq')
    
    if res.converged:
        best_a = res.root
        print(f"\nSUCCESS! To get Ct=0.75, the ideal rotor must be designed for a = {best_a:.4f}")
        
        # Generate the final winning geometry and print it
        r, c, tw = generate_ideal_rotor(best_a, ap_in=last_ap)
        print("\n--- FINAL OPTIMIZED GEOMETRY ---")
        print(f"Root: Chord = {c[0]:.2f} m | Twist = {np.degrees(tw[0]):.2f} deg")
        print(f"Mid : Chord = {c[25]:.2f} m | Twist = {np.degrees(tw[25]):.2f} deg")
        print(f"Tip : Chord = {c[-1]:.2f} m | Twist = {np.degrees(tw[-1]):.2f} deg")
        return r, c, tw
    else:
        print("Could not find exact match.")    

# =============================================================================
# 4. RUNNER
# =============================================================================
def plot_geometry(r, c, tw):
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    r_R = r / 50.0
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Normalized Radius $r/R$')
    ax1.set_ylabel(r'Chord $c$ [m]', color=color1)
    line1 = ax1.plot(r_R, c, '-', color=color1, label=r'Chord $c$')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel(r'Twist $\theta$ [deg]', color=color2)
    line2 = ax2.plot(r_R, np.degrees(tw), '-', color=color2, label=r'Twist $\theta$')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Optimized Blade Geometry')
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, 'optimized_blade_geometry.png'), dpi=150)
    plt.close(fig)
    print(f"Optimized blade geometry plot saved to: {os.path.join(img_dir, 'optimized_blade_geometry.png')}")

if __name__ == "__main__":
    r, c, tw = get_geometry_blade("constant", 100)
    run_bem_solver(plotting=True, tsr_range=[6.0, 8.0, 10.0], r_arr=r, c_arr=c, twist_arr=tw, plotname="initial")
    discretization_study()
    r, c, tw = design_for_exact_ct()
    plot_geometry(r, c, tw)
    run_bem_solver(plotting=True, tsr_range=[8.0], r_arr=r, c_arr=c, twist_arr=tw, plotname="optimized")

