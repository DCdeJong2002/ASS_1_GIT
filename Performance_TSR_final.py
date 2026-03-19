import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load data
data = pd.read_excel('polar DU95W180 (3).xlsx', skiprows=3)
polar_alpha, polar_cl, polar_cd = data['Alfa'], data['Cl'], data['Cd']

## ================================================================== ##
## Specifications
## ================================================================== ##

Pitch = -2   # Set blade pitch angle (degrees)
delta_r_R = 0.005   # Set spanwise resolution

Radius = 50   # (m)
NBlades = 3   # Number of blades
U0 = 10   # Freestream velocity (m/s)

RootLocation_R, TipLocation_R = 0.2, 1.0   # Blade starts at 20% r/R
r_R_bins = np.arange(RootLocation_R, TipLocation_R + delta_r_R/2, delta_r_R)

## ================================================================== ##
## Induction and correction functions
## ================================================================== ##

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
    temp_tip = -NBlades/2 * (tipradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - axial_induction)**2))
    Ftip = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0, 1)))
    
    temp_root = NBlades / 2 * (rootradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - axial_induction)**2))
    Froot = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0, 1)))
    
    return Froot * Ftip

def LoadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Calculate the local forces (normal and tangential) and circulation on a blade element.
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

def SolveStreamtube(U0, r1_R, r2_R, rootradius_R, tipradius_R, Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Solve the momentum balance for a single streamtube using BEM method.
    """
    Area = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_mid = (r1_R + r2_R) / 2
    r_local = r_mid * Radius
    
    a, aline = 0.1, 0
    fnorm_history = np.zeros(200) # Added tracker for convergence plotting
    
    for i in range(200):
        Urotor = U0 * (1 - a)
        Utan = (1 + aline) * Omega * r_local
        
        fnorm, ftan, gamma, alpha, phi = LoadBladeElement(Urotor, Utan, chord, twist, polar_alpha, polar_cl, polar_cd)
        fnorm_history[i] = fnorm # Log normal force for this specific iteration
        
        CT = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0**2)
        
        anew = ainduction(CT)
        F = max(PrandtlTipRootCorrection(r_mid, rootradius_R, tipradius_R, Omega*Radius/U0, NBlades, anew), 0.0001)
        anew /= F
 
        a = 0.75 * a + 0.25 * anew
        
        aline = (ftan * NBlades) / (2 * np.pi * U0 * (1 - a) * Omega * 2 * r_local**2)
        aline /= F
        
    return [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi], fnorm_history

## ================================================================== ##
## Solving for different TSRs and collecting results
## ================================================================== ##

# Storage for Plot 4
tsr_performance = {} 
results_tsr8 = []
ct_history_tsr8 = []
results_tsr8 = []
ct_history_tsr8 = []

print(f"{'TSR':<10} | {'CT':<10} | {'CP':<10}")
print("-" * 35)

for TSR in np.arange(6, 11, 1):
    Omega = U0 * TSR / Radius

    temp_results = []
    histories = []
    
    for i in range(len(r_R_bins) - 1):
        r_mid = (r_R_bins[i] + r_R_bins[i+1]) / 2
        
        chord_function = 3 * (1 - r_mid) + 1
        twist_function = -(14 * (1 - r_mid) + Pitch)

        res, f_hist = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1], RootLocation_R, TipLocation_R, Omega, Radius, NBlades, chord_function, twist_function, polar_alpha, polar_cl, polar_cd)
        temp_results.append(res)
        histories.append(f_hist)
    
    res_arr = np.array(temp_results)
    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

    CT = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP = np.sum(dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega / (0.5 * U0**3 * np.pi * Radius**2))

    print(f"{TSR:<10.1f} | {CT:<10.4f} | {CP:<10.4f}")
    
    # Store the results for this TSR
    tsr_performance[TSR] = {'CT': CT, 'CP': CP}
    
    if TSR == 8: 
        results_tsr8 = res_arr
        
        # Integrate the history of all streamtubes at each iteration step
        dr_col = dr[:, np.newaxis] 
        hist_arr = np.array(histories) # Shape: (spanwise_elements, 200_iterations)
        ct_history_tsr8 = np.sum(hist_arr * dr_col * NBlades / (0.5 * U0**2 * np.pi * Radius**2), axis=0)

## ================================================================== ##
## Plotting results for TSR=8
## ================================================================== ##

base_path = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(base_path, "plots_BEM")
os.makedirs(save_folder, exist_ok=True)

# Helper function to save and show to keep the code clean
def save_and_show(filename):
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

## ================================================================== ##
## Comparison: Influence of Tip Correction (TSR=8)
## ================================================================== ##

results_tsr8_no_corr = []
TSR_comp = 8
Omega_comp = U0 * TSR_comp / Radius

# Run the simulation again for TSR=8 but without Prandtl correction (F=1)
for i in range(len(r_R_bins) - 1):
    rm = (r_R_bins[i] + r_R_bins[i+1]) / 2
    chord = 3 * (1 - rm) + 1
    twist = -(14 * (1 - rm) + Pitch)
    
    # Simple logic to solve without correction (manual override of F=1)
    a_nc, aline_nc = 0.1, 0
    for _ in range(300):
        Urot = U0 * (1 - a_nc)
        Utan = (1 + aline_nc) * Omega_comp * rm * Radius
        fn, ft, gam, alp, phi = LoadBladeElement(Urot, Utan, chord, twist, polar_alpha, polar_cl, polar_cd)
        CT_loc = (fn * Radius * delta_r_R * NBlades) / (0.5 * (2*np.pi*rm*Radius*delta_r_R*Radius) * U0**2)
        anew_nc = ainduction(CT_loc)
        # F is skipped here (effectively F=1.0)
        a_nc = 0.75 * a_nc + 0.25 * anew_nc
        aline_nc = (ft * NBlades) / (2 * np.pi * U0 * (1 - a_nc) * Omega_comp * 2 * (rm * Radius)**2)
        
    results_tsr8_no_corr.append([a_nc, aline_nc, rm, fn, ft])

res_nc = np.array(results_tsr8_no_corr)

'''
# Plot a: anlge of attack and inflow angle
plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 6], 'b-', label=r'Angle of attack ($\alpha$)')
plt.plot(results_tsr8[:, 2], results_tsr8[:, 7], 'r-', label=r'Inflow angle ($\phi$)')
plt.title('Spanwise distribution of angle of attack and inflow angle (TSR=8)')
plt.xlabel('r/R')
plt.ylabel('Angle [deg]')
plt.grid(True)
plt.legend()
save_and_show("1_Alpha_Phi_Distribution.png")

# --- Plot 2: Spanwise distribution of axial and azimuthal inductions ---
plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 0], 'b-', label=r'Axial induction ($a$)')
plt.plot(results_tsr8[:, 2], results_tsr8[:, 1], 'r-', label=r'Azimuthal induction ($a^\prime$)')
plt.title('Spanwise distribution of axial and azimuthal inductions (TSR=8)')
plt.xlabel('r/R')
plt.ylabel('Induction factor')
plt.grid(True)
plt.legend()
save_and_show("2_Induction_Factors.png")

# --- Plot 3: Spanwise distribution of thrust and azimuthal loading ---
# Normalizing by 0.5 * rho * U0^2 * Radius (Non-dimensional sectional loading)
plt.figure(figsize=(9, 5))
cn = results_tsr8[:, 3] / (0.5 * U0**2 * Radius) # Normal force coefficient per unit span
ct = results_tsr8[:, 4] / (0.5 * U0**2 * Radius) # Tangential force coefficient per unit span
plt.plot(results_tsr8[:, 2], cn, 'b-', label=r'Thrust loading ($C_n$)')
plt.plot(results_tsr8[:, 2], ct, 'r-', label=r'Azimuthal loading ($C_t$)')
plt.title('Spanwise distribution of thrust and azimuthal loading (TSR=8)')
plt.xlabel('r/R')
plt.ylabel(r'Sectional load coefficient $F / (\frac{1}{2} U_\infty^2 R)$')
plt.grid(True)
plt.legend()
save_and_show("3_Spanwise_Loading.png")

# --- Plot 4: Total thrust and torque versus tip-speed ratio ---
# Extracting data from our performance dictionary
tsr_plot_list = sorted(tsr_performance.keys())
ct_plot_list = [tsr_performance[t]['CT'] for t in tsr_plot_list]
# Torque Coefficient CQ = CP / TSR
cq_plot_list = [tsr_performance[t]['CP'] / t for t in tsr_plot_list]

# --- Plot 4a: Total Thrust Coefficient (CT) vs. TSR ---
plt.figure(figsize=(9, 5))
plt.plot(tsr_plot_list, ct_plot_list, 'bo-', label=r'Total thrust coefficient ($C_T$)')
plt.title('Total thrust coefficient vs. Tip-Speed Ratio')
plt.xlabel('Tip-Speed Ratio (TSR)')
plt.ylabel(r'Thrust coefficient $C_T$')
plt.grid(True)
plt.legend()
save_and_show("4a_Thrust_vs_TSR.png")

# --- Plot 4b: Total Torque Coefficient (CQ) vs. TSR ---
plt.figure(figsize=(9, 5))
plt.plot(tsr_plot_list, cq_plot_list, 'ro-', label=r'Total torque coefficient ($C_Q$)')
plt.title('Total torque coefficient vs. Tip-Speed Ratio')
plt.xlabel('Tip-Speed Ratio (TSR)')
plt.ylabel(r'Torque coefficient $C_Q$')
plt.grid(True)
plt.legend()
save_and_show("4b_Torque_vs_TSR.png")
'''

'''
# --- Plot 6a: Influence of Tip Correction on Axial Induction ---
plt.figure(figsize=(9, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 0], 'b-', label='With correction')
plt.plot(res_nc[:, 2], res_nc[:, 0], 'r-', label='No correction')
plt.title('Influence of Prandtl tip/root correction on axial induction (TSR=8)')
plt.xlabel('r/R')
plt.ylabel('Axial induction factor $a$')
plt.grid(True)
plt.legend()
# Saving as 6a
save_and_show("6a_Induction_Correction_Influence.png")

# --- Plot 6b: Influence of Tip Correction on Normal Loading ---
plt.figure(figsize=(9, 5))
norm_val = 0.5 * U0**2 * Radius
plt.plot(results_tsr8[:, 2], results_tsr8[:, 3]/norm_val, 'b-', label='With correction')
plt.plot(res_nc[:, 2], res_nc[:, 3]/norm_val, 'r-', label='No correction')
plt.title('Influence of Prandtl tip/root correction on normal loading (TSR=8)')
plt.xlabel('r/R')
# Using double backslashes for LaTeX to avoid the parser error we saw earlier
plt.ylabel(r'$F_{norm} / (0.5 \rho U_{\infty}^2 R)$') 
plt.grid(True)
plt.legend()
# Saving as 6b
save_and_show("6b_Loading_Correction_Influence.png")
'''


'''
# --- Plot 5a: BEM Convergence History (Total Rotor CT) ---

plt.figure(figsize=(7, 5))
plt.plot(range(1, 201), ct_history_tsr8, 'b-', linewidth=2)
plt.xlim(1,100)
plt.title('Convergence history of total thrust coefficient ($C_T$) (TSR=8)')
plt.xlabel('Iteration step')
plt.ylabel('Total $C_T$')
plt.grid(True)
save_and_show("5_Convergence_History.png")

# --- Plot 5b: BEM Convergence Residuals (Log Scale) ---
plt.figure(figsize=(7, 5))

# 1. Calculate the absolute difference between consecutive iterations
# np.diff computes ct[i] - ct[i-1]
residuals = np.abs(np.diff(ct_history_tsr8))

# 2. Use semilogy for the log-y axis
# The x-axis starts at 2 because the first 'difference' is between step 1 and 2
plt.semilogy(range(2, 201), residuals, 'r-', linewidth=2, label='Residual $|C_{T,i} - C_{T,i-1}|$')
plt.xlim(1,100)
plt.title('Convergence residuals of total thrust coefficient (TSR=8)')
plt.xlabel('Iteration step')
plt.ylabel('Absolute difference in $C_T$')
plt.grid(True, which="both", ls="-", alpha=0.5) # 'both' shows major and minor log lines
plt.legend()

save_and_show("5_Convergence_Residuals_Log.png")
'''

'''
## ================================================================== ##
## Influence of Number of Annuli - Separate Plots (TSR=8)
## ================================================================== ##

# List of resolutions to test
N_values = [8, 20, 100]

for N in N_values:
    # 1. Create a fresh figure for each N
    plt.figure(figsize=(7, 5))
    
    # Define bins for this specific resolution
    r_R_bins_N = np.linspace(RootLocation_R, TipLocation_R, N + 1)
    results_N = []
    
    # 2. Run the BEM logic for each annulus
    for i in range(N):
        rm = (r_R_bins_N[i] + r_R_bins_N[i+1]) / 2
        chord = 3 * (1 - rm) + 1
        twist = -(14 * (1 - rm) + Pitch)
        
        # Solving using your SolveStreamtube function
        res, _ = SolveStreamtube(U0, r_R_bins_N[i], r_R_bins_N[i+1], RootLocation_R, 
                                 TipLocation_R, Omega_comp, Radius, NBlades, 
                                 chord, twist, polar_alpha, polar_cl, polar_cd)
        results_N.append(res)
    
    res_N_arr = np.array(results_N)
    
    # 3. Plotting logic
    # Use markers for N=10 and N=30 to show the discretization points
    fmt = '-'
    plt.plot(res_N_arr[:, 2], res_N_arr[:, 3]/(0.5*U0**2*Radius), fmt, label=f'Annuli = {N}')
    
    plt.title(f'Normal loading distribution ($C_n$) with {N} Annuli')
    plt.xlabel('r/R')
    # Using the "safe" double backslash for LaTeX labels
    plt.ylabel(r'$C_n = F_{norm} / (0.5 \rho U_{\infty}^2 R)$')
    plt.grid(True)
    plt.legend()
    
    # 4. Save each plot with a unique filename
    save_and_show(f"7_Annuli_Influence_N{N}.png")
'''

## ================================================================== ##
## Influence of Spacing Method (TSR=8, N=40)
## ================================================================== ##
N_fixed = 40
spacing_methods = {}

# Constant Spacing
spacing_methods['Constant'] = np.linspace(RootLocation_R, TipLocation_R, N_fixed + 1)

# Cosine Spacing (Clusters points at Tip and Root)
beta = np.linspace(0, np.pi, N_fixed + 1)
spacing_methods['Cosine'] = RootLocation_R + (TipLocation_R - RootLocation_R) * 0.5 * (1 - np.cos(beta))

plt.figure(figsize=(9, 5))

# Style mapping for clarity
line_styles = {'Constant': '-', 'Cosine': '--'}
colors = {'Constant': 'blue', 'Cosine': 'red'}

for label, bins in spacing_methods.items():
    results_spacing = []
    for i in range(N_fixed):
        rm = (bins[i] + bins[i+1]) / 2
        chord = 3 * (1 - rm) + 1
        twist = -(14 * (1 - rm) + Pitch)
        res, _ = SolveStreamtube(U0, bins[i], bins[i+1], RootLocation_R, TipLocation_R, 
                                 Omega_comp, Radius, NBlades, chord, twist, 
                                 polar_alpha, polar_cl, polar_cd)
        results_spacing.append(res)
    
    res_S_arr = np.array(results_spacing)
    
    # Apply custom styles here
    plt.plot(res_S_arr[:, 2], res_S_arr[:, 3]/(0.5*U0**2*Radius), 
             marker='o', 
             linestyle=line_styles[label], 
             color=colors[label],
             markersize=5, 
             linewidth=2,
             label=f'{label} Spacing')

plt.title('Comparison of spacing methods (N=40)')
plt.xlabel('r/R')
# Using the "safe" double backslash for LaTeX labels to avoid parser errors
plt.ylabel(r'$C_n = F_{norm} / (0.5 \rho U_{\infty}^2 R)$')
plt.grid(True)
plt.legend()

# Zoom in on the tip to see the difference clearly
plt.xlim(0.85, 1.01) 
plt.ylim(0.5,  1.5)
save_and_show("8_Spacing_Method_Comparison.png")