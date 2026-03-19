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

for TSR in [6, 8, 10]:
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

plt.figure(figsize=(9, 5))
plt.plot(tsr_plot_list, ct_plot_list, 'bo-', label=r'Total thrust coefficient ($C_T$)')
plt.plot(tsr_plot_list, cq_plot_list, 'ro-', label=r'Total torque coefficient ($C_Q$)')
plt.title('Total Thrust and Torque Coefficients vs. Tip-Speed Ratio')
plt.xlabel('Tip-Speed Ratio (TSR)')
plt.ylabel('Coefficient')
plt.grid(True)
plt.legend()
save_and_show("4_Total_Performance_vs_TSR.png")

# Plot 5: BEM Convergence History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('BEM Solver Convergence for Total Thrust (TSR=8)', fontsize=14)

# Absolute CT Plot
ax1.plot(range(200), ct_history_tsr8, 'b-', linewidth=2)
ax1.set_title(r'Total $C_T$ vs Iterations')
ax1.set_xlabel('Iteration Step')
ax1.set_ylabel(r'Total $C_T$')
ax1.grid(True)

# Difference in CT Plot (Log Scale)
diff_CT = np.abs(np.diff(ct_history_tsr8))
ax2.semilogy(range(1, 200), diff_CT, 'r-', linewidth=2)
ax2.set_title(r'Absolute Difference ($\Delta C_T$)')
ax2.set_xlabel('Iteration Step')
ax2.set_ylabel(r'$|C_{T_{i}} - C_{T_{i-1}}|$')
ax2.grid(True)

plt.tight_layout()

plt.show()