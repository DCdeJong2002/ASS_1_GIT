import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
data = pd.read_excel('polar DU95W180 (3).xlsx', skiprows=3)
polar_alpha, polar_cl, polar_cd = data['Alfa'], data['Cl'], data['Cd']



## ================================================================== ##
## Previous chord, twist, pitch specs
## ================================================================== ##

def InitialChordFunction(r_R):
    return 3 * (1 - r_R) + 1

def InitialTwistFunction(r_R, pitch):
    return -(14 * (1 - r_R) + pitch)

pitch_initial = -2



# Find aerodynamic optimum (max Cl/Cd) for the blade design
glide_ratio = polar_cl / polar_cd
alpha_opt = polar_alpha[np.argmax(glide_ratio)]

## ================================================================== ##
## Specifications & Core BEM Functions
## ================================================================== ##

Radius = 50.0      # (m)
NBlades = 3        # Number of blades
U0 = 10.0          # Freestream velocity (m/s)
rho = 1.225        # Air density
Root_R, Tip_R = 0.2, 1.0 
delta_r_R = 0.005  # Resolution
r_R_bins = np.arange(Root_R, Tip_R + delta_r_R/2, delta_r_R)

def ainduction(CT):
    """Axial induction factor with Glauert's correction."""
    CT1, CT2 = 1.816, 2 * np.sqrt(1.816) - 1.816
    if np.isscalar(CT):
        if CT >= CT2: return 1 + (CT - CT1) / (4 * (np.sqrt(1.816) - 1))
        return 0.5 - 0.5 * np.sqrt(max(0, 1 - CT))
    else:
        a = np.zeros(np.shape(CT))
        a[CT >= CT2] = 1 + (CT[CT >= CT2] - CT1) / (4 * (np.sqrt(1.816) - 1))
        a[CT < CT2] = 0.5 - 0.5 * np.sqrt(np.maximum(0, 1 - CT[CT < CT2]))
        return a

def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    ai = np.clip(axial_induction, -1, 0.999)
    # Tip and Root factors
    f_t = -NBlades/2 * (tipradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - ai)**2))
    F_t = 2 / np.pi * np.arccos(np.clip(np.exp(f_t), 0, 1))
    f_r = NBlades/2 * (rootradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - ai)**2))
    F_r = 2 / np.pi * np.arccos(np.clip(np.exp(f_r), 0, 1))
    return F_t * F_r

def LoadBladeElement(vnorm, vtan, chord, twist, p_a, p_cl, p_cd):
    vmag2 = vnorm**2 + vtan**2
    phi = np.arctan2(vnorm, vtan)
    alpha = twist + np.degrees(phi)
    cl, cd = np.interp(alpha, p_a, p_cl), np.interp(alpha, p_a, p_cd)
    fnorm = 0.5 * vmag2 * chord * (cl * np.cos(phi) + cd * np.sin(phi))
    ftan = 0.5 * vmag2 * chord * (cl * np.sin(phi) - cd * np.cos(phi))
    gamma = 0.5 * np.sqrt(vmag2) * cl * chord
    return fnorm, ftan, gamma, alpha, np.degrees(phi)

def SolveStreamtube(U0, r1_R, r2_R, root_R, tip_R, Omega, Radius, NBlades, chord, twist, p_a, p_cl, p_cd):
    Area = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_mid = (r1_R + r2_R) / 2
    a, aline = 0.25, 0.0 # Start closer to target induction
    for i in range(250):
        fn, ft, gam, alp, phi = LoadBladeElement(U0*(1-a), (1+aline)*Omega*r_mid*Radius, chord, twist, p_a, p_cl, p_cd)
        CT_loc = (fn * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0**2)
        anew = ainduction(CT_loc)
        F = max(PrandtlTipRootCorrection(r_mid, root_R, tip_R, Omega*Radius/U0, NBlades, anew), 0.0001)
        anew /= F
        if abs(a - anew) < 1e-6: break
        a = 0.75 * a + 0.25 * anew
        aline = (ft * NBlades) / (2 * np.pi * U0 * (1 - a) * Omega * 2 * (r_mid * Radius)**2 * F)
    return [a, aline, r_mid, fn, ft, gam, alp, phi]

## ================================================================== ##
## Master Performance Loop (The Ground Truth)
## ================================================================== ##

def RunFullSimulation(c_slope, c_const, t_slope, pitch, TSR_val):
    Omega = U0 * TSR_val / Radius
    temp_res = []
    for i in range(len(r_R_bins) - 1):
        rm = (r_R_bins[i] + r_R_bins[i+1]) / 2
        chord = c_slope * (1 - rm) + c_const
        twist = -(t_slope * (1 - rm) + pitch)
        res = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1], Root_R, Tip_R, Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd)
        temp_res.append(res)
    
    res_arr = np.array(temp_res)
    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius
    # This is the user's "Correct" CT formula
    CT_final = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP_final = np.sum(dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega / (0.5 * U0**3 * np.pi * Radius**2))
    return CT_final, CP_final, res_arr

## ================================================================== ##
## Design Optimization & Precise Scaling
## ================================================================== ##

# 1. Generate Ideal Baseline (a=0.25)
r_range = np.linspace(0.2, 1.0, 50)
c_req, t_req = [], []
for rr in r_range:
    phi = np.arctan2(U0 * 0.75, 8.0 * U0 * rr)
    F = max(PrandtlTipRootCorrection(rr, Root_R, 1.0, 8.0, NBlades, 0.25), 0.01)
    Cx = np.interp(alpha_opt, polar_alpha, polar_cl) * np.cos(phi)
    c = (8 * np.pi * rr * Radius * F * 0.25 * np.sin(phi)**2) / (NBlades * (1-0.25)**2 * Cx)
    c_req.append(c)
    t_req.append(np.degrees(phi) - alpha_opt)

c_slope, c_const = np.polyfit(1 - r_range, c_req, 1)
t_slope, pitch = np.polyfit(1 - r_range, t_req, 1)

# 2. Scaling Loop: Adjust geometry until "Correct CT" hits 0.7500
print("Scaling chord to reach exact CT target...")
for i in range(15):
    CT_current, _, _ = RunFullSimulation(c_slope, c_const, t_slope, pitch, 8.0)
    if abs(CT_current - 0.75) < 0.00005: break
    # Scale chord based on the deficit
    scale = 0.75 / CT_current
    c_slope *= scale
    c_const *= scale

## ================================================================== ##
## Final Multi-TSR Results
## ================================================================== ##

print("\n" + "-"*40)
print(f"{'TSR':<10} | {'CT':<10} | {'CP':<10}")
print("-" * 40)

results_tsr8 = None
for tsr_choice in [6, 8, 10]:
    ct_val, cp_val, res_data = RunFullSimulation(c_slope, c_const, t_slope, pitch, tsr_choice)
    print(f"{tsr_choice:<10.1f} | {ct_val:<10.4f} | {cp_val:<10.4f}")
    if tsr_choice == 8: results_tsr8 = res_data
    if tsr_choice == 8: ct_val_tsr8 = ct_val
    if tsr_choice == 8: cp_val_tsr8 = cp_val

print("-" * 40)
print(f"Final Twist: {t_slope:.4f}*(1-r/R)")
print(f"Final Pitch: {pitch:.4f} deg")
print(f"Final Chord: ({c_slope:.4f}*(1-r/R) + {c_const:.4f}) m")



## ================================================================== ##
## Baseline Calculation & Comparison Plots
## ================================================================== ##

# 1. Define and Run the "Old" (Baseline) Simulation
c_slope_old, c_const_old = 3.0, 1.0
t_slope_old, pitch_old = 14.0, -2.0

ct_old, cp_old, results_old = RunFullSimulation(c_slope_old, c_const_old, t_slope_old, pitch_old, 8.0)

# 2. Create Plotting Dashboard
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Comparison: Baseline vs. Optimized Blade (TSR=8)\n'
             f'Old: CP={ct_old:.4f}, CT={cp_old:.4f} | New: CP={cp_val_tsr8:.4f}, CT={ct_val_tsr8:.4f}', fontsize=16)

r_R = results_tsr8[:, 2]

# --- Plot 1: Geometry (Chord and Twist) ---
ax1 = axs[0, 0]
ax1.set_title('Blade Geometry')
# Chord
ax1.plot(r_R, c_slope_old*(1-r_R)+c_const_old, 'k--', label='Old Chord [m]')
ax1.plot(r_R, c_slope*(1-r_R)+c_const, 'b-', label='New Chord [m]')
ax1.set_ylabel('Chord [m]', color='b')
ax1.tick_params(axis='y', labelcolor='b')
# Twist on twin axis
ax1b = ax1.twinx()
ax1b.plot(r_R, t_slope_old*(1-r_R)+pitch_old, 'k:', label='Old Twist [deg]')
ax1b.plot(r_R, t_slope*(1-r_R)+pitch, 'r-', label='New Twist [deg]')
ax1b.set_ylabel('Twist [deg]', color='r')
ax1b.tick_params(axis='y', labelcolor='r')
ax1.grid(True)
ax1.legend(loc='upper right', fontsize='small')

# --- Plot 2: Induction Factors (a and a') ---
ax2 = axs[0, 1]
ax2.set_title('Induction Factors')
ax2.plot(r_R, results_old[:, 0], 'k--', label='Old $a$')
ax2.plot(r_R, results_tsr8[:, 0], 'r-', label='New $a$ (Target 0.25)')
ax2.plot(r_R, results_old[:, 1], 'k:', label="Old $a'$")
ax2.plot(r_R, results_tsr8[:, 1], 'g-', label="New $a'$")
ax2.set_ylabel('Induction Factor')
ax2.grid(True)
ax2.legend()

# --- Plot 3: Aerodynamic Angles (Alpha and Phi) ---
ax3 = axs[1, 0]
ax3.set_title('Angles of Attack and Inflow')
ax3.plot(r_R, results_old[:, 6], 'k--', label=r'Old $\alpha$')
ax3.plot(r_R, results_tsr8[:, 6], 'b-', label=r'New $\alpha$ (Target $\alpha_{opt}$)')
ax3.plot(r_R, results_old[:, 7], 'k:', label=r'Old $\phi$')
ax3.plot(r_R, results_tsr8[:, 7], 'm-', label=r'New $\phi$')
ax3.set_ylabel('Angle [deg]')
ax3.grid(True)
ax3.legend()

# --- Plot 4: Sectional Loads (Fnorm and Ftan) ---
ax4 = axs[1, 1]
ax4.set_title('Sectional Loads (Non-dimensioned)')
# Normalize forces: F / (0.5 * rho * U0^2 * R)
norm = 0.5 * rho * U0**2 * Radius
ax4.plot(r_R, results_old[:, 3]/norm, 'k--', label='Old $F_{norm}$')
ax4.plot(r_R, results_tsr8[:, 3]/norm, 'r-', label='New $F_{norm}$')
ax4.plot(r_R, results_old[:, 4]/norm, 'k:', label='Old $F_{tan}$')
ax4.plot(r_R, results_tsr8[:, 4]/norm, 'g-', label='New $F_{tan}$')
ax4.set_ylabel(r'Force Coefficient $F / (\frac{1}{2} \rho U_0^2 R)$')
ax4.grid(True)
ax4.legend()

for ax in axs.flat:
    ax.set_xlabel('r/R')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()