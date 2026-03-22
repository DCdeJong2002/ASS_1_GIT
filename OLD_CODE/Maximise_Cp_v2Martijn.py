import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Airfoil Data ---
data = pd.read_excel('polar DU95W180 (3).xlsx', skiprows=3)
p_alpha, p_cl, p_cd = data['Alfa'], data['Cl'], data['Cd']

# Find aerodynamic optimum (max Cl/Cd)
glide_ratio = p_cl / p_cd
alpha_opt = p_alpha[np.argmax(glide_ratio)]

## ================================================================== ##
## Robust BEM Functions
## ================================================================== ##

Radius, NBlades, U0, rho = 50.0, 3, 10.0, 1.225
Root_R, Tip_R = 0.2, 1.0
delta_r_R = 0.005
r_R_bins = np.arange(Root_R, Tip_R + delta_r_R/2, delta_r_R)

def ainduction(CT):
    """Axial induction with Glauert's correction and stability clipping."""
    CT1, CT2 = 1.816, 2 * np.sqrt(1.816) - 1.816
    if np.isscalar(CT):
        if CT >= CT2: 
            val = 1 + (CT - CT1) / (4 * (np.sqrt(1.816) - 1))
        else:
            val = 0.5 - 0.5 * np.sqrt(max(0, 1 - CT))
        return np.clip(val, -0.5, 0.95) 
    else:
        a = np.zeros(np.shape(CT))
        a[CT >= CT2] = 1 + (CT[CT >= CT2] - CT1) / (4 * (np.sqrt(1.816) - 1))
        a[CT < CT2] = 0.5 - 0.5 * np.sqrt(np.maximum(0, 1 - CT[CT < CT2]))
        return np.clip(a, -0.5, 0.95)

def Prandtl(r_R, TSR, a):
    """Prandtl correction with corrected root sign and stable exponents."""
    ai = np.clip(a, -1, 0.95)
    sqrt_term = np.sqrt(1 + ((TSR * r_R)**2) / ((1 - ai)**2))
    # Tip: exponent must be negative
    f_t = -NBlades/2 * (1.0 - r_R) / r_R * sqrt_term
    F_t = 2 / np.pi * np.arccos(np.clip(np.exp(f_t), 0, 1))
    # Root: exponent must be negative
    f_r = -NBlades/2 * (r_R - 0.2) / 0.2 * sqrt_term
    F_r = 2 / np.pi * np.arccos(np.clip(np.exp(f_r), 0, 1))
    return max(F_t * F_r, 0.01)

def RunBEM(c_slope, c_const, t_slope, pitch, TSR_val):
    """The core solver for performance calculation."""
    Omega = U0 * TSR_val / Radius
    temp_res = []
    for i in range(len(r_R_bins) - 1):
        rm = (r_R_bins[i] + r_R_bins[i+1]) / 2
        chord = c_slope * (1 - rm) + c_const
        twist = -(t_slope * (1 - rm) + pitch)
        
        a, ap = 0.2, 0.01
        for _ in range(150):
            a = np.clip(a, -0.5, 0.95)
            phi = np.arctan2(U0*(1-a), (1+ap)*Omega*rm*Radius)
            alpha = twist + np.degrees(phi)
            cl = np.interp(alpha, p_alpha, p_cl)
            cd = np.interp(alpha, p_alpha, p_cd)
            
            v2 = (U0*(1-a))**2 + ((1+ap)*Omega*rm*Radius)**2
            fn = 0.5 * v2 * chord * (cl * np.cos(phi) + cd * np.sin(phi))
            ft = 0.5 * v2 * chord * (cl * np.sin(phi) - cd * np.cos(phi))
            
            Area = np.pi * ((r_R_bins[i+1]*Radius)**2 - (r_R_bins[i]*Radius)**2)
            CT_loc = (fn * Radius * delta_r_R * NBlades) / (0.5 * Area * U0**2)
            
            anew = ainduction(CT_loc)
            F = Prandtl(rm, TSR_val, anew)
            anew /= F
            
            if abs(a - anew) < 1e-6: break
            a = 0.8 * a + 0.2 * anew
            ap_new = (ft * NBlades) / (2 * np.pi * U0 * (1 - a) * Omega * 2 * (rm * Radius)**2 * F)
            ap = 0.8 * ap + 0.2 * np.clip(ap_new, -0.5, 0.5)
            
        # Added phi and alpha to the return array for plotting
        temp_res.append([a, ap, rm, fn, ft, np.degrees(phi), alpha])
    
    res = np.array(temp_res)
    dr = delta_r_R * Radius
    CT_final = np.sum(dr * res[:, 3] * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP_final = np.sum(dr * res[:, 4] * res[:, 2] * NBlades * Radius * Omega / (0.5 * U0**3 * np.pi * Radius**2))
    return CT_final, CP_final, res

## ================================================================== ##
## Pitch-Sweep Optimization
## ================================================================== ##

# 1. Baseline initialization
r_ref = np.linspace(0.2, 1.0, 50)
c_ideal, t_ideal = [], []
for rr in r_ref:
    phi = np.arctan((1 - 0.25) / (8.0 * rr))
    F_guess = Prandtl(rr, 8.0, 0.25)
    cl_val = np.interp(alpha_opt, p_alpha, p_cl)
    c = (8 * np.pi * rr * Radius * F_guess * 0.25 * np.sin(phi)**2) / (NBlades * (0.75)**2 * cl_val * np.cos(phi))
    c_ideal.append(c)
    t_ideal.append(np.degrees(phi) - alpha_opt)

cs_i, cc_i = np.polyfit(1 - r_ref, c_ideal, 1)
ts_i, p_i = np.polyfit(1 - r_ref, t_ideal, 1)

# 2. Iterative optimization
best_CP = 0
final_geom = (cs_i, cc_i, ts_i, p_i)

cp_convergence = [] # Tracker for plotting

print("Optimizing pitch and scaling chord for CT=0.75...")
for test_pitch in np.arange(p_i - 2.5, p_i + 2.5, 0.25):
    curr_cs, curr_cc = cs_i, cc_i
    for _ in range(8):
        ct, cp, _ = RunBEM(curr_cs, curr_cc, ts_i, test_pitch, 8.0)
        if ct > 0: # Avoid division by zero
            curr_cs *= (0.75 / ct)
            curr_cc *= (0.75 / ct)
    
    cp_convergence.append(cp) # Log the verified CP for this pitch step
    
    if cp > best_CP:
        best_CP = cp
        final_geom = (curr_cs, curr_cc, ts_i, test_pitch)

f_cs, f_cc, f_ts, f_p = final_geom

## ================================================================== ##
## Final Multi-TSR Table Output
## ================================================================== ##

print("\n" + "-"*40)
print(f"{'TSR':<10} | {'CT':<10} | {'CP':<10}")
print("-" * 40)

for t in [6.0, 8.0, 10.0]:
    ct, cp, _ = RunBEM(f_cs, f_cc, f_ts, f_p, t)
    print(f"{t:<10.1f} | {ct:<10.4f} | {cp:<10.4f}")

print("-" * 40)
print(f"Final Twist: {f_ts:.4f}*(1-r/R)")
print(f"Final Pitch: {f_p:.4f} deg")
print(f"Final Chord: ({f_cs:.4f}*(1-r/R) + {f_cc:.4f}) m")

## ================================================================== ##
## Plotting
## ================================================================== ##

# Get data for the optimized geometry at the design TSR (8.0)
_, _, res_opt = RunBEM(f_cs, f_cc, f_ts, f_p, 8.0)

# Unpack results array
a_arr = res_opt[:, 0]
ap_arr = res_opt[:, 1]
rm_arr = res_opt[:, 2]
fn_arr = res_opt[:, 3]
ft_arr = res_opt[:, 4]
phi_arr = res_opt[:, 5]
alpha_arr = res_opt[:, 6]

# Calculate geometric distributions
chord_arr = f_cs * (1 - rm_arr) + f_cc
twist_arr = -(f_ts * (1 - rm_arr) + f_p)

# Create figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('BEM Optimization Results (TSR = 8.0)', fontsize=16, fontweight='bold')

# Plot 1: CP Convergence
axs[0, 0].plot(range(1, len(cp_convergence) + 1), cp_convergence, 'bo-', linewidth=2)
axs[0, 0].set_title('CP Convergence during Pitch Sweep')
axs[0, 0].set_xlabel('Pitch Iteration')
axs[0, 0].set_ylabel('Power Coefficient (CP)')
axs[0, 0].grid(True)

# Plot 2: Angles (Alpha and Phi)
axs[0, 1].plot(rm_arr, phi_arr, 'g-', label='Inflow Angle ($\phi$)', linewidth=2)
axs[0, 1].plot(rm_arr, alpha_arr, 'm-', label='Angle of Attack ($\\alpha$)', linewidth=2)
axs[0, 1].axhline(alpha_opt, color='r', linestyle='--', label='$\\alpha_{opt}$ (Target)')
axs[0, 1].set_title('Aerodynamic Angles vs r/R')
axs[0, 1].set_xlabel('r/R')
axs[0, 1].set_ylabel('Angle [degrees]')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: Geometry - Twist
axs[1, 0].plot(rm_arr, twist_arr, 'c-', linewidth=2)
axs[1, 0].set_title('Twist Distribution vs r/R')
axs[1, 0].set_xlabel('r/R')
axs[1, 0].set_ylabel('Twist [degrees]')
axs[1, 0].grid(True)

# Plot 4: Geometry - Chord
axs[1, 1].plot(rm_arr, chord_arr, 'k-', linewidth=2)
axs[1, 1].set_title('Chord Distribution vs r/R')
axs[1, 1].set_xlabel('r/R')
axs[1, 1].set_ylabel('Chord [m]')
axs[1, 1].grid(True)

# Plot 5: Induction Factors
axs[2, 0].plot(rm_arr, a_arr, 'b-', label='Axial Induction ($a$)', linewidth=2)
axs[2, 0].plot(rm_arr, ap_arr, 'r-', label="Tangential Induction ($a'$)", linewidth=2)
axs[2, 0].set_title('Induction Factors vs r/R')
axs[2, 0].set_xlabel('r/R')
axs[2, 0].set_ylabel('Induction Factor')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Plot 6: Sectional Forces
axs[2, 1].plot(rm_arr, fn_arr, 'b-', label='Normal Force ($F_n$)', linewidth=2)
axs[2, 1].plot(rm_arr, ft_arr, 'r-', label='Tangential Force ($F_t$)', linewidth=2)
axs[2, 1].set_title('Sectional Forces vs r/R')
axs[2, 1].set_xlabel('r/R')
axs[2, 1].set_ylabel('Force [N/m]')
axs[2, 1].legend()
axs[2, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()