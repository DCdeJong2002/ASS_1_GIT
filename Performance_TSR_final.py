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
    # Define CT1 and CT2 for Glauert's correction
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1

    # Calculate induction factor a based on CT using Glauert's correction
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
    # Calculate tip correction factor Ftip
    temp_tip = -NBlades/2 * (tipradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - axial_induction)**2))
    Ftip = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_tip), 0, 1)))
    
    # Calculate root correction factor Froot
    temp_root = NBlades / 2 * (rootradius_R - r_R) / r_R * np.sqrt(1 + ((TSR * r_R)**2) / ((1 - axial_induction)**2))
    Froot = np.array(2 / np.pi * np.arccos(np.clip(np.exp(temp_root), 0, 1)))
    
    # Return total correction factor F
    return Froot * Ftip


def LoadBladeElement(vnorm, vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    Calculate the local forces (normal and tangential) and circulation on a blade element.
    """
    # Calculate velocities and flow angle
    vmag2 = vnorm**2 + vtan**2
    phi = np.arctan2(vnorm, vtan)
    
    # Determine angle of attack and interpolate airfoil coefficients
    alpha = twist + np.degrees(phi)
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    
    # Calculate forces and circulation
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
    # Calculate streamtube area and local radius
    Area = np.pi * ((r2_R * Radius)**2 - (r1_R * Radius)**2)
    r_mid = (r1_R + r2_R) / 2
    r_local = r_mid * Radius
    
    # Initial induction factors
    a, aline = 0.1, 0
    
    for i in range(200):
        # Local velocities at the rotor
        Urotor = U0 * (1 - a)
        Utan = (1 + aline) * Omega * r_local
        
        # Loads from the blade element
        fnorm, ftan, gamma, alpha, phi = LoadBladeElement(Urotor, Utan, chord, twist, polar_alpha, polar_cl, polar_cd)
        
        # Calculate local thrust coefficient (CT) for the streamtube
        CT = (fnorm * Radius * (r2_R - r1_R) * NBlades) / (0.5 * Area * U0**2)
        
        # Update axial induction with Glauert and Prandtl corrections
        anew = ainduction(CT)
        F = max(PrandtlTipRootCorrection(r_mid, rootradius_R, tipradius_R, Omega*Radius/U0, NBlades, anew), 0.0001)
        anew /= F
 
        a = 0.75 * a + 0.25 * anew
        
        # Update tangential induction factor
        aline = (ftan * NBlades) / (2 * np.pi * U0 * (1 - a) * Omega * 2 * r_local**2)
        aline /= F
        
    return [a, aline, r_mid, fnorm, ftan, gamma, alpha, phi]



## ================================================================== ##
## Solving for different TSRs and collecting results
## ================================================================== ##

results_tsr8 = []

print(f"{'TSR':<10} | {'CT':<10} | {'CP':<10}")
print("-" * 35)

for TSR in [6, 8, 10]:
    # Calculate rotational speed for TSR
    Omega = U0 * TSR / Radius

    temp_results = []
    for i in range(len(r_R_bins) - 1):
        r_mid = (r_R_bins[i] + r_R_bins[i+1]) / 2
        
        ## ========================================================== ##
        ## Definitions for chord and twist functions  
              
        chord_function = 3 * (1 - r_mid) + 1
        twist_function = -(14 * (1 - r_mid) + Pitch)

        ## ========================================================== ##

        # Solve streamtube and add results
        res = SolveStreamtube(U0, r_R_bins[i], r_R_bins[i+1], RootLocation_R, TipLocation_R, Omega, Radius, NBlades, chord_function, twist_function, polar_alpha, polar_cl, polar_cd)
        temp_results.append(res)
    
    res_arr = np.array(temp_results)
    dr = (r_R_bins[1:] - r_R_bins[:-1]) * Radius

    # Calculate global performance coefficients
    CT = np.sum(dr * res_arr[:, 3] * NBlades / (0.5 * U0**2 * np.pi * Radius**2))
    CP = np.sum(dr * res_arr[:, 4] * res_arr[:, 2] * NBlades * Radius * Omega / (0.5 * U0**3 * np.pi * Radius**2))

    print(f"{TSR:<10.1f} | {CT:<10.4f} | {CP:<10.4f}")
    
    if TSR == 8: 
        results_tsr8 = res_arr



## ================================================================== ##
## Plotting results for TSR=8
## ================================================================== ##

# Plot 1: Axial and tangential induction
plt.figure(figsize=(10, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 0], 'r-', label=r'$a$')
plt.plot(results_tsr8[:, 2], results_tsr8[:, 1], 'g--', label=r'$a^\prime$')
plt.title('Induction Factors along the Span (TSR=8)')
plt.xlabel('r/R'); plt.grid(); plt.legend(); plt.show()

# Plot 2: Normal and tangential force
plt.figure(figsize=(10, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 3]/(0.5*U0**2*Radius), 'r-', label=r'$F_{norm}$')
plt.plot(results_tsr8[:, 2], results_tsr8[:, 4]/(0.5*U0**2*Radius), 'g--', label=r'$F_{tan}$')
plt.title(r'Non-dimensioned Forces along the Span ($F / \frac{1}{2} \rho U_\infty^2 R$)')
plt.xlabel('r/R'); plt.grid(); plt.legend(); plt.show()

# Plot 3: Circulation
plt.figure(figsize=(10, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 5]/(np.pi*U0**2/(NBlades*(U0*8/Radius))), 'b-', label=r'$\Gamma$')
plt.title('Non-dimensioned Circulation Distribution')
plt.xlabel('r/R'); plt.grid(); plt.legend(); plt.show()

# Plot 4: Angle of Attack and inflow angle
plt.figure(figsize=(10, 5))
plt.plot(results_tsr8[:, 2], results_tsr8[:, 6], 'k-', label=r'$\alpha$')
plt.plot(results_tsr8[:, 2], results_tsr8[:, 7], 'm--', label=r'$\phi$')
plt.title('Spanwise Angle of Attack and Inflow Angle (deg)')
plt.xlabel('r/R'); plt.grid(); plt.legend(); plt.show()